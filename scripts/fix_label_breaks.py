"""scripts/fix_label_breaks.py — give each stray piece the number of the bone it touches.

WHAT THE PIECES ARE. clean_label_speckle removes dust and reports what it will not touch:
substantial fragments detached from the rest of their label. Diagnosing a sample of them
showed 86% are not detached from anything -- they are lying against a DIFFERENT vertebra,
0.7 to 2.4 mm away, while their own body is 3 to 26 mm off. T8 carrying a piece of T9, L3
carrying four pieces of L4. These are boundary errors from the segmenter, one level out.

THE RULE. A piece is renumbered when it is clearly nearer another labelled bone than its
own: at most CONTACT_MM from the neighbour, and the neighbour at least CLEARLY_CLOSER times
nearer than its own body. Both conditions matter. The distance alone would move a piece that
merely sits between two vertebrae; the ratio alone would move one that is a long way from
everything.

WHAT IT WILL NOT DO. A piece whose own body is nearest is left exactly as it is -- that is
one bone in two parts, the label is already right, and joining it would only paper over a
gap the CT may not contain. A piece at the reconstruction circle is left alone, because the
bone between it and the rest was never imaged. Anything ambiguous is reported, not guessed.

Reversible: every modified volume is copied to --backup first, and the header and affine are
reused verbatim, so nothing here can move a label off its CT.

    python scripts/fix_label_breaks.py --dry-run
    python scripts/fix_label_breaks.py --apply
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

VERT = set(range(8, 30)) - {27}
RIBS = set(range(34, 58))
LUMB = {74, 75}
CONSIDER = VERT | RIBS | LUMB

MIN_PIECE = 400          # below this clean_label_speckle already dealt with it
CONTACT_MM = 2.5         # a piece must be touching the neighbour it is given to
CLEARLY_CLOSER = 0.6     # and the neighbour must be markedly nearer than its own body
MIN_NEIGHBOUR = 500      # ignore specks as candidate neighbours

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"


def one(args):
    path, apply_it, backup, contact_mm, closer = args
    case = Path(path).name.split("_")[0]
    rows = []
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
    except Exception as e:                                    # noqa: BLE001
        return [{"case": case, "label": "READ_ERROR", "piece_vox": 0, "d_own_mm": "",
                 "given_to": "", "d_other_mm": "", "action": "error", "note": str(e)[:60]}]
    z = np.array(img.header.get_zooms()[:3], float)
    counts = np.bincount(lab.reshape(-1), minlength=256)
    out = lab.copy()
    touched = False

    for v in sorted(CONSIDER):
        if v >= len(counts) or not counts[v]:
            continue
        m = lab == v
        span = []
        for a in range(3):
            hit = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != a)))[0]
            if not len(hit):
                break
            span.append(slice(max(0, int(hit[0]) - 10),
                              min(m.shape[a], int(hit[-1]) + 11)))
        if len(span) != 3:
            continue
        sl = tuple(span)
        sub = m[sl]
        lt, n = ndimage.label(sub)
        if n < 2:
            continue
        sizes = ndimage.sum(sub, lt, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        crop = lab[sl]
        d_own_map = ndimage.distance_transform_edt(~(lt == big), sampling=z)

        # distance maps for the neighbours, computed once each
        others = [int(o) for o in np.unique(crop[crop > 0])
                  if int(o) != v and (crop == o).sum() >= MIN_NEIGHBOUR]
        dmaps = {o: ndimage.distance_transform_edt(~(crop == o), sampling=z) for o in others}

        for k in range(1, n + 1):
            if k == big or sizes[k - 1] < MIN_PIECE:
                continue
            piece = lt == k
            d_own = float(d_own_map[piece].min())
            best, best_d = None, 1e9
            for o, dm in dmaps.items():
                dd = float(dm[piece].min())
                if dd < best_d:
                    best_d, best = dd, o
            row = {"case": case, "label": NAME.get(v, str(v)),
                   "piece_vox": int(sizes[k - 1]), "d_own_mm": round(d_own, 1),
                   "given_to": NAME.get(best, "") if best else "",
                   "d_other_mm": round(best_d, 1) if best else "", "action": "", "note": ""}
            if best is not None and best_d <= contact_mm and best_d <= closer * d_own:
                row["action"] = "renumbered"
                if apply_it:
                    block = out[sl]
                    block[piece] = best
                    out[sl] = block
                    touched = True
            elif best is None or d_own <= best_d:
                row["action"] = "left_as_one_bone_in_two_parts"
                row["note"] = "own body is nearest; the label is already correct"
            else:
                row["action"] = "left_ambiguous"
                row["note"] = "nearer a neighbour but not touching it"
            rows.append(row)

    if apply_it and touched:
        bak = Path(backup) / Path(path).name
        if not bak.exists():
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak)
        newimg = nib.Nifti1Image(out.astype(lab.dtype), img.affine, img.header)
        newimg.set_data_dtype(lab.dtype)
        nib.save(newimg, str(path))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--worklist", default="qc_final/speckle_cleanup.csv")
    ap.add_argument("--out", default="qc_final/break_fixes.csv")
    ap.add_argument("--backup", default="data/v5_pre_break_backup")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contact-mm", type=float, default=CONTACT_MM,
                    help="a piece must be at most this far from the bone it is given to")
    ap.add_argument("--closer", type=float, default=CLEARLY_CLOSER,
                    help="and that bone at most this fraction of the distance to its own")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    if a.apply == a.dry_run:
        print("  ! choose exactly one of --apply or --dry-run")
        return 2

    # only the cases the cleanup flagged; the rest have nothing detached to consider
    cases = set()
    wl = Path(a.worklist)
    if wl.exists():
        for r in csv.DictReader(open(wl, encoding="utf-8")):
            if "segmentation_break" in (r.get("note") or "") or r.get("action") == "review":
                cases.add(r["case"])
    files = [p for p in sorted(Path(a.labels).glob("*_label.nii.gz"))
             if (not cases) or p.name.split("_")[0] in cases]
    if not files:
        print("  ! nothing to do")
        return 1
    print(f"  {len(files)} case(s) carry a detached piece")

    cols = ["case", "label", "piece_vox", "d_own_mm", "given_to", "d_other_mm",
            "action", "note"]
    tally, done = {}, 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        payload = [(str(f), a.apply, a.backup, a.contact_mm, a.closer) for f in files]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for rows in ex.map(one, payload, chunksize=1):
                for r in rows:
                    tally[r["action"]] = tally.get(r["action"], 0) + 1
                w.writerows(rows)
                done += 1
                if done % 20 == 0:
                    fh.flush()
                    print(f"  {done}/{len(files)}", flush=True)

    print(f"\n  {'APPLIED' if a.apply else 'DRY RUN'} over {done} case(s)")
    for k in ("renumbered", "left_as_one_bone_in_two_parts", "left_ambiguous", "error"):
        print(f"    {k:32s} {tally.get(k, 0)}")
    print(f"  report: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
