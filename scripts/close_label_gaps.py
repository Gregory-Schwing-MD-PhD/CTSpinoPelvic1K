"""scripts/close_label_gaps.py — rejoin a bone the segmenter split, where the gap is a seam.

WHAT IS LEFT AFTER RENUMBERING. fix_label_breaks moves a piece to the bone it is touching,
which handled the boundary errors. What remains is a bone genuinely in two parts under one
correct label: a vertebra whose lamina lost contact with its body, a sacrum split across a
segment. The label is right. It is simply not connected.

WHY THAT MATTERS FOR A RELEASE. "Every structure is one connected component" is a property a
user can rely on -- to take a largest-component, to compute a length, to mesh a surface. If
it is nearly true, every downstream tool has to carry a special case, and most will not: the
rib-length measurement in this repository already takes the largest component and would
silently report half a rib.

WHAT THIS WILL AND WILL NOT DO. A gap of one or two millimetres between two parts of one
bone is a seam the segmenter dropped, and it is bridged: the voxels lying between the two
parts, closer to both than the bridge is wide, become the label. Anything wider is left
alone and reported. Bridging five millimetres would be inventing bone, and on a spine-limited
scan some of those gaps are places the scanner genuinely saw nothing.

THE BRIDGE IS THIN BY CONSTRUCTION. Only background voxels are taken, never another
structure's, and only those within the gap corridor -- so it cannot fatten a bone or eat a
neighbour.

    python scripts/close_label_gaps.py --dry-run
    python scripts/close_label_gaps.py --apply --max-gap-mm 2.5
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
MIN_PIECE = 400

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"


def one(args):
    path, apply_it, backup, max_gap = args
    case = Path(path).name.split("_")[0]
    rows = []
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
    except Exception as e:                                     # noqa: BLE001
        return [{"case": case, "label": "READ_ERROR", "piece_vox": 0, "gap_mm": "",
                 "bridge_vox": 0, "action": "error", "note": str(e)[:60]}]
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
            pad = int(np.ceil(max_gap / z[a])) + 3
            span.append(slice(max(0, int(hit[0]) - pad),
                              min(m.shape[a], int(hit[-1]) + pad + 1)))
        if len(span) != 3:
            continue
        sl = tuple(span)
        sub = m[sl]
        lt, n = ndimage.label(sub)
        if n < 2:
            continue
        sizes = ndimage.sum(sub, lt, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        core = lt == big
        d_core = ndimage.distance_transform_edt(~core, sampling=z)
        crop = out[sl]

        for k in range(1, n + 1):
            if k == big or sizes[k - 1] < MIN_PIECE:
                continue
            piece = lt == k
            gap = float(d_core[piece].min())
            row = {"case": case, "label": NAME.get(v, str(v)),
                   "piece_vox": int(sizes[k - 1]), "gap_mm": round(gap, 1),
                   "bridge_vox": 0, "action": "", "note": ""}
            if gap > max_gap:
                row["action"] = "left_open"
                row["note"] = f"gap wider than {max_gap} mm; bridging it would invent bone"
                rows.append(row)
                continue
            # THE CORRIDOR BETWEEN THE TWO PARTS: background voxels whose combined distance
            # to each part barely exceeds the gap itself. Anything off to the side has a
            # larger sum and is excluded, so the bridge is a seam and not a blob.
            d_piece = ndimage.distance_transform_edt(~piece, sampling=z)
            corridor = (d_core + d_piece) <= (gap + max(z) * 1.2)
            bridge = corridor & (crop == 0)
            nb = int(bridge.sum())
            if not nb:
                row["action"] = "left_open"
                row["note"] = "no free corridor between the parts"
                rows.append(row)
                continue
            row["action"] = "bridged"
            row["bridge_vox"] = nb
            if apply_it:
                crop[bridge] = v
                out[sl] = crop
                touched = True
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
    ap.add_argument("--cases", default=None,
                    help="csv with a `case` column; default is every volume")
    ap.add_argument("--out", default="qc_final/gap_closures.csv")
    ap.add_argument("--backup", default="data/v5_pre_gap_backup")
    ap.add_argument("--max-gap-mm", type=float, default=2.5)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    if a.apply == a.dry_run:
        print("  ! choose exactly one of --apply or --dry-run")
        return 2

    want = None
    if a.cases and Path(a.cases).exists():
        want = {r["case"] for r in csv.DictReader(open(a.cases, encoding="utf-8"))}
    files = [p for p in sorted(Path(a.labels).glob("*_label.nii.gz"))
             if want is None or p.name.split("_")[0] in want]
    if not files:
        print("  ! nothing to do")
        return 1
    print(f"  {len(files)} volume(s); bridging gaps up to {a.max_gap_mm} mm")

    cols = ["case", "label", "piece_vox", "gap_mm", "bridge_vox", "action", "note"]
    tally, done, vox = {}, 0, 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        payload = [(str(f), a.apply, a.backup, a.max_gap_mm) for f in files]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for rows in ex.map(one, payload, chunksize=1):
                for r in rows:
                    tally[r["action"]] = tally.get(r["action"], 0) + 1
                    vox += int(r["bridge_vox"])
                w.writerows(rows)
                done += 1
                if done % 20 == 0:
                    fh.flush()
                    print(f"  {done}/{len(files)}", flush=True)

    print(f"\n  {'APPLIED' if a.apply else 'DRY RUN'} over {done} volume(s)")
    print(f"    parts rejoined      : {tally.get('bridged', 0)}")
    print(f"    left open           : {tally.get('left_open', 0)}")
    print(f"    voxels added        : {vox}")
    print(f"  report: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
