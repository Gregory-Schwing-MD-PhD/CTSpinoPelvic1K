"""scripts/qc_version_progression.py — did each release actually improve on the last?

WHY THIS EXISTS. A dataset article that says "v3, then v4, then v5" is describing effort,
not quality. The reader has no way to tell whether the later versions are better or merely
later. Running the SAME checks against every version turns the version history into a
measurement, and it is the only honest way to claim the corpus improved.

The checks are chosen so that each one is a property a *release* should have, and so that
each is answerable without ground truth the corpus does not have:

  structural       every label id is in the published scheme; no strays.
  sidedness        left ids on the left, right ids on the right, with the axis read from
                   the affine rather than assumed.
  connectivity     how many labels are in more than one connected piece. A fragmented
                   label is either a scan-truncation (legitimate) or speckle (not), and
                   the count falling across versions is the speckle being removed.
  rib incidence    every rib assigned to the vertebra its number implies. This is the
                   check the whole rib effort was for, so it is the one that should move
                   most between v3 (no ribs) and v5.
  count coherence  the rib-free interval is well defined -- a lowest rib-bearing vertebra
                   exists, a sacrum exists, and the count between them is 4, 5 or 6.
                   Anything else means the anchors are broken.
  completeness     what fraction of records carry each structure class at all.

Nothing here needs a model or a human. Every number is a property of the labels.

    python scripts/qc_version_progression.py \\
        --versions v3=data/hf_export_v3/labels v4=data/hf_export_v4/labels \\
                   v5=data/v5_final --out qc_final/version_progression.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402

VERT = set(range(1, 26)) | {28}
RIBS_L = set(range(34, 46)) | {74}
RIBS_R = set(range(46, 58)) | {75}
SACRUM, S1 = 26, 29
HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33
LEFT = RIBS_L | {HIP_L, FEM_L}
RIGHT = RIBS_R | {HIP_R, FEM_R}
VALID = (VERT | RIBS_L | RIBS_R | {SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R}
         | set(range(58, 80)) | {27, 255, 0})
MIN_VOX = 200


def one(path: str) -> dict:
    stem = Path(path).name.split("_")[0]
    r = {"case": stem}
    try:
        img = nib.load(path)
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
        codes = nib.aff2axcodes(img.affine)
    except Exception as exc:
        return {"case": stem, "error": type(exc).__name__}

    ids = set(int(x) for x in np.unique(lab))
    r["n_ids"] = len(ids - {0})
    r["stray_ids"] = len(ids - VALID)

    # --- sidedness, with the axis read from the affine ---------------------------------
    lr = [i for i, k in enumerate(codes) if k in ("L", "R")]
    r["sidedness_ok"] = 1
    if lr:
        ax = lr[0]
        lm = np.isin(lab, list(LEFT))
        rm = np.isin(lab, list(RIGHT))
        if lm.any() and rm.any():
            lc = float(np.argwhere(lm)[:, ax].mean())
            rc = float(np.argwhere(rm)[:, ax].mean())
            wrong = (lc >= rc) if codes[ax] == "R" else (lc <= rc)
            r["sidedness_ok"] = 0 if wrong else 1

    # --- connectivity: labels in more than one meaningful piece ------------------------
    frag = 0
    for v in sorted(ids & (VERT | RIBS_L | RIBS_R)):
        m = lab == v
        if m.sum() < MIN_VOX:
            continue
        n = ndimage.label(m)[1]
        if n > 1:
            sizes = ndimage.sum(m, ndimage.label(m)[0], range(1, n + 1))
            if int((sizes >= MIN_VOX).sum()) > 1:
                frag += 1
    r["fragmented_labels"] = frag

    # --- what the release contains -----------------------------------------------------
    r["has_ribs"] = int(bool(ids & (RIBS_L | RIBS_R)))
    r["has_femora"] = int(bool(ids & {FEM_L, FEM_R}))
    r["has_s1"] = int(S1 in ids)
    r["has_lumbar_rib"] = int(bool(ids & {74, 75}))
    r["has_ignore"] = int(255 in ids)
    r["n_vertebrae"] = len([v for v in ids & VERT if (lab == v).sum() >= MIN_VOX])

    # --- rib incidence and the rib-free count ------------------------------------------
    # A rib belongs to the vertebra its number implies: left rib 34+k and right 46+k both
    # belong to T(k+1) = id 8+k. Offsets are the defect the rib work existed to remove.
    off = 0
    checked = 0
    verts = {v: None for v in ids & VERT if (lab == v).sum() >= MIN_VOX}
    for v in list(verts):
        verts[v] = np.argwhere(lab == v)[::7] * sp
    for rid in sorted(ids & (RIBS_L | RIBS_R)):
        if rid in (74, 75):
            continue
        k = (rid - 34) if rid < 46 else (rid - 46)
        expect = 8 + k
        rp = np.argwhere(lab == rid)[::7] * sp
        if not len(rp) or not verts:
            continue
        best, bid = 1e9, None
        for v, vp in verts.items():
            if vp is None or not len(vp):
                continue
            d = float(np.sqrt(((rp[:, None, :] - vp[None, ::5, :]) ** 2).sum(-1)).min())
            if d < best:
                best, bid = d, v
        if bid is None or best > 40.0:
            continue
        checked += 1
        if bid != expect:
            off += 1
    r["ribs_checked"] = checked
    r["ribs_offset"] = off

    rib_bearing = sorted({8 + ((x - 34) if x < 46 else (x - 46))
                          for x in ids & (RIBS_L | RIBS_R) if x not in (74, 75)})
    lowest_rb = max(rib_bearing) if rib_bearing else None
    lumbar_present = sorted(v for v in ids & set(range(20, 26))
                            if (lab == v).sum() >= MIN_VOX)
    if lowest_rb is not None and (SACRUM in ids or S1 in ids):
        n_free = len([v for v in verts if v > lowest_rb])
        r["rib_free_count"] = n_free
        r["count_coherent"] = int(n_free in (4, 5, 6))
    else:
        r["rib_free_count"] = ""
        r["count_coherent"] = ""
    return r


def _coerce(row):
    """Restore the types csv threw away, leaving genuinely empty fields as ''.

    An empty string is meaningful here and must not become 0: `count_coherent` is blank when
    the anchors are missing, and summarise() filters those out before taking a percentage.
    Turning them into zeros would quietly report every anchorless case as incoherent.
    """
    out = {}
    for k, v in row.items():
        if v == "" or v is None:
            out[k] = ""
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def summarise(name, rows):
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    if not n:
        return {"version": name, "cases": 0}
    def frac(k):
        v = [r[k] for r in ok if r.get(k) not in ("", None)]
        return round(100.0 * sum(v) / len(v), 1) if v else ""
    tot_ribs = sum(r.get("ribs_checked", 0) for r in ok)
    tot_off = sum(r.get("ribs_offset", 0) for r in ok)
    coh = [r["count_coherent"] for r in ok if r.get("count_coherent") not in ("", None)]
    return {
        "version": name,
        "cases": n,
        "read_errors": len(rows) - n,
        "stray_ids_cases": sum(1 for r in ok if r.get("stray_ids", 0)),
        "sidedness_fail": sum(1 for r in ok if r.get("sidedness_ok") == 0),
        "fragmented_labels_total": sum(r.get("fragmented_labels", 0) for r in ok),
        "cases_with_fragment": sum(1 for r in ok if r.get("fragmented_labels", 0)),
        "pct_with_ribs": frac("has_ribs"),
        "pct_with_femora": frac("has_femora"),
        "pct_with_s1": frac("has_s1"),
        "pct_with_ignore": frac("has_ignore"),
        "lumbar_rib_cases": sum(r.get("has_lumbar_rib", 0) for r in ok),
        "median_vertebrae": int(np.median([r.get("n_vertebrae", 0) for r in ok])),
        "ribs_checked": tot_ribs,
        "ribs_offset": tot_off,
        "rib_offset_pct": round(100.0 * tot_off / tot_ribs, 3) if tot_ribs else "",
        "count_coherent_pct": round(100.0 * sum(coh) / len(coh), 1) if coh else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", required=True,
                    help="name=path pairs, e.g. v3=data/hf_export_v3/labels")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--out", default="qc_final/version_progression.csv")
    ap.add_argument("--per-case-out", default="qc_final/version_progression_percase.csv")
    ap.add_argument("--force", action="store_true",
                    help="recompute a version even if its part file already exists")
    a = ap.parse_args()

    # WRITE EACH VERSION AS IT FINISHES, AND SKIP ONE ALREADY DONE.
    #
    # The first version of this held every result in memory and wrote only after all four
    # versions completed. Job 40011919 got through v2, v3 and v5pre, died at 600/802 of v5
    # on an eight-hour wall clock, and produced NOTHING -- three complete versions of work
    # discarded because the fourth did not finish. A long job that writes once at the end
    # converts any timeout into total loss.
    #
    # Each version now lands in its own file the moment it is done, and a version whose file
    # already exists is skipped, so resubmitting resumes instead of restarting. The combined
    # outputs are assembled at the end from whatever per-version files exist, so they are
    # correct after a partial run too.
    part_dir = Path(a.out).parent / "version_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    for spec in a.versions:
        name, _, path = spec.partition("=")
        part = part_dir / f"{name}_percase.csv"
        if part.exists() and not a.force:
            print(f"  {name}: already done ({part}), skipping", flush=True)
            continue
        files = sorted(str(p) for p in Path(path).glob("*.nii.gz"))
        if a.limit:
            files = files[: a.limit]
        if not files:
            print(f"  ! {name}: nothing under {path}")
            continue
        print(f"  {name}: {len(files)} case(s) from {path}", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, x in enumerate(ex.map(one, files, chunksize=2), 1):
                x["version"] = name
                rows.append(x)
                if i % 100 == 0:
                    print(f"    {i}/{len(files)}", flush=True)
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        tmp = part.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        tmp.replace(part)                      # atomic: a killed job leaves no half file
        print(f"    wrote {part}", flush=True)

    # --- assemble from the parts, in the order the versions were named ------------------
    summaries, percase = [], []
    for spec in a.versions:
        name = spec.partition("=")[0]
        part = part_dir / f"{name}_percase.csv"
        if not part.exists():
            print(f"  ! {name}: no results, omitted from the summary")
            continue
        # csv gives back strings. summarise() was written against in-memory dicts and does
        # arithmetic on these fields, so reading a part file straight into it raises
        # "unsupported operand type(s) for +: 'int' and 'str'" -- which is the whole resume
        # path failing, and would only have surfaced on the grid.
        rows = [_coerce(r) for r in csv.DictReader(open(part))]
        percase.extend(rows)
        summaries.append(summarise(name, rows))

    if not summaries:
        return 1
    cols = list(summaries[0])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(summaries)
    pc_cols = []
    for r in percase:
        for k in r:
            if k not in pc_cols:
                pc_cols.append(k)
    with open(a.per_case_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pc_cols)
        w.writeheader(); w.writerows(percase)

    print("\n" + "=" * 78)
    print("VERSION PROGRESSION")
    print("=" * 78)
    keys = ["cases", "stray_ids_cases", "sidedness_fail", "cases_with_fragment",
            "fragmented_labels_total", "pct_with_ribs", "pct_with_femora",
            "pct_with_s1", "pct_with_ignore", "lumbar_rib_cases", "ribs_checked",
            "ribs_offset", "rib_offset_pct", "count_coherent_pct"]
    head = f"{'metric':26s}" + "".join(f"{s['version']:>12s}" for s in summaries)
    print(head); print("-" * len(head))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{str(s.get(k,'')):>12s}" for s in summaries))
    print(f"\n  wrote {a.out} and {a.per_case_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
