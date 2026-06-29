"""qc_trust_ts_preflight.py — predict the trust-TS rib worklist over the WHOLE dataset,
straight off the v3 TS labels, with NO GPU and NO Möller (so you can run it before the
full v4 rebuild and see whether trusting TS works for all cases, not just a sample).

For every v3 label it reports, per side, the TS rib numbers present, any GAP (a missing
number between present ones -> would remain under trust-TS), and any DUP (a number in 2+
substantial pieces after the same stray gate build_v4_ribs uses -> TS fragmentation the
Möller graft is meant to bridge). Prints a summary (how many cases are clean / gap / dup)
and writes a CSV of the flagged cases. The graft can only REDUCE the dup count (it merges
fragments), so this is an UPPER BOUND on the post-graft review list.

  python scripts/qc_trust_ts_preflight.py --labels data/hf_export_v3/labels --csv trust_ts_preflight.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

LEFT_OFF, RIGHT_OFF = 33, 45            # rib_left_N -> 33+N, rib_right_N -> 45+N (34..57)
MIN_VOX = 50                            # a piece below this is a speckle
STRAY_ABS_VOX, STRAY_FRAC = 500, 0.10  # same gate as build_v4_ribs._drop_same_id_strays


def _rid(off, n):
    return off + n


def analyse(lab) -> dict:
    st = ndimage.generate_binary_structure(3, 3)
    objs = ndimage.find_objects(lab.astype(np.int32))
    res = {"left_gaps": [], "right_gaps": [], "dups": []}
    for side, off in (("L", LEFT_OFF), ("R", RIGHT_OFF)):
        present = [n for n in range(1, 13)
                   if _rid(off, n) - 1 < len(objs) and objs[_rid(off, n) - 1]]
        if not present:
            continue
        gaps = [n for n in range(min(present), max(present) + 1) if n not in present]
        res["left_gaps" if side == "L" else "right_gaps"] = gaps
        for n in present:
            sl = objs[_rid(off, n) - 1]
            cc, k = ndimage.label(lab[sl] == _rid(off, n), structure=st)
            big = sorted((int(s) for s in np.bincount(cc.ravel())[1:] if s >= MIN_VOX),
                         reverse=True)
            if len(big) > 1:
                survivors = [big[0]] + [s for s in big[1:]
                                        if s >= max(STRAY_ABS_VOX, STRAY_FRAC * big[0])]
                if len(survivors) > 1:        # a substantial 2nd piece survives the stray gate
                    res["dups"].append(f"{side}{n}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True, help="v3 labels dir (*_label.nii.gz)")
    ap.add_argument("--csv", type=Path, default=Path("trust_ts_preflight.csv"))
    a = ap.parse_args()

    files = sorted(a.labels.glob("*_label.nii.gz"))
    n_clean = n_gap = n_dup = 0
    rows = []
    for i, f in enumerate(files):
        cid = f.name.split("_")[0]
        lab = np.asanyarray(nib.load(str(f)).dataobj)
        r = analyse(lab)
        gap = bool(r["left_gaps"] or r["right_gaps"])
        dup = bool(r["dups"])
        if gap:
            n_gap += 1
        if dup:
            n_dup += 1
        if not gap and not dup:
            n_clean += 1
        else:
            rows.append({"ct": cid,
                         "left_gaps": r["left_gaps"], "right_gaps": r["right_gaps"],
                         "dups": ",".join(r["dups"])})
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(files)}  clean={n_clean} gap={n_gap} dup={n_dup}", flush=True)

    with a.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ct", "left_gaps", "right_gaps", "dups"])
        w.writeheader()
        w.writerows(rows)

    tot = len(files)
    print(f"\n=== trust-TS pre-flight over {tot} cases ===")
    print(f"  clean (no gap, no substantial dup) : {n_clean}  ({100*n_clean/tot:.0f}%)")
    print(f"  has a GAP (remains under trust-TS) : {n_gap}")
    print(f"  has a DUP (graft target; upper bnd): {n_dup}")
    print(f"  flagged cases written to           : {a.csv}")
    print("  NOTE: the Möller graft can only REDUCE the dup count (it bridges fragments);")
    print("        gaps are genuine TS misses and would remain for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
