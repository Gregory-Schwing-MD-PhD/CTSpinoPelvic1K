"""For the crossover records, measure against the anatomy rather than against each other.

The sweep flags a record when a quarter of one hip label sits beyond the plane midway between
the two hip centroids. That plane is a proxy for the midline and it moves: if one label is
already wrong, it drags the plane toward the error and changes the number the threshold is
compared against.

The spine does not move. The vertebral bodies and the sacrum sit on the midline by
definition, so their centroid gives a midline that does not depend on either hip being
labelled correctly -- and a hip bone crossing THAT is a real finding, not a threshold artefact.

Reports both numbers per record so the difference is visible, and the voxel count of the
largest piece on the wrong side, since a compact block of one hip wearing the other's name is
the 1035 fault and a thin rind along the symphysis is not.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

HIP_L, HIP_R = 30, 31
MIDLINE_IDS = list(range(20, 26)) + [26, 29]        # lumbar bodies and sacrum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    rows = []

    for cid in a.cases:
        img = nib.load(str(src / "labels" / f"{cid}_label.nii.gz"))
        arr = np.asanyarray(img.dataobj)
        codes = nib.aff2axcodes(img.affine)
        ax = next(i for i, k in enumerate(codes) if k in ("L", "R"))
        toward_right = codes[ax] == "R"

        spine = np.isin(arr, MIDLINE_IDS)
        if not spine.any():
            print(f"  {cid}: no midline structure to measure against")
            continue
        midline = float(np.argwhere(spine)[:, ax].mean())

        L, R = arr == HIP_L, arr == HIP_R
        if not (L.any() and R.any()):
            continue
        cl = float(np.argwhere(L)[:, ax].mean())
        cr = float(np.argwhere(R)[:, ax].mean())
        between = (cl + cr) / 2

        def beyond(mask, plane):
            idx = np.argwhere(mask)[:, ax]
            # the left hip belongs on the side away from 'right'
            wrong = idx > plane if toward_right else idx < plane
            return float(wrong.mean())

        lw_mid = beyond(L, midline)
        rw_mid = float((np.argwhere(R)[:, ax] < midline).mean()) if toward_right \
            else float((np.argwhere(R)[:, ax] > midline).mean())
        lw_bet = beyond(L, between)
        rw_bet = float((np.argwhere(R)[:, ax] < between).mean()) if toward_right \
            else float((np.argwhere(R)[:, ax] > between).mean())

        # biggest contiguous block of the left label sitting on the right of the midline
        wrong_side = L & ((np.indices(arr.shape)[ax] > midline) if toward_right
                          else (np.indices(arr.shape)[ax] < midline))
        lab, n = ndi.label(wrong_side)
        biggest = int(np.bincount(lab.ravel())[1:].max()) if n else 0

        rows.append({"case": cid, "left_beyond_midline": round(lw_mid, 3),
                     "right_beyond_midline": round(rw_mid, 3),
                     "left_beyond_between": round(lw_bet, 3),
                     "right_beyond_between": round(rw_bet, 3),
                     "largest_wrong_side_piece": biggest})
        print(f"  {cid}  vs spine midline: left {lw_mid:5.1%} right {rw_mid:5.1%}   "
              f"vs midplane: left {lw_bet:5.1%} right {rw_bet:5.1%}   "
              f"largest wrong-side block {biggest:,} vox")

    if rows:
        worst = sorted(rows, key=lambda r: -max(r["left_beyond_midline"],
                                                r["right_beyond_midline"]))
        print("\n  measured against the spine, worst first:")
        for r in worst:
            m = max(r["left_beyond_midline"], r["right_beyond_midline"])
            verdict = ("a real crossover" if m > 0.25 and r["largest_wrong_side_piece"] > 20000
                       else "at the symphysis, not a labelling fault")
            print(f"    {r['case']}  {m:5.1%} across, largest block "
                  f"{r['largest_wrong_side_piece']:>8,}  -- {verdict}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
