"""Two hip-laterality measures disagree completely. Only one can be right.

measure_hip_crossover says 26-49% of a hip label sits across the midline in eighteen records.
relabel_hips_by_midline, given those same eighteen files, found zero voxels to move. Both
claim to answer the same question -- which side of the midline is this voxel on -- one in
voxel index along the axis nibabel calls left-right, the other in world X through the affine.

Acting on either without settling it risks relabelling eighteen correct records, which is the
worse error: the flags are 2.2% of the release and a bad fix touches every one of them.

So compute both, per case, side by side, and include CONTROLS -- records the sweep did not
flag. If the controls also show a third of the label across, the metric is miscalibrated and
there was never anything wrong with the eighteen.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

HIP_L, HIP_R = 30, 31
LUMBAR = tuple(range(20, 26))
MIDLINE_IDS = list(LUMBAR) + [26, 29]


def both_ways(path):
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    codes = nib.aff2axcodes(img.affine)
    ax = next((i for i, k in enumerate(codes) if k in ("L", "R")), None)
    L, R = arr == HIP_L, arr == HIP_R
    if ax is None or not (L.any() and R.any()):
        return None

    # --- measure A: voxel index along the axis nibabel calls left-right ---------------
    spine = np.isin(arr, MIDLINE_IDS)
    mid_idx = float(np.argwhere(spine)[:, ax].mean())
    toward_right = codes[ax] == "R"
    li = np.argwhere(L)[:, ax]
    a_left = float((li > mid_idx).mean() if toward_right else (li < mid_idx).mean())

    # --- measure B: world X through the affine, as the relabel script does ------------
    lum = np.isin(arr, LUMBAR)
    com = np.array(np.nonzero(lum)).mean(axis=1)
    mid_x = float(apply_affine(img.affine, com)[0])
    lx = apply_affine(img.affine, np.array(np.nonzero(L)).T)[:, 0]
    b_left = float((lx > mid_x).mean())          # RAS+: +X is the patient's right

    # is the left hip label even on the correct side overall?
    cl = float(np.argwhere(L)[:, ax].mean())
    cr = float(np.argwhere(R)[:, ax].mean())
    return dict(codes="".join(codes), ax=ax, mid_idx=mid_idx, mid_x=mid_x,
                a_left=a_left, b_left=b_left, cl=cl, cr=cr,
                det=float(np.linalg.det(img.affine[:3, :3])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--flagged", nargs="+", required=True)
    ap.add_argument("--controls", type=int, default=20)
    a = ap.parse_args()
    src = Path(a.src)

    allf = sorted((src / "labels").glob("*_label.nii.gz"))
    flagged = set(a.flagged)
    ctrl = [p for p in allf if p.name[:4] not in flagged]
    step = max(1, len(ctrl) // a.controls)
    controls = [ctrl[i * step] for i in range(min(a.controls, len(ctrl)))]

    print(f"  {'case':<6} {'grp':<8} {'axes':<5} {'A idx%':>8} {'B world%':>9} "
          f"{'mid idx':>8} {'mid X':>9} {'det':>7}")
    rows = {"flagged": [], "control": []}
    for grp, paths in (("flagged", [src / "labels" / f"{c}_label.nii.gz" for c in a.flagged]),
                       ("control", controls)):
        for p in paths:
            r = both_ways(p)
            if r is None:
                continue
            rows[grp].append(r)
            print(f"  {p.name[:4]:<6} {grp:<8} {r['codes']:<5} {r['a_left']:>7.1%} "
                  f"{r['b_left']:>8.1%} {r['mid_idx']:>8.1f} {r['mid_x']:>9.1f} "
                  f"{r['det']:>7.1f}")

    print("\n  summary")
    for grp in ("flagged", "control"):
        rs = rows[grp]
        if not rs:
            continue
        A = np.array([r["a_left"] for r in rs])
        B = np.array([r["b_left"] for r in rs])
        print(f"    {grp:<8} n={len(rs):<3} measure A median {np.median(A):6.1%} "
              f"(max {A.max():5.1%})   measure B median {np.median(B):6.1%} "
              f"(max {B.max():5.1%})")

    if rows["control"]:
        cA = np.array([r["a_left"] for r in rows["control"]])
        fA = np.array([r["a_left"] for r in rows["flagged"]])
        print(f"""
  Read it this way. If the controls sit near zero on measure A and the flagged sit at a
  third, the flags are real and measure B is the one that is broken. If BOTH groups sit at a
  third, measure A is miscalibrated and the eighteen were never wrong.
    controls  median {np.median(cA):.1%}
    flagged   median {np.median(fA):.1%}""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
