"""metal_census.py — every dense object in the volume, not only the ones near the spine.

seed_hardware.py searches a 12 mm shell around the spine masks, which is the right region
for finding instrumentation and the wrong region for proving there is none: a pedicle screw
head sits proud of the lamina, a rod runs posterior to it, and an iliac or S2AI screw is
somewhere else entirely. "I found two cages" and "there are only two cages" are different
claims and the second one needs the whole frame.

The whole frame brings a problem the shell was hiding. These are CT COLONOGRAPHY series:
the stool is tagged with oral contrast that saturates the scanner exactly as metal does, so
a plain threshold over the volume returns bowel in quantity. Contrast and metal separate on
shape and place rather than on density -- tagged stool is amorphous, follows the colon, and
sits well away from bone; an implant is compact or linear, and it is ON the skeleton.

So every dense component is measured and placed, and the report says what each one looks
like rather than assuming. Read only.

    python scripts/metal_census.py --ct thoracic_fix/0068/0068_ct.nii.gz \
        --label thoracic_fix/0068/0068_label_proposed.nii.gz --hu 2500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

BONE_IDS = list(range(8, 30)) + [30, 31, 32, 33]     # spine, sacrum, hips, femurs
VERT_NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
             **{v: f"L{v - 19}" for v in range(20, 26)}, 26: "sacrum", 29: "S1",
             30: "hip_L", 31: "hip_R", 32: "femur_L", 33: "femur_R"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--hu", type=float, default=2500.0)
    ap.add_argument("--min-voxels", type=int, default=40)
    a = ap.parse_args()

    img = nib.load(a.label)
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    ct = np.asanyarray(nib.load(a.ct).dataobj)
    sp = np.array(img.header.get_zooms()[:3], float)
    vox_mm3 = float(np.prod(sp))

    bone = np.isin(lab, BONE_IDS)
    # distance to the nearest labelled bone voxel, in mm -- the single most useful number
    # for telling an implant from a bolus of tagged stool
    dist = ndimage.distance_transform_edt(~bone, sampling=sp)

    bright = ct > a.hu
    cc, ncc = ndimage.label(bright)
    print(f"threshold {a.hu:.0f} HU over the WHOLE volume: "
          f"{int(bright.sum()):,} voxels in {ncc} components\n")

    sizes = ndimage.sum(bright, cc, range(1, ncc + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= a.min_voxels]
    print(f"{len(keep)} component(s) of {a.min_voxels}+ voxels:\n")
    print(f"  {'vox':>7}  {'mm3':>8}  {'L x W x T (mm)':>22}  {'aspect':>6}  "
          f"{'dist to bone':>12}  what it looks like")
    print("  " + "-" * 96)

    rows = []
    for i in sorted(keep, key=lambda k: -sizes[k - 1]):
        m = cc == i
        idx = np.argwhere(m)
        pts = idx * sp
        q = pts - pts.mean(0)
        vt = np.linalg.svd(q, full_matrices=False)[2]
        ext = [float((q @ vt[k]).max() - (q @ vt[k]).min()) for k in range(3)]
        aspect = ext[0] / max(ext[1], 1e-6)
        d = float(dist[m].min())
        near = sorted({VERT_NAME.get(int(v), str(int(v)))
                       for v in np.unique(lab[ndimage.binary_dilation(m, iterations=3)])
                       if int(v) in VERT_NAME})
        # SHAPE AND PLACE, in that order. Nothing more than 15 mm from any labelled bone is
        # instrumentation on this skeleton; a long thin thing on the bone is a screw or rod;
        # a compact thing on the bone is a cage or a fragment.
        if d > 15:
            what = "away from the skeleton -- tagged stool / contrast"
        elif aspect >= 3.0:
            what = f"LINEAR on bone -- screw or rod (near {','.join(near) or '?'})"
        elif ext[0] >= 18 and ext[1] >= 8:
            what = f"COMPACT block on bone -- cage (near {','.join(near) or '?'})"
        else:
            what = f"small dense fleck on bone (near {','.join(near) or '?'})"
        print(f"  {int(sizes[i-1]):>7,}  {sizes[i-1]*vox_mm3:>8,.0f}  "
              f"{ext[0]:>6.1f} x{ext[1]:>5.1f} x{ext[2]:>5.1f}  {aspect:>6.1f}  "
              f"{d:>10.1f} mm  {what}")
        rows.append((int(sizes[i - 1]), d, aspect, what))

    on_bone = [r for r in rows if r[1] <= 15]
    off_bone = [r for r in rows if r[1] > 15]
    print(f"\n  {len(on_bone)} component(s) on the skeleton, "
          f"{sum(r[0] for r in on_bone):,} voxels")
    print(f"  {len(off_bone)} component(s) away from it, "
          f"{sum(r[0] for r in off_bone):,} voxels -- not instrumentation")
    lin = [r for r in on_bone if r[2] >= 3.0]
    print(f"\n  screw/rod-shaped objects on the skeleton: {len(lin)}"
          + ("" if lin else "   <- no posterior instrumentation in this volume"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
