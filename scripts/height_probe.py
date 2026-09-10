"""Why is posterior body height ~6 mm above Panjabi 1992?

THE ESTIMATOR IN extract_level_gradients.py IS AN UPPER ENVELOPE, NOT A LANDMARK.
It takes a 10 mm mid-sagittal slab, collapses it over the whole slab width, computes a
superior-inferior extent for every anteroposterior position, splits at the midpoint, and
reports the MAXIMUM column in each half. Panjabi's VBHp is a caliper reading at the
posterior wall of one dried vertebra. A maximum over a 10 x 15 mm patch is >= a reading at
one point by construction, so some of the gap is guaranteed before any anatomy is involved.

This measures four estimators on the same masks so the size of that effect is a number
rather than an argument:

  max_half   what the release publishes: tallest column in the posterior half
  wall_p90   90th percentile of the columns in the posterior 20% of the body
  wall_med   median column height in the posterior 20% of the body
  wall_col   the single column at the posterior-most extent of the body

If wall_med lands near Panjabi's 23-24 mm and max_half sits near 30, the difference is the
estimator. If every estimator sits near 30, the difference is the population: dried
elderly cadaveric specimens against living patients aged 50 and over, and the release is
measuring what it says it measures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}


def probe(path):
    img = nib.as_closest_canonical(nib.load(str(path)))       # RAS: x=LR, y=PA, z=SI
    lab = np.asanyarray(img.dataobj)
    sp = np.asarray(img.header.get_zooms()[:3], float)
    out = {}
    for vid, name in LUMBAR.items():
        whole = lab == vid
        if whole.sum() < 500:
            continue
        # CARVE THE BODY OFF THE POSTERIOR ELEMENTS, the same single plane the release
        # uses. The first run of this probe measured the WHOLE vertebra, so its
        # "posterior 20%" landed on the spinous process and its numbers meant nothing.
        fronts = []
        zz = np.nonzero(whole.any(axis=(0, 1)))[0]
        for z in zz[len(zz) // 3: 2 * len(zz) // 3 + 1]:
            sl = whole[:, :, z]
            if sl.sum() < 40:
                continue
            hole = ndimage.binary_fill_holes(sl) & ~sl
            if not hole.any():
                continue
            fronts.append(int(np.nonzero(hole.any(axis=0))[0].max()))
        if not fronts:
            continue
        f = int(np.median(fronts))
        m = np.zeros_like(whole)
        m[:, f:, :] = whole[:, f:, :]     # RAS: +y is anterior, so this keeps the body
        if m.sum() < 500:
            continue
        idx = np.argwhere(m)
        # mid-sagittal slab, 10 mm wide, exactly as the release does
        bx = float(np.median(idx[:, 0]))
        xlo = max(0, int(round(bx - 5.0 / sp[0])))
        xhi = int(round(bx + 5.0 / sp[0])) + 1
        slab = m[xlo:xhi]
        ys = np.nonzero(slab.any(axis=(0, 2)))[0]
        col = {}
        for y in ys:
            c = slab[:, y, :]
            if c.sum() < 3:
                continue
            zc = np.nonzero(c.any(axis=0))[0]
            col[y] = (zc.max() - zc.min() + 1) * sp[2]
        if len(col) < 6:
            continue
        yy = np.array(sorted(col))
        hh = np.array([col[y] for y in yy])
        ymid = (yy.min() + yy.max()) / 2.0
        post = hh[yy <= ymid]
        # posterior 20% of the body's anteroposterior span
        cut = yy.min() + 0.20 * (yy.max() - yy.min())
        band = hh[yy <= cut]
        if not len(post) or not len(band):
            continue
        out[name] = {
            "max_half": float(post.max()),
            "wall_p90": float(np.percentile(band, 90)),
            "wall_med": float(np.median(band)),
            "wall_col": float(hh[yy.argmin()]),
        }
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/v5_final")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    files = sorted(root.glob("*_label.nii.gz"))[:n]
    print(f"{len(files)} case(s) from {root}\n", flush=True)
    acc = {}
    for i, f in enumerate(files, 1):
        try:
            for lv, d in probe(f).items():
                for k, v in d.items():
                    acc.setdefault((lv, k), []).append(v)
        except Exception as exc:
            print(f"  {f.name}: {type(exc).__name__}: {exc}")
        if i % 20 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    PAN = {"L1": 23.8, "L2": 24.3, "L3": 23.8, "L4": 24.1, "L5": 22.9}
    print(f"\n{'level':6s} {'n':>5s} " + " ".join(f"{k:>10s}" for k in
          ("max_half", "wall_p90", "wall_med", "wall_col")) + f" {'Panjabi':>9s}")
    for lv in LUMBAR.values():
        row = [np.median(acc.get((lv, k), [np.nan])) for k in
               ("max_half", "wall_p90", "wall_med", "wall_col")]
        cnt = len(acc.get((lv, "max_half"), []))
        print(f"{lv:6s} {cnt:5d} " + " ".join(f"{v:10.1f}" for v in row)
              + f" {PAN[lv]:9.1f}")
    print("\nmedians over cases. Panjabi 1992 Table 2 VBHp, mean of twelve specimens.")


if __name__ == "__main__":
    main()
