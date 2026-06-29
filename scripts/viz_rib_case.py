"""viz_rib_case.py — QC render of a v4 case: ribs coloured by rib id, drawn OVER the
spine/pelvis (vertebrae + sacrum + femurs) for context, optionally over a CT bone backdrop.
Coronal + sagittal max-projections, anatomically oriented (superior up).

  python scripts/viz_rib_case.py --label .../0231_label.nii.gz [--ct .../0231_ct.nii.gz] \
      --highlight 45,57 --out 0231.png

Spine/pelvis (ids 1-33) = translucent steel-blue context; ribs (34-57) = spectral by number;
--highlight rings the given rib ids in red. Volumes are reoriented to canonical RAS first, so
the view is correct regardless of how the affine stores the axes (the cause of earlier
"weird angle" renders). A duplicate id = same colour in two blobs; a hyperplastic TP shows as
a ringed nub hugging a vertebra; a gap = a missing rib beside its level.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

RIB_LO, RIB_HI = 34, 57
SPINE_HI = 33                                            # 1..33 = vertebrae/sacrum/S1/hips/femurs


def _canon(path):
    """Load + reorient to canonical RAS: data axes become 0=L->R, 1=P->A, 2=I->S."""
    return np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)


def _disp(arr2d):
    """A (in-plane, S-I) projection -> image with superior at top."""
    return np.flipud(arr2d.T)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, type=Path)
    p.add_argument("--ct", type=Path, help="optional CT for the grayscale backdrop")
    p.add_argument("--highlight", default="", help="comma rib ids to ring red, e.g. 45,57")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    lab = _canon(a.label)
    ct = _canon(a.ct).astype(float) if a.ct else None
    hl = {int(x) for x in a.highlight.split(",") if x.strip()}
    rib_cmap = cm.get_cmap("turbo", RIB_HI - RIB_LO + 1)   # vivid, no gray (clashed with CT)
    spine_cmap = ListedColormap(["#3f6fb0"])

    # canonical axes: 0 = L->R, 1 = P->A, 2 = I->S. coronal projects A-P (axis1); sagittal R-L (axis0).
    views = [("coronal (frontal)", 1), ("sagittal (lateral)", 0)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for axp, (name, axis) in zip(axs, views):
        if ct is not None:
            axp.imshow(_disp(np.clip(ct, 200, 1500).max(axis=axis)), cmap="gray")
        sp = _disp(((lab >= 1) & (lab <= SPINE_HI)).any(axis=axis))
        axp.imshow(np.ma.masked_where(~sp, sp), cmap=spine_cmap, alpha=0.45)
        ribs = _disp(np.where((lab >= RIB_LO) & (lab <= RIB_HI), lab, 0).max(axis=axis))
        axp.imshow(np.ma.masked_where(ribs == 0, ribs), cmap=rib_cmap,
                   vmin=RIB_LO, vmax=RIB_HI, alpha=0.85)
        if hl:
            axp.contour(np.isin(ribs, list(hl)).astype(float), levels=[0.5],
                        colors="red", linewidths=1.6)
        present = sorted(int(v) for v in np.unique(ribs) if v)
        axp.set_title(f"{name}\nrib ids: {present}", fontsize=8)
        axp.axis("off")
    fig.suptitle(f"{a.label.name}   spine=blue, ribs=turbo by number, highlight(red)={sorted(hl)}",
                 fontsize=10)
    out = a.out or Path(a.label.name.replace(".nii.gz", "") + "_ribs.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
