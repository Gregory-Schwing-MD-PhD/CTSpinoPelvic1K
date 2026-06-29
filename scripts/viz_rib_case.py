"""viz_rib_case.py — QC render of a v4 case: ribs coloured by rib id, drawn OVER the
spine/pelvis (vertebrae + sacrum + femurs) for context, optionally over a CT bone backdrop.
Coronal + sagittal max-projections, for spotting duplicate / stray ribs and gaps in context.

  python scripts/viz_rib_case.py --label .../0231_label.nii.gz [--ct .../0231_ct.nii.gz] \
      --highlight 45,57 --out 0231.png

Spine/pelvis (ids 1-33) = translucent steel-blue context; ribs (34-57) = spectral by number;
--highlight rings the given rib ids in red. A duplicate id = same colour in two blobs; a
hyperplastic TP shows as a ringed nub hugging a vertebra; a gap = a missing rib beside its level.
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


def _axes(affine):
    codes = nib.aff2axcodes(affine)
    lr = next(i for i, c in enumerate(codes) if c in "RL")
    ap = next(i for i, c in enumerate(codes) if c in "AP")
    return lr, ap, codes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, type=Path)
    p.add_argument("--ct", type=Path, help="optional CT for the grayscale backdrop")
    p.add_argument("--highlight", default="", help="comma rib ids to ring red, e.g. 45,57")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    L = nib.load(str(a.label))
    lab = np.asanyarray(L.dataobj)
    lr, ap, codes = _axes(L.affine)
    ct = np.asanyarray(nib.load(str(a.ct)).dataobj).astype(float) if a.ct else None
    hl = {int(x) for x in a.highlight.split(",") if x.strip()}
    rib_cmap = cm.get_cmap("nipy_spectral", RIB_HI - RIB_LO + 1)
    spine_cmap = ListedColormap(["#3f6fb0"])            # one steel-blue for all spine/pelvis

    views = [("coronal (frontal)", ap), ("sagittal (lateral)", lr)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for axp, (name, axis) in zip(axs, views):
        if ct is not None:
            axp.imshow(np.rot90(np.clip(ct, 200, 1500).max(axis=axis)), cmap="gray")
        spine = np.rot90(((lab >= 1) & (lab <= SPINE_HI)).any(axis=axis))
        axp.imshow(np.ma.masked_where(~spine, spine), cmap=spine_cmap, alpha=0.45)
        ribs = np.rot90(np.where((lab >= RIB_LO) & (lab <= RIB_HI), lab, 0).max(axis=axis))
        axp.imshow(np.ma.masked_where(ribs == 0, ribs), cmap=rib_cmap,
                   vmin=RIB_LO, vmax=RIB_HI, alpha=0.85)
        if hl:
            axp.contour(np.isin(ribs, list(hl)).astype(float), levels=[0.5],
                        colors="red", linewidths=1.6)
        present = sorted(int(v) for v in np.unique(ribs) if v)
        axp.set_title(f"{name}\nrib ids: {present}", fontsize=8)
        axp.axis("off")
    fig.suptitle(f"{a.label.name}   spine=blue, ribs=spectral, highlight(red)={sorted(hl)}   "
                 f"axcodes={codes}", fontsize=10)
    out = a.out or Path(a.label.name.replace(".nii.gz", "") + "_ribs.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
