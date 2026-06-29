"""viz_rib_case.py — quick QC render of a v4 rib case: the rib cage coloured by rib id over a
CT bone backdrop, coronal + sagittal MIP-style views, for spotting duplicate / stray ribs.

  python scripts/viz_rib_case.py --label data/hf_export_v4/labels/0008_label.nii.gz \
      --ct data/hf_export_v4/ct/0008_ct.nii.gz --highlight 45 --out 0008_ribs.png

A duplicate id shows as the SAME colour in two separated blobs; a stray shows as a small patch
off the cage. --highlight rings the given rib ids in red (e.g. the dup/gap ids from the QC).
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

RIB_LO, RIB_HI = 34, 57


def _axes(affine):
    codes = nib.aff2axcodes(affine)                      # e.g. ('R','A','S')
    lr = next(i for i, c in enumerate(codes) if c in "RL")
    ap = next(i for i, c in enumerate(codes) if c in "AP")
    return lr, ap, codes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, type=Path)
    p.add_argument("--ct", type=Path, help="optional CT for the grayscale backdrop")
    p.add_argument("--highlight", default="", help="comma rib ids to ring red, e.g. 45,54,55")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    L = nib.load(str(a.label))
    lab = np.asanyarray(L.dataobj)
    lr, ap, codes = _axes(L.affine)
    ct = np.asanyarray(nib.load(str(a.ct)).dataobj).astype(float) if a.ct else None
    hl = {int(x) for x in a.highlight.split(",") if x.strip()}
    cmap = cm.get_cmap("nipy_spectral", RIB_HI - RIB_LO + 1)

    views = [("coronal (frontal)", ap), ("sagittal (lateral)", lr)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for axp, (name, axis) in zip(axs, views):
        ribs = np.where((lab >= RIB_LO) & (lab <= RIB_HI), lab, 0).max(axis=axis)
        ribs = np.rot90(ribs)
        if ct is not None:
            bg = np.rot90(np.clip(ct, 200, 1500).max(axis=axis))
            axp.imshow(bg, cmap="gray")
        axp.imshow(np.ma.masked_where(ribs == 0, ribs), cmap=cmap,
                   vmin=RIB_LO, vmax=RIB_HI, alpha=0.75)
        if hl:
            axp.contour(np.isin(ribs, list(hl)).astype(float), levels=[0.5],
                        colors="red", linewidths=1.6)
        present = sorted(int(v) for v in np.unique(ribs) if v)
        axp.set_title(f"{name}\nrib ids present: {present}", fontsize=8)
        axp.axis("off")
    fig.suptitle(f"{a.label.name}   highlight={sorted(hl)}   axcodes={codes}", fontsize=10)
    out = a.out or Path(a.label.name.replace(".nii.gz", "") + "_ribs.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
