"""scripts/render_reading_sheet.py — the four views that decide a six-lumbar reading, per record.

For each record: (a) axial through the labelled L1 at pedicle height, where a thoracic body
shows coronally facing facets and a costal facet and a lumbar body shows sagittal facets;
(b) coronal slab through the labelled L1 for any rudimentary rib; (c) coronal through the
labelled L6 and the sacral alae, where a lumbarized S1 shows its process reaching or fused to
the ala; (d) midline sagittal for the disc under the labelled L6 and the sacral height.
Labels are drawn as outlines so the bone itself stays visible.

    python scripts/render_reading_sheet.py --data ~/data/CTSpinoPelvic1K --cases 0376 1053 --out results/reading_sheets

Volumes are canonicalised for display only (as_closest_canonical); nothing is written back.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

T12, L1, L5, L6, SACRUM, S1 = 19, 20, 24, 25, 26, 29
OUTLINE = {19: "#ffb000", 20: "#00d5ff", 24: "#7cff00", 25: "#ff2fd6", 26: "#ff4040", 29: "#ffffff",
           30: "#c0c0c0", 31: "#c0c0c0"}


def load(data: Path, case: str):
    ct = nib.as_closest_canonical(nib.load(str(data / "ct" / f"{case}_ct.nii.gz")))
    lb = nib.as_closest_canonical(nib.load(str(data / "labels" / f"{case}_label.nii.gz")))
    return (np.asanyarray(ct.dataobj).astype(np.float32), np.asanyarray(lb.dataobj).astype(np.int16),
            np.asarray(ct.header.get_zooms()[:3], float))


def centroid(lab, vid):
    idx = np.argwhere(lab == vid)
    return None if len(idx) == 0 else idx.mean(0)


def window(a, lo=-200, hi=1200):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def panel(ax, img2d, lab2d, zoom, title, ids):
    ax.imshow(window(img2d).T, cmap="gray", origin="lower", aspect=zoom[1] / zoom[0])
    for vid in ids:
        m = lab2d == vid
        if m.any():
            ax.contour(m.T.astype(float), levels=[0.5], colors=[OUTLINE.get(vid, "yellow")], linewidths=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default="results/reading_sheets")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    for case in a.cases:
        ct, lab, zoom = load(data, case)          # canonical RAS: x=left->right, y=post->ant, z=inf->sup
        c_l1, c_l6, c_sac = centroid(lab, L1), centroid(lab, L6), centroid(lab, SACRUM)
        c_low = c_l6 if c_l6 is not None else centroid(lab, L5)
        if c_l1 is None or c_low is None or c_sac is None:
            print(case, "missing a label (L1, L6/L5 or sacrum); skipped"); continue
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))
        # (a) axial through labelled L1, slightly above its centre (pedicle level)
        z = int(round(c_l1[2] + 4 / zoom[2]))
        panel(ax[0, 0], ct[:, :, z], lab[:, :, z], (zoom[0], zoom[1]),
              f"{case} (a) axial, labelled L1 at pedicle level: facet orientation, costal facet?", (T12, L1))
        # (b) coronal slab through labelled L1 (max-intensity over 12 mm) for a rudimentary rib
        y0, y1 = int(c_l1[1] - 6 / zoom[1]), int(c_l1[1] + 6 / zoom[1])
        mip = ct[:, y0:y1, :].max(1)
        lab_c = lab[:, int(c_l1[1]), :]
        panel(ax[0, 1], mip, lab_c, (zoom[0], zoom[2]),
              f"{case} (b) coronal MIP through labelled L1: any rib or costal facet?", (T12, L1))
        ax[0, 1].set_ylim(c_l1[2] - 70 / zoom[2], c_l1[2] + 70 / zoom[2])
        # (c) coronal through the labelled lowest body and the sacral alae
        y = int(round(c_low[1] - 8 / zoom[1]))
        panel(ax[1, 0], ct[:, y, :], lab[:, y, :], (zoom[0], zoom[2]),
              f"{case} (c) coronal, labelled {'L6' if c_l6 is not None else 'L5'} and the alae: process to ala?", (L5, L6, SACRUM, S1, 30, 31))
        ax[1, 0].set_ylim(c_sac[2] - 60 / zoom[2], c_low[2] + 50 / zoom[2])
        # (d) midline sagittal: disc under the lowest body, sacral height
        x = int(round(c_low[0]))
        panel(ax[1, 1], ct[x, :, :], lab[x, :, :], (zoom[1], zoom[2]),
              f"{case} (d) midline sagittal: disc beneath, sacral height", (T12, L1, L5, L6, SACRUM, S1))
        ax[1, 1].set_ylim(c_sac[2] - 80 / zoom[2], c_l1[2] + 40 / zoom[2])
        fig.suptitle(f"{case}: outlines  T12 orange, L1 cyan, L5 green, L6 magenta, sacrum red, S1 white", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{case}_reading_sheet.png", dpi=150)
        plt.close(fig)
        print("wrote", out / f"{case}_reading_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
