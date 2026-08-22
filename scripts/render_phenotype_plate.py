"""scripts/render_phenotype_plate.py — the phenotype plate for the dataset article.

WHY A FIGURE OF RENDERS AT ALL. A dataset article about transitional anatomy that shows
no anatomy asks the reader to take the phenotypes on trust. The variants this corpus
exists to capture are visible in a single posterior view of the lumbosacral junction --
that is how they are read clinically -- so the figure shows exactly that, one column per
phenotype, at one shared viewing angle so the columns differ only in the patient.

THE PHENOTYPE IS THE RIB-FREE COUNT, NOT A LEVEL NAME. Each column is captioned by the
number of rib-free vertebrae between the lowest rib-bearing vertebra and the sacrum.
Five is typical. Four and six are the transitional configurations, and naming them "L5
sacralised" or "S1 lumbarised" would import a convention the image cannot settle -- a
spine-limited field of view cannot tell those two apart, which is the whole argument of
the paper. The count can be checked against the picture; a level name cannot.

Surfaces and colours come from the same renderer and the same ITK-SNAP descriptor the
gallery uses, so a reader comparing figure to website sees one object, not two.

    python scripts/render_phenotype_plate.py --labels data/v5_final \
        --cases '0004:five;0008:four' --out paper/mpda/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402
from render_turntable import read_colours, render, MIN_VOX         # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(fp, downsample):
    """Canonicalise to ANALYSE only -- nothing here is ever written back.

    Returns the cropped label array and the post-downsample voxel size along S.
    """
    img = nib.as_closest_canonical(nib.load(str(fp)))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zs_mm = float(img.header.get_zooms()[2]) * downsample
    for vid in np.unique(lab):
        if vid and (lab == vid).sum() < MIN_VOX:
            lab[lab == vid] = 0
    lab = lab[::downsample, ::downsample, ::downsample]
    occ = np.argwhere(lab > 0)
    if not len(occ):
        return None, zs_mm
    lo, hi = occ.min(0), occ.max(0) + 1
    return lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], zs_mm


def crop_lumbosacral(lab, zs_mm, above_mm=200.0):
    """Keep the pelvis and a fixed height of spine above the top of the sacrum.

    ANCHORED ON THE SACRUM, NOT ON A FRACTION OF THE LABELLED HEIGHT. Cases differ in how
    much thorax the scan covered, so a fractional crop frames a different part of the
    skeleton in every column -- an earlier version framed mid-thorax in three panels of
    four and cut the junction out of the figure entirely. Anchoring on the sacrum puts the
    same anatomy in every column, which is the only way the columns are comparable.

    The span above the anchor is in millimetres and identical for every case, so a taller
    patient shows fewer levels rather than a rescaled skeleton.
    """
    sac = np.isin(lab, [LS.SACRUM_ID, LS.S1_ID])
    zs = np.nonzero(sac.any(axis=(0, 1)))[0]
    if not len(zs):                                        # no sacrum: fall back to base
        zs = np.nonzero(lab.any(axis=(0, 1)))[0]
        if not len(zs):
            return lab
        top = int(zs.min()) + int(round(0.45 * (zs.max() - zs.min())))
    else:
        top = int(zs.max())
    hi = min(lab.shape[2], top + int(round(above_mm / max(zs_mm, 1e-6))))
    lo = int(np.nonzero(lab.any(axis=(0, 1)))[0].min())
    return lab[:, :, lo:hi]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True,
                    help="semicolon list of case:caption, e.g. '0004:five;0008:four'. "
                         "Semicolon, not comma -- captions contain commas.")
    ap.add_argument("--descriptor", default="data/itksnap_v5_labels.txt")
    ap.add_argument("--angle", type=float, default=0.0, help="0 = posterior view")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--above_mm", type=float, default=200.0,
                    help="height of spine kept above the top of the sacrum")
    ap.add_argument("--out", default="paper/mpda/figures")
    ap.add_argument("--name", default="fig2_phenotypes")
    a = ap.parse_args()

    colours = read_colours(a.descriptor)
    spec = []
    for tok in a.cases.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        case, _, cap = tok.partition(":")
        spec.append((case, cap or case))

    panels = []
    for case, cap in spec:
        fp = Path(a.labels) / f"{case}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        lab, zs_mm = load(fp, a.downsample)
        if lab is None:
            continue
        lab = crop_lumbosacral(lab, zs_mm, a.above_mm)
        fr = render(lab, colours, a.angle, LS.IGNORE_LABEL)
        if fr is None:
            print(f"  ! {case} rendered empty")
            continue
        panels.append((case, cap, fr))
        print(f"  {case}: {fr.shape[1]}x{fr.shape[0]}")

    if not panels:
        print("  nothing rendered")
        return 1

    # A COMMON CANVAS FOR EVERY PANEL. matplotlib sizes each axes to its own image, so
    # panels of different pixel aspect got different widths and their titles ran into one
    # another. Padding to one canvas first makes every axes identical, which also means a
    # reader comparing two columns is comparing anatomy and not layout.
    H = max(p[2].shape[0] for p in panels)
    W = max(p[2].shape[1] for p in panels)
    bg = panels[0][2][0, 0]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(7.0, 7.0 / len(panels) * H / W + 0.35))
    if len(panels) == 1:
        axes = [axes]
    letters = "abcdefgh"
    for ax, (case, cap, fr), L in zip(axes, panels, letters):
        pad = np.empty((H, W, 3), np.uint8)
        pad[:] = bg
        oy, ox = (H - fr.shape[0]) // 2, (W - fr.shape[1]) // 2
        pad[oy:oy + fr.shape[0], ox:ox + fr.shape[1]] = fr
        ax.imshow(pad, interpolation="bilinear")
        # set_axis_off() would hide the xlabel with everything else, so the frame and
        # ticks are removed individually and the label kept
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # captions below the panel: above it they collide with the neighbouring column,
        # and a caption that collides is worse than no caption
        ax.set_xlabel(f"({L}) {cap}", fontsize=7.5, labelpad=4)

    fig.subplots_adjust(wspace=0.02)
    for ext in ("pdf", "png"):
        p = Path(a.out) / f"{a.name}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=400)
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
