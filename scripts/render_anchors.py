"""scripts/render_anchors.py — show the two anchors, and the count taken between them.

WHAT THIS FIGURE IS FOR. The paper says twice that naming a lumbar vertebra needs two fixed
ends, and then asks the reader to accept that on trust. This shows them, on real released
labels, in the view a surgeon or radiologist would look at.

  ROSTRAL ANCHOR: the lowest rib-bearing vertebra, together with the rib that makes it one.
  CAUDAL ANCHOR:  S1, carved out of the sacrum as its own label.

Everything between the two is the interval the dataset actually reports, and it is drawn in
a third colour and counted on the figure. Everything else is muted, because the point is not
the skeleton, it is the two ends and the gap.

WHY THIS IS NOT THE PHENOTYPE PLATE. That figure shows the configurations side by side and
captions them by count. This one explains where the count comes from. A reader who has not
met the problem needs the second before the first is meaningful.

THE COLOURS ARE ASSIGNED BY ROLE, NOT BY LEVEL. The rostral anchor is the lowest rib-bearing
vertebra whatever its number: in a case with a lumbar rib that is a vertebra a level below
where it would ordinarily sit, and colouring by role rather than by identifier is what makes
the three panels comparable. A figure that coloured T12 specifically would be asserting the
count it is supposed to be deriving.

    python scripts/render_anchors.py --labels data/hf_export_v5/labels \\
        --cases '0007:five rib-free (typical);0008:four, via a lumbar rib;0005:six rib-free'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_turntable import render                                # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})

SACRUM, S1 = 26, 29
LUMBAR = list(range(20, 26))
THORACIC = list(range(8, 20))
RIB_L, RIB_R = 34, 46            # rib_left_1 .. rib_left_12 = 34..45; right = 46..57
LUM_RIB = (74, 75)

C_ANCHOR_TOP = np.array([214, 69, 65], np.float32)     # rostral: lowest rib-bearing + rib
C_ANCHOR_BOT = np.array([48, 110, 190], np.float32)    # caudal: S1
C_BETWEEN = np.array([232, 176, 68], np.float32)       # the rib-free interval
C_MUTED = np.array([196, 196, 192], np.float32)
C_SACRUM = np.array([150, 150, 148], np.float32)


def rib_level(rib_id):
    """-> 1..12 for a numbered rib id, else None."""
    if RIB_L <= rib_id <= RIB_L + 11:
        return rib_id - RIB_L + 1
    if RIB_R <= rib_id <= RIB_R + 11:
        return rib_id - RIB_R + 1
    return None


def analyse(lab):
    """-> (rostral vertebra id, rib ids on it, interval vertebra ids, has lumbar rib).

    The rostral anchor is the LOWEST vertebra that carries a rib, found from the labels
    rather than assumed to be T12: a lumbar rib moves it down a level, and that is the
    whole point of the figure.
    """
    present = set(int(v) for v in np.unique(lab))
    lumbar_rib = any(r in present for r in LUM_RIB)

    # the lowest vertebra bearing a numbered rib
    rib_ids = sorted(r for r in present if rib_level(r) is not None)
    top_level = max((rib_level(r) for r in rib_ids), default=None)
    rostral, on_it = None, []
    if top_level is not None:
        vid = 7 + top_level                       # T-n has id 7+n
        if vid in present:
            rostral = vid
            on_it = [r for r in rib_ids if rib_level(r) == top_level]

    if lumbar_rib:
        # a rib on a lumbar body: that body becomes the rostral anchor
        for v in LUMBAR:
            if v in present:
                rostral = v
                on_it = [r for r in LUM_RIB if r in present]
                break

    between = [v for v in LUMBAR if v in present and (rostral is None or v > rostral)]
    return rostral, on_it, between, lumbar_rib


def colours_for(lab):
    rostral, on_it, between, lumbar_rib = analyse(lab)
    col = {}
    for v in THORACIC + LUMBAR:
        col[v] = C_MUTED
    for r in list(range(RIB_L, RIB_L + 12)) + list(range(RIB_R, RIB_R + 12)):
        col[r] = C_MUTED
    col[SACRUM] = C_SACRUM
    for v in between:
        col[v] = C_BETWEEN
    if rostral is not None:
        col[rostral] = C_ANCHOR_TOP
    for r in on_it:
        col[r] = C_ANCHOR_TOP
    col[S1] = C_ANCHOR_BOT
    for extra in (30, 31, 32, 33):
        col[extra] = np.array([225, 225, 222], np.float32)
    return col, rostral, between, lumbar_rib


def crop(lab, rostral, margin_mm, zooms):
    """Frame from the bottom of the sacrum to just above the ROSTRAL ANCHOR.

    A fixed height above the sacrum does not work here and the first version of this figure
    showed why: a spine with six rib-free vertebrae is longer, so its lowest rib-bearing
    vertebra sits higher and fell outside a 170 mm window entirely. The panel then had no
    upper anchor to draw, which is the one thing the figure exists to show. Framing on the
    anchor itself guarantees both ends are in every panel, and the panels are then directly
    comparable in the only respect that matters: how much spine lies between them.
    """
    sac = np.nonzero((lab == SACRUM) | (lab == S1))
    if not len(sac[2]):
        return lab
    zlo = max(0, int(sac[2].min()) - 5)
    if rostral is not None and (lab == rostral).any():
        ztop = int(np.nonzero((lab == rostral))[2].max())
    else:
        ztop = int(sac[2].max())
    zhi = min(lab.shape[2], ztop + int(round(margin_mm / max(zooms[2], 1e-6))))
    out = lab.copy()
    out[:, :, :zlo] = 0
    out[:, :, zhi:] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--cases", required=True, help="semicolon list of case:caption")
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--margin_mm", type=float, default=18.0,
                    help="how much to keep above the rostral anchor")
    ap.add_argument("--mm_per_px", type=float, default=0.75,
                    help="common scale for every panel")
    ap.add_argument("--panel_in", type=float, default=3.4)
    ap.add_argument("--height_in", type=float, default=5.2)
    ap.add_argument("--legend_cols", type=int, default=1)
    ap.add_argument("--out", default="paper/mpda/figures")
    ap.add_argument("--name", default="fig_anchors")
    a = ap.parse_args()

    items = [c.split(":", 1) for c in a.cases.split(";") if c.strip()]
    fig, axes = plt.subplots(1, len(items), figsize=(a.panel_in * len(items), a.height_in))
    if len(items) == 1:
        axes = [axes]

    # EVERY PANEL ON ONE MILLIMETRE SCALE, ALIGNED AT THE SACRUM. Voxel sizes differ between
    # cases, so rendering each at its own resolution makes a panel with larger voxels look
    # like a larger patient -- and the quantity this figure is about is exactly how much
    # spine sits between the anchors. Each render is resampled to a common mm-per-pixel and
    # the panels are padded to a shared canvas with their sacral bases on one line.
    panels = []
    for case, caption in items:
        f = Path(a.labels) / f"{case}_label.nii.gz"
        img = nib.as_closest_canonical(nib.load(str(f)))
        lab = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()[:3]
        # colours and anchors come from the FULL volume; cropping first is what made the
        # six-rib-free case report no rostral anchor at all
        col, rostral, between, lumbar_rib = colours_for(lab)
        lab = crop(lab, rostral, a.margin_mm, zooms)
        # render() already transposes so superior is up and flips to read as a radiograph
        # does (render_turntable.py:97). Transposing again here laid the spine on its side.
        rgb = render(lab, col, a.angle, 255).astype(np.float32)
        # rows are craniocaudal (zooms[2]), columns are left-right (zooms[0])
        fy, fx = zooms[2] / a.mm_per_px, zooms[0] / a.mm_per_px
        rgb = ndimage.zoom(rgb, (fy, fx, 1.0), order=1, mode="nearest")
        panels.append((case, caption, rgb, len(between)))
        print(f"  {case}: rostral={rostral} between={between} lumbar_rib={lumbar_rib}")

    H = max(p_.shape[0] for _, _, p_, _ in panels)
    W = max(p_.shape[1] for _, _, p_, _ in panels)
    for ax, (case, caption, rgb, n) in zip(axes, panels):
        canvas = np.full((H, W, 3), 250.0, np.float32)
        h, w = rgb.shape[:2]
        x0 = (W - w) // 2
        canvas[H - h:, x0:x0 + w] = rgb          # bottom-aligned: sacral bases on one line
        ax.imshow(np.clip(canvas, 0, 255).astype(np.uint8), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the caption already states the count; repeating it on a second line collided
        # across panels at strip proportions and added nothing
        ax.set_title(caption, fontsize=8.5)

    import matplotlib.patches as mp
    handles = [
        mp.Patch(color=C_ANCHOR_TOP / 255, label="rostral anchor: lowest rib-bearing "
                                                 "vertebra, with its rib"),
        mp.Patch(color=C_BETWEEN / 255, label="the rib-free vertebrae counted between them"),
        mp.Patch(color=C_ANCHOR_BOT / 255, label="caudal anchor: S1, carved from the sacrum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=a.legend_cols, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.13 if a.legend_cols == 1 else 0.08, 1, 1))
    out = Path(a.out) / f"{a.name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
