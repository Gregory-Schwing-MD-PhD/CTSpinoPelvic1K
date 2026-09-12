"""scripts/render_anchors.py — show the two anchors, and the count taken between them.

WHAT THIS FIGURE IS FOR. The paper says twice that naming a lumbar vertebra needs two fixed
ends, and then asks the reader to accept that on trust. This shows them, on real released
labels, in the view a surgeon or radiologist would look at.

  ROSTRAL ANCHOR: the lowest rib-bearing vertebra, together with the rib that makes it one.
  CAUDAL ANCHOR:  the sacrum. Earlier releases carved its cranial part as a separate S1
                  class; v11 withdraws that, because the plane estimate behind it was
                  degenerate on three quarters of the cohort. Both ids are taken
                  together here, so a v10 volume renders identically to a v11 one.

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
import matplotlib.patheffects as pe

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_turntable import render, BG                                # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"], "font.size": 9})

SACRUM, S1 = 26, 29
LUMBAR = list(range(20, 26))
THORACIC = list(range(8, 20))
import label_scheme as LS  # noqa: E402
RIB_L, RIB_R = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 1   # first rib id per side
N_RIBS = LS.N_RIBS
LUM_RIB = (LS.LUMBAR_RIB_LEFT, LS.LUMBAR_RIB_RIGHT)

C_ANCHOR_TOP = np.array([214, 69, 65], np.float32)     # rostral: lowest rib-bearing + rib
C_ANCHOR_BOT = np.array([48, 110, 190], np.float32)    # caudal: S1
C_BETWEEN = np.array([232, 176, 68], np.float32)       # the rib-free interval
C_MUTED = np.array([196, 196, 192], np.float32)
C_SACRUM = np.array([150, 150, 148], np.float32)


def rib_level(rib_id):
    """-> 1..12 for a numbered rib id, else None."""
    if RIB_L <= rib_id <= RIB_L + N_RIBS - 1:
        return rib_id - RIB_L + 1
    if RIB_R <= rib_id <= RIB_R + N_RIBS - 1:
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
    for r in list(range(RIB_L, RIB_L + N_RIBS)) + list(range(RIB_R, RIB_R + N_RIBS)):
        col[r] = C_MUTED
    # the caudal anchor: the sacrum entire, whether or not a carved S1 is present
    col[SACRUM] = C_ANCHOR_BOT
    for v in between:
        col[v] = C_BETWEEN
    if rostral is not None:
        col[rostral] = C_ANCHOR_TOP
    for r in on_it:
        col[r] = C_ANCHOR_TOP
    col[S1] = C_ANCHOR_BOT          # v10 volumes only; merged into 26 in v11
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


def fit_titles(fig, axes, gap_pt=12.0, floor=7.5):
    """Scale every panel title by ONE factor until no two ADJACENT titles touch.

    Matplotlib does not clip a title to its axes, so in a strip of panels a long
    caption silently runs into its neighbour. That shipped in Figure 2 of the dataset
    article: "(c) four, stump ribs on T12, fused junction" overlapped "(d)" by 2.8pt.

    The criterion is pairwise, not per-axes. Titles wider than the image beneath them
    are normal and look fine; two of the four here are. What must hold is that
    neighbours keep daylight between them. Fitting each title inside its own axes
    instead demanded a 0.63 shrink and drove the type below the legend under it.

    One factor for the whole strip: four captions set at four sizes reads as a
    mistake, so the tightest pair sets the size and the rest follow.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = gap_pt * fig.dpi / 72.0
    boxes = [ax.title.get_window_extent(renderer=r) for ax in axes]
    worst = 1.0
    for a, b in zip(boxes, boxes[1:]):
        pitch = (b.x0 + b.x1) / 2 - (a.x0 + a.x1) / 2
        half = (a.width + b.width) / 2
        if half + pad > pitch:
            worst = min(worst, (pitch - pad) / half)
    if worst >= 1.0:
        return None
    for ax in axes:
        ax.title.set_fontsize(max(floor, ax.title.get_fontsize() * worst))
    fig.canvas.draw()
    return worst


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
    ap.add_argument("--scale_mode",
                    choices=("image", "span", "vertebra", "mm"),
                    default="image",
                    help="image: render at true scale, then scale each whole panel up "
                         "so the smaller ones match the largest. span: equal "
                         "anchor-to-anchor height. vertebra: equal vertebral size. "
                         "mm: true millimetre scale, no rescaling.")
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
        rgb, ids2d = render(lab, col, a.angle, 255, return_ids=True)
        rgb = rgb.astype(np.float32)
        # rows are craniocaudal (zooms[2]), columns are left-right (zooms[0])
        fy, fx = zooms[2] / a.mm_per_px, zooms[0] / a.mm_per_px
        rgb = ndimage.zoom(rgb, (fy, fx, 1.0), order=1, mode="nearest")
        # order=0 on the id map: interpolating ids invents labels that do not exist
        ids2d = ndimage.zoom(ids2d, (fy, fx), order=0, mode="nearest")
        panels.append((case, caption, rgb, len(between), ids2d, list(between), rostral))
        print(f"  {case}: rostral={rostral} between={between} lumbar_rib={lumbar_rib}")

    def ink_box(rgb):
        """Bounding box of everything that is not background."""
        m = np.abs(rgb - BG.reshape(1, 1, 3)).sum(axis=2) > 12
        if not m.any():
            return None
        rr = np.nonzero(m.any(axis=1))[0]
        cc = np.nonzero(m.any(axis=0))[0]
        return int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1

    # ---- 1. CROP THE BORDER OFF. Everything after this is arithmetic; leaving the
    # renderer's padding in place is what made three previous alignment attempts fail.
    cropped = []
    for (case, caption, rgb, n, ids2d, between, rostral) in panels:
        bb = ink_box(rgb)
        if bb is None:
            cropped.append((case, caption, rgb, n, ids2d, between, rostral))
            continue
        r0, r1, c0, c1 = bb
        cropped.append((case, caption, rgb[r0:r1, c0:c1], n,
                        ids2d[r0:r1, c0:c1], between, rostral))
        print(f"  {case}: cropped to {r1-r0}x{c1-c0} (was {rgb.shape[0]}x{rgb.shape[1]})")
    panels = cropped

    # ---- 2. ONE SPAN. Scale every panel to the tallest, so the drawn heights match.
    target_h = max(p_.shape[0] for _, _, p_, _, _, _, _ in panels)
    scaled = []
    for (case, caption, rgb, n, ids2d, between, rostral) in panels:
        f = target_h / rgb.shape[0]
        if abs(f - 1.0) > 1e-6:
            rgb = ndimage.zoom(rgb, (f, f, 1.0), order=1, mode="nearest")
            ids2d = ndimage.zoom(ids2d, (f, f), order=0, mode="nearest")
        scaled.append((case, caption, rgb, n, ids2d, between, rostral))
        print(f"  {case}: scaled x{f:.3f} -> {rgb.shape[0]}x{rgb.shape[1]}")
    panels = scaled

    # ---- 3. ONE CANVAS, one offset. A little breathing room so the numerals beside the
    # rightmost column are not clipped by the panel edge.
    PAD_X = int(0.10 * max(p_.shape[1] for _, _, p_, _, _, _, _ in panels))
    PAD_Y = 4
    H = target_h + 2 * PAD_Y
    W = max(p_.shape[1] for _, _, p_, _, _, _, _ in panels) + 2 * PAD_X
    print(f"  every panel on a {H}x{W} canvas, top-aligned, spans equal")

    for ax, (case, caption, rgb, n, ids2d, between, rostral) in zip(axes, panels):
        canvas = np.full((H, W, 3), BG.reshape(1, 1, 3), np.float32)
        h, w = rgb.shape[:2]
        x0 = (W - w) // 2
        y0 = PAD_Y                      # top-aligned: the highest drawn pixel is on one line
        canvas[y0:y0 + h, x0:x0 + w] = rgb

        # numerals, from the id map so a shared colour cannot merge the column
        for k, vid in enumerate(sorted(between), start=1):
            m = (ids2d == vid)
            if not m.any():
                continue
            rr, cc = np.nonzero(m)
            cy_, cx_ = rr.mean() + y0, cc.mean() + x0
            tx = cx_ + 0.5 * (cc.max() - cc.min()) + 0.030 * W
            t = ax.annotate(str(k), xy=(cx_, cy_), xytext=(tx, cy_),
                            ha="center", va="center", fontsize=7.6, color="#141414",
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="#9a9a9a",
                                            shrinkA=2.0, shrinkB=2.0))
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

        ax.imshow(np.clip(canvas, 0, 255).astype(np.uint8), interpolation="bilinear")
        # PIN THE LIMITS TO THE CANVAS. ax.annotate expands the axes data limits to include
        # its text, and the numerals sit at a different x in every panel -- so matplotlib
        # autoscaled each axes differently and the same canvas landed at a different height
        # in each. That is what was left of the misalignment after the borders were cropped:
        # 12.5 pt of it, produced entirely by the labels rather than by the anatomy.
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the caption already states the count; repeating it on a second line collided
        # across panels at strip proportions and added nothing
        ax.set_title(caption, fontsize=8.5)

    f = fit_titles(fig, list(axes))
    if f:
        print(f"  titles shrunk to {f:.3f} so panel captions keep daylight")

    import matplotlib.patches as mp
    handles = [
        mp.Patch(color=C_ANCHOR_TOP / 255, label="rostral anchor: lowest rib-bearing "
                                                 "vertebra, with its rib"),
        mp.Patch(color=C_BETWEEN / 255, label="the rib-free vertebrae counted between them"),
        mp.Patch(color=C_ANCHOR_BOT / 255, label="caudal anchor: the sacrum"),
    ]
    # THE SAME GAP ABOVE AND BELOW THE KEY, AND THE SAME GAP FIG 1 HAS TO ITS CAPTION.
    # Figure 1 sits 5.8 pt above its caption and that is the distance to match: content to
    # key, then key to caption. The second half is set by how much white the tight crop
    # leaves under the key, since LaTeX adds a fixed 3.6 pt caption skip on top of it -- so
    # pad_inches goes NEGATIVE here, trimming the descender space matplotlib's tight bbox
    # reserves below the text but does not draw in. Both numbers are checked by rasterising
    # the reprint, not the figure: the figures are set at different scales on the page.
    fig.legend(handles=handles, loc="lower center", ncol=a.legend_cols, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, 0.069))
    fig.tight_layout(rect=(0, 0.13 if a.legend_cols == 1 else 0.08, 1, 1))
    out = Path(a.out) / f"{a.name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.001, dpi=300)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
