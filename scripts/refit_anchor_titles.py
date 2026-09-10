"""scripts/refit_anchor_titles.py -- redraw Figure 2's titles so panel (c) stops
colliding with panel (d).

THE BUG. render_anchors.py sets every panel title at a fixed 8.5pt, centred on its
panel. Matplotlib does not clip a title, so a caption wider than the space between
two panel centres runs into its neighbour. Measured on the shipped figure, panel (c)
ends at x=438.6pt and panel (d) starts at x=435.8pt: a 2.8pt overlap.

THE RIGHT CRITERION IS NOT "each title fits inside its own axes". In a tight strip
of panels the titles are routinely wider than the image beneath them and that looks
fine; (a) and (b) are both wider here and nobody noticed. What must hold is that
ADJACENT titles do not touch. Two titles centred a pitch p apart collide when
(w_left + w_right)/2 > p. Fitting to the axes width instead demanded a 0.63 shrink,
which drove the type below the legend beneath it -- worse than the overlap it fixed.
On the pairwise rule the same figure needs 0.95, which is invisible.

WHY THIS SCRIPT EXISTS rather than just re-running render_anchors.py: that script
needs the label volumes, and the four case identifiers it was invoked with were never
recorded, in a shell script, a note or the commit message. Guessing them would change
which patients the figure shows. So the four rendered panels are lifted out of the
existing PDF and recomposited at the original's exact geometry: identical pixels,
identical placement, correct titles.

The permanent fix is the same rule inside render_anchors.py. This is the one-off
repair of the already-shipped figure.

    python scripts/refit_anchor_titles.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from PIL import Image

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"],
                     "font.size": 9})

C_ANCHOR_TOP = np.array([214, 69, 65], np.float32) / 255
C_ANCHOR_BOT = np.array([48, 110, 190], np.float32) / 255
C_BETWEEN = np.array([232, 176, 68], np.float32) / 255

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "mpda" / "figures" / "fig_anchors.pdf"

TITLES = [
    "(a) five rib-free, last rib on T12",
    "(b) four, long rib on L1",
    "(c) four, stump ribs on T12, fused junction",
    "(d) six rib-free, last rib on T12",
]
LEGEND = [
    (C_ANCHOR_TOP, "rostral anchor: lowest rib-bearing vertebra, with its rib"),
    (C_BETWEEN, "the rib-free vertebrae counted between them"),
    (C_ANCHOR_BOT, "caudal anchor: S1, carved from the sacrum"),
]
TITLE_PT = 8.5
GAP_PT = 12.0        # daylight between neighbouring titles; 4pt cleared the collision
                     # but still read as one run-on line, which is what was reported


def fit_titles(fig, axes, gap_pt=GAP_PT, floor=7.5):
    """Scale every title by ONE factor until no two ADJACENT titles touch.

    One factor for all of them: a strip whose four captions are set at four sizes
    reads as a mistake. The tightest pair sets the size and the rest follow.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi_scale = fig.dpi / 72.0                      # window extents are in pixels
    boxes = [ax.title.get_window_extent(renderer=r) for ax in axes]
    worst = 1.0
    for a, b in zip(boxes, boxes[1:]):
        pitch = (b.x0 + b.x1) / 2 - (a.x0 + a.x1) / 2
        need = (a.width + b.width) / 2 + gap_pt * dpi_scale
        if need > pitch:
            worst = min(worst, (pitch - gap_pt * dpi_scale) / ((a.width + b.width) / 2))
    if worst >= 1.0:
        return None
    for ax in axes:
        ax.title.set_fontsize(max(floor, ax.title.get_fontsize() * worst))
    fig.canvas.draw()
    return worst


def main():
    doc = pymupdf.open(FIG)
    page = doc[0]
    W, H = page.rect.width, page.rect.height

    # panels, with the rectangle each was drawn into, left to right
    found = []
    for info in page.get_images(full=True):
        rects = page.get_image_rects(info[0])
        if not rects:
            continue
        pix = pymupdf.Pixmap(doc, info[0])
        if pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        found.append((rects[0], np.array(Image.open(io.BytesIO(pix.tobytes("png"))))))
    found.sort(key=lambda t: t[0].x0)
    if len(found) != len(TITLES):
        raise SystemExit(f"expected {len(TITLES)} panels, found {len(found)}")
    print(f"  lifted {len(found)} panels, {found[0][1].shape[1]}x{found[0][1].shape[0]} px, "
          f"drawn {found[0][0].width:.1f}x{found[0][0].height:.1f}pt each")

    # exact page size, axes placed exactly where the originals were drawn: no
    # tight_layout and no tight bbox, both of which move the panels
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    axes = []
    for (r, img), title in zip(found, TITLES):
        ax = fig.add_axes([r.x0 / W, (H - r.y1) / H, r.width / W, r.height / H])
        ax.imshow(img, interpolation="bilinear", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=TITLE_PT)
        axes.append(ax)

    handles = [mp.Patch(color=c, label=t) for c, t in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, 0.0))

    f = fit_titles(fig, axes)
    print(f"  titles {TITLE_PT}pt -> {TITLE_PT * f:.2f}pt (factor {f:.3f})" if f
          else "  titles already clear")

    fig.savefig(FIG, dpi=300)
    plt.close(fig)
    doc.close()

    d = pymupdf.open(FIG)
    spans = sorted((s for b in d[0].get_text("dict")["blocks"] for l in b.get("lines", [])
                    for s in l["spans"] if s["text"].strip().startswith("(")),
                   key=lambda s: s["bbox"][0])
    print(f"  page {d[0].rect.width:.1f}x{d[0].rect.height:.1f}pt, {len(spans)} titles")
    bad = 0
    for a, b in zip(spans, spans[1:]):
        gap = b["bbox"][0] - a["bbox"][2]
        bad += gap < 0
        print(f"    gap {gap:6.1f}pt after {a['text'][:36]!r}" + ("  <-- OVERLAP" if gap < 0 else ""))
    if bad:
        raise SystemExit("titles still overlap")


if __name__ == "__main__":
    main()
