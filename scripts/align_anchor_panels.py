"""Put Figure 2's four renders at one scale and one vertical datum.

THE DEFECT. The four panel frames are placed perfectly -- identical 93.5 x 109.0 pt boxes on
a common baseline -- but the renders INSIDE them were made at different zooms and camera
heights. Measured: the ink spans 79.9 to 88.8 pt tall (an 11% scale spread) and its top edge
scatters over 19 pt. Panel (c) is both the smallest and the lowest, which is why it reads as
the odd one out.

WHAT "SAME SIZE" HAS TO MEAN HERE. Not equal ink height: a six-lumbar column (d) really is
longer than a four-lumbar one (b), and flattening that would erase the anatomy the figure
exists to show. It needs a feature the CAMERA fixed rather than one biology varied, and the
candidates were measured rather than assumed. Over the four panels the lumbar body width
varies by 4.3% and the S1 endplate width by 4.5%, while the anchor's on-screen depth varies
by 13.8% and the rib span by 36.2% -- the last being the phenotype itself. So the scale
comes from the mean of the two stable ones, and the corrections it asks for are all under 5%.

The iliac span was tried first and was wrong: the ilia flare at different angles between
these patients, so bone width at the anchor level is anatomy. Scaling on it demanded a 16%
enlargement of panel (d) and made it the largest in the strip instead of matching it.

AND ONE DATUM. The figure's argument is that the count runs between two anchors, so the
caudal anchor is what the eye should be able to track across the strip. The blue S1 is
uniquely coloured, so its centroid is found directly and every panel is translated to put it
on a common line. After that the rostral anchor's height varies across panels, which is the
finding, rather than the camera.

The frames, titles and legend are untouched vector content; only the raster inside each
frame is rebuilt.
"""
from __future__ import annotations

import sys

import numpy as np
import pymupdf
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper/mpda/figures/fig_anchors.pdf"
DST = sys.argv[2] if len(sys.argv) > 2 else SRC

doc = pymupdf.open(SRC)
page = doc[0]

placements = []
for im in page.get_images(full=True):
    xref = im[0]
    for r in page.get_image_rects(xref):
        placements.append((r, xref))
placements.sort(key=lambda t: t[0].x0)
assert len(placements) == 4, f"expected 4 panels, found {len(placements)}"


def rgb_of(xref):
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n >= 4 or pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[..., :3].copy()


def blue_mask(a):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (b > 90) & (b - r > 45) & (b - g > 25)


def ink_mask(a):
    return a.min(axis=2) < 245


panels = []
for i, (r, xref) in enumerate(placements):
    a = rgb_of(xref)
    bm, im_ = blue_mask(a), ink_mask(a)
    ys, xs = np.nonzero(bm)
    assert ys.size > 20, f"panel {i}: caudal anchor not found ({ys.size} blue px)"
    cy, cx = ys.mean(), xs.mean()

    # scale proxy: mean of the two features measured as camera-fixed (see docstring)
    bxs = np.nonzero(bm.any(axis=0))[0]
    blue_w = float(bxs[-1] - bxs[0] + 1)
    r_, g_, b_ = (a[..., k].astype(int) for k in range(3))
    red = (r_ > 110) & (r_ - g_ > 50) & (r_ - b_ > 50)
    orange = (r_ > 150) & (g_ > 90) & (r_ - b_ > 60) & (abs(r_ - g_) < 110) & ~red
    oxs = np.nonzero(orange.any(axis=0))[0]
    orange_w = float(oxs[-1] - oxs[0] + 1) if oxs.size else blue_w
    span = 0.5 * (blue_w + orange_w)
    panels.append(dict(i=i, rect=r, xref=xref, arr=a, cy=cy, cx=cx, span=span,
                       h=a.shape[0], w=a.shape[1]))
    print(f"panel {i}: anchor at ({cx:6.1f},{cy:6.1f}) px, blue_w {blue_w:5.1f} orange_w {orange_w:5.1f} -> scale ref {span:6.1f} px")

# one scale for all: the median span, so no panel is blown up much
target_span = float(np.median([p["span"] for p in panels]))
# one datum: the median anchor height as a fraction of frame, so the strip keeps its place
target_cy = float(np.median([p["cy"] / p["h"] for p in panels]))
print(f"\ntarget iliac span {target_span:.1f} px; "
      f"anchor datum at {target_cy:.3f} of frame height")

for p in panels:
    s = target_span / p["span"]
    img = Image.fromarray(p["arr"])
    nw, nh = max(1, int(round(p["w"] * s))), max(1, int(round(p["h"] * s)))
    img = img.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGB", (p["w"], p["h"]), (255, 255, 255))
    # place so the anchor lands on the common datum, horizontally centred on the frame
    ox = int(round(p["w"] / 2.0 - p["cx"] * s))
    oy = int(round(target_cy * p["h"] - p["cy"] * s))
    canvas.paste(img, (ox, oy))

    out = np.asarray(canvas)
    ys, xs = np.nonzero(blue_mask(out))
    print(f"panel {p['i']}: scale {s:.3f}, offset ({ox:+d},{oy:+d}) -> "
          f"anchor now at y={ys.mean():.1f} ({ys.mean()/p['h']:.3f} of frame)")
    p["out"] = canvas

# repaint: cover each frame in white, then draw the corrected raster in the same rect
for p in panels:
    page.draw_rect(p["rect"], color=None, fill=(1, 1, 1), overlay=True)
for p in panels:
    import io as _io
    buf = _io.BytesIO()
    p["out"].save(buf, format="PNG")
    page.insert_image(p["rect"], stream=buf.getvalue(), overlay=True)

doc.save(DST, garbage=4, deflate=True, incremental=False)
print(f"\nwrote {DST}")
