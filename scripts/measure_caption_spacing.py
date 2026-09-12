"""Caption and key spacing in the reprint, measured ink to ink.

    python scripts/measure_caption_spacing.py

Every figure should show the same gap between its artwork and its caption, and a
figure carrying a key should show that gap twice: above the key and below it.

THREE TRAPS, ALL HIT ON THE WAY HERE.

  1. Scanning ink across the full page width is wrong for a single-column float: the other
     column carries body text at the same heights, so the "last ink above the caption" is
     that text and the number never responds to the figure.
  2. Deriving the column from the words on the caption's FIRST LINE fails the same way -- a
     line of the other column sits at the same y and joins the x-span. The column comes from
     the caption BLOCK instead, which PyMuPDF splits by column.
  3. A text block's bbox top sits about 1.7pt above the glyphs, so mixing block tops with
     word tops silently shifts every number. Everything below is INK: the topmost and
     bottommost drawn pixel, which is also what the eye measures.

A figure carrying a key reports two distances. The key should sit the same distance from
the panels it explains as from the caption that follows it.
"""
from __future__ import annotations

import re

import numpy as np
import pymupdf

DPI = 400
KEYWORD = {2: "rostral", 3: "arthroplasty", 4: "standing"}
PDF = "paper/mpda/CTSpinoPelvic1K_reprint_check.pdf"


def ink_mask(page):
    pm = page.get_pixmap(dpi=DPI, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    return (a.min(axis=2) < 245), pm.height / page.rect.height, pm.width / page.rect.width


out = []
for page in pymupdf.open(PDF):
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        m = re.match(r"^FIG\.\s*(\d+)", text.strip(), re.I)
        if not m:
            continue
        n = int(m.group(1))
        mask, sy, sx = ink_mask(page)
        col = mask[:, int(x0 * sx): int(x1 * sx)]
        rows = np.nonzero(col.any(axis=1))[0] / sy          # every inked row, in points

        cap_top = rows[rows >= y0 - 1].min()                # first ink of the caption
        words = page.get_text("words")

        if n in KEYWORD:
            hits = [w for w in words
                    if KEYWORD[n] in w[4].lower() and w[3] < y0 and w[0] >= x0 - 1]
            kt = max(h[1] for h in hits)
            kb = max(h[3] for h in hits if abs(h[1] - kt) < 2)
            band = rows[(rows >= kt - 2) & (rows <= kb + 2)]
            key_top, key_bot = band.min(), band.max()
            content = rows[rows < key_top - 0.5].max()
            out.append((n, f"content->key {key_top - content:5.2f}   "
                           f"key->caption {cap_top - key_bot:5.2f}"))
        else:
            content = rows[rows < cap_top - 0.5].max()
            out.append((n, f"content->caption {cap_top - content:5.2f}"))

for n, line in sorted(out):
    print(f"Fig {n}: {line}")
