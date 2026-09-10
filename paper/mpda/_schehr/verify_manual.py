"""Check the claims that rested on sources I could not read, now that the PDFs are here.
Writes _schehr/manual_check.txt in UTF-8 (the console codepage cannot take the ligatures
these PDFs contain)."""
import re
import unicodedata
from pathlib import Path

import pymupdf

REFS = Path(r"C:\Users\grego\OneDrive\Desktop\CTSpinoPelvic1K-1\MANUAL_REFS")
OUT = Path(__file__).resolve().parent / "manual_check.txt"

TARGETS = {
    "KONIN_WALZ.pdf": ("Konin & Walz 2010 - the LSTV prevalence range",
                       [r"\d{1,2}(\.\d)?%[^.]{0,80}(population|prevalen|LSTV|transitional)",
                        r"(prevalen|incidence)[^.]{0,120}\d{1,2}(\.\d)?%",
                        r"ranges? from[^.]{0,80}%"]),
    "ACR.pdf": ("ACR Low Back Pain 2021 - is the imaging lumbar-only?",
                [r"MRI lumbar spine", r"CT lumbar spine", r"radiography lumbar spine",
                 r"Usually Appropriate[^.]{0,80}", r"lumbar spine without[^.]{0,60}"]),
    "NASS.pdf": ("NASS stenosis 2013 - imaging extent",
                 [r"MRI[^.]{0,100}lumbar", r"recommend[^.]{0,140}imaging",
                  r"CT[^.]{0,60}myelog[^.]{0,60}"]),
    "VERIDAH.pdf": ("VERIDAH - training set size and provenance",
                    [r"1[,. ]?536", r"training[^.]{0,90}(scans|subjects|images|CT)",
                     r"in-house", r"internal dataset[^.]{0,80}"]),
    "BENZEL_CH_1.pdf": ("Benzel ch.1 - morphometric values with or without spread",
                        [r"pedicle[^.]{0,120}(width|height|diameter|mm)",
                         r"vertebral body[^.]{0,80}(height|mm)",
                         r"(standard deviation|\bSD\b|range of|mean of)[^.]{0,90}",
                         r"Table[^.]{0,80}(dimension|morphometr|pedicle)"]),
}


def clean(t):
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"-\n", "", t)
    return re.sub(r"\s+", " ", t)


lines = []
for fname, (label, pats) in TARGETS.items():
    p = REFS / fname
    lines.append("=" * 100)
    lines.append(label)
    if not p.exists():
        lines.append("  MISSING")
        continue
    d = pymupdf.open(p)
    txt = clean(" ".join(pg.get_text() for pg in d))
    lines.append(f"  {len(d)} pages, {len(txt)} chars")
    seen = set()
    for pat in pats:
        for m in re.finditer(pat, txt, re.I):
            s = txt[max(0, m.start() - 160): m.end() + 200].strip()
            k = s[:70]
            if k in seen:
                continue
            seen.add(k)
            lines.append(f"  ... {s}")
            if len(seen) > 14:
                break
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT} ({len(lines)} lines)")
