"""Check the two Panjabi papers against the sentence that cites them:

    "vertebral body, endplate and canal dimensions differ by level"
    [panjabi1991, panjabi1992, benzel2015]

and against the neighbouring claim that reference morphometry is reported without spread.
Writes _schehr/panjabi_check.txt in UTF-8."""
import re
import unicodedata
from pathlib import Path

import pymupdf

REFS = Path(r"C:\Users\grego\OneDrive\Desktop\CTSpinoPelvic1K-1\MANUAL_REFS")
OUT = Path(__file__).resolve().parent / "panjabi_check.txt"

PATS = [
    r"(vertebral body|endplate|end-plate|end plate)[^.]{0,120}(height|width|depth|dimension)",
    r"(spinal canal|canal)[^.]{0,100}(width|depth|diameter|dimension|area)",
    r"(varied|vary|varies|increase[sd]?|decrease[sd]?|differ)[^.]{0,110}(level|T1|L1|caudal|cranial)",
    r"(mean|average)[^.]{0,60}(standard deviation|SD|\u00b1)",
    r"\b(\d{1,2}) (fresh|cadaver|specimens?|vertebrae)\b",
    r"(specimens?|cadaver)[^.]{0,90}(studied|measured|obtained|used)",
    r"quantitative[^.]{0,90}(anatomy|measurement|three-dimensional)",
    r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|\d{2,3})[^.]{0,40}(vertebrae|specimens)",
]


def clean(t):
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"-\n", "", t)
    return re.sub(r"\s+", " ", t)


lines = []
for fname, label in (("panjabi1991.pdf", "Panjabi 1991 - Thoracic human vertebrae"),
                     ("panjabi1992.pdf", "Panjabi 1992 - Human lumbar vertebrae")):
    p = REFS / fname
    lines.append("=" * 100)
    lines.append(label)
    if not p.exists():
        lines.append("  MISSING")
        continue
    d = pymupdf.open(p)
    txt = clean(" ".join(pg.get_text() for pg in d))
    lines.append(f"  {len(d)} pages, {len(txt)} chars")
    lines.append(f"  --- opening ---")
    lines.append("  " + txt[:700])
    lines.append("  --- matches ---")
    seen = set()
    for pat in PATS:
        for m in re.finditer(pat, txt, re.I):
            s = txt[max(0, m.start() - 150): m.end() + 190].strip()
            k = s[:60]
            if k in seen:
                continue
            seen.add(k)
            lines.append(f"  ... {s}")
            if len(seen) > 16:
                break
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
