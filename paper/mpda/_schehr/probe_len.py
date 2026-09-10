"""Probe: how much text must come out for the reprint to fit ten pages?

Cuts N characters of body prose (whole paragraphs from the end of the Discussion, which
is where prose is most compressible) and reports the resulting page count. Nothing is
written back to main.tex; this only measures.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPDA = HERE.parent
WORK = Path("/tmp/probe")

src = (MPDA / "main.tex").read_text(encoding="utf-8")
target = int(sys.argv[1]) if len(sys.argv) > 1 else 0

body_start = src.index(r"\section{Introduction}")
bib_start = src.index(r"\begin{thebibliography}")
head, body, tail = src[:body_start], src[body_start:bib_start], src[bib_start:]

# drop trailing sentences from the longest paragraphs until `target` characters are gone
paras = body.split("\n\n")
order = sorted(range(len(paras)), key=lambda i: -len(paras[i]))
removed = 0
for i in order:
    if removed >= target:
        break
    p = paras[i]
    if p.lstrip().startswith(("%", "\\begin", "\\section", "\\subsection")):
        continue
    if len(p) < 300:
        continue
    keep = max(200, len(p) - (target - removed))
    cut = p[keep:]
    removed += len(cut)
    paras[i] = p[:keep] + ("" if p.endswith("\n") else "")

WORK.mkdir(exist_ok=True)
(WORK / "figures").mkdir(exist_ok=True)
out = head + "\n\n".join(paras) + tail
out = re.sub(r",preprint(,linenumbers)?\]\{revtex4-2\}", ",reprint]{revtex4-2}", out, count=1)
(WORK / "main.tex").write_text(out, encoding="utf-8")
for f in (MPDA / "figures").glob("*.pdf"):
    shutil.copy(f, WORK / "figures" / f.name)
for n in ("figure_captions.tex", "census_table.tex"):
    if (MPDA / n).exists():
        shutil.copy(MPDA / n, WORK / n)

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=WORK, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import pymupdf
d = pymupdf.open(WORK / "main.pdf")
last = d[-1]
y = max(b[3] for b in last.get_text("blocks"))
print(f"cut {removed:5d} chars -> {len(d)} pages, last page fills to y={y:.0f}")
