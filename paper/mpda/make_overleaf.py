r"""make_overleaf.py -- build the Overleaf project (overleaf/ and the zip) from main.tex.

The Overleaf copy differs from main.tex in two ways only:
  * the inline thebibliography is replaced by \bibliographystyle{medphy}\bibliography{ctspinopelvic1k},
    so Overleaf runs bibtex live and a reference added from Mendeley resolves without a
    local rebuild (one \bibliography command, never two -- the aapm substyle refuses two);
  * the document class is set to the journal's two-column reprint layout so the page
    count Overleaf shows is the one the ten-page limit applies to.

    python paper/mpda/make_overleaf.py
"""
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "overleaf"
ZIP = HERE / "CTSpinoPelvic1K_MDPA_overleaf.zip"

src = (HERE / "main.tex").read_text(encoding="utf-8")
a = src.index(r"\begin{thebibliography}")
b = src.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")
src = src[:a] + "\\bibliographystyle{medphy}\n\\bibliography{ctspinopelvic1k}\n" + src[b:]
src = re.sub(r",preprint(,linenumbers)?\]\{revtex4-2\}", ",reprint]{revtex4-2}", src, count=1)
assert src.count(r"\bibliography{") == 1

OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "main.tex").write_text(src, encoding="utf-8")
for name in ["ctspinopelvic1k.bib", "medphy.bst", "figure_captions.tex"]:
    shutil.copy(HERE / name, OUT / name)
figs = sorted(re.findall(r"includegraphics\[[^\]]*\]\{figures/([a-z_0-9]+)\.pdf\}", src))
for f in set(figs):
    shutil.copy(HERE / "figures" / f"{f}.pdf", OUT / "figures" / f"{f}.pdf")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(OUT).as_posix())
print(f"wrote {ZIP.name}: main.tex + bib + bst + captions + {len(set(figs))} figures")
