r"""paper/mpda/make_supplemental_files.py -- one PDF per supplemental item.

The journal takes supporting information as separately numbered uploads ("Supplemental
Material 1", "Supplemental Material 2", ...), not as a single combined document. This
splits supplement_body.tex into one standalone PDF per float and names them the way the
form expects.

ONE SOURCE STILL. The floats are not retyped here: each block is lifted out of
supplement_body.tex, which is the same file the arXiv build appends to the article and the
combined supplementary.pdf inputs. Three copies of a figure caption would drift, and the
article cites these as Fig. S1..S3 and Table S1..S3, so the numbering has to agree
everywhere.

    python paper/mpda/make_supplemental_files.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "dist" / "submission" / "supplemental"

PREAMBLE = r"""\documentclass[aapm,mph,amsmath,amssymb,reprint]{revtex4-2}
\usepackage{graphicx}
% built in _supp_build/, one level under the figures
\graphicspath{{../}}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\newsavebox{\fitbox}
\newcommand{\maxwidth}[1]{%
  \sbox{\fitbox}{#1}%
  \ifdim\wd\fitbox>\linewidth
    \resizebox{\linewidth}{!}{\usebox{\fitbox}}%
  \else
    \usebox{\fitbox}%
  \fi
}
\begin{document}
\onecolumngrid
\begin{center}
{\large\bfseries Supplemental Material @@NUM@@}\\[3pt]
CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame,\\
annotated for lumbosacral transitional anatomy
\end{center}
\vspace{6pt}
@@COUNTERS@@
"""

TAIL_BIB = r"""
\begin{thebibliography}{1}
\bibitem{vialle2005}
R.~Vialle, N.~Levassor, L.~Rillardon, A.~Templier, W.~Skalli, and P.~Guigui,
\newblock Radiographic analysis of the sagittal alignment and balance of the spine in asymptomatic subjects,
\newblock J. Bone Joint Surg. Am. {\bf 87}, 260--267 (2005),
\newblock doi:10.2106/JBJS.D.02043.
\end{thebibliography}
"""


def float_blocks(src: str):
    """Each float in supplement_body.tex, in order, with its kind and S-number."""
    out, fig_n, tab_n = [], 0, 0
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}", src):
        kind = m.group(1)
        star = src[m.end() - 2:m.end() - 1] == "*"
        endtok = "\\end{%s%s}" % (kind, "*" if star else "")
        e = src.find(endtok, m.end())
        if e < 0:
            continue
        block = src[m.start():e + len(endtok)]
        if kind == "figure":
            fig_n += 1
            out.append(("figure", fig_n, block))
        else:
            tab_n += 1
            out.append(("table", tab_n, block))
    return out


def main() -> int:
    body = (HERE / "supplement_body.tex").read_text(encoding="utf-8")
    # the census table is \input, not written inline; pull it in so it can be split out too
    census = (HERE / "census_table.tex").read_text(encoding="utf-8")
    body = body.replace("\\input{census_table}", census)

    blocks = float_blocks(body)
    if not blocks:
        print("no floats found in supplement_body.tex")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    work = HERE / "_supp_build"
    work.mkdir(exist_ok=True)
    made = []
    for i, (kind, n, block) in enumerate(blocks, 1):
        label = "Fig. S%d" % n if kind == "figure" else "Table S%d" % n
        # force the S-numbering this item carries in the article
        counters = ("\\setcounter{figure}{%d}\\renewcommand{\\thefigure}{S\\arabic{figure}}"
                    % (n - 1)) if kind == "figure" else (
                   "\\setcounter{table}{%d}\\renewcommand{\\thetable}{S\\arabic{table}}"
                   % (n - 1))
        tex = PREAMBLE.replace("@@NUM@@", str(i)).replace("@@COUNTERS@@", counters) + chr(10) + block + chr(10)
        if "\\cite{" in block:
            tex += TAIL_BIB
        tex += "\n\\end{document}\n"
        stem = "Supplemental_Material_%d" % i
        (work / (stem + ".tex")).write_text(tex, encoding="utf-8", newline="\n")
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", stem],
                           cwd=work, capture_output=True, text=True, timeout=600)
        pdf = work / (stem + ".pdf")
        log = (work / (stem + ".log")).read_text(encoding="utf-8", errors="replace")
        errs = [l for l in log.splitlines() if l.startswith("!")]
        if not pdf.exists():
            print("  FAILED %s (%s): no pdf" % (stem, label))
            return 1
        if errs:
            print("  FAILED %s (%s): %s" % (stem, label, errs[0]))
            return 1
        # every item must actually contain its float, not just compile
        import fitz
        txt = "".join(p.get_text() for p in fitz.open(str(pdf)))
        if label.replace(" ", "") not in txt.replace(" ", ""):
            print("  FAILED %s: %s not present in the rendered page" % (stem, label))
            return 1
        shutil.copy(pdf, OUT / (stem + ".pdf"))
        made.append((stem, label))
        print("  %-26s %s" % (stem + ".pdf", label))

    shutil.rmtree(work, ignore_errors=True)
    print("\n%d supplemental files in %s" % (len(made), OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
