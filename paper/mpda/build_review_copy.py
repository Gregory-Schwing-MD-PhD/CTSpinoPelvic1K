"""build_review_copy.py -- one PDF and one Word file for co-author review.

TWO ARTEFACTS, ONE PURPOSE. Co-authors need to read the whole thing and mark it up, which
the submission set does not support: it is deliberately split into three documents, and the
manuscript is anonymised so it does not even carry their names.

  CTSpinoPelvic1K_review.pdf   title page, then the manuscript, then the supplement, in one
                               file. Built from the double-spaced line-numbered build, not
                               the two-column one, because that is the version you can
                               actually write comments on -- and line numbers give everyone
                               a shared way to refer to a sentence.

  CTSpinoPelvic1K_review.docx  the same text in Word, for anyone who reviews with track
                               changes.

WHAT THE WORD FILE IS AND IS NOT. It is a readable manuscript, not the submission artefact.
REVTeX constructs have no Word equivalent, so tables lose their rules, figures are
referenced rather than embedded at print resolution, and the typesetting is pandoc's.
Comments should come back as text; the LaTeX source stays the master copy.

    python build_review_copy.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
WSL = "/mnt/c/Users/grego/OneDrive/Desktop/CTSpinoPelvic1K-1/paper/mpda"


def merge_pdf():
    import fitz
    out = fitz.open()
    parts = []
    for name in ("title_page.pdf", "main.pdf", "supplementary.pdf"):
        f = HERE / name
        if not f.exists():
            raise SystemExit(f"missing {name} -- compile it first")
        d = fitz.open(f)
        out.insert_pdf(d)
        parts.append(f"{name} ({len(d)}p)")
        d.close()
    dest = HERE / "CTSpinoPelvic1K_review.pdf"
    out.save(dest)
    print(f"  {dest.name}: {len(out)} pages  =  " + " + ".join(parts))
    out.close()


def latex_for_pandoc() -> str:
    """Flatten the manuscript into LaTeX pandoc can read.

    The manuscript is written for REVTeX, whose constructs pandoc does not know. Rather
    than let it drop them silently, each is rewritten into a plain equivalent here.
    """
    s = (HERE / "main.tex").read_text(encoding="utf-8")
    body = s[s.index(r"\begin{abstract}"):s.index(r"\end{document}")]

    # the co-authors' own names belong on their review copy, so the title page's
    # author block replaces the anonymised placeholder
    tp = (HERE / "title_page.tex").read_text(encoding="utf-8")
    authors = re.findall(r"\\item ([^\\\n]+?)\\,\$\^\{(\d)\}\$", tp)
    author_line = "; ".join(f"{n.strip()}" for n, _ in authors)

    title = re.search(r"\\title\{(.*?)\}", s, re.S).group(1)
    title = re.sub(r"\s+", " ", title).strip()

    # \fitfloat{graphic}{caption} -> the graphic, then its caption
    body = re.sub(r"\\fitfloat\{(.*?)\}\{(.*?)\}\s*(?=\\end\{figure)",
                  r"\1\n\\caption{\2}", body, flags=re.S)
    # \maxwidth{...} is a fitting wrapper with no meaning outside LaTeX
    body = re.sub(r"\\maxwidth\{%?\s*", "", body)
    body = body.replace(r"\begin{ruledtabular}", "").replace(r"\end{ruledtabular}", "")
    body = body.replace(r"\colrule", r"\hline")
    body = body.replace(r"\si{\degree}", r"$^\circ$").replace(r"\textdegree{}", r"$^\circ$")
    body = body.replace(r"\textdegree", r"$^\circ$")
    # stray closing braces left by the wrappers above
    body = re.sub(r"\n\}\n(?=\s*\\end\{table)", "\n", body)

    head = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage{graphicx}\n\\usepackage{booktabs}\n"
        "\\title{" + title + "}\n"
        "\\author{" + author_line + "}\n"
        "\\begin{document}\n\\maketitle\n"
        "\\begin{center}\\textit{Draft for co-author review --- not the submission copy.}"
        "\\end{center}\n\n"
    )
    return head + body + "\n\\end{document}\n"


def make_docx():
    src = HERE / "_review_src.tex"
    src.write_text(latex_for_pandoc(), encoding="utf-8")
    dest = HERE / "CTSpinoPelvic1K_review.docx"
    cmd = (f"cd '{WSL}' && pandoc _review_src.tex -f latex -o "
           f"'{dest.name}' --resource-path=.:figures 2>&1 | head -12")
    out = subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                         capture_output=True, text=True, timeout=300)
    warn = out.stdout.strip()
    if dest.exists():
        print(f"  {dest.name}: {dest.stat().st_size // 1024} KB")
    else:
        print("  docx FAILED")
    if warn:
        print("  pandoc said:")
        for line in warn.splitlines():
            print(f"    {line}")
    src.unlink(missing_ok=True)


if __name__ == "__main__":
    print("building the co-author review copies")
    merge_pdf()
    make_docx()
