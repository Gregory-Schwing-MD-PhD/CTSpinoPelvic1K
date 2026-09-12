"""paper/mpda/make_coauthor_docs.py -- the two files a coauthor actually opens.

  CTSpinoPelvic1K_reprint.pdf            two columns, real authors, the published geometry.
                                         This is the one the ten-page limit is measured on.
  CTSpinoPelvic1K_single_column.docx     one column, editable, for track-changes review.

WHY THE DOCX IS BUILT FROM A STRIPPED PREAMBLE. pandoc cannot read REVTeX: patching its
preamble construct by construct produced a new error on every run -- \\affiliation, \\hd,
\\colrule, the custom lengths. Discarding the preamble entirely and keeping only
\\begin{document}...\\end{document} under a minimal article class works first time, because
what pandoc needs is the BODY. Everything dropped is layout, and the docx is not the layout
artefact; the PDF is.

    python paper/mpda/make_coauthor_docs.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WSL_HERE = "/mnt/c/Users/grego/OneDrive/Desktop/CTSpinoPelvic1K-1/paper/mpda"


def wsl(cmd: str, timeout=900):
    return subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=timeout)


def build_pdf() -> Path | None:
    """The reprint build is the only thing that measures the page limit."""
    r = wsl(f"cd {WSL_HERE} && ./build.sh --reprint 2>&1 | tail -25")
    out = r.stdout + r.stderr
    m = re.search(r"PAGES: (\d+)", out)
    print(f"  reprint: {m.group(1) if m else '?'} pages")
    if re.search(r"^!", out, re.M):
        print("  ! LaTeX errors -- not shipping this PDF")
        return None
    src = HERE / "CTSpinoPelvic1K_reprint_check.pdf"
    if not src.exists():
        print("  ! no reprint PDF produced")
        return None
    dst = HERE / "CTSpinoPelvic1K_reprint.pdf"
    shutil.copyfile(src, dst)
    return dst


def flatten_for_pandoc(tex: str) -> str:
    """Body only, minimal preamble, REVTeX-isms neutralised."""
    body = tex.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]

    # the anonymisation switch: keep the real names
    body = re.sub(r"\\ifreview(.*?)\\else(.*?)\\fi", r"\2", body, flags=re.S)
    body = re.sub(r"\\ifreview(.*?)\\fi", "", body, flags=re.S)

    # REVTeX front matter pandoc has no notion of
    for cmd in ("affiliation", "altaffiliation", "thanks", "homepage",
                "collaboration", "noaffiliation"):
        body = re.sub(r"\\" + cmd + r"\{(?:[^{}]|\{[^{}]*\})*\}", "", body)
    body = body.replace(r"\maketitle", "")

    # table rules and the custom width macros
    body = body.replace(r"\colrule", r"\hline").replace(r"\botrule", "")
    body = re.sub(r"\\begin\{ruledtabular\}|\\end\{ruledtabular\}", "", body)
    # \maxwidth{% ... \end{tabular} } -- strip the opener AND its orphaned closer, or
    # pandoc stops at "unexpected }" expecting \end{table}
    body = re.sub(r"\\maxwidth\{%?", "", body)
    body = re.sub(r"(\\end\{tabular\})\s*\n\s*\}", r"\1", body)
    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
    body = re.sub(r"\\(hd|has|no|qual)\b", "", body)

    # figure* / table* are not article environments
    body = body.replace(r"\begin{figure*}", r"\begin{figure}")
    body = body.replace(r"\end{figure*}", r"\end{figure}")
    body = body.replace(r"\begin{table*}", r"\begin{table}")
    body = body.replace(r"\end{table*}", r"\end{table}")
    body = re.sub(r"\\si\{\\degree\}", r"$^\\circ$", body)
    body = body.replace(r"\textdegree{}", r"$^\circ$")

    pre = "\n".join([
        r"\documentclass[12pt]{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{url}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{setspace}\onehalfspacing",
        r"\begin{document}",
    ])
    return pre + body + "\n\\end{document}\n"


def build_docx() -> Path | None:
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    flat = HERE / "_flat_single_column.tex"
    flat.write_text(flatten_for_pandoc(tex), encoding="utf-8")
    dst = HERE / "CTSpinoPelvic1K_single_column.docx"
    r = wsl(f"cd {WSL_HERE} && pandoc _flat_single_column.tex "
            f"--from=latex --to=docx --resource-path=.:figures "
            f"-o {dst.name} 2>&1 | tail -12")
    print("  pandoc:", (r.stdout + r.stderr).strip()[:300] or "clean")
    flat.unlink(missing_ok=True)
    return dst if dst.exists() else None


def main() -> int:
    print("building the two coauthor documents")
    pdf = build_pdf()
    docx = build_docx()
    for p in (pdf, docx):
        if p and p.exists():
            print(f"  wrote {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
        else:
            print("  ! one output missing")
    return 0 if (pdf and docx) else 1


if __name__ == "__main__":
    sys.exit(main())
