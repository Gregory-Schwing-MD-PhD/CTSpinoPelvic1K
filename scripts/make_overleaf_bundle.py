"""scripts/make_overleaf_bundle.py — a self-contained folder and zip for Overleaf.

Overleaf wants an upload it can compile with nothing else present. That is almost what
paper/mpda already is, with three differences worth handling rather than leaving a
collaborator to discover:

  THE FIGURES MUST COME ALONG. main.tex includes four PDFs from figures/. Two are generated
  from the morphometrics by make_figures.py and two by render scripts that need the label
  volumes; a collaborator on Overleaf has neither, so the built PDFs are shipped and the
  generators are not.

  THE CLASS FILE DOES NOT. revtex4-2 with the aapm substyle is in Overleaf's TeX Live, and
  shipping a copy would pin collaborators to whatever version was current here.

  THE BUILD SCRIPT DOES NOT EITHER. paper/mpda/build.sh drives a local TinyTeX and refers to
  WSL paths; on Overleaf it is noise at best. Overleaf compiles main.tex directly.

The bibliography is inline `\\thebibliography`, so there is no .bib to carry and no bibtex
pass to configure. That is deliberate for a paper with seventeen references and it means the
Overleaf project compiles in one pass.

    python scripts/make_overleaf_bundle.py --out dist/overleaf
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

README = """# CTSpinoPelvic1K — Medical Physics dataset article

Upload this whole folder to Overleaf (New Project -> Upload Project, or drag the zip in).

## Compiling

- Main document: `main.tex`
- Compiler: **pdfLaTeX**
- Two passes are needed for cross-references. Overleaf does this automatically; if a
  reference shows as `??`, recompile once.
- No bibtex/biber pass. The bibliography is an inline `thebibliography` environment, so
  everything is in `main.tex`.

## The page limit is real, and it is not the number Overleaf shows you

Medical Physics allows **ten published pages**. `main.tex` is set to `preprint`, which is
single column and double spaced, so Overleaf will show roughly twice that. To check the real
count, change one word on line 16:

    \\documentclass[aapm,mph,amsmath,amssymb,preprint]{revtex4-2}
    \\documentclass[aapm,mph,amsmath,amssymb,reprint]{revtex4-2}

and recompile. It is currently **10 pages** in `reprint`. Please change it back to
`preprint` before committing, and if you add text, check the reprint count — the article is
at the limit, so anything added has to be paid for.

## Figures

`figures/` holds four built PDFs. They are generated from the released measurements and the
label volumes, neither of which is in this project, so edit the captions here and ask for a
regenerated figure if the plot itself needs to change.

| file | what it is |
|---|---|
| `fig_priorart.pdf` | the comparison against other public collections |
| `fig_anchors.pdf` | the two anchors and the interval between them, rendered from the labels |
| `fig_countfree.pdf` | count-free measures |
| `fig_validation.pdf` | derived measures against published reference ranges |
| `fig_opportunistic.pdf` | opportunistic screening measures |

Figure 1 is drawn in TikZ inside `main.tex` and needs no file.

## Before submission

Grep the source for `[CO-AUTHOR]` and `PENDING`. Outstanding at the time of writing:

- the Zenodo DOI is a placeholder (`\\datasetdoi` on line 27)
- every author must confirm their own affiliation and supply a conflict-of-interest
  statement; the current declaration covers all authors and only the first has been asked
- the IRB determination is marked PENDING in the Ethics section and must not be submitted
  as written
- the repository URL in Data Availability
- twelve of seventeen references still have unverified volume/page fields; the five that
  were checked are named in a comment above the bibliography
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="paper/mpda")
    ap.add_argument("--out", default="dist/overleaf")
    ap.add_argument("--zip", default=None, help="default: <out>.zip")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)

    tex = src / "main.tex"
    if not tex.exists():
        print(f"  ! {tex} not found")
        return 1
    shutil.copy(tex, out / "main.tex")

    # only the figures main.tex actually includes, so a stale PDF cannot ride along
    body = tex.read_text(encoding="utf-8")
    wanted = set()
    for line in body.split("\n"):
        if "includegraphics" in line and "figures/" in line:
            frag = line.split("figures/", 1)[1]
            wanted.add(frag.split("}")[0].strip())
    missing = []
    for name in sorted(wanted):
        p = src / "figures" / name
        if p.exists():
            shutil.copy(p, out / "figures" / name)
        else:
            missing.append(name)
    if missing:
        print(f"  ! main.tex includes figures that do not exist: {missing}")
        return 1

    (out / "README.md").write_text(README, encoding="utf-8")

    zpath = Path(a.zip) if a.zip else out.with_suffix(".zip")
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out))

    n = sum(1 for _ in out.rglob("*") if _.is_file())
    kb = zpath.stat().st_size / 1024
    print(f"  {out}: {n} file(s), {len(wanted)} figure(s)")
    for name in sorted(wanted):
        print(f"    figures/{name}")
    print(f"  {zpath}: {kb:.0f} kB")
    print("\n  Overleaf: New Project -> Upload Project -> that zip. Compiler pdfLaTeX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
