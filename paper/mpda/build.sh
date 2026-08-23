#!/usr/bin/env bash
# Compile the MDPA paper, and check the two things that only show up on a rendered page.
#
#   bash paper/mpda/build.sh              # preprint (working copy), single column
#   bash paper/mpda/build.sh --reprint    # two column, to check the PUBLISHED page count
#
# WHY THIS EXISTS. There was no LaTeX on the Windows host, none in WSL, and the grid's is
# unusable for this paper: it carries revtex4-1 rather than revtex4-2, and a PGF too old for
# the arrows.meta library, so Figure 1 could not even be drawn there. TinyTeX installs
# per-user without sudo and carries revtex4-2 with the aapm substyle:
#
#   wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
#   export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
#   tlmgr install revtex pgf siunitx booktabs geometry hyperref amsmath tools graphics \
#                 oberdiek etoolbox xcolor l3packages l3kernel textcase natbib url fancyhdr
#
# CHECK THE REPRINT PAGE COUNT, NOT THE PREPRINT ONE. The MPDA policy allows ten PUBLISHED
# pages. The working file is `preprint`, which is single-column and double-spaced, so its
# page count says almost nothing about the limit — at the time of writing the paper is 20
# preprint pages and 10 published ones. Float placement also differs between the two, so a
# figure that behaves in one can misbehave in the other; both are worth looking at.
#
# AND LOOK AT THE FIGURES. Every defect in Figure 1 — boxes printing through the row
# beneath, a connector drawn through its own caption text, 6pt type in a field of white
# space, an arrowhead eating a label, a reserved TikZ key silently dropping four node
# shapes — was invisible in the source and obvious in the render. Rasterise the page and
# look at it:
#
#   python -c "import fitz; fitz.open('paper/mpda/CTSpinoPelvic1K_dataset_article.pdf')[4]
#              .get_pixmap(dpi=125).save('/tmp/p.png')"
set -uo pipefail
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"

if [ "$MODE" = "--reprint" ]; then
  WORK=/tmp/mpda_reprint
  OUTNAME=CTSpinoPelvic1K_reprint_check.pdf
else
  WORK=/tmp/mpda_build
  OUTNAME=CTSpinoPelvic1K_dataset_article.pdf
fi

mkdir -p "$WORK/figures"
if [ "$MODE" = "--reprint" ]; then
  sed 's/,preprint\]{revtex4-2}/,reprint]{revtex4-2}/' "$HERE/main.tex" > "$WORK/main.tex"
else
  cp "$HERE/main.tex" "$WORK/main.tex"
fi
cp "$HERE"/figures/*.pdf "$WORK/figures/" 2>/dev/null

cd "$WORK"
grep -m1 documentclass main.tex
pdflatex -interaction=nonstopmode main.tex > p1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > p2.log 2>&1

echo "=== errors ==="
grep -A3 '^!' p2.log | head -40

echo "=== overfull boxes over 5pt (what runs off the page) ==="
grep 'Overfull \\hbox' p2.log | sed -E 's/.*\(([0-9.]+)pt too wide\).*/\1 &/' \
  | awk '$1 > 5' | sort -rn | head -12

echo "=== undefined references and citations ==="
grep -i 'undefined' p2.log | head -10

echo "=== pages ==="
# pymupdf, not a regex over the raw bytes: modern pdflatex writes compressed object
# streams, so /Type /Page never appears as literal text and the regex silently reports 0.
#   pip install --break-system-packages pymupdf
python3 -c "
import pymupdf
d = pymupdf.open('main.pdf')
print('PAGES:', d.page_count, '(MPDA limit is 10 PUBLISHED pages -- the reprint count)')
"

cp main.pdf "$HERE/$OUTNAME" && echo "wrote paper/mpda/$OUTNAME"
