#!/usr/bin/env bash
# Build the arXiv submission tarball and prove it compiles from nothing but itself.
#
#   bash paper/mpda/make_arxiv.sh
#
# arXiv compiles the source it is given, in a clean tree, with no network and no access to
# this machine. So the only test that means anything is to unpack the tarball somewhere else
# and run pdflatex there -- a package installed locally, a figure resolved by an absolute
# path, or a .bbl left behind all work here and fail there.
#
# WHAT GOES IN: main.tex and the figure PDFs it includes, nothing else. No .aux, no .log, no
# build directory. The bibliography is a thebibliography environment inside main.tex rather
# than BibTeX, so there is no .bbl to ship -- which is the usual arXiv failure and is avoided
# here by construction.
#
# WHICH DOCUMENTCLASS: reprint (two-column, ten pages), decided 2026-09-07. arXiv posts the
# authors' own manuscript, and the two-column form is the one a reader expects and the one
# the journal will typeset; the double-spaced preprint form exists for the review PDF only.
# Authors and affiliations come from the \else branch of \ifreview in main.tex, which must
# match title_page.tex.
set -euo pipefail
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/arxiv"
TAR="$HERE/CTSpinoPelvic1K_arxiv.tar.gz"

rm -rf "$OUT" && mkdir -p "$OUT/figures"
# arXiv posts the authors' own manuscript: names in, redactions out, no caption list,
# no line numbers. The three switches are single words, so no escaping is needed.
sed -e 's/,preprint,linenumbers]{revtex4-2}/,reprint]{revtex4-2}/' \
    -e 's/reviewtrue/reviewfalse/' -e 's/captionlisttrue/captionlistfalse/' \
    "$HERE/main.tex" > "$OUT/main.tex"

cp "$HERE/figure_captions.tex" "$OUT/" 2>/dev/null || true
# only the figures main.tex actually includes
grep -o 'figures/[A-Za-z0-9_.-]*\.pdf' "$HERE/main.tex" | sort -u > /tmp/figs.txt
while read -r f; do
  [ -f "$HERE/$f" ] || { echo "  MISSING $f"; exit 1; }
  cp "$HERE/$f" "$OUT/$f"
done < /tmp/figs.txt
echo "  included $(wc -l < /tmp/figs.txt) figure(s)"

# --- compile in a clean tree, as arXiv will -----------------------------------------
CLEAN=$(mktemp -d)
cp -r "$OUT"/. "$CLEAN"/
cd "$CLEAN"
pdflatex -interaction=nonstopmode main.tex > a1.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > a2.log 2>&1 || true

echo "  --- clean-tree compile ---"
if [ ! -f main.pdf ]; then
  echo "  FAILED to produce a PDF"
  grep -A3 '^!' a2.log | head -30
  exit 1
fi
grep -A3 '^!' a2.log | head -20 || true
echo "  undefined: $(grep -ci 'undefined' a2.log || true) line(s)"
python3 -c "
import pymupdf
d = pymupdf.open('main.pdf')
print(f'  compiles clean: {d.page_count} pages (reprint, two-column form)')
t = d[0].get_text()
for must in ('Gregory Schwing', 'Nizar Alnabahneh', 'OpenSpineConsortium'):
    print(f'  page 1 carries {must!r}: {must in t}')
print(f'  any redaction marker left: {\"removed for double-anonymized\" in \" \".join(p.get_text() for p in d)}')
"

cd "$HERE"
tar -czf "$TAR" -C "$OUT" .
echo "  wrote $TAR ($(du -h "$TAR" | cut -f1))"
echo "  contents:"
tar -tzf "$TAR" | sed 's/^/    /'
rm -rf "$CLEAN"
