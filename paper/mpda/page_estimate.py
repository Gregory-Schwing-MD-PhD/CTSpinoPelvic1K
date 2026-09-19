r"""paper/mpda/page_estimate.py -- the journal's own published-page estimate.

Medical Physics estimates published pages as

    pages = 4.3 + words/3000 + (figures + tables)/1.9

and it is the number the editorial office applies at submission, so it is the one that
has to come in under ten -- whatever the REVTeX reprint happens to typeset in. The two
disagree, and the formula is the stricter of the pair.

WHAT THE FLOAT TERM MEANS. Each float costs 1/1.9 = 0.526 of a page whatever its size, so
a five-row table costs the same as a full-width figure. Eleven floats spend 5.8 pages
before a single word is counted; with the 4.3 base that is 10.1 already, so no amount of
prose cutting can reach ten while eleven floats remain. Counting floats first is the
whole point of running this.

    python paper/mpda/page_estimate.py
    python paper/mpda/page_estimate.py --limit 10
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def body_words(tex: str) -> int:
    """Words in the article body: abstract through the last section, no bibliography."""
    # drop everything before the abstract and the bibliography onward
    i = tex.find(r"\begin{abstract}")
    j = tex.find(r"\begin{thebibliography}")
    s = tex[i if i > 0 else 0: j if j > 0 else len(tex)]

    s = re.sub(r"(?m)^\s*%.*$", "", s)                 # whole-line comments
    s = re.sub(r"(?<!\\)%.*", "", s)                   # trailing comments
    # figure and table bodies are counted as floats, not as words; captions go with them
    s = re.sub(r"\\begin\{(figure|table|tikzpicture)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\\(label|ref|cite|citep|citet)\s*\{[^}]*\}", " x ", s)
    s = re.sub(r"\\url\s*\{[^}]*\}", " x ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)             # remaining control sequences
    s = re.sub(r"[{}$&~^_\\]", " ", s)
    words = [w for w in re.split(r"\s+", s) if re.search(r"[A-Za-z0-9]", w)]
    return len(words)


def count_floats(tex: str) -> tuple[int, int, list[str]]:
    j = tex.find(r"\begin{thebibliography}")
    s = tex[:j if j > 0 else len(tex)]
    s = re.sub(r"(?m)^\s*%.*$", "", s)
    figs = re.findall(r"\\begin\{figure\*?\}", s)
    tabs = re.findall(r"\\begin\{table\*?\}", s)
    names = re.findall(r"\\label\{(fig:[^}]*|tab:[^}]*)\}", s)
    return len(figs), len(tabs), names


ap = argparse.ArgumentParser()
ap.add_argument("--tex", default=str(HERE / "main.tex"))
ap.add_argument("--limit", type=float, default=10.0)
a = ap.parse_args()

tex = Path(a.tex).read_text(encoding="utf-8")
w = body_words(tex)
nf, nt, names = count_floats(tex)
floats = nf + nt

base, wterm, fterm = 4.3, w / 3000.0, floats / 1.9
total = base + wterm + fterm

print("  words (body, floats and bibliography excluded) : %5d" % w)
print("  figures                                        : %5d" % nf)
print("  tables                                         : %5d" % nt)
print("  floats                                         : %5d" % floats)
print()
print("  4.3 (base)          = %6.2f" % base)
print("  words / 3000        = %6.2f" % wterm)
print("  floats / 1.9        = %6.2f" % fterm)
print("  " + "-" * 30)
print("  ESTIMATED PAGES     = %6.2f   (limit %.0f)" % (total, a.limit))
print()
print("  floats present:", ", ".join(names))
print()

# what it would take to come in under the limit
room = a.limit - base - fterm
print("  with %d floats the formula allows %.0f words of body text" % (floats, max(room, 0) * 3000))
if room < 0:
    need = 0
    while base + (floats - need) / 1.9 > a.limit:
        need += 1
    print("  THE FLOATS ALONE EXCEED THE LIMIT: %d float(s) must go before any prose cut counts"
          % need)
else:
    over = w - room * 3000
    if over > 0:
        print("  over by %.0f words (%.2f pages)" % (over, over / 3000.0))

if total > a.limit:
    print("\n  OVER by %.2f pages" % (total - a.limit))
    sys.exit(1)
print("\n  under the limit by %.2f pages" % (a.limit - total))
