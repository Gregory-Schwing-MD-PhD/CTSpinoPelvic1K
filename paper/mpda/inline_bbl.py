r"""inline_bbl.py -- paste the bibtex-generated bibliography into main.tex.

REVTeX's aapm substyle issues its own \bibdata for endnotes, so adding a second
\bibliography command makes bibtex refuse the run ("Illegal, another \bibdata command").
The template README asks for the generated .bbl to be pasted inline anyway -- "Copy the
contents of this .bbl file into your main latex document, replacing the \bibliography
command" -- so that is what happens here, and the conflict never arises.

The source of truth stays ctspinopelvic1k.bib: regenerate with

    pdflatex bibtest && bibtex bibtest && python inline_bbl.py

so a corrected reference is fixed once, in the .bib, rather than in hand-typed LaTeX.

The authors' \sloppy and their record of what was checked are carried across.
"""
from pathlib import Path
import re

HERE = Path(__file__).parent
bbl = (HERE / "bibtest.bbl").read_text(encoding="utf-8")

# take only the entries; the surrounding environment is rebuilt with the local preamble
body = bbl[bbl.index(r"\begin{thebibliography}"):bbl.index(r"\end{thebibliography}")]
body = body.split("\n", 1)[1]                      # drop the \begin line

header = r"""\begin{thebibliography}{20}
% GENERATED, DO NOT HAND-EDIT. Produced from ctspinopelvic1k.bib through the journal's
% medphy.bst and pasted here as the template README instructs, because REVTeX's aapm
% substyle emits its own \bibdata and a second \bibliography command makes bibtex refuse
% the run. To change a reference, edit the .bib and re-run:
%     pdflatex bibtest && bibtex bibtest && python inline_bbl.py
%
% Every entry was verified against the publisher record, Crossref or arXiv in September
% 2026. Verification corrected three errors that had survived in the hand-typed list:
%   * osc2026    first author is Schehr, not Schwing
%   * moller2026 the title belonged to a different paper; it is now the published
%                preprint, which also settles the citable form the co-author was to choose
%   * ctspine1k  the preprint has appeared in Mach. Learn. Biomed. Imaging
% A trade-press item was removed rather than published unverified: its host returns 403 to
% automated retrieval, so the attribution rested on search snippets and not on the page. The
% claim it supported now cites a peer-reviewed review of the same subject.
%
% \sloppy in the reference list only: a news-article URL with no hyphens to break on
% overflows the two-column measure, and loosening inter-word spacing among the references
% is a smaller cost than a line running into the margin.
\sloppy
"""

p = HERE / "main.tex"
s = p.read_text(encoding="utf-8")
a = s.index(r"\begin{thebibliography}")
b = s.index(r"\end{thebibliography}")
old = s[a:b]
s = s[:a] + header + body + s[b:]
p.write_text(s, encoding="utf-8")

print(f"inlined {len(re.findall(r'.bibitem', body))} verified entries "
      f"(replaced {len(re.findall(r'.bibitem', old))} hand-written)")
