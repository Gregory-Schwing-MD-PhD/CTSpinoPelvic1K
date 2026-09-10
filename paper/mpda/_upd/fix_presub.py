# -*- coding: utf-8 -*-
"""Two defects found in the pre-submission check of the assembled packet.

1. The 'Fused and split records' subsection is EMPTY. Its content was folded
   into Sec. II.A (Sources) during the page-limit consolidation and the heading
   was left behind, so both the submitted preprint and the ten-page reprint
   print "II.B. Fused and split records" followed straight by "II.C." with no
   body between them.

2. The self-citation is NOT redacted under review, though the preamble comment
   and the cover letter both say it is. Reference 32 prints
   "A. Schehr, J. Kim, and G. Schwing, OpenSpineConsortium: ..." in full, which
   names two of the fourteen authors and un-blinds the consortium macro
   everywhere it is used. Wrapped in a review conditional so the entry, and
   therefore its citation number, survives while the identifying text does not.
"""
from pathlib import Path

MPDA = Path(__file__).resolve().parent.parent
p = MPDA / "main.tex"
s = p.read_text(encoding="utf-8")
B = chr(92)

# ---- 1. drop the empty subsection ---------------------------------------------------
empty = B + "subsection{Fused and split records}\n\n \n\n"
assert s.count(empty) == 1, "empty subsection not matched verbatim"
s = s.replace(empty, "")
print("removed the empty 'Fused and split records' subsection")

# ---- 2. redact the self-citation under review ---------------------------------------
head = B + "bibitem{osc2026}\n"
body = ("A.~Schehr, J.~Kim, and G.~Schwing,\n"
        + B + "newblock {OpenSpineConsortium}: an open-source framework for medical"
        + " student engagement in computational spine imaging research,\n"
        + B + "newblock Cureus  (2026),\n"
        + B + "newblock doi:10.7759/cureus.112661.\n")
assert s.count(head + body) == 1, "osc2026 bibitem not matched verbatim"
redacted = (B + "newblock [Reference removed for double-anonymized review: a published"
            + " description of the authors' student research program.]\n")
new = (head + B + "ifreview\n" + redacted + B + "else\n" + body + B + "fi\n")
s = s.replace(head + body, new)
print("redacted the osc2026 self-citation under the review conditional")

p.write_text(s, encoding="utf-8", newline="")
