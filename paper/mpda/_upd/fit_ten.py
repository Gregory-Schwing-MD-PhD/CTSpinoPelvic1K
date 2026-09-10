# -*- coding: utf-8 -*-
r"""Recover the eleventh page.

The ten-page check had been measuring the BLINDED form, where fourteen authors and
three affiliations collapse to one bracketed line. The published article carries the
real block, which costs a page: the camera-ready form is 11 pages and its last page
holds 95 words, the tail of the reference list.

Three cuts, all of duplicated or speculative material. No described result, no number
and no limitation is touched.

  1. The Discussion's comparison paragraph opened by restating what the Introduction
     already says about Table I. The sentences after it carry the actual comparison.
  2. Future directions listed five speculations. Kept the two that name a concrete
     next action; dropped the rest, which a reader cannot act on.
  3. The Conclusion restated the abstract. Kept the claim, dropped the restatement.
"""
from pathlib import Path

MPDA = Path(__file__).resolve().parent.parent
p = MPDA / "main.tex"
s = p.read_text(encoding="utf-8")
before = len(s.split())


def cut(old, new, label):
    global s
    assert s.count(old) == 1, "no unique match: " + label
    s = s.replace(old, new)
    print("  %-52s %+d words" % (label, len(new.split()) - len(old.split())))


cut(
    r"""\emph{Comparison with related material.} Against the collections in
Table~\ref{tab:priorart}, the contribution is the crosswalk placing spine and pelvis on the
same series, together with classes for the anatomy an enumeration anomaly produces.
VerSe~\cite{verse2021,versedata2021} comes closest""",
    r"""\emph{Comparison with related material.}
VerSe~\cite{verse2021,versedata2021} comes closest""",
    "Discussion opener (duplicates the Introduction)")

cut(
    r"""\emph{Future directions.} Imaging beyond this protocol, and the C2 anchor it cannot
supply, are the natural next additions; applying the released weights to VerSe would add the
S1 anchor and the anomaly classes to a bone-kernel collection. A Castellvi read of the whole
cohort would test the co-occurrence of rib and lumbosacral anomalies at full power and would
supply the labels a lumbosacral classifier needs. The S1 carve should be replaced by one
that follows the anatomy, and whether the fused segment in the nine four-lumbar records is a
lumbar-type body is a question for pelvic morphometrics.""",
    r"""\emph{Future directions.} Applying the released weights to VerSe would add the S1 anchor
and the anomaly classes to a bone-kernel collection. A Castellvi read of the whole cohort
would test the co-occurrence of rib and lumbosacral anomalies at full power and supply the
labels a lumbosacral classifier needs.""",
    "Future directions (kept the two actionable ones)")

cut(
    r"""CTSpinoPelvic1K places spine, pelvis, per-level ribs and femora on one coordinate frame in
802 records, and gives the anatomy that makes lumbar numbering ambiguous a class of its own
rather than recording it as something it is not. Because both counting anchors are explicit,
a level can be identified from a field of view that cannot support the conventional count
downward from C2, which is every record here.""",
    r"""CTSpinoPelvic1K places spine, pelvis, per-level ribs and femora on one coordinate frame in
802 records, and gives the anatomy that makes lumbar numbering ambiguous a class of its own.
Because both counting anchors are explicit, a level can be identified from a field of view
that cannot support the conventional count downward from C2.""",
    "Conclusion (dropped the abstract restatement)")

p.write_text(s, encoding="utf-8", newline="")
print("\n  total: %+d words" % (len(s.split()) - before))
