r"""Tighten the paragraphs that repeat themselves, to bring the article to the ten published
pages the policy allows. No fact, number or citation is removed; only wording that restates
something the same paragraph or a neighbouring one has already said.
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")
before = len(s)

edits = [
    # 1. two lead-in sentences where one does, and the size argument stated twice
    ("""\\emph{But the vertebrae themselves are known to differ.} Counting is not the only route to
an identity. Level-specific morphometry is long established. Pedicle width and height change
systematically from the thoracic to the lumbar spine,\\cite{zindrick1987} vertebral body,
endplate and canal dimensions differ by level,\\cite{panjabi1991,panjabi1992,benzel2015} and
the transverse process and the iliolumbar ligament mark the last lumbar vertebra
independently of any count.\\cite{hughes2006,koninwalz2010} If those differences hold at the boundary itself, the phenotype is decidable locally, and on this corpus it does hold. Separating T12 from L1 on shape alone, with each patient's own size divided out and
cross-validation grouped by case, gives an area under the curve of 0.990 and 97.3\\%
accuracy over 1485 vertebrae. Size is removed because raw millimeters separate thoracic
from lumbar trivially and teach nothing about the junction, where the two are nearly the
same size. That figure is a logistic regression on five released per-level measures and six
ratios between them, each measure divided by the same patient's median across their
levels, with the folds grouped by case and the curve taken on pooled out-of-fold scores
(\\texttt{test\\_morphometric\\_separability.py} in the released code).""",
     """\\emph{But the vertebrae themselves are known to differ.} Level-specific morphometry is long
established: pedicle width and height change systematically from the thoracic to the lumbar
spine,\\cite{zindrick1987} vertebral body, endplate and canal dimensions differ by
level,\\cite{panjabi1991,panjabi1992,benzel2015} and the transverse process and the
iliolumbar ligament mark the last lumbar vertebra independently of any
count.\\cite{hughes2006,koninwalz2010} If those differences hold at the boundary itself the
phenotype is decidable locally, and on this corpus it does. Separating T12 from L1 on shape
alone gives an area under the curve of 0.990 and 97.3\\% accuracy over 1485 vertebrae: a
logistic regression on five released per-level measures and six ratios between them, each
measure divided by the same patient's median across their levels, so that size, which
separates thoracic from lumbar trivially and says nothing at the junction, is removed; folds
are grouped by case and the curve is taken on pooled out-of-fold scores
(\\texttt{test\\_morphometric\\_separability.py} in the released code)."""),

    # 2. "The cohort suits that question" is answered by the two sentences that follow it
    ("""radiographs. The cohort suits that question. An abdominopelvic CT runs from the diaphragm to the pelvic
floor, so it holds the T12-to-S1 span of the planning study together with the lowest ribs
and the whole pelvis, and it lacks C2 exactly as that study does.""",
     """radiographs. An abdominopelvic CT runs from the diaphragm to the pelvic floor, so it holds
that span together with the lowest ribs and the whole pelvis, and it lacks C2 exactly as the
planning study does."""),

    # 3. the out-of-fold guarantee is stated twice in one sentence
    ("""five-fold nnU-Net v2 ensemble, applied out-of-fold, so that each record is completed by the fold that
never trained on it, so no record is completed by a model that has seen it.""",
     """five-fold nnU-Net v2 ensemble applied out-of-fold, so that no record is completed by a
model that has seen it."""),

    # 4. the numbering dilemma is put twice, once abstractly and once concretely
    ("""\\textbf{A rib on a lumbar body receives its own class}, and this is the one class here with
no published counterpart. A lumbar rib and a stump rib are the same object under two counts,
so assigning it a number would commit the label to one reading of an enumeration anomaly. A
scheme that numbers every rib 1--12 has nowhere to put a thirteenth, so the annotator must
either call it rib~12, asserting the vertebra beneath it is thoracic, the very question at
issue, or discard it.""",
     """\\textbf{A rib on a lumbar body receives its own class}, and this is the one class here with
no published counterpart. A lumbar rib and a stump rib are the same object under two counts,
so a scheme that numbers every rib 1--12 leaves the annotator either calling it rib~12,
asserting the vertebra beneath it is thoracic, the very question at issue, or discarding
it."""),

    # 5. the consequence of the gap is spelled out twice
    ("""No public collection can therefore record a six-lumbar spine
together with both of its borders, however good the segmentation, and a segmenter trained on
them inherits the gap, so that, faced with an enumeration anomaly, it must shift the whole column by a
level or absorb a vertebra into its neighbor.""",
     """No public collection can therefore record a six-lumbar spine
together with both of its borders, and a segmenter trained on them inherits the gap: faced
with an enumeration anomaly it must shift the whole column by a level or absorb a vertebra
into its neighbor."""),

    # 6. the crosswalk's value is stated twice in the same paragraph
    ("""It is not a file merge, because every patient was scanned prone and supine and neither
source recorded which series it drew on, so each annotation had to be traced back to its
own series (Sec.~\\ref{sec:sources}). Done once, that crosswalk yields a spine-and-pelvis
frame at 802 patients for the cost of the crosswalk, and the bones neither set had, ribs, femora,
S1 and hardware, could then be annotated against it rather than from scratch. Two of the
most-used public bone annotation sets were halves of one dataset; this release is the whole.""",
     """It is not a file merge: every patient was scanned prone and supine and neither source
recorded which series it drew on, so each annotation had to be traced back to its own series
(Sec.~\\ref{sec:sources}). Done once, that crosswalk yields a spine-and-pelvis frame at 802
patients, and the bones neither set had, ribs, femora, S1 and hardware, could then be
annotated against it rather than from scratch. Two of the most-used public bone annotation
sets were halves of one dataset; this release is the whole."""),

    # 7. the ambiguity is asserted and then restated
    ("""Each
set yields the same count (Fig.~\\ref{fig:anchors}), and the morphology that would
separate the readings is not
decisive to a human reader in every case. Whether it differs enough for a learned classifier to separate them, or at
least to assign each reading a probability, is the question this release is built to
support. Absent that, what remains is the count itself.""",
     """Each set yields the same count (Fig.~\\ref{fig:anchors}), and the morphology that would
separate the readings is not decisive to a human reader in every case. Whether it differs
enough for a learned classifier to separate them, or to assign each reading a probability,
is the question this release is built to support. Absent that, what remains is the count
itself."""),

    # 8. the phenotype contrast is set up twice
    ("""\\emph{The limitation, stated plainly.} No scan in this corpus contains C2, so the
conventional count cannot be performed, which is a property of abdominal imaging, not of the annotation. A thirteenth thoracic vertebra and a lumbar rib are different
phenotypes, distinguished as separate subtypes in cadaveric classifications of the
thoracolumbar junction~\\cite{duplessis2018,poolman2023}. The first is an additional rib-bearing segment,
usually above five lumbar vertebrae, so the column holds 25 presacral vertebrae; the second
is a rudimentary rib on a lumbar-type L1 in a column of the usual 24.""",
     """\\emph{The limitation, stated plainly.} No scan in this corpus contains C2, so the
conventional count cannot be performed, a property of abdominal imaging rather than of the
annotation. A thirteenth thoracic vertebra and a lumbar rib are different phenotypes,
distinguished as separate subtypes in cadaveric classifications of the thoracolumbar
junction~\\cite{duplessis2018,poolman2023}: the first is an additional rib-bearing segment,
usually above five lumbar vertebrae, so the column holds 25 presacral vertebrae; the second
is a rudimentary rib on a lumbar-type L1 in a column of the usual 24."""),

    # 9. a stump rib's definition is parenthetical twice over
    ("""A \\emph{stump
rib} (a hypoplastic twelfth rib, a cardinal indicator of a thoracolumbar transitional
vertebra) is still a rib, so it leaves the count unchanged, and only its length records the
anomaly.""",
     """A \\emph{stump rib}, a hypoplastic twelfth rib and a cardinal indicator of a thoracolumbar
transitional vertebra, is still a rib, so it leaves the count unchanged and only its length
records the anomaly."""),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected one match, found {n}:\n---\n{old[:150]}\n---"
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; {before} -> {len(s)} chars ({before - len(s)} saved)")
