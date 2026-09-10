r"""Consolidate the limitations into one place, remove the prior-art argument's repetitions,
and correct the three claims the citation check flagged as broader than their source.

Every fact and every citation in the removed text is carried into the replacement; nothing
is dropped except wording that had already been said elsewhere.
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")
before = len(s)

edits = [
    # ---------------------------------------------------------------- limitations, merged
    # the orphan sentence goes; its point is the limitations block itself
    ("""The release states its own limitations in enough detail that a reader
can find them without being told.

""", ""),

    ("""\\emph{Coverage.} Thoracic ground truth is field-of-view limited (Fig.~\\ref{fig:fov}) and
does not extend to T1.
Postural angles are supine; pelvic incidence is not postural and needs no such caveat. The
cohort is a colorectal screening population aged 50 and over, so its distributions should
not be read as representative of a surgical one.""",
     """\\emph{Limitations: coverage and cohort.}\\label{sec:limitations} Thoracic ground truth is
field-of-view limited (Fig.~\\ref{fig:fov}) and does not reach T1. Postural angles are
supine, though pelvic incidence is not postural and needs no such caveat. The cohort is one
acquisition protocol of a colorectal screening population aged 50 and over, so its
distributions should not be read as a surgical one, and generalization to pathologic anatomy
is untested."""),

    ("""\\emph{Strength of the labels varies by structure, and the release does not average over
that.} Vertebral labels derive from radiologist-supervised source annotations, corrected
where those sources were wrong. Pelvic labels on records that lacked one are
pseudolabeled. The rib layer is a pseudolabel whose human review was triaged by an
automated rule rather than exhaustive, so its guarantee is the absence of the failure modes
that rule detects. Transitional labels derive from two independent sources and are not
uniformly adjudicated, which is precisely why the measures reported here are count-free,
and the Castellvi grades are a consensus of two radiology resident physicians.
S1 is carved from the sacrum by an automatic estimate of the S1--S2 boundary, and in
100 records that estimate is implausible, making S1 more than half the sacrum's height or
less than fifteen percent of it. Those records are flagged in the quality-control table; exclude them from any measurement that depends on the sacral endplate.""",
     """\\emph{Label strength varies by structure, and the release does not average over that.}
Vertebral labels derive from radiologist-supervised source annotations, corrected where
those sources were wrong; pelvic labels on records that lacked one are pseudolabeled. The
rib layer is a pseudolabel whose review was triaged rather than exhaustive, so its guarantee
is the absence of the failure modes that rule detects. Transitional labels come from two
independent sources and are not uniformly adjudicated, which is why the measures reported
here are count-free, and the Castellvi grades are a consensus of two radiology resident
physicians. S1 is carved by an automatic estimate of the S1--S2 boundary; in 100 records
that estimate is implausible, making S1 more than half the sacrum's height or under fifteen
percent of it, and those records are flagged in the quality-control table for exclusion from
any measurement that depends on the sacral endplate."""),

    ("""\\emph{Structures are not universally present.} Sacrum, hips and femora are present in all 802 records; one record carries no S1 and one no T12, each outside the field of view. Nine records carry no L5 identifier because their source annotation counted four lumbar vertebrae, all nine are Castellvi IIIb, and the fused transitional segment carries the S1 identifier as the caudal anchor; the manifest names them.""",
     """\\emph{Absent structures, and no classifier at the lower border.} Sacrum, hips and femora are present in all 802 records; one record carries no S1 and one no T12, each outside the field of view. Nine records carry no L5 identifier because their source annotation counted four lumbar vertebrae, all nine are Castellvi IIIb, and the fused transitional segment carries the S1 identifier as the caudal anchor; the manifest names them. No shape-based classifier is attempted at the lumbosacral junction, where 15 records in each rare stratum leave three per fold, too few to estimate from; the 0.990 area under the curve reported above is the thoracolumbar boundary, not this one."""),

    # the Discussion's limitations paragraph is now entirely covered above
    ("""\\emph{Limitations.}\\label{sec:limitations} The cohort comes from one acquisition protocol of a colorectal-screening population aged 50 and over, so how well it generalizes to a pathologic population is untested. Human review of the rib layer was triaged by an automated rule, so it guarantees only the absence of the failures that rule detects. No shape-based classifier is attempted at the lumbosacral junction: 15 records in each rare stratum leave three per fold, too few to estimate from, and such a classifier waits on the whole-cohort Castellvi read named below. The 0.990 area under the curve above is the thoracolumbar boundary, not this one. \n\n""", ""),

    # the guarantee is now stated once, in the limitations
    ("""Records passing the rule were not
individually inspected, so the layer only guarantees freedom from the failure modes the rule detects, no more. The residual rib--vertebra offsets reported in
Sec.~\\ref{sec:validation} are the independent check on what the rule missed.""",
     """Records passing the rule were not individually inspected;
the residual rib--vertebra offsets reported in Sec.~\\ref{sec:validation} are the
independent check on what the rule missed."""),

    # ------------------------------------------------- prior art, said once instead of thrice
    ("""\\emph{Comparison with related material.} Against the collections in
Table~\\ref{tab:priorart}, the contribution is the crosswalk placing spine and pelvis on the
same series, together with classes for the anatomy an enumeration anomaly actually produces. VerSe~\\cite{verse2021,versedata2021} comes closest in intent, enriching for transitional anatomy and grading by Castellvi, but it does not segment the sacrum, so the structure a grade describes has no mask; this release segments it, as L6 or as a separately carved S1. Extending VerSe with a sacrum would not have supplied the pelvis, femora or prone acquisitions, and here both label sets already existed.  What this release adds over TotalSegmentator~\\cite{totalsegmentator} is not more anatomy per scan but the classes an anomaly needs to be recorded accurately. """,
     """\\emph{Comparison with related material.} Against the collections in
Table~\\ref{tab:priorart}, the contribution is the crosswalk placing spine and pelvis on the
same series, together with classes for the anatomy an enumeration anomaly produces.
VerSe~\\cite{verse2021,versedata2021} comes closest in intent but has no sacral mask for the
structure a Castellvi grade describes; this release segments it, as L6 or as a carved S1.
Extending VerSe with a sacrum would not have supplied the pelvis, femora or prone
acquisitions, and what this release adds over TotalSegmentator~\\cite{totalsegmentator} is
the classes an anomaly needs, not more anatomy per scan."""),

    ("""\\emph{A proxy for the preoperative view.} No public preoperative lumbar CT cohort exists that we know of, and VerSe lacks the sacrum and pelvis (Table~\\ref{tab:priorart}), so this release is the closest available stand-in for the view a lumbar surgical case is planned on. It holds the T12-to-S1 span with the lowest ribs and the pelvis, 377 of its 802 records were acquired prone, the position of posterior lumbar surgery, and it carries classes for the anomalies that make level identification fail. It is a screening population at colonography dose with standard soft-tissue kernels, not a surgical series, and it supports method
development for level identification and instrumentation geometry, not validation of a
planning decision.""",
     """\\emph{A proxy for the preoperative view.} No public preoperative lumbar CT cohort exists
that we know of, so this release is the closest available stand-in for the view a lumbar
case is planned on: it holds the T12-to-S1 span with the lowest ribs and the pelvis, 377 of
its 802 records were acquired prone, the position of posterior lumbar surgery, and it
carries classes for the anomalies that make level identification fail. It is a screening
population at colonography dose, not a surgical series, so it supports method development
for level identification and instrumentation geometry rather than validation of a planning
decision."""),

    ("""\\emph{Future directions.} Imaging beyond this protocol, and the C2 anchor it cannot supply, are the natural next additions; applying the released weights to VerSe would add the S1 anchor and the anomaly classes to a bone-kernel collection. A Castellvi read of the whole cohort, rather than of the 33 flagged records, would test the co-occurrence of rib and lumbosacral anomalies at full power. The S1 carve should be replaced by one that follows the anatomy, so that the caudal anchor lands consistently across normal and sacralized junctions, and whether the fused segment in the nine four-lumbar records is a lumbar-type body is a question of pelvic morphology, for deep-learning morphometrics of the pelves read in isolation.""",
     """\\emph{Future directions.} Imaging beyond this protocol, and the C2 anchor it cannot
supply, are the natural next additions; applying the released weights to VerSe would add the
S1 anchor and the anomaly classes to a bone-kernel collection. A Castellvi read of the whole
cohort would test the co-occurrence of rib and lumbosacral anomalies at full power and would
supply the labels a lumbosacral classifier needs. The S1 carve should be replaced by one
that follows the anatomy, and whether the fused segment in the nine four-lumbar records is a
lumbar-type body is a question for pelvic morphometrics."""),

    # ------------------------------------------------------- claims narrowed to their source
    # Epstein lists anomalies among the predominant factors; "most often cited" overstates it
    ("""and the cause most often cited is the transitional anatomy this
cohort was assembled around.\\cite{mody2008,epstein2021}""",
     """and among the predominant contributing factors is the
transitional anatomy this cohort was assembled around.\\cite{mody2008,epstein2021}"""),

    # the review reports increasing patient-specific modelling; it draws no contrast with norms
    ("""\\emph{Patient-specific models.} Planning is moving toward patient-specific biomechanical
models rather than population norms.\\cite{fea2026}""",
     """\\emph{Patient-specific models.} Finite-element studies increasingly apply patient-specific
modeling to alignment planning and implant design.\\cite{fea2026}"""),

    # both sources are cadaveric classifications of the thoracolumbar junction; say so
    ("""A thirteenth thoracic vertebra and a lumbar rib are different
phenotypes~\\cite{duplessis2018,poolman2023}.""",
     """A thirteenth thoracic vertebra and a lumbar rib are different
phenotypes, distinguished as separate subtypes in cadaveric classifications of the
thoracolumbar junction~\\cite{duplessis2018,poolman2023}."""),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected one match, found {n}:\n---\n{old[:160]}\n---"
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; main.tex {before} -> {len(s)} chars ({before - len(s)} saved)")
