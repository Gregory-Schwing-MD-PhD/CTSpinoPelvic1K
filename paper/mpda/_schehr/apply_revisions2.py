r"""Second pass on Ashley Schehr's revisions: correct the subtype counts against the
released manifest, and tighten the three added passages so the article still typesets to
the ten published pages the policy allows.

The counts below are read from the release itself, not recalled:
    data/zenodo_deposit/manifest.json     lstv_label: lumbarization 14, sacralization 17,
                                          semi-sacralization 2, normal 769
                                          has_l6 True 18; n_lumbar_labels 4 in nine
    data/zenodo_deposit/splits_5fold.json subtype_counts lumb 15, sacralization 15,
                                          semisacralization 2, ambiguous 2, normal 768
                                          every fold: three lumb, three sacralization
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")

edits = [
    # --- the AUC methods paragraph, tightened -----------------------------------------
    (r"""The classifier is a logistic regression on five released per-level measures
(endplate width, anterior and posterior body height, canal width and transverse-process
span), each divided by the same patient's median of that measure across their measured
levels, plus six ratios between measures, which carry no size at all; T12 is the positive
class and L1 the negative; the five folds are grouped by case, so no patient contributes
to both training and validation; and the area under the curve is computed on the pooled
out-of-fold scores (\texttt{scripts/test\_morphometric\_separability.py} in the released
code). Schinz et al.\ reached the same conclusion from the other direction on 1,242
whole-thoracolumbar CTs: a vertebral-shape-based labeling of the junction agreed with
nerve morphology in every case, against 92.6\% to 97.2\% for counting- and rib-based
rules~\cite{schinz2026}. No such classifier is attempted at the lumbosacral junction here;
Section~\ref{sec:limitations} says why.""",
     r"""That figure is a logistic regression on five released per-level measures and six
ratios between them, each measure divided by the same patient's median across their
levels, with the folds grouped by case and the curve taken on pooled out-of-fold scores
(\texttt{test\_morphometric\_separability.py} in the released code). Schinz et al.\ reach
the same conclusion from the other side: on 1{,}242 whole-thoracolumbar CTs a shape-based
labeling of the junction matched nerve morphology in every case, against 92.6--97.2\% for
counting- and rib-based rules.\cite{schinz2026} The lumbosacral junction is not treated
this way here, for the reason in Section~\ref{sec:limitations}."""),

    # --- the counts, corrected against the release and stated once --------------------
    (r"""Stratification is on the transitional subtype rather than a binary LSTV flag. Across the
33 graded records the subtype reads sacralization in 15 (nine by a rib-free count of four,
six by the source label), lumbarization in 14, semi-sacralization in two and ambiguous in
two; the fourteen, two and one of Section~\ref{sec:anchors} are the subset of these that
also carry an L6. Every validation fold therefore carries three sacralization and about
three lumbarization records. With 15 and 14 across 802 that is the most stratification can
guarantee, and it is why a fold-level metric on the rare classes carries an error bar far
wider than its own decimal places.""",
     r"""Stratification is on the transitional subtype rather than a binary LSTV flag. Across all
802 records the source transitional label reads lumbarization in 14, sacralization in 17
and semi-sacralization in two; separately, 18 records carry a sixth lumbar-type vertebra,
nine carry only four lumbar labels, and 33 carry a consensus Castellvi grade. A record
called sacralization by its four lumbar labels enters the same stratum as one called
sacralization by the source label, so the two rare strata hold 15 records each and every
validation fold carries three of each. That is the most stratification can guarantee, and
it is why a fold-level metric on the rare classes carries an error bar far wider than its
own decimal places."""),

    # --- limitations, tightened and corrected -----------------------------------------
    (r"""The lumbosacral junction carries only 33 graded records and no shape-based or rule-based classifier is attempted there: with 15 sacralization and 14 lumbarization records, a case-grouped cross-validation would rest on three records per fold, too few for a stable estimate, and the classifier waits on the whole-cohort Castellvi read named below. The 0.990 area under the curve reported in the Introduction addresses the thoracolumbar boundary, not the lumbosacral one, and should not be read as evidence for the lumbosacral boundary. """,
     r"""No shape-based classifier is attempted at the lumbosacral junction: 15 records in each rare stratum leave three per fold, too few to estimate from, and such a classifier waits on the whole-cohort Castellvi read named below. The 0.990 area under the curve above is the thoracolumbar boundary, not this one. """),

    # --- the ACR citation, addressing "this source is for MRI, not CT" ----------------
    (r"""By guideline those are lumbar-only imaging studies~\cite{acr2021,nass2013,acrct2022}, T12 to S1 (the span the ACR practice parameter specifies for the lumbar spine, written for MRI and followed by lumbar CT~\cite{acrmri2023}), and whole-spine""",
     r"""By guideline those are lumbar-only imaging studies~\cite{acr2021,nass2013,acrct2022}, T12 to S1, the span the ACR spine parameters state for the lumbar spine~\cite{acrmri2023}, and whole-spine"""),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected one match, found {n}:\n{old[:120]}"
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("applied", len(edits), "tightening edits")
