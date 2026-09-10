r"""Apply Ashley Schehr's revisions (tracked changes and comments in
'CTSpinoPelvic1K_schehr revisions + sources notes.docx', 9 Sept 2026) to main.tex.

Every replacement asserts that its target string is present exactly once, so a drifted
manuscript fails loudly instead of silently skipping an edit.
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")

edits = [
    # --- tracked changes -------------------------------------------------------------
    # abstract: "a lumbar study, T12 to S1," -> "lumbar-only imaging (T12 to S1)"; cite the CT parameter too
    (r"a lumbar surgical case is planned on a lumbar study~\cite{acr2021,acrmri2023}, T12 to S1, without C2.",
     r"a lumbar surgical case is planned on lumbar-only imaging~\cite{acr2021,acrct2022,acrmri2023} (T12 to S1) without C2."),
    # abstract limitations: "recumbent" -> "are supine"
    (r"postural" + "\n" + r"angles recumbent; no held-out test set;",
     r"postural" + "\n" + r"angles are supine; no held-out test set;"),
    # introduction: "lumbar studies" -> "lumbar-only imaging studies"; address the MRI-not-CT comment
    (r"By guideline those are lumbar studies~\cite{acr2021,nass2013,acrmri2023}, T12 to S1, and whole-spine",
     r"By guideline those are lumbar-only imaging studies~\cite{acr2021,nass2013,acrct2022}, T12 to S1 (the span the ACR practice parameter specifies for the lumbar spine, written for MRI and followed by lumbar CT~\cite{acrmri2023}), and whole-spine"),
    # validation: stray comma
    (r"802 records while four carried \texttt{left\_hip} on the patient's right, and pooling all",
     r"802 records while four carried \texttt{left\_hip} on the patient's right and pooling all"),
    # --- comments 12, 13, 14: how the AUC was computed; compare with Schinz; why not LSTV ---
    (r"""accuracy over 1485 vertebrae. Size is removed because raw millimeters separate thoracic
from lumbar trivially and teach nothing about the junction, where the two are nearly the
same size.
""",
     r"""accuracy over 1485 vertebrae. Size is removed because raw millimeters separate thoracic
from lumbar trivially and teach nothing about the junction, where the two are nearly the
same size. The classifier is a logistic regression on five released per-level measures
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
Section~\ref{sec:limitations} says why.
"""),
    # --- comments 45, 46: reconcile the sacralization and lumbarization counts ---------
    (r"""Stratification is on the transitional subtype rather than a binary LSTV flag, so every validation fold carries three sacralization and three lumbarization records. With 15 of
each across 802 that is the most stratification can guarantee, and it is why a fold-level
metric on the rare classes carries an error bar far wider than its own decimal places.""",
     r"""Stratification is on the transitional subtype rather than a binary LSTV flag. Across the
33 graded records the subtype reads sacralization in 15 (nine by a rib-free count of four,
six by the source label), lumbarization in 14, semi-sacralization in two and ambiguous in
two; the fourteen, two and one of Section~\ref{sec:anchors} are the subset of these that
also carry an L6. Every validation fold therefore carries three sacralization and about
three lumbarization records. With 15 and 14 across 802 that is the most stratification can
guarantee, and it is why a fold-level metric on the rare classes carries an error bar far
wider than its own decimal places."""),
    # --- comment 47: cite the primary source for the wrong-level rate --------------------
    (r"cohort was assembled around.\cite{epstein2021} Per-level body height, canal width,",
     r"cohort was assembled around.\cite{mody2008,epstein2021} Per-level body height, canal width,"),
    # --- comment 12 and Ashley's inserted limitation text -------------------------------
    (r"""\emph{Limitations.} The cohort comes from one acquisition protocol of a colorectal-screening population aged 50 and over, so how well it generalizes to a pathologic population is untested. Human review of the rib layer was triaged by an automated rule, so it guarantees only the absence of the failures that rule detects. """,
     r"""\emph{Limitations.}\label{sec:limitations} The cohort comes from one acquisition protocol of a colorectal-screening population aged 50 and over, so how well it generalizes to a pathologic population is untested. Human review of the rib layer was triaged by an automated rule, so it guarantees only the absence of the failures that rule detects. The lumbosacral junction carries only 33 graded records and no shape-based or rule-based classifier is attempted there: with 15 sacralization and 14 lumbarization records, a case-grouped cross-validation would rest on three records per fold, too few for a stable estimate, and the classifier waits on the whole-cohort Castellvi read named below. The 0.990 area under the curve reported in the Introduction addresses the thoracolumbar boundary, not the lumbosacral one, and should not be read as evidence for the lumbosacral boundary. """),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected exactly one match, found {n}:\n{old[:120]}"
    s = s.replace(old, new)

# the anchors subsection needs a label for the count cross-reference
old = r"\subsection{Label scheme and counting anchors}"
assert s.count(old) == 1
if r"\label{sec:anchors}" not in s:
    s = s.replace(old, old + r"\label{sec:anchors}")

p.write_text(s, encoding="utf-8")
print("applied", len(edits), "edits to main.tex")
