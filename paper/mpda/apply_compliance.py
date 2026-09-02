"""apply_compliance.py -- bring main.tex to PUB 3-C + Medical Physics author guidelines, on v7.

Every edit below is anchored on an exact string from the current main.tex and asserts before
replacing, so a silent no-op is impossible: if the manuscript moves under this script, it
fails loudly rather than producing a file that looks edited and is not.

WHAT CHANGES AND WHY

  DOUBLE-BLIND. Medical Physics moved to double-anonymised review: "Do not include names or
  institutional affiliations anywhere in the manuscript". Authors and affiliations move to a
  separate title page, and the repository URL -- which carries the corresponding author's
  name -- is replaced by the DOI, which does not.

  PUB 3-C, which superseded the MPDA policy on 2026-05-16 and renamed the article type to
  "Dataset and Software Article". Four hard requirements were unmet: a "computational tools"
  subheading, a metadata-completeness table, a link to the material at the end of the
  introduction, and explicit Discussion and Conclusion sections.

  GENERATIVE AI. The guidelines require any AI manuscript-preparation tool to be declared in
  the Methods section under COPE guidance.

  v7. The dataset was re-released after the manuscript was last written. S1 is now cut by a
  plane rather than by intersection with a network's output, and the record that lacked a
  pelvis has been completed -- so the census paragraph was factually wrong.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
p = HERE / "main.tex"
s = p.read_text(encoding="utf-8")
orig_len = len(s)
edits = []


def sub(old: str, new: str, label: str):
    global s
    assert old in s, f"ANCHOR NOT FOUND for {label!r}"
    assert s.count(old) == 1, f"anchor not unique for {label!r} ({s.count(old)} matches)"
    s = s.replace(old, new)
    edits.append(label)


# ---------------------------------------------------------------- 1. preamble
sub("""%  THREE CONSTRAINTS SHAPE THIS FILE, all from the MPDA policy:""",
    """%  Article type: DATASET AND SOFTWARE ARTICLE (DS). AAPM policy PUB 3-C superseded the
%  Medical Physics Dataset Article policy (PUB 3-B) on 2026-05-16 and renamed the type.
%
%  THREE CONSTRAINTS SHAPE THIS FILE, all from the PUB 3-C policy:""",
    "preamble: PUB 3-C / DS article type")

sub(r"""\usepackage[colorlinks=true,allcolors=blue]{hyperref}""",
    r"""\usepackage[colorlinks=true,allcolors=blue]{hyperref}
% Continuous line numbering is a submission requirement, not a preference.
\usepackage[mathlines]{lineno}""",
    "preamble: lineno package")

sub(r"""\newcommand{\datasetdoi}{10.5281/zenodo.22139643}""",
    r"""% The CONCEPT DOI, which always resolves to the current version. PUB 3-C added
% "Allow record versioning and file management" to the repository requirements, and a
% referee instructed to "verify existence/accessibility of material via supplied DOI"
% should land on the release this manuscript describes rather than a superseded one.
\newcommand{\datasetdoi}{10.5281/zenodo.22139642}
% The exact version this manuscript describes, for reproducibility.
\newcommand{\datasetversiondoi}{10.5281/zenodo.22242745}
\newcommand{\datasetversion}{v7}""",
    "preamble: concept + version DOI")

# ------------------------------------------------------------- 2. blinding
sub(r"""\author{Gregory Schwing}""",
    r"""% DOUBLE-ANONYMISED REVIEW: author names and affiliations live in title_page.tex and
% must NOT appear here. Restore this block only for the camera-ready version.
%\author{Gregory Schwing}""",
    "blind: first author")

for nm in ["Ashley Schehr", "Annika Tekumulla", "Margret Khoushi", "Ryan Christian",
           "Dane Hubers", "Faris Mahjoub", "Hassan Saad", "Mia Sooch",
           "Sathya Siddapureddy", "Michael McLellan", "Jerick Kim"]:
    sub("\\author{%s}" % nm, "%%\\author{%s}" % nm, f"blind: {nm}")

sub(r""" \affiliation{Department of Surgery, Detroit Medical Center / Wayne State University, Detroit, Michigan, USA}""",
    r"""%\affiliation{Department of Surgery, Detroit Medical Center / Wayne State University, Detroit, Michigan, USA}""",
    "blind: affiliation 1")
sub(r""" \affiliation{Wayne State University School of Medicine, Detroit, Michigan, USA}""",
    r"""%\affiliation{Wayne State University School of Medicine, Detroit, Michigan, USA}
\author{[Author names removed for double-anonymised review]}
\affiliation{[Affiliations removed for double-anonymised review]}""",
    "blind: affiliation 2 + placeholder")

# ------------------------------------------------- 3. Introduction + material link
sub(r"""\section{Purpose}""", r"""\section{Introduction}""", "section: Purpose -> Introduction")

sub("""a class to each structure an enumeration anomaly produces: a sixth lumbar vertebra, and a
rib borne by a lumbar vertebra. A scheme without those classes must record such anatomy as
something it is not.""",
    r"""a class to each structure an enumeration anomaly produces: a sixth lumbar vertebra, and a
rib borne by a lumbar vertebra. A scheme without those classes must record such anatomy as
something it is not.

\emph{Where the material is.} The release described here (\datasetversion) is archived at
\url{https://doi.org/\datasetdoi}, which resolves to the current version; the exact version
described by this manuscript is \datasetversiondoi. Documentation, the label scheme, the
reference loader and the per-record quality-control tables are distributed with the archive
and are described in Sec.~\ref{sec:format}.""",
    "intro: link to material at end of Introduction (PUB 3-C B.2.1)")

# --------------------------------------------- 4. computational tools + AI declaration
sub(r"""\subsection{Validation}""",
    r"""\subsection{Computational tools}

\emph{Access.} Every processing step described above is performed by code released under an
open-source licence and archived with the dataset; no step depends on software that is not
publicly obtainable. The release archive carries the reference loader, the label-scheme
definition and the quality-control scripts that generated the tables in this manuscript.

\emph{Versions and parameters.} Segmentation of femora, the sacral sub-division and the
per-level rib numbering used TotalSegmentator~\cite{totalsegmentator} (\texttt{total} task,
release 2.x) on a single graphics processing unit. The binary rib network is
M\"oller's~\cite{ribseg} released weights, applied through the nnU-Net v2
framework~\cite{nnunet}. The pelvic completion for records whose pelvis was absent used a
five-fold nnU-Net v2 ensemble, applied out-of-fold: each record is completed by the fold that
never trained on it, so no record is completed by a model that has seen it. Image handling
throughout used NiBabel and SciPy; no image is resampled or reoriented when a label is
written, so every label shares its image's grid and affine exactly.

\emph{Generative artificial intelligence.} Large-language-model assistance was used for
manuscript preparation --- drafting and editing of text, and generation of analysis and
figure code --- under author review. It was not used to generate, annotate or interpret any
imaging data, and no scientific claim in this manuscript originates from it. All numbers
reported here are computed by the released code from the released volumes.

\subsection{Validation}""",
    "methods: computational tools subsection + AI declaration")

# ---------------------------------------------- 5. S1 carve method (undocumented before)
sub("""Neither anchor is novel. TotalSegmentator~\\cite{totalsegmentator} carries a separate S1""",
    r"""\emph{How S1 is cut.} The sacrum's outer boundary is the source annotation and is never
altered; S1 is the cranial part of that same bone, separated by a plane. The plane's normal
is the sacrum's own cranio-caudal axis --- taken as the principal component most closely
aligned with the superior--inferior direction, since the bone is nearly as wide as it is
tall and its longest axis is therefore not reliably the cranio-caudal one --- made orthogonal
to the patient's left--right axis. That left--right axis is measured rather than assumed, as
the normal of the reflection plane that best maps the sacrum onto itself, which removes any
rotation of the patient within the scanner: across this cohort that rotation has a median of
4.5\textdegree{} and reaches 16.1\textdegree{}, and 508 of 801 records exceed
3\textdegree{}. The level of the cut preserves the sub-division volume of the preceding
release, so the plane changes the shape and orientation of the S1/S2 boundary and not how
much of the sacrum is called S1. Per-record geometry ships with the archive.

Neither anchor is novel. TotalSegmentator~\cite{totalsegmentator} carries a separate S1""",
    "methods: document the S1 carve (v7)")

# ------------------------------------------------------- 6. v7 census correction
sub("""\\emph{Structures are not universally present.} The same census shows the release is not
uniformly complete: one record carries no sacrum, hips or femora, and two carry no S1. Two
lack T12 and nine lack an L5 identifier, in every case because the structure lies outside
the field of view or, for L5, because the lowest lumbar segment is labelled L6 or
incorporated into the sacrum. A user filtering on the presence of the caudal anchor should
filter explicitly rather than assume it.""",
    r"""\emph{Structures are not universally present.} The same census shows the release is not
uniformly complete. Sacrum, hips and femora are present in all 802 records. One record
carries no S1, one lacks T12, and nine lack an L5 identifier --- in every case because the
structure lies outside the field of view or, for L5, because the lowest lumbar segment is
labelled L6 or incorporated into the sacrum. A user filtering on the presence of the caudal
anchor should filter explicitly rather than assume it.

\emph{The partial-annotation sentinel is unused.} Identifier 255 marks a region that was not
traced and is not background. In the preceding release exactly one record used it; that
record has since been completed and no record in this release carries it. The identifier is
retained because the protocol is part of the scheme and a later release may again ship a
partially traced record.""",
    "results: v7 census correction (sacrum/hips/femora complete, ignore unused)")

# --------------------------------------------------- 7. Discussion + Conclusion
sub(r"""\section*{Data Availability}""",
    r"""\section{Discussion}

\emph{What the release enables.} The material supports three kinds of reuse without further
annotation: segmentation and level-numbering benchmarks on a cohort where the count is
genuinely ambiguous rather than merely unlabelled; morphometric description of the
lumbosacral junction using landmarks that are present in the field of view; and
templating-style work on the instrumented subset, where the metal is labelled rather than
absorbed into the bone beside it. Because the release places spine, pelvis, ribs and femora
in one coordinate frame, measurements that require both a spinal and a pelvic landmark ---
sacral slope and pelvic incidence among them --- can be computed without registering two
annotations to each other.

\emph{Limitations with respect to future use.} The cohort is a colon-screening population
imaged supine and prone, so it is neither a trauma nor a deformity nor a paediatric series,
and generalisation beyond abdominal computed tomography of adults is not established here.
No scan contains C2, so the conventional cervical-down count cannot be performed on any
record; that is a property of the imaging rather than of the annotation, and it is the
reason the measures reported here are count-free. Label strength is not uniform: vertebral
labels derive from radiologist-supervised sources, pelvic labels on records that lacked one
are completed by a model, and the rib layer is a pseudolabel whose human review was triaged
rather than exhaustive. The sub-division of the sacrum into S1 inherits its level from an
automatic method, and in 100 records that level places S1 at more than half the sacrum or
under fifteen percent of it, which no first sacral segment is; those records are flagged in
the released quality-control table and should be filtered before any measurement that
depends on the sacral endplate. The shipped folds are cross-validation folds covering the
whole cohort, so there is no held-out test set, and with fifteen lumbarisation and fifteen
sacralisation records a per-fold estimate on the rare classes rests on roughly three cases.

\emph{Comparison with related material.} Sources annotating the spine alone cannot supply a
pelvic landmark, and sources annotating the pelvis alone cannot supply a vertebral count;
the contribution here is the crosswalk that places both on the same series, together with
classes for the anatomy an enumeration anomaly actually produces. Where an existing scheme
must record a sixth lumbar vertebra or a rib borne by a lumbar vertebra as something else,
this release records it as itself.

\section{Conclusion}

CTSpinoPelvic1K is an openly licensed release of 802 abdominal computed-tomography records in
which the spine, pelvis, per-level ribs and femora share one coordinate frame, and in which
the anatomy that makes lumbar numbering ambiguous --- a sixth lumbar vertebra, a rib borne by
a lumbar vertebra, a transitional lumbosacral junction --- has a class of its own rather than
being recorded as something it is not. Two counting anchors are provided explicitly, the last
rib-bearing thoracic vertebra above and a sacral sub-division below, so that a level can be
identified from a field of view that cannot support the conventional count. The release is
archived with a persistent identifier, ships the loader and quality-control tables needed to
verify it, and states per structure how strong each label is. Its intended reuse is
segmentation and level-numbering benchmarking, morphometric description of the lumbosacral
junction, and templating work on the instrumented subset.

\section*{Data Availability}""",
    "add Discussion and Conclusion sections (PUB 3-C B.2.4, B.2.5)")

# ------------------------------------------------ 8. Data Availability, blinded
sub(r"""The dataset is permanently archived at \url{https://doi.org/\datasetdoi}. Code is at \url{https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K}, tagged at the release commit.""",
    r"""The dataset is permanently archived at \url{https://doi.org/\datasetdoi}, which resolves to
the current version; the release described here is \datasetversion{} (\datasetversiondoi).
The archive carries the reference loader, the label-scheme definition and the
quality-control tables, and the analysis code is distributed with it under an open-source
licence. The repository URL is omitted here because it identifies the authors; it is given
on the title page and will be restored to this section for the camera-ready version.""",
    "data availability: blinded, concept + version DOI")

p.write_text(s, encoding="utf-8")
print(f"main.tex: {orig_len} -> {len(s)} chars, {len(edits)} edits")
for e in edits:
    print(f"  - {e}")
