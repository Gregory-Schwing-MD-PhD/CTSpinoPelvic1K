r"""compress.py -- fit the manuscript back to ten pages after the compliance additions.

The manuscript was exactly at the ten-page limit before PUB 3-C compliance work; the
required additions -- a computational-tools subsection, a metadata-completeness table,
Discussion and Conclusion sections, and the S1 method -- took it to twelve. Excess pages are
billed at $200 each.

WHAT IS CUT, AND WHAT IS NOT. Only text written during the compliance pass is tightened.
None of the authors' existing content is removed: deciding which of their material to lose
is an editorial judgement for them, not a formatting fix. The completeness table is
condensed by grouping fields that share a denominator, which loses no information because
the grouped rows carried identical counts.
"""
from pathlib import Path

p = Path(__file__).parent / "main.tex"
s = p.read_text(encoding="utf-8")
before = len(s)


def sub(old, new, label):
    global s
    assert old in s, f"anchor missing: {label}"
    s = s.replace(old, new, 1)
    print(f"  - {label}: {len(old)} -> {len(new)} chars")


# ---- completeness table: group rows that share a denominator --------------
old_rows = """Group & Field & Records & Available \\\\
\\hline
Identity and files & volume identifier, image path, label path & 802 & 100.0\\% \\\\
                   & record configuration, match type & 802 & 100.0\\% \\\\
\\hline
Acquisition & patient position & 802 & 100.0\\% \\\\
            & tube potential, section thickness, reconstruction kernel & 802 & 100.0\\% \\\\
            & scanner manufacturer and model & 768 & 95.8\\% \\\\
\\hline
Crosswalk & spine series identifier & 782 & 97.5\\% \\\\
          & pelvic series identifier & 713 & 88.9\\% \\\\
\\hline
Demographics & sex & 749 & 93.4\\% \\\\
             & age & 709 & 88.4\\% \\\\
\\hline
Transitional anatomy & transitional label and class & 802 & 100.0\\% \\\\
                     & Castellvi grade & 33 & 4.1\\% \\\\
\\hline
Label origin & spine annotation origin & 802 & 100.0\\% \\\\
             & pelvic annotation origin & 802 & 100.0\\% \\\\
             & instrumentation flag & 802 & 100.0\\% \\\\
             & partial-annotation flag & 802 & 100.0\\% \\\\
\\hline
Quality control & image/label alignment check & 802 & 100.0\\% \\\\
                & bone-fraction check at the spine & 782 & 97.5\\% \\\\"""

new_rows = """Field & Records & Available \\\\
\\hline
Identifier, image path, label path, configuration, match type & 802 & 100.0\\% \\\\
Patient position, tube potential, section thickness, kernel & 802 & 100.0\\% \\\\
Transitional label and class & 802 & 100.0\\% \\\\
Spine and pelvic annotation origin & 802 & 100.0\\% \\\\
Instrumentation and partial-annotation flags & 802 & 100.0\\% \\\\
Image/label alignment check & 802 & 100.0\\% \\\\
Spine series identifier; bone-fraction check & 782 & 97.5\\% \\\\
Scanner manufacturer and model & 768 & 95.8\\% \\\\
Sex & 749 & 93.4\\% \\\\
Age & 709 & 88.4\\% \\\\
Pelvic series identifier & 713 & 88.9\\% \\\\
Castellvi grade & 33 & 4.1\\% \\\\"""
sub(old_rows, new_rows, "completeness table: group equal-denominator rows")
sub(r"\begin{tabular}{llrr}", r"\begin{tabular}{lrr}", "table: four columns to three")
sub(r"\begin{table*}[t]", r"\begin{table}[t]", "table: full-width to single column")
sub(r"\end{table*}", r"\end{table}", "table: close single column")

# ---- Discussion: same content, less prose --------------------------------
sub("""\\emph{What the release enables.} The material supports three kinds of reuse without further
annotation: segmentation and level-numbering benchmarks on a cohort where the count is
genuinely ambiguous rather than merely unlabelled; morphometric description of the
lumbosacral junction using landmarks that are present in the field of view; and
templating-style work on the instrumented subset, where the metal is labelled rather than
absorbed into the bone beside it. Because the release places spine, pelvis, ribs and femora
in one coordinate frame, measurements that require both a spinal and a pelvic landmark ---
sacral slope and pelvic incidence among them --- can be computed without registering two
annotations to each other.""",
"""\\emph{What the release enables.} Three kinds of reuse need no further annotation:
segmentation and level-numbering benchmarks on a cohort where the count is genuinely
ambiguous rather than merely unlabelled; morphometric description of the lumbosacral
junction from landmarks inside the field of view; and templating work on the instrumented
subset, where metal is labelled rather than absorbed into the bone beside it. Because spine,
pelvis, ribs and femora share one coordinate frame, a measurement needing both a spinal and
a pelvic landmark --- sacral slope and pelvic incidence among them --- requires no
registration between two annotations.""",
    "discussion: tighten paragraph 1")

sub("""\\emph{Limitations with respect to future use.} The cohort is a colon-screening population
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
sacralisation records a per-fold estimate on the rare classes rests on roughly three cases.""",
"""\\emph{Limitations with respect to future use.} The cohort is a colon-screening population
imaged supine and prone --- not a trauma, deformity or paediatric series --- so
generalisation beyond adult abdominal computed tomography is not established here. No scan
contains C2, so the conventional cervical-down count cannot be performed on any record;
that is a property of the imaging rather than the annotation, and it is why the measures
reported here are count-free. Label strength is not uniform: vertebral labels derive from
radiologist-supervised sources, pelvic labels on records that lacked one are model-completed,
and the rib layer is a pseudolabel whose review was triaged rather than exhaustive. The
sacral sub-division inherits its level from an automatic method, and in 100 records that
level places S1 above half the sacrum or below fifteen percent of it, which no first sacral
segment is; those records are flagged in the released quality-control table and should be
filtered before any measurement depending on the sacral endplate.""",
    "discussion: tighten paragraph 2")

sub("""\\emph{Comparison with related material.} Sources annotating the spine alone cannot supply a
pelvic landmark, and sources annotating the pelvis alone cannot supply a vertebral count;
the contribution here is the crosswalk that places both on the same series, together with
classes for the anatomy an enumeration anomaly actually produces. Where an existing scheme
must record a sixth lumbar vertebra or a rib borne by a lumbar vertebra as something else,
this release records it as itself.""",
"""\\emph{Comparison with related material.} Sources annotating the spine alone cannot supply a
pelvic landmark, and sources annotating the pelvis alone cannot supply a vertebral count.
The contribution here is the crosswalk placing both on the same series, with classes for the
anatomy an enumeration anomaly actually produces: where another scheme must record a sixth
lumbar vertebra or a lumbar-borne rib as something else, this release records it as itself.""",
    "discussion: tighten paragraph 3")

# ---- Conclusion ----------------------------------------------------------
sub("""CTSpinoPelvic1K is an openly licensed release of 802 abdominal computed-tomography records in
which the spine, pelvis, per-level ribs and femora share one coordinate frame, and in which
the anatomy that makes lumbar numbering ambiguous --- a sixth lumbar vertebra, a rib borne by
a lumbar vertebra, a transitional lumbosacral junction --- has a class of its own rather than
being recorded as something it is not. Two counting anchors are provided explicitly, the last
rib-bearing thoracic vertebra above and a sacral sub-division below, so that a level can be
identified from a field of view that cannot support the conventional count. The release is
archived with a persistent identifier, ships the loader and quality-control tables needed to
verify it, and states per structure how strong each label is. Its intended reuse is
segmentation and level-numbering benchmarking, morphometric description of the lumbosacral
junction, and templating work on the instrumented subset.""",
"""CTSpinoPelvic1K is an openly licensed release of 802 abdominal computed-tomography records
placing spine, pelvis, per-level ribs and femora in one coordinate frame, in which the
anatomy that makes lumbar numbering ambiguous --- a sixth lumbar vertebra, a lumbar-borne
rib, a transitional lumbosacral junction --- carries a class of its own rather than being
recorded as something it is not. Two counting anchors are explicit, the last rib-bearing
thoracic vertebra above and a sacral sub-division below, so a level can be identified from a
field of view that cannot support the conventional count. The release is archived under a
persistent identifier, ships the loader and quality-control tables needed to verify it, and
states per structure how strong each label is.""",
    "conclusion: tighten")

p.write_text(s, encoding="utf-8")
print(f"main.tex: {before} -> {len(s)} chars ({before-len(s)} removed)")
