---
title: "Multiview Radiographic Spine Assessment: fracture detection with anatomically correct level assignment"
subtitle: "Research proposal — Detroit Medical Center / Wayne State University"
author: "Gregory Schwing, MD, PhD — Department of Surgery"
date: "September 2026"
---

# Summary

Radiographs are the first imaging a trauma patient receives, and a spine radiograph is
read for two things at once: is a vertebra broken, and which vertebra is it. Automated
systems have become good at the first question and have largely assumed the second. This
proposal builds a multiview radiographic model that answers both, and adds the spinopelvic
geometry a surgical plan is built from.

The work begins with radiographs, refines on paired computed tomography, and is anchored on
an existing openly released CT resource — CTSpinoPelvic1K — that was built specifically to
make level identification testable rather than assumed.

# Background, and where the opportunity is

**Fracture detection on radiographs is close to solved, in isolation.** Zhang and
colleagues trained on a multicenter radiograph cohort and localized fresh vertebral
compression fractures with external-validation AUC of 0.90 (95% CI 0.84–0.95) and 0.84
(0.72–0.93) across two independent test cohorts.^1^ Comparable models now approach
subspecialist performance for the binary question.

**Automated spinopelvic measurement is also mature.** A 2025 meta-analysis of 15 studies
and more than 10,000 radiographs reported a pooled mean absolute error of 4.1° (95% CI
2.7–5.5) for pelvic incidence, with intraclass correlation coefficients above 0.81 against
human measurement.^2^

**Where this sits in trauma care.** Radiographs are what the patient gets first, and three
decisions follow from them: whether a fracture has been missed and will surface on a later
CT, whether a CT is obtained that the films could have settled, and — if the injury is
operative — which vertebra the plan is written against. The third is where an automated
reader can do harm rather than none: a confident level that is wrong is worse than no level
at all, and it is wrong most often in the patients whose anatomy is unusual.

**Both lines of work take the vertebral level as given, and that is the weak joint.** A
detector that reports a fracture at "L1" is counting, and the count is exactly what fails in
the population where it matters. Lumbosacral transitional vertebrae are reported in 4–30% of
people depending on definition,^3^ and the accepted standard for resolving numeration is
whole-spine imaging counted caudally from C2^4^ — which a trauma lateral of the
thoracolumbar spine cannot provide. Wrong-level spine surgery runs at roughly one in 3,110
procedures, and transitional anatomy is the cause most often cited.^5^

The ambiguity is specific rather than vague. Four rib-free lumbar vertebrae may mean an L1
bearing a lumbar rib or an L5 assimilated to the sacrum; six may mean a true sixth lumbar
vertebra or a T12 whose ribs are aplastic. The observation is identical within each pair, and
only the global count separates them.

**What we already hold that others do not.** CTSpinoPelvic1K is an openly licensed release of
802 CT records placing spine, pelvis, per-level ribs and femora on one coordinate frame, with
explicit classes for a sixth lumbar vertebra, a thirteenth thoracic vertebra, a separately
carved first sacral segment, and lumbar ribs, so a variant is recorded as itself rather than
forced into an ordinary level.^6^ It was built to support one question directly relevant
here: whether **local morphology alone** can name a level **without** the global count. It
also ships per-level morphometry across 802 records and validated derived spinopelvic
parameters, which give this project both a pretraining corpus and a reference distribution.

# Aims

**Aim 1 — Multiview fracture detection on DMC trauma radiographs.** Train a two-view
(AP and lateral) model for vertebral fracture detection and localization. The two views carry
different information and should not be collapsed: the lateral carries fracture morphology
(anterior wedging, endplate depression, height loss) and the sagittal geometry; the AP carries
the transverse processes, the twelfth rib, and the iliac crest — the features that decide
which vertebra is which.

**Aim 2 — Anatomically correct level assignment.** Predict level identity from local
morphology and in-view anchors (lowest rib-bearing vertebra above, sacral base below) rather
than by counting from the top of the film, and emit an explicit flag when the anatomy is
transitional. This is the aim that distinguishes the proposal from published fracture
detectors, and CTSpinoPelvic1K provides the supervision to train and evaluate it.

**Aim 3 — Automated spinopelvic parameters and anomaly reporting.** Report pelvic
incidence, pelvic tilt, sacral slope, lumbar lordosis and PI–LL mismatch from the same views,
benchmarked against the published error bands.^2^ Report numeric anomalies (L6, lumbar rib,
transitional junction) as findings in their own right.

# Approach

**Phase 1 — Radiographs first.** Assemble a retrospective DMC trauma radiograph cohort
(AP and lateral thoracolumbar and lumbar series). Fracture labels from the dictated report
with radiologist adjudication of a review subset. *[Case volume and date range to be
determined with Dr. Bartholomew.]*

**Phase 2 — CT-derived supervision.** Two routes, both available now. Where a patient has
both radiographs and CT from the same encounter, the CT settles both fracture and level and
becomes the reference standard for that case. Independently, digitally reconstructed
radiographs generated from the CTSpinoPelvic1K volumes provide radiograph-like images whose
level labels are already correct in transitional anatomy — a pretraining corpus that cannot
be obtained from radiographs alone, because on a radiograph the correct level is precisely
what is unknown.

**Phase 3 — Refinement on DMC CT.** Fine-tune the level and parameter heads on local CT to
capture scanner, protocol and population differences from the screening cohort
CTSpinoPelvic1K was drawn from.

**Evaluation.** Reported in the terms the decisions are made in, not only as summary
statistics. For fracture detection: sensitivity for injuries missed on the initial film and
identified on subsequent CT, alongside AUC for comparison with published benchmarks.^1^ For
level assignment: agreement with CT ground truth, reported **separately for anatomically
ordinary and transitional cases** — the stratification is the point, since aggregate accuracy
hides failure in exactly the 4–30% where the question is hard. For spinopelvic parameters:
agreement with manual measurement against the published error bands.^2^

# Why this is feasible here

The CT resource, the label scheme, the quality-control tooling and the derived morphometry
already exist and are openly archived.^6^ A trained annotation pipeline is running through
OpenSpineConsortium, with medical students working case-by-case against a written protocol
and every correction attributed. What the project needs and does not yet have is local trauma
imaging and the clinical judgement to aim it, which is the collaboration being proposed.

# What is being asked

- **Access to a retrospective DMC trauma imaging cohort** — paired radiographs and CT from
  the same encounter — under an approved protocol.
- **Clinical framing and endpoint selection.** Which outputs would change a decision, and at
  what operating point a missed injury is worse than a false alarm. This is the part the
  published work gets wrong most often and the part a trauma surgeon is best placed to fix.
- **Co-investigator standing** on the protocol, and guidance on the trauma-service workflow
  any eventual tool would have to fit.

Radiological adjudication of the label subset is already covered: two radiology
collaborators support the existing dataset work, so no reading burden falls on the trauma
service.

*[Cohort size, date range, scope, effort and authorship to be discussed. IRB determination
to be filed once the data request is defined.]*

# References

1. Zhang H, Xu R, Guo X, et al. Deep learning-based automated high-accuracy location and
   identification of fresh vertebral compression fractures from spinal radiographs: a
   multicenter cohort study. *Front Bioeng Biotechnol.* 2024;12:1397003.
   doi:10.3389/fbioe.2024.1397003

2. Glaser D, AlMekkawi AK, Caruso JP, et al. Deep learning for automated spinopelvic
   parameter measurement from radiographs: a meta-analysis. *Artif Intell Surg.*
   2025;5:1–15. doi:10.20517/ais.2024.36

3. Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings,
   and clinical relevance. *AJNR Am J Neuroradiol.* 2010;31(10):1778–1786.
   doi:10.3174/ajnr.A2036

4. Lian J, Levine N, Cho W. A review of lumbosacral transitional vertebrae and associated
   vertebral numeration. *Eur Spine J.* 2018;27(5):995–1004. doi:10.1007/s00586-018-5554-8

5. Epstein NE. A perspective on wrong level, wrong side, and wrong site spine surgery.
   *Surg Neurol Int.* 2021;12:286. doi:10.25259/SNI_402_2021

6. Schwing G, et al. CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate
   frame, annotated for lumbosacral transitional anatomy. Zenodo.
   doi:10.5281/zenodo.22139642 (concept identifier; v7 archived at
   doi:10.5281/zenodo.22242745). Dataset descriptor in preparation.
