---
title: "Multiview Radiographic Fracture Detection with Correct Anatomic Naming"
subtitle: "Research proposal — Detroit Medical Center / Wayne State University"
author: "Gregory Schwing, MD, PhD — Department of Surgery"
date: "September 2026"
---

# Summary

A trauma radiograph is read for two things: is a bone broken, and which bone is it.
Automated systems have become good at the first and have largely assumed the second. This
proposal builds a multiview model that answers both across the structures a trauma series
actually covers — vertebrae, sacrum, pelvis and hips, ribs, and proximal femur — and adds
the spinopelvic geometry a surgical plan is built from.

It starts with radiographs, refines on paired CT, and is anchored on an existing openly
released CT resource, CTSpinoPelvic1K, that already carries every one of those structures on
one coordinate frame.

# Background and gap

**Detection is close to solved in isolation.** Zhang and colleagues localized fresh vertebral
compression fractures on radiographs with external-validation AUC of 0.90 (95% CI 0.84–0.95)
and 0.84 (0.72–0.93) across two independent cohorts.^1^ Automated spinopelvic measurement is
likewise mature: a 2025 meta-analysis of 15 studies and more than 10,000 radiographs reports
pooled mean absolute error of 4.1° for pelvic incidence, with ICC above 0.81 against human
measurement.^2^

**Naming is not.** A detector reporting a fracture at "L1" is counting, and the count fails
where it matters most. Lumbosacral transitional vertebrae occur in 4–30% of people depending
on definition,^3^ the accepted standard for resolving numeration is whole-spine imaging
counted caudally from C2^4^ — which a trauma film cannot provide — and wrong-level spine
surgery runs at roughly one in 3,110 procedures, with transitional anatomy the usual cause.^5^
The ambiguity is specific: four rib-free lumbar vertebrae may mean an L1 bearing a lumbar rib
or an L5 assimilated to the sacrum; six may mean a true sixth lumbar vertebra or a T12 whose
ribs are aplastic. The observation is identical within each pair.

The problem recurs beyond the spine. Ribs are numbered by counting, and side is routinely
mislabeled — quality control on our own CT corpus caught four records carrying a left hip on
the patient's right, a failure invisible downstream.

**In trauma this is where an automated reader can do harm rather than none.** A confident
site that is wrong is worse than no site at all, and it is wrong most often in the patients
whose anatomy is unusual.

**What we already hold.** CTSpinoPelvic1K is an openly licensed release of 802 CT records
placing spine, sacrum, hips, femora and per-level ribs on one coordinate frame, with explicit
classes for a sixth lumbar vertebra, a thirteenth thoracic vertebra, a separately carved
first sacral segment, and lumbar ribs.^6^ It was built to test whether **local morphology
alone** can name a structure **without** the global count, and it ships per-level morphometry
and validated spinopelvic parameters — a pretraining corpus and a reference distribution in
one.

# Aims

**1. Multiview detection across the trauma series.** A two-view (AP and lateral) model for
fracture detection and localization across vertebrae, sacrum, pelvis and hips, ribs, and
proximal femur. The views are not collapsed: the lateral carries fracture morphology and
sagittal geometry; the AP carries transverse processes, the twelfth rib and the iliac crest —
the features that decide identity.

**2. Correct anatomic naming.** Vertebral level, rib number and side predicted from local
morphology and in-view anchors rather than by counting from the edge of the film, with an
explicit flag when the anatomy is transitional. This is what distinguishes the work from
published detectors, and CTSpinoPelvic1K supplies the supervision.

**3. Spinopelvic parameters and anomaly reporting.** Pelvic incidence, tilt, sacral slope,
lumbar lordosis and PI–LL mismatch from the same views, benchmarked against published error
bands,^2^ with numeric anomalies (L6, lumbar rib, transitional junction) reported as findings.

# Approach

**Radiographs first.** A retrospective DMC trauma cohort, AP and lateral, with fracture
labels from the dictated report and radiologist adjudication of a review subset.

**CT-derived supervision.** Where a patient has both radiographs and CT from one encounter,
the CT settles both fracture and site. Independently, digitally reconstructed radiographs
from the CTSpinoPelvic1K volumes give radiograph-like images whose labels are already correct
in transitional anatomy — supervision unobtainable from radiographs alone, since on a
radiograph the correct name is precisely what is unknown.

**Refinement on DMC CT** for local scanner, protocol and population differences.

**Evaluation, in the terms the decisions are made in.** Sensitivity for injuries missed on
the initial film and identified on subsequent CT, alongside AUC for comparison with published
benchmarks.^1^ Naming accuracy against CT, reported **separately for ordinary and transitional
anatomy** — aggregate accuracy hides failure in exactly the 4–30% where the question is hard.
Spinopelvic parameters against the published error bands.^2^

*Scope note: the CT corpus covers the axial skeleton, hips and proximal femur. Long bones
below the femur would need a separate imaging source and are a later extension, not part of
the initial aims.*

# What is being asked

- **Access to a retrospective DMC trauma imaging cohort** — paired radiographs and CT from
  the same encounter — under an approved protocol.
- **Clinical framing and endpoint selection:** which outputs would change a decision, and at
  what operating point a missed injury is worse than a false alarm.
- **Co-investigator standing** and guidance on the trauma-service workflow any eventual tool
  would have to fit.

Radiological adjudication is already covered by two radiology collaborators supporting the
existing dataset work; no reading burden falls on the trauma service.

*[Cohort size, date range, scope, effort and authorship to be discussed. IRB determination to
be filed once the data request is defined.]*

# References

1. Zhang H, Xu R, Guo X, et al. Deep learning-based automated high-accuracy location and
   identification of fresh vertebral compression fractures from spinal radiographs: a
   multicenter cohort study. *Front Bioeng Biotechnol.* 2024;12:1397003.
   doi:10.3389/fbioe.2024.1397003
2. Glaser D, AlMekkawi AK, Caruso JP, et al. Deep learning for automated spinopelvic
   parameter measurement from radiographs: a meta-analysis. *Artif Intell Surg.* 2025;5:1–15.
   doi:10.20517/ais.2024.36
3. Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings,
   and clinical relevance. *AJNR Am J Neuroradiol.* 2010;31(10):1778–1786.
   doi:10.3174/ajnr.A2036
4. Lian J, Levine N, Cho W. A review of lumbosacral transitional vertebrae and associated
   vertebral numeration. *Eur Spine J.* 2018;27(5):995–1004. doi:10.1007/s00586-018-5554-8
5. Epstein NE. A perspective on wrong level, wrong side, and wrong site spine surgery.
   *Surg Neurol Int.* 2021;12:286. doi:10.25259/SNI_402_2021
6. Schwing G, et al. CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame,
   annotated for lumbosacral transitional anatomy. Zenodo. doi:10.5281/zenodo.22139642
   (concept identifier; v7 at doi:10.5281/zenodo.22242745). Dataset descriptor in preparation.
