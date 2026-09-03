---
title: "Automated fracture detection and level identification on trauma radiographs"
subtitle: "Analysis plan under the DMC Trauma QI protocol (PI: Rahul Vaidya, MD)"
author: "Gregory Schwing, MD, PhD — Department of Surgery, Detroit Medical Center / Wayne State University"
date: "September 2026"
---

# What we want to do

We want to train a model that reads a trauma spine radiograph and reports three things:
whether there is a fracture, which vertebra it is in, and whether it is displaced. We then
want to check how often it is right using the imaging and reports already collected under the
Trauma QI protocol.

This is a retrospective analysis of existing imaging. Nothing the model produces goes into
a chart, gets shown to a treating physician, or affects any patient's care. The output is a
paper and a set of model weights.

# Why this is worth doing

Commercial fracture detection already exists and works. GLEAMER's BoneView is FDA-cleared for
the appendicular skeleton, ribs, and thoracolumbar spine, runs in over 300 hospitals, and in
its reader study improved fracture sensitivity by about 10 points and cut false negatives by
29%.^1^ We are not trying to beat that.

The problem these tools do not solve is naming the level. A detector that says "fracture at
L1" gets there by counting vertebrae on the film, and counting is unreliable in exactly the
patients where it matters. Lumbosacral transitional vertebrae occur in somewhere between 4%
and 30% of people depending on the definition,^2^ and the accepted way to number the spine is
to count down from C2 on whole-spine imaging^3^ — which no one gets in the trauma bay.
Wrong-level spine surgery happens roughly once per 3,100 procedures and transitional anatomy
is the most common reason.^4^

For a trauma surgeon the practical version is this: a confident wrong level on the initial
film is worse than no level at all, and it will be wrong most often in the patient whose
anatomy is unusual. Same problem for ribs (numbered by counting) and for side (routinely
mislabeled — our own CT quality control caught four cases with a left hip labeled on the
patient's right).

We have a resource nobody else has for this. CTSpinoPelvic1K is a public CT dataset we
built: 802 scans with the spine, sacrum, hips, femora, and every rib labeled on one
coordinate frame, with explicit classes for L6, a thirteenth thoracic vertebra, a separate
S1, and lumbar ribs.^5^ It was built specifically to test whether a vertebra can be named
from its own appearance without counting from C2. That is the supervision a radiograph model
needs and cannot get from radiographs.

# Why radiographs, and whether we actually need CT

**Radiographs are the target, so they are non-negotiable.** The model has to work on the
film a patient actually gets first. Training and evaluation both have to be on real trauma
radiographs from this population, AP and lateral, because the AP carries the twelfth rib and
the transverse processes (what decides the level) and the lateral carries the fracture
morphology and the sagittal alignment.

**CT is needed for less than it looks like.** It is not needed to train the detector. The
fracture and outline heads train on public radiograph datasets, and the level-naming head
pretrains on radiograph-like projections rendered from our own CT dataset. What DMC CT gives
us is ground truth for two specific questions the radiograph cannot answer about itself:

1. *Was the level right?* On a transitional spine, the film cannot settle which vertebra is
   which. The CT can. This is the one comparison that makes the project worth doing, and it
   only needs CT for patients who had both studies in the same encounter.
2. *Was the fracture missed?* The clinically meaningful error is a fracture not seen on the
   initial film and found later on CT. To count those we need to know a CT was done and
   what it showed. For this the **CT report** is enough; we do not need the images.

So the honest data request is: all trauma spine radiographs in the log; CT *reports* for the
same encounters; and CT *images* only for the paired subset used for the level-naming
check. Everything else about the model can be built without touching DMC CT at all.

# What we would pull from the protocol

From the trauma log and PACS, for patients already captured under the Trauma QI protocol:

- Thoracolumbar and lumbar spine radiographs (AP and lateral), with the dictated report.
- The CT report for any spine CT in the same encounter.
- CT images for the subset of encounters with both a radiograph and a spine CT.
- Age, sex, mechanism, and whether the injury was operative. No names, MRNs, or dates
  leave the DMC environment; the study ID is generated at pull.

A retrospective, de-identified pull. Volume to be set with the lab once we know the log's
date range.

# What we would do with it

**Stage 1 — build the model on public data.** This happens first and needs nothing from DMC.
Fracture detection and outlining train on FracAtlas (4,083 radiographs with segmentation
masks)^6^ and GRAZPEDWRI-DX (20,000 wrist films with boxes); level naming trains on
projections rendered from CTSpinoPelvic1K plus VinDr-RibCXR for rib numbering; the
spinopelvic measurements are already benchmarked on the CT. By the time we ask for DMC data
we have a working model with published-benchmark numbers.

**Stage 2 — refine on DMC radiographs.** Fine-tune on the de-identified trauma films so the
model has seen our scanners, our protocols, and our patients.

**Stage 3 — evaluate against CT.** On a held-out set never used for training, report:

- Sensitivity for fractures that were missed on the initial film and found on CT. That is
  the number that matters clinically, not just an AUC.
- Level agreement with CT, reported separately for normal and transitional spines. Averaging
  the two hides the failure in the 4–30% where it actually happens.
- Displacement call against the report.
- Spinopelvic parameters against manual measurement, compared with the published error bands
  (pooled mean absolute error for pelvic incidence is about 4°).^7^

# Where the data lives

All DMC imaging is de-identified before it leaves the DMC environment and is then stored and
processed only on Wayne State research computing — the WSU high-performance cluster, where
our CT pipeline already runs. Nothing goes to a personal machine, a personal cloud account,
commercial cloud compute, or any third-party AI service. The models are open-weight and
trained locally, so no patient image ever leaves institutional storage. Access to the study
directory is limited to named study personnel. Any linkage to the record stays inside DMC
under the protocol.

What gets published is model weights and aggregate numbers, not images. If a figure needs an
example radiograph we use a public-dataset case.

# What we are asking of the lab

- A query of the trauma log for encounters with spine radiographs and the corresponding CT
  reports, and an imaging pull for those encounters.
- Confirmation from the IRB manager that this analysis sits within the protocol's scope as
  written, or what would need to be added if it does not.
- A point of contact for questions about the log's fields.

Radiology reads for label adjudication are already covered by two radiology residents who
work with us on the CT dataset. There is no reading burden on the trauma service.

# References

1. GLEAMER BoneView: FDA 510(k) clearance, March 2022; deployment and reader-study figures as
   reported by the manufacturer. https://www.gleamer.ai
2. Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings,
   and clinical relevance. *AJNR Am J Neuroradiol.* 2010;31(10):1778–1786.
   doi:10.3174/ajnr.A2036
3. Lian J, Levine N, Cho W. A review of lumbosacral transitional vertebrae and associated
   vertebral numeration. *Eur Spine J.* 2018;27(5):995–1004. doi:10.1007/s00586-018-5554-8
4. Epstein NE. A perspective on wrong level, wrong side, and wrong site spine surgery.
   *Surg Neurol Int.* 2021;12:286. doi:10.25259/SNI_402_2021
5. Schwing G, et al. CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame,
   annotated for lumbosacral transitional anatomy. Zenodo. doi:10.5281/zenodo.22139642
   (v7 at doi:10.5281/zenodo.22242745). Dataset descriptor in preparation.
6. Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: a dataset for fracture
   classification, localization and segmentation of musculoskeletal radiographs.
   *Sci Data.* 2023;10:521. doi:10.1038/s41597-023-02432-4
7. Glaser D, AlMekkawi AK, Caruso JP, et al. Deep learning for automated spinopelvic
   parameter measurement from radiographs: a meta-analysis. *Artif Intell Surg.*
   2025;5:1–15. doi:10.20517/ais.2024.36
