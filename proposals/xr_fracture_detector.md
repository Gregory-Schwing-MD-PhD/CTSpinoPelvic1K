---
title: "Multiview Radiographic Fracture Detection with Correct Anatomic Naming"
subtitle: "Research proposal — Detroit Medical Center / Wayne State University"
author: "Gregory Schwing, MD, PhD — Department of Surgery"
date: "September 2026"
---

# Summary

Commercial AI already tells a radiologist *that* a bone is broken, and does it well enough to
be running in hundreds of hospitals. What it does not do is say **which** bone, in a way that
survives anatomic variation, or **outline** the fracture, or characterize it.

This proposal builds a multiview model that produces a structured finding rather than a
flag — detection, an outline of the fracture, the named bone or level with side, and whether
the fragment is displaced — across the structures a trauma series covers: vertebrae, sacrum,
pelvis and hips, ribs, and proximal femur. It is trained on public data, refined on DMC
imaging, and is **research only**: no output enters the medical record and no clinician sees
it during care.

# What the system would output

For every radiograph, per finding:

| Output | Form | Why it is not already available |
|---|---|---|
| Fracture present | probability | Solved commercially; included as the baseline |
| Fracture outline | pixel mask | Cleared products localize with a box, not a boundary |
| Bone / level name | label + side | The failure mode this project exists to fix |
| Displacement | none / minimal / displaced | Not offered by cleared products; drives management |
| Spinopelvic parameters | PI, PT, SS, LL, PI–LL | Available separately, not alongside the fracture read |
| Anatomic anomaly | L6, lumbar rib, transitional | Not offered anywhere |

# Prior art, and what hospitals are actually running

**Detection is a commercial product.** GLEAMER's BoneView received FDA 510(k) clearance in
March 2022 for detection and localization of fractures across the appendicular skeleton, rib
cage, and thoracic and lumbar spine. It is deployed in more than 300 institutions across 13
countries, is used by over 3,500 radiologists and emergency physicians, and is distributed
through Aidoc, Fujifilm, Ferrum Health and Blackford. Reported effects are a 10.4-point gain
in fracture-detection sensitivity, a 29% reduction in false negatives, and a 15% reduction in
reading time.^1^ Competing detection work reports external-validation AUC of 0.90 (95% CI
0.84–0.95) for vertebral compression fracture on radiographs.^2^

**Spinopelvic measurement is also solved.** A 2025 meta-analysis of 15 studies and more than
10,000 radiographs reports pooled mean absolute error of 4.1° for pelvic incidence, ICC above
0.81 against human measurement.^3^

**Three things remain open, and they are the proposal.**

*Naming.* A detector reporting a fracture at "L1" is counting, and the count fails where it
matters. Lumbosacral transitional vertebrae occur in 4–30% of people depending on
definition,^4^ the accepted standard for numeration is whole-spine imaging counted caudally
from C2^5^ — which a trauma film cannot provide — and wrong-level spine surgery runs at
roughly one in 3,110 procedures with transitional anatomy the usual cause.^6^ The ambiguity
is specific: four rib-free lumbar vertebrae may mean an L1 bearing a lumbar rib or an L5
assimilated to the sacrum; six may mean a true sixth lumbar vertebra or a T12 with aplastic
ribs. The observation is identical within each pair. The same counting problem applies to
ribs, and side is routinely mislabeled — quality control on our own CT corpus caught four
records carrying a left hip on the patient's right.

*Outlining.* Cleared products draw a box. A boundary is what supports measurement —
fragment size, angulation, displacement distance.

*Characterization.* Displacement is what changes management, and no cleared product reports
it.

**What we already hold.** CTSpinoPelvic1K is an openly licensed release of 802 CT records
placing spine, sacrum, hips, femora and per-level ribs on one coordinate frame, with explicit
classes for a sixth lumbar vertebra, a thirteenth thoracic vertebra, a separately carved
first sacral segment, and lumbar ribs.^7^ It was built to test whether local morphology alone
can name a structure without the global count, and it ships per-level morphometry and
validated spinopelvic parameters.

# Public data available for pretraining

No DMC data is needed to build the first working model. The following are openly available
and cover the three outputs above:

| Dataset | Size | What it supplies |
|---|---|---|
| FracAtlas^8^ | 4,083 radiographs | Fracture **segmentation masks** — the outline task |
| GRAZPEDWRI-DX | 20,327 images, 6,091 patients | 67,771 annotated objects: fractures, implants, boxes and polygons |
| VinDr-SpineXR | 10,469 spine radiographs | Lesion boxes, 13 categories — spine-specific |
| VinDr-RibCXR | chest radiographs | Individual **rib segmentation and labeling** |
| MURA | 40,561 radiographs | Large-scale normal/abnormal pretraining (no localization) |
| RSNA Cervical Spine Fracture | CT | Fracture supervision in 3D |
| CTSpinoPelvic1K^7^ | 802 CT | Correct level, rib number and side under variant anatomy |

The gap none of them fill is the pairing of a radiograph with a correct anatomic name under
transitional anatomy — which is why the CT corpus matters: digitally reconstructed
radiographs generated from it carry labels that are already correct, supervision unobtainable
from radiographs alone, because on a radiograph the correct name is precisely what is
unknown.

# What needs to be done

**Stage 1 — Public-data model (no DMC data, no approvals needed).** Train detection and
outline heads on FracAtlas and GRAZPEDWRI-DX; the naming head on DRRs from CTSpinoPelvic1K
plus VinDr-RibCXR; the parameter head on the existing CT-derived measurements. Deliverable: a
model with published-benchmark performance, built entirely on public data.

**Stage 2 — DMC refinement (research use only).** A retrospective, de-identified cohort of
DMC trauma radiographs with paired CT from the same encounter. CT settles both fracture and
name and becomes the reference standard. Fine-tune for local scanner, protocol and
population.

**Stage 3 — Validation and evaluation.** Held-out DMC cases never used in training, reported
in the terms decisions are made in: sensitivity for injuries missed on the initial film and
found on subsequent CT; naming accuracy against CT, **stratified into ordinary and
transitional anatomy**, since aggregate accuracy hides failure in exactly the 4–30% where the
question is hard; outline agreement (Dice) against CT-derived fracture extent; displacement
against the dictated report; spinopelvic parameters against the published error bands.^3^

*Scope note: the CT corpus reaches the axial skeleton, hips and proximal femur. Long bones
below the femur would need a separate imaging source and are a later extension.*

# Regulatory and institutional pathway

**This is research, not clinical use, and the distinction is deliberate.** All analysis is
retrospective on completed encounters. No output is placed in the medical record, returned to
a treating clinician, or used in any care decision. A model used this way is not a medical
device; prospective use at the point of care would be, and is explicitly out of scope.

**IRB — this study requires approval, and that is not the pathway our existing work used.**
The CTSpinoPelvic1K dataset work received a Not Human Participant Research determination
because it is built entirely from publicly released, already de-identified imaging. **This
proposal is categorically different: it draws imaging and reports from DMC patients' own
records, which is human subjects research and requires IRB review and approval.** That the
data is de-identified before analysis does not change it, because assembling the cohort
requires reaching identifiers in the first place.

A full protocol will therefore be submitted to the WSU IRB (irb.wayne.edu, 313-577-1628).
Retrospective review of records collected for clinical purposes is commonly eligible for
expedited review, and studies of this design commonly request a waiver of informed consent
and a HIPAA waiver of authorization — but the review category and the waivers are the IRB's
determination to make, and are named here as what will be requested rather than what applies.
De-identification and DICOM header scrubbing occur under the approved protocol, before any
imaging is used for model development.

**AI platform authorization — the step most people miss.** The determination letter issued
for our existing dataset work carries a condition that will apply here too and is easy to
overlook: *"for the use of AI platforms in research,
please consult with C&IT regarding authorization to use external software."* The route is the
Academic Research Technology team at **services@wayne.edu**. This is worth resolving early,
and it argues for training on **WSU-managed compute** rather than commercial cloud or external
model APIs — doing so keeps the data inside the institution and avoids the external-software
authorization question entirely.

# What is being asked

- **Access to a retrospective DMC trauma imaging cohort** — paired radiographs and CT from
  the same encounter — under an approved IRB protocol.
- **Clinical framing and endpoint selection:** which outputs would change a decision, and at
  what operating point a missed injury is worse than a false alarm.
- **Co-investigator standing**, and guidance on where such a tool could eventually fit if the
  research phase justified a clinical evaluation.

Radiological adjudication is covered by two radiology collaborators supporting the existing
dataset work; no reading burden falls on the trauma service.

*[Cohort size, date range, scope, effort and authorship to be discussed.]*

# References

1. GLEAMER BoneView: FDA 510(k) clearance, March 2022; deployment and reader-study figures as
   reported by the manufacturer and trade press. <https://www.gleamer.ai>
2. Zhang H, Xu R, Guo X, et al. Deep learning-based automated high-accuracy location and
   identification of fresh vertebral compression fractures from spinal radiographs: a
   multicenter cohort study. *Front Bioeng Biotechnol.* 2024;12:1397003.
   doi:10.3389/fbioe.2024.1397003
3. Glaser D, AlMekkawi AK, Caruso JP, et al. Deep learning for automated spinopelvic
   parameter measurement from radiographs: a meta-analysis. *Artif Intell Surg.* 2025;5:1–15.
   doi:10.20517/ais.2024.36
4. Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings,
   and clinical relevance. *AJNR Am J Neuroradiol.* 2010;31(10):1778–1786.
   doi:10.3174/ajnr.A2036
5. Lian J, Levine N, Cho W. A review of lumbosacral transitional vertebrae and associated
   vertebral numeration. *Eur Spine J.* 2018;27(5):995–1004. doi:10.1007/s00586-018-5554-8
6. Epstein NE. A perspective on wrong level, wrong side, and wrong site spine surgery.
   *Surg Neurol Int.* 2021;12:286. doi:10.25259/SNI_402_2021
7. Schwing G, et al. CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame,
   annotated for lumbosacral transitional anatomy. Zenodo. doi:10.5281/zenodo.22139642
   (concept; v7 at doi:10.5281/zenodo.22242745). Dataset descriptor in preparation.
8. Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: a dataset for fracture
   classification, localization and segmentation of musculoskeletal radiographs.
   *Sci Data.* 2023;10:521. doi:10.1038/s41597-023-02432-4
