---
license: cc-by-nc-4.0
pretty_name: CTSpinoPelvic1K
task_categories:
  - image-segmentation
size_categories:
  - 1K<n<10K
tags:
  - medical-imaging
  - computed-tomography
  - spine
  - pelvis
  - lumbosacral-transitional-vertebra
  - segmentation
---

# CTSpinoPelvic1K

A CT-native benchmark for lumbosacral transitional vertebra (LSTV) segmentation,
fusing CTSpine1K and CTPelvic1K over the shared COLONOG patient cohort into a
unified 10-class spinopelvic labelmap (0 background; 1-5 L1-L5; 6 L6/lumbarized
S1; 7 sacrum; 8 left hip; 9 right hip).

## Branches

- **main** — original release: source annotations only, so each case is
  spine-only, pelvic-only, or fused (partial annotation via nnU-Net ignore-label).
- **v2** — model-completed: every case densely labelled by an out-of-fold,
  5-fold nnU-Net (the unified 10-class spinopelvic map, no ignore-label voxels).
  Recommended release for the LSTV / spinopelvic benchmark.
- **v3** — bone-augmented: v2 plus a single TotalSegmentator pass per case adding
  GT-vertebra-matched ribs, both femurs, and an S1 body carved from the sacrum
  (see *v3 label scheme* below). The v2 spinopelvic labels are untouched — bone
  lands only on background and S1 merely subdivides the existing sacrum.

## v3 label scheme (bone-augmented)

v3 keeps the v2 spinopelvic classes (0–9) and appends bone structures from one
TotalSegmentator inference per case:

| id | class | source |
|---|---|---|
| 0 | background | |
| 1–6 | L1–L6 | radiologist GT (v2) |
| 7 | sacrum | radiologist GT (v2) |
| 8 / 9 | left hip / right hip | radiologist GT (v2) |
| 10–21 | rib_left_1 … rib_left_12 | TS, numbered from the GT thoracic vertebrae |
| 22–33 | rib_right_1 … rib_right_12 | TS, numbered from the GT thoracic vertebrae |
| 34 / 35 | femur_left / femur_right | TS |
| 36 | S1 (carved from the sacrum) | (GT sacrum) ∩ (TS vertebrae_S1) |
| 37 | ignore | sentinel |

A rib is emitted **only** where a GT thoracic vertebra backs it, and its number
comes from that radiologist vertebra (not from TS) — so nothing rests on TS's
vertebra numbering. Femurs are added directly. S1 is the part of the GT sacrum that
TS identifies as the S1 body, so the sacrum's outer boundary stays radiologist GT
and only its internal S1 split comes from TS. Bone labels are written on background
and never overwrite a v2 voxel; the S1 carve relabels sacrum voxels in place without
changing the sacrum's extent.

**Bone coverage is heterogeneous by design.** The TotalSegmentator pass runs on the
802 released per-patient representatives (342 fused + 440 spine-only + the 20 pure
pelvic-only orphans); the 351 separate-mode pelvic acquisitions carry the v2 labels
only (their patient's spine acquisition is the bone-labelled representative).

## Code and trained models

All code for this benchmark is public and reachable from here:

- **Dataset construction, TotalSegmentator benchmark, and QC** — patient-anchored
  fusion, bone-HU series placement, affine / Y-axis correction, the unified
  10-class label scheme, the zero-shot TotalSegmentator scoring
  (`benchmark_totalseg.py`), and the topology/structure QC:
  https://github.com/anonymous-mlhc/CTSpinoPelvic1K

- **nnU-Net training / inference pipeline** — ignore-label partial-annotation
  protocol, 5-fold stratified cross-validation, and the fused-only / partial-label
  ablations:
  https://github.com/anonymous-mlhc/spinopelvic-seg

- **Trained 5-fold checkpoints** (HuggingFace model repository):
  https://huggingface.co/anonymous-mlhc/spinopelvic-seg-checkpoints

## Data accounting

Counts in the manuscript are reported at three different granularities — masks,
placed cases, and exported volumes — which is the source of the apparent
inconsistencies. The table below gives every level from the released manifest,
the single source of truth.

| Level | Definition | Count |
|---|---|---|
| Cohort patients | COLONOG patients searched | 825 |
| TCIA series | candidate scans searched (intra-patient) | 3,451 |
| Placed masks | source masks assigned to a series | 1,498 (784 spine + 714 pelvic) |
| Alignment failures | masks with no valid series | 0 |
| Released patients | patients with at least one placed mask | 802 |
| CT volumes (cases) | released image + label pairs | 1,153 |

Which region a radiologist originally traced, for the 1,153 volumes (the rest is
model-completed in v2/v3, which carry **no** ignore-label voxels):

| Annotated region | Volumes |
|---|---|
| Fused (spine + pelvis) | 342 |
| Spine-only | 440 |
| Pelvic-only | 371 |
| Partial total (ignore-label in `main`) | 811 |

### Reconciliation with the manuscript

- **Volumes = 1,153.** The draft abstract (1,137) and Table 3 (1,163) used earlier
  placement runs and are superseded by this released count.
- **1,498 is masks, not volumes.** A fused case carries both a spine and a pelvic
  mask, so 1,498 placed masks (784 spine + 714 pelvic) reduce to 1,153 volumes.
- **LSTV total = 53** (28 sacralization, 22 lumbarization, 3 semi-sacralization);
  the "54" in Table 3 was a draft count.
- **Figure 4** used a separate-mode decomposition (fused / separate /
  spine-only / pelvic-only); the annotated-region counts above are the released
  breakdown.
- **Separate-series cases** (a patient whose spine and pelvic masks were placed
  on different acquisitions) are not fused; they enter as separate spine-only and
  pelvic-only volumes and are counted in the 811 partial total.

## Mask-to-scan assignment

The released masks carry no SeriesInstanceUID, so each must be matched to one of
its patient's TCIA series. The table below is the **metadata-only** confidence,
*before* bone-HU - from weak proxies alone (filename SeriesNumber, NIfTI affine
geometry, slice count), only 5 of 1,498 masks match with certainty and 107 are
ambiguous:

| Confidence (metadata only, pre bone-HU) | Masks |
|---|---|
| Certain | 5 |
| High | 675 |
| Medium | 706 |
| Low | 5 |
| Ambiguous | 107 |
| **Total placed** | **1,498** |

Bone-HU maximization then places **every** mask on its best-matching scan
(0 unresolved) - so the final dataset has no ambiguous placements; the table only
shows why a metadata lookup is insufficient and the fusion pipeline is required.

## LSTV phenotypes

Counts are per volume - one labeled CT scan (image + mask), 1,153 total. These 53
LSTV volumes come from 33 distinct patients: a separate-mode patient (whose spine
and pelvic masks lie on different acquisitions) appears as two scans, a spine-only
and a pelvic-only, both carrying the phenotype. Each is Castellvi-classified below.

| Phenotype | Volumes |
|---|---|
| Normal | 1,100 |
| Sacralization | 28 |
| Lumbarization | 22 |
| Semi-sacralization | 3 |
| **Total LSTV** | **53** |

LSTV labels retain both the CTPelvic1K morphological qualifier and the CTSpine1K
vertebral-count heuristic, with a flag marking cases where the two disagree.

### Radiologist Castellvi classification

Every LSTV case carries an independent radiologist Castellvi read
(`lstv_phenotypes.csv`): transitional type, left/right laterality, non-rib-bearing
vertebra count, and per-case notes, including the ambiguous
cross-source-disagreement cases. The two counts differ only by granularity: the
53 LSTV labels above are per volume, whereas the radiologist read is per patient
(33 LSTV patients); a separate-mode patient is two scans (spine-only and
pelvic-only), so the volume count is higher. All LSTV patients are classified.

### Lumbarization candidates surfaced by completion

In the model-completed (v2) labels, **34 volumes carry a full six-vertebra lumbar
column (L1-L6)**, versus 16 in the original partial source labels. **9 of these 34
are not flagged LSTV by either source criterion** (e.g. pelvic-only cases the
source could not assess for spine anatomy) - additional lumbarization candidates
for review beyond the 53 source-flagged phenotypes, raising the LSTV candidate
pool to 62 volumes (53 source-flagged + 9 completion-surfaced). By contrast, the
four-lumbar (sacralization-count) total is unchanged (9 in both trees), so
completion surfaces lumbarization candidates but no new sacralization-count
candidates.

## Quality control (model-completed vs radiologist masks)

Ground-truth-free checks on the model-completed (v2) labels, with the
radiologist-annotated tree as the standard.

| Check | Completed (v2) | Radiologist baseline |
|---|---|---|
| Vertebra class mixing — stranded-voxel fraction (mean) | 0.0006 | 0.0003 |
| Structure — left/right hip swap | 0% | 0% |
| Structure — vertebra gaps / incomplete pelvis | 0 | 0 |
| Structure — duplicated structures | subset of baseline | reference |

Completion introduces no structural defect beyond the radiologist masks; the only
residual is marginal vertebra class mixing, a known CNN failure mode. The v3 bone
pass adds only background structures (and the in-place S1 split), so it does not
change these spinopelvic QC numbers.

## Splits

Five-fold splits (`splits_5fold.json`), stratified by LSTV phenotype and fusion
status.

| Split | Volumes | Patients |
|---|---|---|
| Train | 805 | 630 |
| Validation | 174 | 166 |
| Test | 174 | 168 |
| **Total** | **1,153** | **802** |

Per-class metrics are reported on the fused cases, which carry complete ground
truth.
