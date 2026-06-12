---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
tags:
- medical
- ct
- lumbar-spine
- pelvis
- colonography
- lstv
- sacralization
- lumbarization
size_categories:
- 1K<n<10K
---

# CTSpinoPelvic1K

A fused spine + pelvis 3D CT segmentation dataset built for one job the common
tools get wrong: **numbering the lumbar spine correctly in patients with a
lumbosacral transitional vertebra (LSTV).** Built by patient-level crosswalk
between three public sources:

1. **TCIA CT COLONOGRAPHY** — DICOM CT volumes (prone + supine per patient)
2. **CTSpine1K (COLONOG subset)** — VerSe-convention vertebral label masks
3. **CTPelvic1K dataset2** — sacrum + bilateral hip label masks

Annotations are placed onto the TCIA CT volume with the highest bone
coverage (HU > 200), separately per anatomy. For most patients both
annotations land on the same series (**fused** cases); for the rest,
spine and pelvic labels target different prone/supine acquisitions
(**separate** cases, exported as two records per patient — one
`spine_only`, one `pelvic_native`).

## Design principles (the decisions that make it LSTV-aware)

1. **Vertebrae are radiologist ground truth, full stop** — we never ship a
   pseudolabelled spine. The L1–L6 / L6 / sacrum calls and the transitional
   adjudications come from CTSpine1K's radiologists.
2. **A built-in counting anchor** — the **last rib-bearing vertebra** (T12 =
   class 31), the vertebra directly above ground-truth L1. It is **not a separate
   class**, just the last thoracic vertebra in the native column. You can't tell
   L5 from L6 locally; with a fixed cranial anchor + the sacrum, the lumbar count
   (hence L5-vs-L6) is deterministic, and the model learns rib-bearing-ness from
   the thoracic-vs-lumbar distinction. The anchor is free from GT (present on
   783/784 abdominopelvic studies; earlier exports just dropped the thoracic
   column). See [docs/RIB_ANCHOR_RATIONALE.md](docs/RIB_ANCHOR_RATIONALE.md).
3. **LSTV captured from both ends, and graded** — vertebral count (CTSpine1K)
   *and* pelvic annotation (CTPelvic1K), expert Castellvi-graded (see below).
4. **Honest about the pelvis** — real where fused, pseudolabelled (leak-safe)
   for `spine_only`, with quality reported as held-out Dice on the
   `pelvic_native` scans.

Versions:
- **v1** — the **partial-annotation** artifact (all configs, `ignore` protocol),
  i.e. **the input used to train the pseudolabeller**, now carrying the T12
  anchor. `pelvic_native` keeps `ignore` on the spine (never a faked label).
- **v2** — the **clean, densely-labelled** release: CTSpine1K spine ground truth +
  pelvis (real where fused, pseudolabelled on `spine_only`), `fused + spine_only`
  only. `pelvic_native` is dropped (held out for pelvis validation). This is the
  LSTV-segmenter training artifact, with no `ignore` voxels on the shipped cases.
- **v3** *(roadmap)* — adds the student-annotated rib cage (classes 35+, reserved now).

> **Reviewing segmentations?** If you were asked to help correct the AI-drafted
> labels, see **[docs/REVIEW.md](docs/REVIEW.md)** — account/token setup,
> install, and how to connect to the distributed review system (and, for the
> project maintainer, how to stand up the HuggingFace Space backend).

## Labels

| ID | Name                  | Source                                            |
|---:|-----------------------|---------------------------------------------------|
| 0  | background            | —                                                 |
| 1  | L1                    | CTSpine1K (VerSe 20 → 1)                           |
| 2  | L2                    | CTSpine1K (VerSe 21 → 2)                           |
| 3  | L3                    | CTSpine1K (VerSe 22 → 3)                           |
| 4  | L4                    | CTSpine1K (VerSe 23 → 4)                           |
| 5  | L5                    | CTSpine1K (VerSe 24 → 5)                           |
| 6  | L6 / LSTV             | CTSpine1K (VerSe 25 → 6) — lumbarized S1          |
| 7  | sacrum                | CTPelvic1K (dataset2 1 → 7); CTSpine1K VerSe 26 fallback |
| 8  | left hip              | CTPelvic1K (dataset2 2 → 8)                        |
| 9  | right hip             | CTPelvic1K (dataset2 3 → 9)                        |
| 10 | **ignore**            | partial-annotation only — un-traced region, NOT bg |
| 13–19 | **C1–C7**          | CTSpine1K (VerSe 1–7 → 13–19), native contiguous   |
| 20–32 | **T1–T13**         | CTSpine1K (VerSe 8–19, 28 → 20–32); **T12 = 31** is the counting anchor |
| 35–60 | **rib cage** *(reserved)* | rib_left/right 1–13 — reserved, not yet populated |

CTPelvic1K's sacrum takes priority over CTSpine1K's sacrum (VerSe label 26)
to avoid the two labelling conventions colliding on lumbosacral transitional
vertebrae. The vertebral column is native and contiguous (C1–C7 = 13–19,
T1–T13 = 20–32); the counting anchor is the last thoracic vertebra (T12 = 31),
not a stored class. The rib-cage IDs (35+) are **reserved and empty** so v3 is
purely additive.

## LSTV — captured from both ends, expert-graded

Transitional status is read **independently** from the vertebral count
(`lstv_vertebral`, CTSpine1K) and the pelvic/sacral annotation (`lstv_pelvic`,
CTPelvic1K filename qualifier), with an `lstv_agreement` flag. The vertebral
calls match **CTSpine1K's published cohort** case-for-case (16 lumbarizations +
9 sacralizations on COLONOG); **8 further sacralization/semi-sacralization cases
come from the CTPelvic1K pelvic annotations**. All **33 transitional cases carry
an expert Castellvi grade** — **25 sourced from CTSpine1K (vertebral) + 8 from
CTPelvic1K (pelvic)**. The full per-case source table is in the dataset card
([docs/dataset_card.md](docs/dataset_card.md)); the grading sheet is
`_lstv_phenotypes.csv`.

## Orientation

All volumes are canonicalised to **PIR** (Posterior–Inferior–Right). The CT
and its label map share exactly the same 4×4 affine; no resampling is needed
before training. PHI fields (`descrip`, `aux_file`, `db_name`, `intent_name`)
are stripped from every NIfTI header.

A post-write HU sanity check is run at export time on each saved CT/label
pair: HU values sampled at hip-mask voxels must be ≥30% bone (HU > 200),
otherwise the case is flagged in the manifest as
`postwrite_hip_bone_pct`. This catches frame-mismatches between CT data
and its affine that earlier passed the simple `affine_equal` check.

## LSTV annotation

Each case carries two complementary LSTV (lumbosacral transitional vertebra)
annotations:

- **`lstv_vertebral`** — derived from CTSpine1K by counting lumbar labels in
  the segmentation (4 → sacralization, 5 → normal, 6 → lumbarization).
- **`lstv_pelvic`** — derived from CTPelvic1K filename qualifiers (any
  substring containing "sacralization" → sacralization).
- **`lstv_agreement`** — `True` when both sources agree, `False` when they
  disagree, `None` when either side is uninformative.
- **`lstv_confusion_zone`** — `True` for cases at the sacralization ↔
  lumbarization boundary where the two signals disagree; flagged for
  downstream audit.
- **`lstv_class`** — integer 0–3 summarising the dominant call (0=normal,
  1=lumbarization, 2=semi-sacralization, 3=sacralization). Pelvic label
  takes priority when both sides disagree.

## Configs

A single patient can contribute multiple records depending on how spine and
pelvic masks align across their prone/supine acquisitions:

| `config`        | Meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `fused`         | Both masks placed on the same CT series; labels 1–9 present   |
| `spine_only`    | Record carries lumbar labels only (1–6); sacrum/hips absent   |
| `pelvic_native` | Record carries sacrum + hip labels only (7–9); lumbar absent  |

Use `match_type` to distinguish where each side's labels came from:

| `match_type`   | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `fused`        | Spine + pelvic masks land on the same TCIA series              |
| `separate`     | Spine and pelvic masks target different series for this patient; patient appears twice (one `spine_only` + one `pelvic_native` record) |
| `spine_only`   | Patient has only a spine mask available                        |
| `pelvic_only`  | Patient has only a pelvic mask available                       |

## Splits

Stratified patient-level 5-fold cross-validation with a held-out test set.
Stratification preserves the rare LSTV classes across every fold so that
no fold is missing lumbarization or sacralization cases at validation time.

The current splits document (`splits_5fold.json`, schema **v4**) uses
LSTV-first stratum ordering: each patient is binned by
`<lstv_subtype>|<match_type>`, and rare buckets are coalesced by dropping
the `match_type` qualifier first (preserving the LSTV signal). An invariant
check at generation time enforces that every fold's validation split
contains at least 3 lumbarization cases (configurable). This was tightened
from schema v3, which sometimes dropped the LSTV tag during coalescing
and produced folds with zero L6 vertebrae in their validation sets.

The export ships several views of the same splits for backward
compatibility:

- **`splits_5fold.json`** — unified splits document (schema v4). Carries the
  test holdout and all five fold assignments in one file, with split
  invariants validated at generation time (patient-level disjointness,
  fold coverage, no overlap between test and trainval, ≥3 lumbarization
  per fold val).
- **`splits/test.json`** — flat list of unique test-set patient tokens
  (legacy path; still shipped for backwards compatibility).
- **`data_splits.json`** — earliest format, mapping `ct_file` entries to
  `train` / `val` / `test`. Consumed only as a last-resort fallback.
- **`splits_summary.json`** — aggregate per-stratum / per-fold counts.
- **`manifest_train.json`, `manifest_validation.json`, `manifest_test.json`** —
  per-record splits derived from the above.

## File format

Each case is a pair of gzipped NIfTI files under `ct/` and `labels/`. The
filename schema (revised Apr 2026) is fully self-describing:

```
fused                      ct/<token:04d>_ct.nii.gz
                           labels/<token:04d>_label.nii.gz

spine-side (separate or
spine_only single-mask)    ct/<token:04d>_spine_ct.nii.gz
                           labels/<token:04d>_spine_label.nii.gz

pelvic-side (separate or
pelvic_only single-mask)   ct/<token:04d>_pelvic_ct.nii.gz
                           labels/<token:04d>_pelvic_label.nii.gz
```

A bare `<token>_ct.nii.gz` therefore unambiguously means a `fused` case
(both regions present in one mask). The `_spine` / `_pelvic` suffix is
applied uniformly to spine-side and pelvic-side files regardless of
whether the source case is `match_type="separate"` (paired but
non-coregistered) or a `spine_only` / `pelvic_only` single-mask case.
Earlier the position (`supine` / `prone`) appeared in every filename;
that was misleading because the prone/supine classifier rarely succeeded,
and `config` is what every downstream consumer actually filters on.
`position` still rides through to the manifest as a metadata column —
it is no longer in the filename.

Per-case metadata lives in `manifest.json` (flat JSON list), with one record
per NIfTI pair. Key fields:

```
token, config, match_type, position, ct_file, label_file, qc_file,
lstv_label, lstv_class, lstv_pelvic, lstv_vertebral,
lstv_agreement, lstv_confusion_zone, has_l6, n_lumbar_labels,
spine_series_uid, pelvic_series_uid,
spine_bone_pct, pelvic_bone_pct,
alignment_ok, ct_resampled_to_mask, postwrite_hip_bone_pct
```

`ct_file` and `label_file` are relative paths with the subdirectory prefix
included (e.g. `"ct/0017_ct.nii.gz"`), so `dataset_root / ct_file` resolves
directly. `manifest.csv` contains the same content for downstream tooling
that prefers tabular ingest.

The two CT-vs-mask diagnostics are worth flagging:

- **`ct_resampled_to_mask`** is `True` when the source CT had to be
  resampled into the placed mask's grid because shape and/or affine
  differed. Most cases hit this; the few that don't are cases where the
  raw CT and placed mask happened to share both shape and affine.
- **`postwrite_hip_bone_pct`** is the percent of voxels under the saved
  label's hip mask that have HU > 200 in the saved CT, computed at export
  time. Values below ~30% indicate a CT/label frame mismatch and are
  logged as warnings during export.

## Quickstart — NIfTI + nibabel

```python
import json, nibabel as nib
from pathlib import Path

root = Path("anonymous-neurips-ED/CTSpinoPelvic1K")  # or local export dir
meta = json.loads((root / "manifest.json").read_text())

rec = meta[0]
ct  = nib.load(str(root / rec["ct_file"]))
lbl = nib.load(str(root / rec["label_file"]))

assert ct.shape == lbl.shape
assert (ct.affine == lbl.affine).all()

print(rec["token"], rec["config"], rec["match_type"], rec["lstv_class"])
```

## Quickstart — dataset_interface (PyTorch)

The repo ships a `dataset_interface.py` with a `CTSpinoPelvic1K` class
(directory-backed, no torch dependency — used by the benchmark and
visualisation scripts) and a `CTSpinoPelvicDataset` PyTorch adapter:

```python
from dataset_interface import CTSpinoPelvicDataset
from torch.utils.data import DataLoader

ds = CTSpinoPelvicDataset(
    root  = "anonymous-neurips-ED/CTSpinoPelvic1K",
    split = ("fold", 0, "train"),
)
dl = DataLoader(ds, batch_size=1, shuffle=True)

for batch in dl:
    ct, label = batch["ct"], batch["label"]   # (B,1,Z,Y,X) / (B,Z,Y,X)
    ...
```

Other supported `split` values: `"trainval"` (whole trainval pool),
`"test"` (fixed test holdout), `("fold", i, "val")` for fold `i`'s
validation side.

## Quickstart — MONAI

```python
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd,
)
from dataset_interface import CTSpinoPelvicDataset

transforms = Compose([
    NormalizeIntensityd(keys="ct", subtrahend=0, divisor=1000),
    RandCropByPosNegLabeld(keys=("ct","label"), label_key="label",
                           spatial_size=(96,96,96), pos=2, neg=1, num_samples=2),
    RandFlipd(keys=("ct","label"), prob=0.5, spatial_axis=(0,1,2)),
])

ds = CTSpinoPelvicDataset(root="anonymous-neurips-ED/CTSpinoPelvic1K",
                          split=("fold", 0, "train"), transform=transforms)
```

## Citation

Please cite all three source datasets alongside this derivative release:

```bibtex
@misc{smith2015ctcolonography,
  author       = {Smith, K. and Clark, K. and Bennett, W. and Nolan, T. and
                  Kirby, J. and Wolfsberger, M. and Moulton, J. and
                  Vendt, B. and Freymann, J.},
  title        = {Data From CT COLONOGRAPHY},
  year         = {2015},
  publisher    = {The Cancer Imaging Archive},
  doi          = {10.7937/K9/TCIA.2015.NWTESAY1},
  howpublished = {\url{https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1}},
}

@article{deng2021ctspine1k,
  title   = {{CTSpine1K}: A Large-Scale Dataset for Spinal Vertebrae
             Segmentation in Computed Tomography},
  author  = {Deng, Yang and Wang, Ce and Hui, Yuan and Li, Qian and
             Li, Jun and Luo, Shiwei and Sun, Mengke and Quan, Quan and
             Yang, Shuxin and Hao, You and Liu, Pengbo and Xiao, Honghu
             and Zhao, Chunpeng and Wu, Xinbao and Zhou, S. Kevin},
  journal = {Machine Learning for Biomedical Imaging},
  volume  = {3},
  number  = {MICCAI Open Data 2024-2025},
  pages   = {824--832},
  month   = {5},
  year    = {2021},
  doi     = {10.59275/j.melba.2025-gf84},
  url     = {https://arxiv.org/pdf/2105.14711},
}

@article{liu2021ctpelvic1k,
  title   = {Deep Learning to Segment Pelvic Bones: Large-Scale {CT}
             Datasets and Baseline Models},
  author  = {Liu, Pengbo and Han, Hu and Du, Yuanqi and Zhu, Heqin and
             Li, Yinhao and Gu, Feng and Xiao, Honghu and Li, Jun and
             Zhao, Chunpeng and Xiao, Li and Wu, Xinbao and Zhou, S. Kevin},
  journal = {International Journal of Computer Assisted Radiology
             and Surgery},
  volume  = {16},
  pages   = {749--756},
  year    = {2021},
}
```

BibTeX entries are also provided in `CITATION.cff`.

## License

- Source datasets — CT COLONOGRAPHY (TCIA), CTSpine1K, CTPelvic1K — retain
  their respective licenses.
- Derivative fused labels, splits, and code: **CC BY-NC 4.0**
  (non-commercial).
