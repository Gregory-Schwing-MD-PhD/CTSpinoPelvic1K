i---
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

A fused spine + pelvis 3D CT segmentation dataset built by patient-level
crosswalk between three public sources:

1. **TCIA CT COLONOGRAPHY** — DICOM CT volumes (prone + supine per patient)
2. **CTSpine1K (COLONOG subset)** — VerSe-convention vertebral label masks
3. **CTPelvic1K dataset2** — sacrum + bilateral hip label masks

Annotations are placed onto the TCIA CT volume with the highest bone
coverage (HU > 200), separately per anatomy. For ~340 patients both
annotations land on the same series (**fused** cases); for the rest,
spine and pelvic labels target different prone/supine acquisitions
(**separate** cases, exported as two records per patient — one
`spine_only`, one `pelvic_native`).

## Labels (10-class)

| ID | Name        | Source                                         |
|----|-------------|------------------------------------------------|
| 0  | background  | —                                              |
| 1  | L1          | CTSpine1K (VerSe label 20 → 1)                 |
| 2  | L2          | CTSpine1K (VerSe label 21 → 2)                 |
| 3  | L3          | CTSpine1K (VerSe label 22 → 3)                 |
| 4  | L4          | CTSpine1K (VerSe label 23 → 4)                 |
| 5  | L5          | CTSpine1K (VerSe label 24 → 5)                 |
| 6  | L6 / LSTV   | CTSpine1K (VerSe label 25 → 6) — lumbarized S1 |
| 7  | sacrum      | CTPelvic1K (dataset2 label 1 → 7)              |
| 8  | left hip    | CTPelvic1K (dataset2 label 2 → 8)              |
| 9  | right hip   | CTPelvic1K (dataset2 label 3 → 9)              |

CTPelvic1K's sacrum takes priority over CTSpine1K's sacrum (VerSe label 26)
to avoid the two labelling conventions colliding on lumbosacral transitional
vertebrae.

## Orientation

All volumes are canonicalised to **PIR** (Posterior–Inferior–Right). The CT
and its label map share exactly the same 4×4 affine; no resampling is needed
before training. PHI fields (`descrip`, `aux_file`, `db_name`, `intent_name`)
are stripped from every NIfTI header.

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

Use `match_type` to distinguish patient-level provenance:

| `match_type`   | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `fused`        | Spine + pelvic masks land on the same TCIA series              |
| `separate`     | Spine and pelvic masks target different series for this patient; patient appears twice (one `spine_only` + one `pelvic_native` record) |
| `spine_only`   | Patient has only a spine mask available                        |
| `pelvic_only`  | Patient has only a pelvic mask available                       |

## Splits

Stratified patient-level 5-fold cross-validation with a held-out test set.
Stratification is on `match_type × has_lstv` so every fold carries the rare
sacralization and lumbarization classes.

- **`splits_5fold.json`** — unified splits document (schema v3). Carries the
  test holdout and all five fold assignments in one file, with split
  invariants validated at generation time (patient-level disjointness,
  fold coverage, no overlap between test and trainval).
- **`splits/test.json`** — flat list of unique test-set patient tokens
  (legacy path; still shipped for backwards compatibility).
- **`data_splits.json`** — earliest format, mapping `ct_file` entries to
  `train` / `val` / `test`. Consumed only as a last-resort fallback.
- **`splits_summary.json`** — aggregate per-stratum / per-fold counts.

## File format

Each case is a pair of gzipped NIfTI files under `ct/` and `labels/`:

```
ct/<token:04d>_<position>_ct.nii.gz         # HU, float32 or int16
labels/<token:04d>_<position>_label.nii.gz  # int16, values 0..9
```

`position` is `supine` or `prone` (the TCIA acquisition position). For
separate-case patients, a `_spine` or `_pelvic` suffix disambiguates the
two records.

Per-case metadata lives in `manifest.json` (flat JSON list), with one record
per NIfTI pair. Key fields:

```
token, config, match_type, position, ct_file, label_file, qc_file,
lstv_label, lstv_class, lstv_pelvic, lstv_vertebral,
lstv_agreement, lstv_confusion_zone, has_l6,
spine_series_uid, pelvic_series_uid,
spine_bone_pct,   pelvic_bone_pct,
n_lumbar_labels, alignment_ok
```

`ct_file` and `label_file` are relative paths with the subdirectory prefix
included (e.g. `"ct/0017_supine_ct.nii.gz"`), so `dataset_root / ct_file`
resolves directly.

## Quickstart — NIfTI + nibabel

```python
import json, nibabel as nib
from pathlib import Path

root = Path("anonymous-mlhc/CTSpinoPelvic1K")  # or local export dir
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
    root  = "anonymous-mlhc/CTSpinoPelvic1K",
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

ds = CTSpinoPelvicDataset(root="anonymous-mlhc/CTSpinoPelvic1K",
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
