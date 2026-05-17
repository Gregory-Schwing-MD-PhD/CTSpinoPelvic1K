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

A fused spine + pelvis 3D CT segmentation dataset built by patient-level
crosswalk between three public sources:

1. **TCIA CT COLONOGRAPHY** — DICOM CT volumes (prone + supine per patient)
2. **CTSpine1K (COLONOG subset)** — VerSe-convention vertebral label masks
3. **CTPelvic1K dataset2** — sacrum + bilateral hip label masks

Annotations are placed onto the TCIA CT volume with the highest bone
coverage (HU > 200), separately per anatomy.  For ~650 patients both
annotations land on the same series (**fused** cases); for the rest, spine
and pelvic labels target different prone/supine acquisitions (**separate**
cases).

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

CTPelvic1K's sacrum takes priority over CTSpine1K's sacrum (label 26) to avoid
the two labelling conventions colliding in cases of lumbosacral transitional
vertebrae.

## Orientation

All volumes are canonicalised to **PIR** (Posterior-Inferior-Right).  The CT
and its label map share exactly the same 4×4 affine; no resampling is needed
before training.

## LSTV annotation

Each case carries two complementary LSTV (lumbosacral transitional vertebra)
annotations:

- **`lstv_vertebral`** — derived from CTSpine1K by counting lumbar labels in the
  segmentation (4 → sacralization, 5 → normal, 6 → lumbarization).
- **`lstv_pelvic`** — derived from CTPelvic1K filename qualifiers (any substring
  containing "sacralization" → sacralization).
- **`lstv_agreement`** — `True` when both sources agree, `False` when they
  disagree, `None` when either side is uninformative.
- **`lstv_class`** — integer 0–3 summarising the dominant call (0=normal,
  1=lumbarization, 2=semi-sacralization, 3=sacralization).  Pelvic label
  takes priority.

## Splits

70 / 15 / 15 train / val / test, stratified by `(lstv_class × match_type)` so
each split contains the rare sacralization and lumbarization classes.

## File format

Each case is a single `.npz` file under `data/<split>/token_<N>.npz`:

```python
import numpy as np, json

d = np.load("token_17.npz", allow_pickle=False)
ct     = d["ct"]       # int16  (Z, Y, X)   HU
label  = d["label"]    # uint8  (Z, Y, X)   0..9
affine = d["affine"]   # float32 (4, 4)    RAS affine
meta   = json.loads(str(d["meta"]))

print(meta["match_type"], meta["lstv_class"], meta["spine_bone_pct"])
```

## Quickstart — PyTorch

```python
from dataset_interface import CTSpinoPelvicDataset
from torch.utils.data import DataLoader

ds  = CTSpinoPelvicDataset(
    root      = "anonymous-neurips-ED/CTSpinoPelvic1K",
    split     = "train",
    cache_dir = "~/.cache/ctspinopelvic1k",
)
dl  = DataLoader(ds, batch_size=1, shuffle=True)

for batch in dl:
    ct, label = batch["ct"], batch["label"]   # (B,1,Z,Y,X) / (B,Z,Y,X)
    ...
```

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
                          split="train", transform=transforms)
```

## Citation

Please cite the source datasets (CTSpine1K, CTPelvic1K, TCIA CT COLONOGRAPHY)
alongside this derivative release.  BibTeX entries are provided in `CITATION.cff`.

## License

- Source datasets — CT COLONOGRAPHY (TCIA), CTSpine1K, CTPelvic1K — retain
  their respective licenses.
- Derivative fused labels, splits, and code: **CC BY-NC 4.0** (non-commercial).
