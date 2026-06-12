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
- castellvi
- counting-anchor
size_categories:
- 1K<n<10K
---

# CTSpinoPelvic1K

A fused **spine + pelvis** 3D CT segmentation dataset built for one job the
common tools get wrong: **numbering the lumbar spine correctly in patients with
a lumbosacral transitional vertebra (LSTV).** It pairs the radiologist vertebral
ground truth of **CTSpine1K** with the pelvis (from **CTPelvic1K** where
available, pseudolabelled otherwise) on **TCIA CT COLONOGRAPHY** volumes, and
adds an explicit **counting anchor** — the last rib-bearing vertebra — that makes
L1–L6 assignment deterministic across spine variants.

---

## Why this dataset exists (the design principles)

Automated spine segmenters mislabel vertebrae in the 5–35% of people with an
LSTV: they count up from the sacrum, assume five lumbar bodies, and shift every
label by one when there are six (or four). In surgical planning that is
indistinguishable from a wrong-level plan. Four decisions shape this dataset:

1. **Vertebrae are radiologist ground truth, full stop.** We **never ship a
   pseudolabelled spine.** Every vertebral label comes from CTSpine1K's
   radiologists, including their adjudication of the transitional cases. This is
   the single most important property: the L1–L6 / L6 / sacrum calls are not a
   model's guess.
2. **A built-in counting anchor.** Numbering is a *non-local counting* problem —
   you cannot tell L5 from L6 by looking at one vertebra. We retain the **last
   rib-bearing vertebra (T12)** as an explicit class (`last_rib_vertebra`, 11):
   the vertebra directly above ground-truth L1. With a fixed cranial anchor in
   the field of view and the sacrum as the caudal anchor, the lumbar count — and
   therefore L5-vs-L6 — becomes deterministic. This anchor is **free from the
   ground truth**: CTSpine1K already segments T12 on 783/784 abdominopelvic
   studies; earlier exports simply discarded it.
3. **LSTV is captured from both ends, and graded.** Transitional status is read
   *independently* from the vertebral count (CTSpine1K) **and** the pelvic/sacral
   annotation (CTPelvic1K), then expert **Castellvi**-graded (see below).
4. **Honest pelvis provenance.** The pelvis is real radiologist GT where it was
   co-annotated (**fused** cases) and a leak-safe model **pseudolabel** otherwise
   (**spine_only** cases); its quality is reported as held-out Dice against real
   CTPelvic1K GT (the held-out `pelvic_native` scans).

---

## Sources

1. **TCIA CT COLONOGRAPHY** — DICOM CT volumes (prone + supine per patient)
2. **CTSpine1K (COLONOG subset)** — VerSe-convention vertebral label masks (the
   spine ground truth)
3. **CTPelvic1K dataset2** — sacrum + bilateral hip label masks (the pelvis
   ground truth)

Annotations are placed onto the TCIA CT volume with the highest bone coverage
(HU > 200), separately per anatomy. For ~650 patients both annotations land on
the same series (**fused** cases); for the rest, the spine label targets one
acquisition (**spine_only**) and the pelvis is pseudolabelled.

---

## Labels

Annotators / consumers work in this scheme (the merged training scheme is a
documented reduction, provided as a conversion script):

| ID | Name                | Source                                              |
|---:|---------------------|-----------------------------------------------------|
| 0  | background          | —                                                   |
| 1  | L1                  | CTSpine1K (VerSe 20 → 1)                             |
| 2  | L2                  | CTSpine1K (VerSe 21 → 2)                             |
| 3  | L3                  | CTSpine1K (VerSe 22 → 3)                             |
| 4  | L4                  | CTSpine1K (VerSe 23 → 4)                             |
| 5  | L5                  | CTSpine1K (VerSe 24 → 5)                             |
| 6  | L6 / LSTV           | CTSpine1K (VerSe 25 → 6) — lumbarized S1            |
| 7  | sacrum              | CTPelvic1K (dataset2 1 → 7); CTSpine1K VerSe 26 fallback |
| 8  | left hip            | CTPelvic1K (dataset2 2 → 8)                          |
| 9  | right hip           | CTPelvic1K (dataset2 3 → 9)                          |
| 10 | **ignore**          | partial-annotation only — un-traced region, NOT bg  |
| 11 | **last_rib_vertebra** | CTSpine1K (VerSe **19 = T12** → 11) — the counting anchor |
| 12 | **rib** *(reserved)*  | v3 student annotation of the anchor's rib — not yet populated |

CTPelvic1K's sacrum takes priority over CTSpine1K's sacrum (VerSe 26) so the two
labelling conventions don't collide at the lumbosacral junction.

**Class 12 (`rib`) is reserved now and left empty** so a future v3 (student-
annotated rib of the anchor vertebra) is purely additive — no renumbering.

---

## The counting anchor (class 11) — the point of the dataset

You cannot number the spine from local appearance; you must **count from a fixed
landmark**. The **last rib-bearing vertebra** is that landmark: the vertebra
directly below it is L1, and you count down to the sacrum. It is labelled
**relationally** ("the vertebra above ground-truth L1"), never as an absolute
thoracic number (T12 vs T13 is unknowable in a lumbosacral field of view and is
not claimed). With the anchor at the top and the sacrum at the bottom, **5
intervening bodies → L5, 6 → L6** — the LSTV question answers itself, in a single
forward pass, on exactly the patients where the stakes are highest. See
`docs/RIB_ANCHOR_RATIONALE.md`.

---

## Versions

- **v1.1** — the full release + the T12 anchor class (all configs). A superset of
  v1; the only change is that class 11 is no longer discarded.
- **v2** — the LSTV-segmenter training artifact: **fused + spine_only** only
  (every case has a radiologist spine). `pelvic_native` (real pelvis, *pseudo*
  spine) is **excluded from the ship** and held back as the **pelvis-pseudolabel
  validation set**.
- **v3** *(roadmap)* — adds the student-annotated **rib** (class 12) of the anchor
  vertebra. The scheme is already reserved for it.

---

## LSTV — status, provenance, and per-case sources

Each case carries two **complementary, independent** LSTV annotations:

- **`lstv_vertebral`** — from CTSpine1K, by counting lumbar labels (4 →
  sacralization, 5 → normal, 6 → lumbarization). This is the **vertebral**
  ground truth.
- **`lstv_pelvic`** — from CTPelvic1K sacral-mask filename qualifiers (a
  "sacralization"/"semi" substring set by the pelvic annotators). This is the
  **pelvic** ground truth.
- **`lstv_agreement`** — `True`/`False`/`None` when the two sources agree /
  disagree / one is uninformative.
- **`lstv_class`** — 0 normal · 1 lumbarization · 2 semi-sacralization ·
  3 sacralization.

The vertebral calls were cross-validated against **CTSpine1K's published
transitional cohort** for COLONOG (16 lumbarizations + 9 sacralizations) and
match case-for-case. **8 further sacralization / semi-sacralization cases come
from the CTPelvic1K pelvic annotations.** All **33 transitional cases carry an
expert Castellvi grade**, sourced as follows:

| token | category            | Castellvi | LSTV source                     |
|------:|---------------------|:---------:|---------------------------------|
| 4     | ambiguous           | IV        | CTSpine1K (vertebral count)     |
| 6     | sacralization       | IV        | CTPelvic1K (pelvic annotation)  |
| 15    | sacralization       | IV        | CTPelvic1K (pelvic annotation)  |
| 22    | semi_sacralization  | IIIa      | CTPelvic1K (pelvic annotation)  |
| 32    | sacralization       | IIIb      | CTPelvic1K (pelvic annotation)  |
| 64    | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 67    | ambiguous           | IIb       | CTSpine1K (vertebral count)     |
| 104   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 107   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 110   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 120   | semi_sacralization  | IIa       | CTPelvic1K (pelvic annotation)  |
| 123   | sacralization       | IIIb      | CTPelvic1K (pelvic annotation)  |
| 125   | sacralization       | IIIb      | CTPelvic1K (pelvic annotation)  |
| 140   | sacralization       | IIIa      | CTPelvic1K (pelvic annotation)  |
| 149   | lumbarization       | IIIa      | CTSpine1K (vertebral count)     |
| 167   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 175   | lumbarization       | Ib        | CTSpine1K (vertebral count)     |
| 189   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 215   | lumbarization       | Ib        | CTSpine1K (vertebral count)     |
| 261   | lumbarization       | IIb       | CTSpine1K (vertebral count)     |
| 267   | lumbarization       | IV        | CTSpine1K (vertebral count)     |
| 344   | lumbarization       | IIb       | CTSpine1K (vertebral count)     |
| 401   | lumbarization       | IIa       | CTSpine1K (vertebral count)     |
| 537   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 554   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 555   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 587   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 615   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 666   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 672   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 699   | lumbarization       | IIIb      | CTSpine1K (vertebral count)     |
| 721   | sacralization_count | IIIb      | CTSpine1K (vertebral count)     |
| 737   | lumbarization       | IIb       | CTSpine1K (vertebral count)     |

**Totals: 25 from CTSpine1K (vertebral) + 8 from CTPelvic1K (pelvic) = 33
Castellvi-graded.** The full sheet (with second reads and notes) is
`_lstv_phenotypes.csv`. `category = sacralization_count` denotes a sacralization
detected by the vertebral count; `sacralization` / `semi_sacralization` denote
the pelvic-annotation source.

---

## The one-vs-two-class boundary convention

When an L6 / last lumbar fuses with the sacrum, identity is decided by the
**L5/L6–S1 disc at the vertebral body**, not by bony continuity (a fused
transverse-process bridge still makes one connected bone but does not collapse a
class — the bamboo-spine principle):

- **Disc present** (even with a fused TP bridge, Castellvi III) → **two classes**
  (last lumbar + sacrum).
- **Disc obliterated** (complete sacralization) → **one class (sacrum)**; record
  the transition as the Castellvi grade, not as voxels.

Full conventions: `docs/LABELING_GUIDE.md`.

---

## Orientation

All volumes are canonicalised to **PIR** (Posterior-Inferior-Right). The CT and
its label map share exactly the same 4×4 affine; no resampling is needed before
training.

## Splits

70 / 15 / 15 train / val / test, **patient-grouped** (prone+supine of one patient
never split across folds) and **stratified by `(lstv_class × match_type)`** so
each split contains the rare sacralization and lumbarization classes.

## File format

Each case is a single `.npz` file under `data/<split>/token_<N>.npz`:

```python
import numpy as np, json

d = np.load("token_17.npz", allow_pickle=False)
ct     = d["ct"]       # int16  (Z, Y, X)   HU
label  = d["label"]    # uint8  (Z, Y, X)   0..9, 11 (anchor); 10=ignore; 12 reserved
affine = d["affine"]   # float32 (4, 4)     RAS affine
meta   = json.loads(str(d["meta"]))

print(meta["match_type"], meta["lstv_class"], meta["has_anchor"])
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

## Citation

Please cite the source datasets (CTSpine1K, CTPelvic1K, TCIA CT COLONOGRAPHY)
alongside this derivative release. BibTeX entries are provided in `CITATION.cff`.

## License

- Source datasets — CT COLONOGRAPHY (TCIA), CTSpine1K, CTPelvic1K — retain their
  respective licenses.
- Derivative fused labels, splits, and code: **CC BY-NC 4.0** (non-commercial).
