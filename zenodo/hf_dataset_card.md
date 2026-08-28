---
license: cc-by-nc-sa-4.0
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

> **This repository holds an earlier export. The current release is v6, archived at
> [10.5281/zenodo.22139643](https://doi.org/10.5281/zenodo.22139643).**
>
> v6 is a different dataset in every respect that matters: 802 records rather than 1153
> files, a scheme of 84 identifiers rather than 10, per-level ribs, femurs, an S1 class
> carved from the sacrum, surgical instrumentation, and a crosswalk giving the TCIA
> SeriesInstanceUID each annotation was drawn on. It also carries corrections that this
> export does not — twenty-two records had left and right hip labels swapped or
> interdigitated here, and the transitional-anatomy flags in this repository's manifest are
> unreliable (see below).
>
> Use the Zenodo record for anything new. What follows documents what is actually in *this*
> repository, because it is still downloadable and it should not mislead anyone who does.

A fused spine + pelvis 3D CT segmentation dataset built by patient-level crosswalk between
three public sources:

1. **TCIA CT COLONOGRAPHY** — DICOM CT volumes (prone + supine per patient)
2. **CTSpine1K (COLONOG subset)** — VerSe-convention vertebral label masks
3. **CTPelvic1K dataset2** — sacrum + bilateral hip label masks

Annotations are placed onto the TCIA CT volume with the highest bone coverage (HU > 200),
separately per anatomy. Where both land on the same series the record is **fused**; otherwise
spine and pelvic labels target different prone/supine acquisitions (**separate**).

## Labels

| ID | Name | Source |
|----|------|--------|
| 0 | background | *fused files only* — see the warning below |
| 1 | L1 | CTSpine1K (VerSe 20 → 1) |
| 2 | L2 | CTSpine1K (VerSe 21 → 2) |
| 3 | L3 | CTSpine1K (VerSe 22 → 3) |
| 4 | L4 | CTSpine1K (VerSe 23 → 4) |
| 5 | L5 | CTSpine1K (VerSe 24 → 5) |
| 6 | L6 | CTSpine1K (VerSe 25 → 6) — six-lumbar spines |
| 7 | sacrum | CTPelvic1K (dataset2 1 → 7) |
| 8 | left hip | CTPelvic1K (dataset2 2 → 8) |
| 9 | right hip | CTPelvic1K (dataset2 3 → 9) |
| **10** | **not annotated (ignore)** | **the region this file does not cover** |

**Identifier 10 is not anatomy and earlier versions of this card omitted it.** In a
`*_spine_label.nii.gz` the pelvis is unannotated, and in a `*_pelvic_label.nii.gz` the spine
is; identifier 10 marks that region. It is not a rare edge case — measured on these files it
covers **99.7%** of a spine-only volume and **99.0%** of a pelvic-only one, and in those files
**identifier 0 does not appear at all**, because 10 has taken the place of background.

Treat 10 as ignore, not as a class. A loss computed over identifiers 0–10 without masking it
trains the network to predict "not annotated" across almost the entire volume. Fused files
(`NNNN_label.nii.gz`, no `_spine_`/`_pelvic_` qualifier) use 0 for background and contain no
10.

CTPelvic1K's sacrum takes priority over CTSpine1K's sacrum (VerSe 26) so the two conventions
do not collide on a transitional vertebra.

## Orientation

All volumes are `('P','I','R')`. Each CT and its label share exactly the same 4×4 affine, so
no resampling or reorientation is needed to overlay them — verify with
`np.allclose(ct.affine, label.affine)` rather than assuming it.

## File format

NIfTI pairs, not archives:

```python
import nibabel as nib
import numpy as np

lab = nib.load("labels/0002_label.nii.gz")
ct = nib.load("ct/0002_ct.nii.gz")
assert np.allclose(lab.affine, ct.affine)

L = np.asanyarray(lab.dataobj)
print(sorted(int(v) for v in np.unique(L)))   # includes 10 on separate-mode files
```

`manifest.json` carries one record per file, and `splits_5fold.json` carries frozen
patient-grouped folds. Earlier revisions of this card described a 70/15/15 train/val/test
split and a `.npz` layout; neither is what this repository contains.

## Transitional-anatomy flags: do not filter on them

`has_l6`, `has_lumbar_rib` and `n_lumbar_labels` in the manifest are unreliable in this
export and were corrected only for v6. Counted in the released label volumes, `has_l6` was
true for one record that contains no L6 and false for all eighteen that do, and
`has_lumbar_rib` was false for every record including the sixteen that carry one. Anyone
selecting the six-lumbar cases on those fields receives one record that is not one, and
anyone selecting lumbar ribs receives an empty set.

Derive the flags from the label volumes, or use the v6 record, where the fields are computed
from the voxels.

## Citation

Please cite the source datasets — CTSpine1K, CTPelvic1K, and TCIA CT COLONOGRAPHY — alongside
this derivative release, and cite the dataset itself at
[10.5281/zenodo.22139643](https://doi.org/10.5281/zenodo.22139643).

## Licence

**CC BY-NC-SA 4.0.** The ShareAlike term is inherited rather than chosen: CTSpine1K, from
which the vertebral annotations derive, is released under CC BY-NC-SA, and a derivative of a
ShareAlike work must carry the same licence. Earlier versions of this card said CC BY-NC 4.0
in the body while the metadata said CC BY-NC-SA 4.0; CC BY-NC-SA 4.0 is correct.

The source datasets retain their own licences. **Research use only** — these labels are not a
medical device and are not validated for clinical decision-making.
