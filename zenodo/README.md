# CTSpinoPelvic1K

Spine, pelvis, per-level ribs and femurs in one coordinate frame, for 802 abdominal CT
records — built so that a lumbar vertebra can be identified when the count itself is in
doubt.

Vertebral numbering is conventionally established by counting down from C2. No abdominal CT
contains C2, so at the thoracolumbar junction a thirteenth thoracic vertebra, a rib borne by
a lumbar vertebra, and a stump rib all produce overlapping appearances, and which label is
correct depends on a count the field of view does not support. This release is annotated to
make that decidable where it can be, and to say so plainly where it cannot.

---

## What is in this deposit, and what is not

**Included** — about 1.8 GB:

| | |
|---|---|
| `labels/` | 802 gzipped NIfTI label volumes, `NNNN_label.nii.gz` |
| `manifest.json` | one record per case: TCIA series identifiers, demographics, scanner, LSTV label, Castellvi grade |
| `splits_5fold.json` | frozen patient-grouped, LSTV-stratified five-fold cross-validation splits |
| `dataset_labels.json` | the label scheme — identifier to structure name |
| `fetch_from_tcia.py` | rebuilds the image half end to end: download, convert, resample |
| `reconstruct_ct.py` | the resampling step alone, for an existing conversion |

**Not included** — the CT images. They are 193 GB against 1.8 GB of labels, they are already
public in [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), and
re-hosting them would duplicate a primary archive for no benefit.

What the source collections never published — and what this deposit does — is **the mapping
from each annotation to the CT series it belongs on.** `manifest.json` carries a
`spine_series_uid` and, where one exists, a `pelvic_series_uid` for every record: TCIA
SeriesInstanceUIDs. Every one of the 802 records has at least one.

---

## Rebuilding the images

**One step is not optional and your masks will not line up without it.**

The released labels do not sit on the grid of the DICOM series as you will download it. Each
label was drawn on one grid, and at export the CT was resampled onto *that* grid so image and
label share an affine. Downloading the series and converting it to NIfTI gives you a volume
with the right anatomy on the wrong grid.

Everything needed to fix that is already in the label file: **the label's own affine is the
target.**

**All three steps, automated.** `fetch_from_tcia.py` reproduces what the release did,
in the same order and with the same settings: download the named series with
`tcia_utils`, convert with `dcm2niix`, resample onto the label grid trilinearly with
-1024 HU outside the original extent.

```bash
pip install tcia_utils nibabel scipy numpy       # and dcm2niix on PATH
python fetch_from_tcia.py --manifest manifest.json --labels labels --out ct
python fetch_from_tcia.py ... --cases 0007 0033  # or just a few
```

It downloads a *named series*, never a patient: every patient here was scanned twice,
prone and supine, and only one of those volumes is the one the annotation was drawn
on. Per record it reports what fraction of the label lands on bone and warns below
50% -- which indicates the wrong series rather than a resampling error.

Because the label's affine is the target, the released PIR orientation comes along
with it. There is nothing separate to rotate.

**If you already have your own conversion**, `reconstruct_ct.py` does the resampling
step alone:

```bash
python reconstruct_ct.py --label labels/0007_label.nii.gz \
    --ct my_conversion/0007.nii.gz --out ct/0007_ct.nii.gz --check
```

`--check` reports the fraction of labelled voxels sitting at bone attenuation. A low number
means you have the **wrong series**, not a resampling problem — every patient in this cohort
was scanned twice, prone and supine, and no amount of interpolation fixes the wrong one.

---

## The label scheme

VerSe-native: vertebrae keep their VerSe identifiers, and every non-VerSe structure takes a
fixed identifier above that range.

| ids | structures |
|---|---|
| 1–7 | C1–C7 |
| 8–19 | T1–T12 |
| 20–25 | **L1–L6** |
| 26 | sacrum |
| 29 | **S1**, carved from the sacrum |
| 30–31 | hips (left, right) |
| 32–33 | femurs (left, right) |
| 34–45 | ribs, left 1–12 |
| 46–57 | ribs, right 1–12 |
| 58–73 | soft tissue — *declared, empty in this release* |
| 74–75 | **lumbar rib** (left, right) |
| 76–79 | hardware — *declared, empty in this release* |
| 255 | ignore |

Two classes exist that no other public CT collection carries, and they are the reason this
scheme is not simply another whole-body label map:

- **L6.** A scheme without one cannot record a six-lumbar spine at all, and must renumber
  the column or drop a level to fit. 17 records here carry an L6.
- **Lumbar rib (74/75).** A scheme that numbers every rib 1–12 has nowhere to put a
  thirteenth: the annotator must either call it rib 12 — which asserts the vertebra beneath
  it is thoracic, the very question at issue — or discard it. 15 records carry one.

---

## The transitional layer

33 records carry a radiologist Castellvi grade in `manifest.json`
(`castellvi_type`, with `castellvi_second_read` and `castellvi_agreement` where a second
read exists).

| grade | records |
|---|---|
| Ib | 2 |
| IIa | 2 |
| IIb | 4 |
| IIIa | 3 |
| IIIb | 18 |
| IV | 4 |

**The grade and the vertebral count are different axes**, and this release exists partly to
show how far apart they run: grade IIIb occurs here at rib-free counts of four, five *and*
six, and seven of the 33 graded records carry a perfectly normal count of five. A corpus
recording only the count would describe those seven as unremarkable.

The grades are **one reader's**, with five cases independently read a second time (three
agreed; of the two that did not, one crosses the boundary between bony fusion and
articulation without it). Confirmation by a board-certified neuroradiologist, blinded to the
existing grades and the counts, is in progress. Treat the current grades accordingly.

---

## Known limitations

Stated at the level of detail needed to catch them independently.

- **Thoracic coverage is field-of-view limited** and does not extend to T1. Two records
  (0068, 1106) carry no thoracic vertebra at all.
- **Structures are not universally present.** One record has no sacrum, hips or femurs; two
  have no S1; nine lack an L5 identifier — in each case because the structure is outside the
  field of view, or because the lowest lumbar segment is labelled L6 or incorporated into
  the sacrum. Filter explicitly rather than assuming.
- **The splits are cross-validation folds covering the whole cohort. There is no held-out
  test set.** Carve one and state which records it holds.
- **Label strength varies by structure and the release does not average over it.** Vertebral
  labels derive from radiologist-supervised source annotations, corrected where those sources
  were wrong. Pelvic labels on records that lacked one are pseudolabelled. The rib layer is a
  pseudolabel whose human review was triaged by an automated rule rather than exhaustive.
- **The rib layer has a specific blind spot.** Ribs come from a binary rib network unioned
  with TotalSegmentator, keeping only voxels connected to a numbered rib. A stump rib that
  the binary network segments but TotalSegmentator never *numbers* has nothing to attach to
  and is dropped — and quality control cannot see it, because an absent rib flags nothing.
- **Soft-tissue (58–73) and hardware (76–79) identifiers are declared and populated by no
  record.** Their absence is absence of annotation, not absence of the structure.
- **Postural angles are supine.** Pelvic incidence is a morphological property and needs no
  such caveat; sacral slope and pelvic tilt do.
- **The cohort is a colorectal screening population aged 50 and over.** Its distributions
  should not be read as representative of a surgical one.

---

## Sources and licence

Imaging from TCIA CT COLONOGRAPHY. Vertebral annotations derive from CTSpine1K and pelvic
annotations from CTPelvic1K, both produced under board-certified radiologist supervision;
neither published the mapping from an annotation to the CT series it was drawn on, which is
the gap this release closes.

Released under **CC BY-NC-SA 4.0**. The underlying TCIA imaging carries its own terms; cite
the source collections alongside this one.

**Research use only.** These labels are not a medical device and are not validated for
clinical decision-making.
