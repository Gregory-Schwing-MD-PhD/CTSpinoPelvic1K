# Hardware review — 52 proposals, 11 real

A metal-threshold pass over the 802 produced 52 candidate hardware annotations. A
radiologist read them and said almost all are artefact. This folder is that split, the rule
that reproduces it, and the eleven cases that are real.

## What is here

| path | what |
|---|---|
| `hardware_manifest.csv` / `.json` | every case, verdict, class, and why |
| `instrumentation/` | renders for the 11 confirmed cases |
| `artefact/` | renders for the 41 rejected |
| `verify/<case>/` | CT + label + descriptor, ready to open in ITK-SNAP |
| `review_index.csv` | the earlier, pre-split ordering (superseded) |

Open a case with:

    ITK-SNAP -g <case>_ct.nii.gz -s <case>_label_hw.nii.gz -l itksnap_v6_labels.txt

## The eleven

| case | class | metal | taken from |
|---|---|---|---|
| 0974 | arthroplasty | 213,587 vox | both femurs, both hips — bilateral |
| 0515 | arthroplasty | 248,500 vox | both femurs, both hips — bilateral |
| 1003 | arthroplasty | 271,122 vox | femur_right |
| 0443 | arthroplasty | 200,684 vox | femur_left |
| 0671 | arthroplasty | 191,369 vox | femur_left |
| 0188 | arthroplasty | 182,906 vox | femur_right |
| 0485 | arthroplasty | 116,137 vox | femur_left |
| 1128 | arthroplasty | 111,872 vox | femur_left |
| 0247 | arthroplasty | 23,204 vox | femur_left — **an order of magnitude smaller than the rest; worth a look** |
| 1035 | SI screw | 8,881 vox | sacrum, S1, both hips |
| 0068 | cage | 5,296 vox | L5, L6 — verified separately in ITK-SNAP |

## How the split was made

Two conditions, both required:

**A real surgical site.** The site comes from the labelled structures the metal actually
touches — a hip joint, the sacroiliac joint, the spine. Most rejected proposals touch
nothing labelled at all, or only "sacrum", which is where rectal contrast and iliac
calcification sit in a colonography series.

**Volume above 2,000 mm³.** Every confirmed implant is 2,586 mm³ or more; every rejected
proposal is 1,768 mm³ or less. The threshold sits in that gap, not in the middle of a cloud.

### Why saturation alone did not work

It was the obvious test and it fails. The scanner clips at 3071 HU, so a real implant has a
saturated core — but nearly every rejected proposal reaches the ceiling too, and several go
far past it: **11,798 HU on 0878, 9,534 on 0763, 7,438 on 0027**. Values above the ceiling
are the signature of streak artefact and reconstruction overshoot around something dense,
not of a denser implant. A test built on "does it saturate" keeps all 52.

## Classes added

The 76–79 block was written for spinal instrumentation and cannot name what is actually in
this cohort:

    80  hardware_arthroplasty   hip or knee prosthesis: stem, head, acetabular cup
    81  hardware_si_screw       iliosacral fixation, crossing the SI joint

Both would otherwise be named **wrongly** rather than left unnamed — a femoral stem is long
and thin, so the screw-or-rod rule claims it. Adding a class costs one line; a wrong subtype
has to be found and undone in every case that used it.

## What this changes beyond the label

**Nine femur labels were substantially prosthesis.** On 1003, 271,006 voxels came out of
`femur_right`. That matters past bookkeeping: pelvic incidence and pelvic tilt are measured
from the femoral head centre, and on these cases the femoral head is an implant. Any
spinopelvic parameter computed from them was measured on metal.

**The instrumented count of 84 in the manifest is wrong.** It comes from an 1800 HU scan,
which is below anything published; the metal-segmentation literature validates 2500 and
3000. 32 of the 84 have no metal near bone at a literature threshold, and of the 52 that
did, 41 are artefact. The real count is **11**.

## Still open

- 0247 is 23,204 voxels against 111,000–271,000 for the other eight arthroplasties. Either a
  partial or resurfacing implant, or a different object. Needs a look.
- Whether the L5–L6 fusion on 0068 is surgical only, or surgical on top of a pre-existing
  transitional vertebra. No measurement separates those.
