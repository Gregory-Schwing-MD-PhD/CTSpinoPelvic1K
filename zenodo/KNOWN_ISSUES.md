# Known issues — CTSpinoPelvic1K v6

What a user will run into, measured rather than estimated. Nothing here is a reason not to
use the dataset; all of it is a reason to filter before a particular analysis.

---

## 1. `null` in a Castellvi field means UNGRADED, not "no transitional vertebra"

**33 of 802** records carry a radiologist Castellvi grade: IIIb ×18, IV ×4, IIb ×4, IIIa ×3,
IIa ×2, Ib ×2. The other **769 are ungraded**, and that is not a negative finding.

These are colonography scans read for polyps. A transitional vertebra is easy to pass over
when it is not what you are looking for, so an absent grade records that nobody looked, not
that nobody found anything. Treating the 769 as negatives would manufacture 769 controls
that no one established.

`castellvi_agreement` is blank wherever there is no second read, so a single read is never
mistaken for a consensus.

---

## 2. Instrumented cases must be excluded from gap-based measurements

**11 records carry surgical hardware** (ids 76–82; see the README). This matters for one
specific reason: **an iatrogenic fusion is indistinguishable from a congenital one to a
distance measurement.** A cage-bridged interspace reads as "no gap" exactly as a
congenitally fused transitional vertebra does.

Filter on `hardware_labelled` in `manifest.json` before any analysis of the gap between the
lowest lumbar vertebra and the sacrum.

In **9 cases the femoral head itself is an implant**. Pelvic incidence and pelvic tilt are
measured from the femoral head centre, so those parameters are measured on metal in those
records — usable if you know it, misleading if you do not.

---

## 3. Prone and supine must not be pooled

Every patient was scanned twice. **Position changes lumbar lordosis and segmental
alignment**, so a value from a prone series and one from a supine series are not the same
quantity, and pooling them across the cohort biases the result.

`position` is in the manifest. Use it. In **351 patients** the two annotations sit on
*different* series, which makes those records a paired within-patient design rather than a
nuisance — the same fact that forbids pooling is what makes flexion–extension mobility
measurable.

---

## 4. Detached label pieces — mostly the edge of the scan, sometimes not

**523 detached pieces across 408 records.** Two different things, and the split matters:

| | n | what it is |
|---|---|---|
| touching the edge of the scan | 331 (63%) | the structure leaves the reconstructed volume; the label is correct and the anatomy is simply cut |
| not at an edge | 192 (37%) | a genuine break, mostly a vertebra separated at the pedicle |

Commonest labels: sacrum ×92, T9 ×83, T8 ×79, T7 ×51, T10 ×47. The thoracic concentration
is expected — these are FOV-limited abdominal scans and the upper thoracic levels are at the
margin of the reconstruction.

Across all 21,474 label components in the deposit: 20,162 are a single clean component, 704
carry loose voxels, 518 sit inside the imaged volume, 62 are near-edge uncertain and 28 lie
on the reconstruction circle.

---

## 5. The thoracic column is FOV-limited, and the top vertebra is usually cut

**553 of 802 records (69%)** have their topmost labelled vertebra cut by the edge of the
field of view, with a median labelled height of 13.6 mm and a range down to 0.8 mm.

Truncated top levels are labelled, deliberately and consistently. If your analysis needs
whole vertebrae, check the extent rather than assuming a labelled level is complete.

---

## 6. Rib numbering comes from a count that cannot always be made

Ribs are numbered by TotalSegmentator, which counts down from the top of what it can see. On
an abdominal scan the top of the field is not the top of the thorax, so **rib numbers are an
inference wherever the upper thorax is out of view** — the same failure this dataset exists
to document for vertebrae.

The *lowest* rib is reliable: the last rib is the last rib whether or not the first eleven
are in frame. Numbers above it are less so. Lumbar ribs have their own classes (74, 75)
rather than being forced to be rib 12.

---

## 7. Open questions, recorded rather than resolved

- **0068**: the L5–L6 interspace is fused by paired interbody cages. Whether that fusion is
  surgical only, or surgical *on top of* a pre-existing transitional vertebra, is not
  determinable from the images. `lstv_vertebral` is unread for this case.
- **0878** and the other rejected hardware proposals: 41 of 52 metal detections were read as
  artefact — contrast, calcification and reconstruction overshoot. 0878 is the closest call,
  at 1,711 mm³ touching a hip. Recorded as artefact.
- **1035**: the sacroiliac screws cross the joint. The hip labels were lateralised for v6;
  the fragments that remain are anatomically real.

---

## 8. v6 against the published v5

v6 is **not** simply v5 plus hardware. Five records — `0179`, `0378`, `0412`, `0787`,
`1153` — were hand-corrected after the v5 export was cut and never re-exported, so those
corrections appear for the first time in v6 (1,025,631 voxels relabelled, chiefly a rib
renumbering on 0179). The remaining 797 records are unchanged apart from the eleven hardware
cases and 0068's renumbering.

If you are comparing against published v5, expect those five to differ for reasons unrelated
to this release.
