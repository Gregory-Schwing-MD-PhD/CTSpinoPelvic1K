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

In **8 records the femoral head itself is a prosthesis** — ten replaced hips, since two of
the eight are bilateral. Pelvic incidence and pelvic tilt are taken from the midpoint of the
two femoral head centres, so in all eight that midpoint is derived from metal rather than
bone, whether one head was replaced or both. Usable if you know it, misleading if you do not.
The eight are `0188`, `0443`, `0485`, `0515`, `0671`, `0974`, `1003`, `1128`; `hardware_label_ids`
contains 80 for each.

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

## 4. Detached label pieces are mostly specks, and a few are real

Counted on the labels in this deposit, with **no minimum size**: **27,738 detached pieces
across 741 of the 802 records**. That number is large because it includes single voxels.
The size distribution is the finding, not the total:

| piece size | n | share | what it is |
|---|---|---|---|
| under 10 voxels | 24,777 | 89.3% | a speck at the boundary between two labels |
| 10–99 | 1,920 | 6.9% | boundary roughness |
| 100–999 | 523 | 1.9% | a fragment |
| 1000 or more | 518 | 1.9% | a real detached piece of bone |

**The last row is the one to act on: 518 pieces of 1000+ voxels, spread over 359 records.**
Everything above it is the segmentation disagreeing with itself by a voxel or two along a
seam, which no analysis of shape, volume or position will notice.

The counting rule, so the number is reproducible: `scipy.ndimage.label` with the default
face-adjacent connectivity, per label id, per volume, counting every component after the
largest. 22,126 label instances are a single clean component; 52,285 components exist in
total.

The commonest labels among detached pieces are `right_hip` (9,574) and `left_hip` (5,282),
which is the pubic symphysis and the sacroiliac joints: two bones that meet along a seam the
segmenter has to cut, so both sides carry specks of the other. Then the ribs — `rib_left_12`
(1,604), `rib_right_6` (1,114), `rib_left_7` (1,003) — where a thin cortical shell breaks
across a slice.

**801 pieces (3%) touch the edge of the reconstructed volume.** For those the label is
correct and the anatomy is simply cut. That share is low here only because specks dominate
the total; among the large pieces, truncation is the usual explanation, and section 5 gives
the extent of it.

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

**0068 — the fusion is surgical; whether it is *only* surgical is not answerable.**
Two threaded cylindrical interbody cages sit in the L5–L6 interspace, 1,277 mm³ and
1,309 mm³, and each one independently abuts both L5 and L6. So the bridge is instrumented
and there is no ambiguity about that. What cannot be recovered from a post-operative image
is whether the interspace was normal before the operation, or whether this patient had a
transitional vertebra that the surgery then fused. `lstv_vertebral` is left unread for this
case rather than guessed. Case 0068 also has six lumbar bodies, numbered against the twelfth
rib rather than by counting down, which is the situation this dataset exists to document.

**The 41 rejected detections — the size separation is real, and one gap in it is narrow.**
52 metal detections survived the 2500 HU floor and a radiologist read every one. What
separated the 11 kept from the 41 rejected was not the attenuation, which saturates at the
scanner's 3071 HU ceiling for implant and dense artefact alike:

| | kept (11) | rejected (41) |
|---|---|---|
| median volume | 74,466 mm³ | 68 mm³ |
| volume range | 2,586 – 134,966 mm³ | 14 – 1,768 mm³ |
| components | median 2 | median 1, up to 18 |
| an identifiable surgical site | 11 of 11 | 6 of 41 |
| peak above the 3071 HU ceiling | 0 of 11 | 4 of 41 |

Three orders of magnitude separate the medians. The gap that is *not* wide is at the
boundary: the smallest implant kept is 0068's cage pair at 2,586 mm³ and the largest thing
rejected is 0317 at 1,768 mm³. Anything in that band is a judgement, and it was made by
reading the image rather than by thresholding the table.

**0878 is the clearest of the rejections and worth stating why**, because its volume alone
would not have rejected it. It reaches 11,798 HU. No implant can: 3071 is the ceiling of the
reconstruction, so a value above it is the reconstruction overshooting, not a denser metal.
It is also 18 disconnected specks rather than a body. Recorded as artefact.

**1035 — this entry previously warned of fragmentation that the v6 labels do not have.**
The left hip and right hip both carried the *left* label over part of their extent, which
made the left hip look like it was in two large pieces. That was a laterality error, not
fragmentation, and it was corrected for v6. As shipped: left hip 99.88% one component with a
single 473 mm³ crumb, right hip one component, sacrum one component, S1 one component. The
sacroiliac screws are two components because there are two screws.

## 8. Hip laterality was wrong in 22 records and is corrected in v6

**If you used v5 for anything measured from the hip or the femoral head, re-run it.**

Four records — `0027`, `0107`, `0790`, `0935` — had `left_hip` and `right_hip` swapped
outright. Eighteen more — `0012`, `0065`, `0135`, `0146`, `0172`, `0186`, `0376`, `0410`,
`0471`, `0513`, `0746`, `0830`, `0917`, `0938`, `0957`, `1124`, `1145`, `1148` — had most of
one hip bone wearing the other hip's label, the same fault previously recorded for 1035.

The evidence, since a laterality claim should carry it:

- **The femurs disagreed with the hips.** Each femur sits in its own hip's socket, and every
  femur pair in the release is correctly sided. In all four transposed records the femurs are
  right and each hip lies beside the *other* femur. No orientation error can do that — a
  prone/supine flip moves both pairs together.
- **Position does not explain it.** The flag rate is 2.39% among prone acquisitions and
  2.84% among supine. If patient position drove it, one would carry the flags and the other
  none.
- **Controls separate cleanly.** Measured as the fraction of a hip label lying across the
  spine midline — the lumbar and sacral centroid, which does not move when a hip label is
  wrong — unflagged records sit at 0.0% (max 4.0%) and the eighteen sat at 31.8% median, up
  to 49.7%.
- **The correction improved the labels.** Hip components fell from 4,315 to 410 and pieces
  under 100 voxels from 4,198 to 347 across those eighteen; each hip is now essentially a
  single component instead of about two-thirds of one, and the two hips come out
  near-symmetric in volume, as a pelvis is.

Laterality is re-derived per voxel from the side of the spine midline, computed through the
affine, so it does not depend on the stored orientation.

**Why this reached v5:** the release check that detects it takes a `--sidedness` argument
that defaults to 0, meaning skipped. It had never been run across all 802 records. It is not
optional in the build that produced this deposit.

---

## 9. v6 against the published v5

v6 is **not** simply v5 plus hardware. Five records — `0179`, `0378`, `0412`, `0787`,
`1153` — were hand-corrected after the v5 export was cut and never re-exported, so those
corrections appear for the first time in v6 (1,025,631 voxels relabelled, chiefly a rib
renumbering on 0179).

Expect v6 to differ from published v5 in: those five records, the eleven hardware cases,
0068's renumbering, and the twenty-two hip corrections in section 8.
