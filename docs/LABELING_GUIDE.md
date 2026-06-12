# CTSpinoPelvic1K — Labeling & Onboarding Guide

Read this before you annotate, correct, or train on a single case. It defines
**the classes**, **the hard problem (LSTV)**, and **the labeling conventions**
that keep the dataset consistent. The conventions are not arbitrary — they exist
because LSTV breaks naive voxel labeling, and inconsistent labels are worse than
useless for training.

---

## 1. What this project is

A spine + pelvis CT segmentation dataset and an nnU-Net model trained from it.
The clinical motivation: the default tools (e.g. TotalSegmentator) **mis-number
the lumbosacral junction in LSTV patients**, and that error propagates into
surgical planning as **wrong-level surgery**. Closing the LSTV gap is the
deliverable. Your annotations are the ground truth that makes that possible, so
their consistency matters more than their speed.

---

## 2. The classes

Annotators work in the **source scheme** (what you draw in ITK-SNAP / Slicer):

| value | name        | description                                                   |
|------:|-------------|---------------------------------------------------------------|
| 0     | background  | everything not below                                          |
| 1     | L1          | 1st lumbar vertebra (count from the top — see §3)             |
| 2     | L2          |                                                               |
| 3     | L3          |                                                               |
| 4     | L4          |                                                               |
| 5     | L5          | normal last lumbar                                            |
| 6     | L6          | **only when a true 6th lumbar is present** (lumbarization)    |
| 7     | sacrum      | the whole sacral mass (incl. a fully-incorporated vertebra — §4) |
| 8     | left_hip    | left ilium                                                    |
| 9     | right_hip   | right ilium                                                   |
| 10    | ignore      | un-annotated region in partial/separate mode (§5) — NOT bg    |

**Downstream training scheme (you don't draw this; the converter makes it):**
L5 and L6 are **merged** into a single `last_lumbar` class and everything above
the sacrum is renumbered contiguously: `0 bg, 1–4 L1–L4, 5 last_lumbar, 6 sacrum,
7 left_hip, 8 right_hip, 9 ignore`. Why we merge is §3.

---

## 3. The hard problem: LSTV

A **lumbosacral transitional vertebra (LSTV)** is a vertebra with mixed
lumbar/sacral character. It has two independent axes:

- **Count** — *lumbarization* (an extra mobile vertebra, "L6") vs *sacralization*
  (the last lumbar incorporated into the sacrum).
- **Morphology** — the transverse-process ↔ sacral-ala relationship, graded by
  **Castellvi** (I dysplastic TP ≥19 mm / II pseudoarthrosis / III bony fusion /
  IV mixed; `a` unilateral, `b` bilateral) and the **Mahato** spectrum.

Two things make this hard for a segmentation network, and each has a convention:

### 3a. Counting is non-local → we MERGE L5/L6
The voxels of L3/L4/L5 look near-identical; the only thing that says "this is the
5th vs the 6th" is *counting down from the top* — a non-local signal a CNN can't
learn locally. So we do **not** ask the network to distinguish L5 from L6: they
are merged into `last_lumbar` for training, and the L5-vs-L6 identity is recovered
**downstream** (instance post-processing + the Castellvi/Mahato grade). Follows
Möller 2026 (VERIDAH §2.2).

**As an annotator you still count and label L1–L6 explicitly** (the radiologist
*can* count; the merge happens later in the pipeline). Count from the
thoracolumbar junction (the last rib-bearing vertebra) down.

### 3b. Fused boundaries are ambiguous → §4
When a vertebra fuses to the sacrum there may be no real boundary to draw. §4 is
the rule for that.

---

## 4. The one-class-vs-two-class rule (the big one)

When an L6 / last lumbar touches or fuses with the sacrum, **the question is not
"is it one continuous bone."** Bony continuity does **not** collapse a semantic
class — by that logic you'd label half of every ankylosed ("bamboo") spine as one
bone. You label by **anatomical identity**, and the criterion is the **L5/L6–S1
intervertebral disc**, *assessed at the vertebral body, not the transverse
process*:

- **Disc present (body distinct), even with a fused TP bony bridge** (Castellvi
  III): the vertebra is still a real vertebra → **TWO classes** (last_lumbar +
  sacrum). The boundary runs at the disc level; only the small lateral TP bridge
  is "soft," and it is handled by a fixed convention (see §4a) — never agonized
  over voxel-by-voxel.
- **Disc obliterated (body fused, complete sacralization)** → **ONE class
  (sacrum).** The vertebra is genuinely incorporated; do not invent a boundary
  through remodeled bone. Record the transitional identity as the **Castellvi/
  Mahato grade (metadata)**, not as voxels — you lose no information.

This maps onto the grading: **Castellvi I–III keep two classes** (disc present);
**complete sacralization merges** (disc gone). Complete-fusion cases are rare —
expect a handful dataset-wide — so flag and adjudicate them individually rather
than re-annotating everything.

### 4a. The fused TP bridge (Castellvi III)
The only genuinely ambiguous bone is the lateral TP→ala bridge. Its boundary is
inter-rater-soft by nature, so: draw it **consistently** (extend the disc-level
body plane laterally through the bridge), accept that perfection is impossible,
and prefer **consistency over voxel-precision**. A deterministic helper
(`ctspino-syn`'s `fuse_split`) exists to draw this the same way every time for
*synthetic* and *pseudolabel* cases — it does **not** overwrite radiologist GT.

---

## 5. Annotation modes & the ignore contract

Cases come in three **configs**:

- **fused** — one CT, both spine and pelvis annotated.
- **separate** — two CTs per patient: a `spine_only` (spine annotated) and a
  `pelvic_native` (pelvis annotated). They are *not* co-registered.
- **pelvic-only / spine-only** — only one region was ever imaged/annotated.

**Partial-annotation ignore contract:** in a partial mask, the region the present
annotator did *not* trace is labeled **`ignore` (10), never background**. Labeling
it background would falsely supervise the network with "nothing here." The
trainer masks `ignore` out of the loss.

---

## 6. Provenance & versions (where a label came from)

Every record carries provenance: `manual` (radiologist), `pseudo` (model
pseudolabel), `pseudo_corrected` (reviewer-corrected pseudolabel). Dataset
versions: **v2** is the published manual+matched release; **v3** is the
fully-pseudolabelled + reviewer-corrected tree.

If you are a **student correcting a pseudolabel**, your correction is
`pseudo_corrected` and **needs a reviewer (radiologist) sign-off** before it is
treated as ground truth.

---

## 7. Hard-won pitfalls (read these)

1. **Trust the spine mask, not the pelvic pseudolabel.** For separate-mode
   patients, the *spine_only* record is authoritative for L6. A pelvic-view
   pseudolabel often can't even see the whole lumbar spine, so it **can't count**
   — never let it promote a patient to "lumbarization."
2. **Never overwrite radiologist GT with an algorithm.** `fuse_split` and other
   tools touch *synthetic*, *pseudolabel*, and *inference-time* labels only.
   Radiologist masks are the gold standard you are trying to *reproduce*.
3. **Class-mixing errors exist even in GT.** If a single vertebra is labeled with
   two classes, or the count is impossible (L1→L2→L4), flag it for correction —
   don't propagate it.
4. **Boundary metrics, not just Dice, for fused cases.** Volumetric Dice barely
   moves on a thin boundary; evaluate the fused interface with a tolerance-banded
   surface metric, and don't over-optimize where the GT itself is inter-rater-soft.

---

## 8. One-paragraph summary for the impatient

Count vertebrae from the top and label L1–L6 + sacrum + hips; use `ignore` (not
background) for un-traced regions in partial masks. A transitional vertebra is
**two classes if its body still has a disc to the sacrum** (even with a fused TP
bridge) and **one class (sacrum) only if the body is fully incorporated and the
disc is gone**. Record the Castellvi/Mahato grade as metadata. Trust radiologist
masks over pseudolabels; flag mixing errors; get reviewer sign-off on
corrections.
