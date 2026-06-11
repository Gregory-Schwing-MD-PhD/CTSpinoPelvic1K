# CTSpinoPelvic1K — Last Rib-Bearing Vertebra & Rib Segmentation Guide

A short, focused annotation task that adds the **counting anchor** the dataset
needs to number vertebrae reliably in a limited lumbosacral field of view. Read
this once; the task itself is fast with AI assistance.

---

## 1. Why this matters (30 seconds)

You cannot tell whether the bottom lumbar is **L5 or L6** by looking at it — they
are near-identical. The only reliable way is to **count from a fixed anchor at
the top of the spine.** That anchor is the **last rib-bearing vertebra (T12)**:
the vertebra immediately below it is **L1**, and from there you count down to the
sacrum.

A lumbosacral CT almost always captures **both** the lowest ribs (T12) at the top
of the volume and the sacrum at the bottom. So if we label the **top anchor
(last rib-bearing vertebra + its rib)** and we already have the **bottom anchor
(sacrum)**, the lumbar count becomes deterministic — and the L5-vs-L6 question
answers itself. That is the entire purpose of this task.

---

## 2. What to segment

Per case, two structures:

1. **The last rib-bearing vertebra** → class **`last_thoracic`** (normally T12).
   Segment the whole vertebra (body + posterior elements) that is in the FOV.
2. **Its rib(s)** → class **`rib`**. You do **not** need the whole rib — the
   **proximal segment at the costovertebral / costotransverse junction is
   enough** to establish that this vertebra bears a true rib. Both sides if
   visible; one side is acceptable.

Nothing else. Do not segment higher thoracic levels or full ribs — out of scope.

---

## 3. How to find the last rib-bearing vertebra

1. Start in the lumbar spine and **scroll cranially** (toward the head).
2. The **last rib-bearing vertebra** is the **lowest vertebra with a true,
   articulating rib** — a rib that articulates at the **costovertebral joint**
   (on the vertebral body) and **costotransverse joint** (on the transverse
   process).
3. The vertebra **immediately below it, with no rib, is L1.**

### Watch for the variants (flag them)
The thoracolumbar junction has its own transitional anatomy that can fool the
count:
- **Rudimentary / lumbar rib at L1** — a short rib-like stub on what is otherwise
  the first lumbar vertebra. If you anchor on this, the whole count shifts by one.
- **T13 rib** — an extra rib-bearing level.
- A small, **non-articulating** stub is **not** a true rib.

Rule of thumb: anchor on the lowest vertebra with a **clearly articulating** rib.
If a level is **borderline** (could be a rudimentary lumbar rib vs a true T12
rib), **segment it anyway and flag it** in the review form — those borderline
cases are themselves clinically interesting and we want them surfaced, not
silently resolved.

---

## 4. Use AI-assisted segmentation — do NOT hand-trace

Hand-tracing a vertebra and rib is slow and unnecessary. Use the project's
**AI-assisted segmentation in ITK-SNAP (the DLS / nnInteractive auto-segmentation
backend).**

1. Open `ct.nii.gz` in ITK-SNAP.
2. Launch the **AI-assisted (nnInteractive / DLS) segmentation** tool.
3. Drop a few interaction points on the **last rib-bearing vertebra**, let the AI
   fill the mask, and correct any obvious leakage with the brush.
4. Repeat for the **rib** (a few points along the proximal rib).
5. **Relabel** the AI output to the correct class values: `last_thoracic` for the
   vertebra, `rib` for the rib. (Confirm the label values against the project
   label sheet before you save.)
6. Save as the per-case rib-anchor label and note borderline cases on the form.

A clean case takes a couple of minutes this way. If the AI server is unavailable,
the ITK-SNAP active-contour (snake) tool on bone is the fallback — still don't
hand-trace voxel by voxel.

---

## 5. Output & hand-back

- One label file per case containing `last_thoracic` + `rib` (added alongside the
  existing spine/pelvis mask, not overwriting it).
- The review form row filled in, with the **count of vertebrae from the last
  rib-bearing vertebra to the sacrum** (this is the number that settles L5 vs L6:
  **5 → no L6, 6 → L6**) and a flag for any borderline rib.

---

## 6. One-paragraph summary

Scroll up from the lumbar spine; find the lowest vertebra with a **true
articulating rib** (that's T12 — the one below it is L1). Segment that vertebra
(`last_thoracic`) and the proximal part of its rib (`rib`) using **ITK-SNAP's
AI-assisted tool**, not by hand. Count vertebrae from there to the sacrum — that
count is the L5-vs-L6 answer. Flag any rudimentary lumbar rib / T13 ambiguity.
