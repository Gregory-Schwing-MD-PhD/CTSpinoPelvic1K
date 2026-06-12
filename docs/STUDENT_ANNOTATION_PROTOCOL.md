# CTSpinoPelvic1K — Student Annotation Protocol (master)

**Goal of the dataset:** a single end-to-end model that segments **and numbers**
the spine from the **last rib-bearing vertebra through the sacrum** — L1–L6 as
distinct classes, all Castellvi classes, and counting anomalies (lumbar rib /
T13 / sacralization / lumbarization) — in a limited lumbosacral field of view.

Your annotations are the ground truth that makes that possible. Read this once;
the per-case work is fast with AI assistance. The two companion docs:
- `LABELING_GUIDE.md` — the *why* behind the conventions (read second).
- `RIB_ANCHOR_REVIEW_GUIDE.md` — deep dive on the rib anchor specifically.

---

## 0. The one idea everything rests on

You **cannot** tell L5 from L6 by looking at one vertebra — they're identical.
The only reliable way to number the spine is to **count from a fixed anchor.**
We have two anchors that are almost always in a lumbosacral CT:

- **Top anchor — the last rib-bearing vertebra.** The vertebra immediately below
  it (with no rib) is **L1.**
- **Bottom anchor — the sacrum.**

Label both anchors, count the vertebrae between them, and the numbering is
deterministic: **5 lumbar → bottom is L5; 6 lumbar → bottom is L6.** That count
*is* the answer to every LSTV question. Everything below serves this.

---

## 1. What to annotate on every case

1. **Last rib-bearing vertebra** → class `last_rib_vertebra`, plus the
   **proximal segment of its rib(s)** → class `rib` (the part at the
   costovertebral/costotransverse junction; you do **not** need the whole rib).
2. **Each lumbar vertebra, L1–L6, as a distinct class** — counted down from the
   anchor. (Do **not** merge them; the model needs the distinct labels to learn
   to number.)
3. **Sacrum** → one class (see the fused rule in §4).
4. **Left / right hip (ilium)** → `left_hip` / `right_hip`, if in FOV.
5. **(LSTV cases only) Castellvi grade** → recorded on the separate
   `_lstv_phenotypes.csv` / review form, **not** as voxels.
6. **Flags** for the edge cases in §5.

Use **AI-assisted segmentation** for all of it (§3). Do not hand-trace.

---

## 2. The classes (label values)

| value | name               | notes                                              |
|------:|--------------------|----------------------------------------------------|
| 0     | background         |                                                    |
| 1–4   | L1–L4              |                                                    |
| 5     | L5                 |                                                    |
| 6     | L6                 | only when a true 6th lumbar is present             |
| 7     | sacrum             | incl. a fully-incorporated vertebra (§4)           |
| 8     | left_hip           | left ilium                                         |
| 9     | right_hip          | right ilium                                        |
| 10    | ignore             | un-annotated region in partial mode — NOT bg       |
| 11    | last_rib_vertebra  | the last rib-bearing vertebra (the top anchor)     |
| 12    | rib                | proximal rib segment of the last rib-bearing vert  |

Confirm these values in ITK-SNAP before saving — a mislabeled value is silently
wrong.

---

## 3. How to count and find the anchor (the core skill)

1. Start in the lumbar spine and **scroll cranially** (toward the head).
2. The **last rib-bearing vertebra** is the **lowest vertebra with a true,
   articulating rib** — a rib that articulates at both the costovertebral joint
   (on the body) and the costotransverse joint (on the transverse process).
3. The vertebra **immediately below it, with no rib, is L1.** Number downward
   from there: L1, L2, … to the sacrum.

**Do NOT assign an absolute thoracic number (T12/T13).** In a lumbosacral FOV you
can't count down from T1, so the absolute number is unknowable — and it doesn't
matter. The anchor is **"the last rib-bearing vertebra,"** a *relational*
identity you can verify; L1 is defined relative to it. Label `last_rib_vertebra`,
not "T12."

---

## 4. The conventions (the rules — see LABELING_GUIDE.md for the why)

- **L5 vs L6: count, never guess.** The anchor + the count decides it.
- **Lumbar rib / T13 trap:** a small **rudimentary rib on L1**, or an extra
  rib-bearing level, can fool the anchor. Anchor on the lowest vertebra with a
  **true, articulating** rib; if a level is borderline (rudimentary vs true rib),
  **segment it and flag it** — don't silently resolve it. Mis-placing the top
  anchor by one shifts the whole count.
- **Fused L6 / sacrum — one class or two?** Decide by the **L5/L6–S1 disc,
  assessed at the vertebral body**, NOT by whether the bone is continuous (a
  fused transverse-process bridge still makes one connected bone, but that
  doesn't change identity — same as a bamboo spine where each vertebra is still
  labeled):
  - **Disc present (body distinct), even with a fused TP bridge** → **two
    classes** (last lumbar + sacrum). Boundary at the disc level; only the small
    lateral TP bridge is "soft."
  - **Disc obliterated (body fused into the sacrum)** → **one class (sacrum).**
    Record the transitional identity as the Castellvi grade, not as voxels.
- **The fused TP-bridge boundary** is inter-rater-soft; draw it **consistently**
  (extend the disc-level body plane laterally), don't agonize over voxels. A
  deterministic helper (`fuse_split`) exists for synthetic/pseudolabel cases —
  it does **not** overwrite your annotations.
- **Partial / separate cases:** label the region you did NOT trace as `ignore`
  (10), **never background.**

---

## 5. Flags & edge cases (surface them, don't bury them)

On the review form, flag:
- **Lumbar rib / T13** ambiguity (rudimentary vs true rib at the anchor).
- **Fused** L6–sacrum (note whether the **disc is present or obliterated** — that
  drives one-vs-two-class).
- **Ambiguous count** (you genuinely can't tell 5 vs 6) → mark indeterminate;
  don't force it.
- **Class-mixing** in any pre-existing mask (a vertebra labeled two numbers).

---

## 6. AI-assisted workflow — use it, do NOT hand-trace

1. Open `ct.nii.gz` in ITK-SNAP.
2. Launch the project's **AI-assisted segmentation (DLS / nnInteractive backend).**
3. Drop a few interaction points per structure (each vertebra, the anchor, the
   rib), let the AI fill the mask, correct obvious leakage with the brush.
4. **Relabel** the AI output to the correct class values (§2).
5. Save; fill the review form; flag edge cases (§5).

A clean case is a few minutes this way. Fallback if the AI server is down:
ITK-SNAP active-contour (snake) on bone — still no voxel-by-voxel tracing.

---

## 7. Output & hand-back

- One label file per case with the classes in §2 (added alongside any existing
  mask, not overwriting it).
- Review form filled: the **count from the last rib-bearing vertebra to the
  sacrum** (5 → L5, 6 → L6), the disc-present/obliterated note for any fused
  case, the Castellvi grade for LSTV cases, and any flags.

---

## 8. One-paragraph summary

Scroll up from the lumbar spine, find the lowest vertebra with a **true
articulating rib** (label `last_rib_vertebra` + its proximal `rib`) — the one
below it is L1. Number L1→L6 **distinctly** down to the **sacrum**, all
AI-assisted in ITK-SNAP. Count vertebrae from the anchor to the sacrum — that
number settles L5 vs L6. For a fused L6/sacrum: **disc present → two classes;
disc gone → one sacrum class.** Don't assign an absolute thoracic number; flag
rudimentary lumbar ribs / T13 / ambiguous counts; use `ignore` (not background)
for un-traced regions.
