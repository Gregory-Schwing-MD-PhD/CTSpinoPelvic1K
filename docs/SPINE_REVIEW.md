# CTSpinoPelvic1K — Spine Review Instructions

**Read this fully before your next case.** This is the method for every spine case.

---

## Why there's another pass

The spinal-column labels in this dataset come from **CTSpine1K**. Our quality control found genuine errors in that original ground truth:

- **Split vertebrae** — one bone broken into two disconnected pieces.
- **Transitional miscounts** — a vertebra mask covering *two* bodies, where there's really a **6th lumbar (L6)**.
- **Missing thoracic vertebrae** — levels clearly inside the scan that were never labelled.

We're correcting the source data, so it has to be right. The QC is now stricter, and **13 earlier spine submissions were re-opened** for redo — if one comes back to you, it will tell you exactly what to fix.

---

## The procedure — every case, in this order

### Step 1 — Find **T12** using the last full rib

You anchor the whole count on T12, and you find T12 from the **ribs** — never by guessing.

- Ribs attach to thoracic vertebrae: rib 1 → T1, rib 2 → T2, … **rib 12 → T12**.
- Therefore: **the lowest vertebra carrying a FULL rib is T12.**
- A **full rib** is long and curving, wrapping toward the front. A small nub (a **stump rib**, roughly ≤ 4 cm) is **not** a full rib — skip it.
- Scroll to the **most caudal (lowest) full rib**, follow it medially to the spine — **that vertebra is T12.**

From T12: everything **below** is lumbar (L1, L2, L3 …), everything **above** is thoracic (T11, T10 …). Trust this over whatever the existing labels say.

### Step 2 — Fix class-mixing / duplicates

Check each vertebra:

- **Split bone** (one label in two disconnected pieces, or a stray blob floating off the bone):
  → **Merge it.** Relabel or connect the stray piece so the vertebra is a single clean connected mask. **Do not renumber anything.**

- **One mask covering TWO vertebral bodies** (a "tall" vertebra spanning two discs — the classic *duplicated L2*):
  → That's a **miscount**. Anchored on T12 from Step 1, count the lumbar bodies. If there are **six**, they are **L1–L6**: separate the fused mask into two bodies and label the extra one **L6** at the bottom.

Only the **lumbar** numbering may change (to introduce an L6). **Never** renumber a thoracic vertebra, hip, femur, or sacrum.

### Step 3 — Extend the spine rostrally (upward)

- **Segment every thoracic vertebra clearly visible in the field of view**, numbering **upward** from T12: T11, T10, T9 …
- Go as high as you can confidently identify a vertebral body. If a vertebra is cut off by the edge of the scan, label what's actually there — don't invent what isn't visible.

---

## Example — case 22

![sagittal example](example_22.png)

T10–T12 have been added rostrally. Below them the lumbar run contains **six** bodies but is numbered L1–L5 — one mask covers two vertebrae. Anchored to T12 (the last full rib), the correct labelling is **L1 → L6**, with the extra body becoming **L6**.

---

## Rules

- **Stay in the spine here.** Only correct vertebrae / sacrum / pelvis in this Space. If you notice a **rib** problem, don't fix it here — flag it (below).
- **Ribs are protected automatically** — you can only edit the spine/pelvis labels.
- **If a transitional level is genuinely unclear, flag it** instead of guessing. It goes straight to the radiologist.
- **The QC runs on every Save** and checks: every bone is one piece, numbering is ascending and contiguous, the **last full rib lands on T12**, and **no vertebra covers two bodies**. It will tell you exactly what's wrong and won't let a bad count through.

---

## Commands

```bash
hf auth login
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-spine.hf.space

python -m reviewtool next                  # claim a case; edit in ITK-SNAP, Save, then Quit
python -m reviewtool resume                # submit
python -m reviewtool next --amend          # redo a case that was re-opened for you
python -m reviewtool flag "possible L6"    # send a transitional case to the radiologist
python -m reviewtool mystats               # your own progress
```

Make sure you `git pull` your reviewtool checkout first — the checks and the `flag` command are new.

**Thank you — this is the last major pass.**
