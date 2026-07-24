# CTSpinoPelvic1K — Spine Review Instructions

**Read this fully before your next case.**

---

## Why there's another pass

The spinal-column labels come from **CTSpine1K**, and our quality control found genuine errors in that original ground truth:

- **Split vertebrae** — one vertebra broken into two disconnected masks.
- **Merged / duplicated vertebrae** — one label covering **two** vertebral bodies (e.g. two bodies both labelled L2).
- **Missing vertebrae** — levels clearly inside the scan that were never labelled.

We are correcting the source data. **13 earlier submissions were re-opened** because the QC is now stricter; if one comes back to you it will say exactly what to fix.

---

## What you fix — three things, all unambiguous

### 1. One vertebra = one label = one connected mask

- If a vertebra is **split into disconnected pieces** → **merge it** (relabel/connect the stray piece).
  *Exception:* if the vertebra is **cut off by the top or bottom of the scan**, it is genuinely in two pieces — that's field-of-view truncation, leave it. The QC knows and won't flag it.
- If **one label covers two vertebral bodies** (a "tall" mask spanning two discs) → **separate them into two masks.** Two vertebrae can never share a label.

### 2. Numbering must be consecutive and in order

- After separating a merged mask you'll have one **extra** body. Renumber **consecutively** so the run has no gaps and no repeats — e.g. six lumbar bodies become **L1 → L6** (the extra one becomes **L6**).
- **Keep the numbering that's already there** where it's consistent. You are making the sequence internally correct — not re-deriving levels from scratch.

### 3. Annotate every vertebra visible in the field of view

- Segment each vertebral body you can clearly see, continuing the **consecutive** numbering upward (…T11, T10, T9).
- If a vertebra is partly cut off at the edge of the scan, label the part that's actually there. Don't invent what isn't visible.

---

## What you do **not** decide

**Do not try to determine absolute vertebral levels from the ribs.** Whether a small rib sits on "T12", "L1" or "T13" cannot be established from these scans — the field of view usually has no reliable anchor, and T13-vs-L1 is a naming convention rather than a measurement. A short 12th rib is also perfectly normal.

So: **don't renumber the whole column to match a rib**, and **don't guess a transitional level.** Fix the internal consistency (Steps 1–3) and flag anything ambiguous.

---

## How to flag a case for the radiologist

While you are **holding** the case (claimed with `next`), run:

```bash
python -m reviewtool flag "possible extra lumbar level"
```

Add the case name if you're holding more than one:

```bash
python -m reviewtool flag 22__pelvic_native "two bodies labelled L2"
```

**What happens:** it goes to the radiologist's queue, comes **off** the student queue, and releases your claim. Then carry on with `python -m reviewtool next`.

**Flag — don't guess — when:** the level identity is ambiguous, the count doesn't work out, the sacrum looks transitional, or you spot a **rib** problem while in the spine Space.

---

## Example — case 22

![sagittal example](example_22.png)

One lumbar mask covers **two** vertebral bodies. Separate them, then renumber the lumbar run consecutively (here L1 → L6). The thoracic vertebrae visible above are segmented and numbered consecutively upward.

---

## Rules

- **Stay in the spine here.** Only correct vertebrae / sacrum / pelvis. Rib problems → flag, don't fix them here.
- **Ribs are protected automatically** — you can only edit the spine/pelvis labels.
- **The QC runs on every Save** and gates on the facts: every bone is one connected mask, no label covers two bodies, numbering is consecutive and ascending. FOV-truncated vertebrae are exempt. Level-identity hints are advisory only and never block you.

---

## Commands

```bash
hf auth login
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-spine.hf.space

python -m reviewtool next                  # claim a case; edit in ITK-SNAP, Save, then Quit
python -m reviewtool resume                # submit
python -m reviewtool next --amend          # redo a case re-opened for you
python -m reviewtool flag "reason"         # send an ambiguous case to the radiologist
python -m reviewtool mystats               # your own progress
```

`git pull` your reviewtool checkout first — the checks and the `flag` command are new.

**Thank you — this is the last major pass.**
