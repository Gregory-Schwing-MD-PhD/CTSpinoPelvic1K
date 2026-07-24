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

### 2. Numbering must be consecutive and in order — with the fewest changes possible

**Change as little as you can.** Keep the existing labels wherever they're consistent; you are repairing the sequence, not re-deriving the spine.

**The one absolute rule: a vertebra with NO rib is a LUMBAR vertebra.**

After separating a merged mask you'll have one **extra** body, so the numbers no longer fit. Resolve it with that rule:

- Look at the **top** of the lumbar run. **Does it carry a rib** (full *or* stump)?
  - **Yes → it is thoracic.** Relabel it **T12** and put its rib on T12. The ribless vertebrae below then renumber **L1, L2, L3, L4, L5** — the count works out and nothing else moves.
  - **No (it's ribless) → it is lumbar**, so the extra body stays lumbar and the run becomes **L1 … L6**.

**Worked example — two vertebrae labelled L2, and the top "L1" has a stump rib:**
that "L1" carries a rib, so it is really **T12**. Shift it up to T12 (rib stays on it), and the five ribless bodies below become L1–L5. No L6 is needed.

*(A stump rib does **not** by itself tell you the level — a short rib can sit on T12 **or** on L1. Use the ribless rule and the surrounding count, not the rib's length.)*

### 3. Annotate every vertebra visible in the field of view

- Segment each vertebral body you can clearly see, continuing the **consecutive** numbering upward (…T11, T10, T9).
- If a vertebra is partly cut off at the edge of the scan, label the part that's actually there. Don't invent what isn't visible.

---

## What you do **not** decide

- **A stump rib does not fix the level by itself.** A short rib can sit on **T12 or on L1** — its length tells you nothing definitive. Don't reason from "the last *full* rib must be T12"; that's not reliable (plenty of people have a naturally short 12th rib).
- **Don't renumber the whole column** to make it match the ribs. Repair the sequence with the fewest changes.
- **Don't guess a transitional level.** If the count still doesn't resolve using the ribless rule, or the sacrum looks transitional — **flag it**.

The only thing you can rely on absolutely: **no rib ⇒ lumbar.**

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

One lumbar mask covers **two** vertebral bodies — two vertebrae cannot share a label, so separate them. Then apply the ribless rule to the top of the lumbar run: if that vertebra carries a rib (full or stump) it is **T12** — shift it up and the ribless bodies below become L1–L5. If it is ribless, it stays lumbar and the run becomes L1–L6. Thoracic vertebrae visible above are segmented and numbered consecutively upward.

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
