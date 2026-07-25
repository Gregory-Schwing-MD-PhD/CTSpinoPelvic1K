# CTSpinoPelvic1K — Spine Review Instructions

**Read this fully before your next case.**

Our field of view is the **lumbar spine plus the lower thoracic only** — the top of the spine is not in view. So we **cannot count from the top.** The only reliable anchor is the **bottom (the lumbosacral junction)**, counting upward.

**Key principle: ribs do not define a vertebra's identity.** We fix the vertebral *number* first (by counting from the sacrum), then a rib simply belongs to whatever vertebra it sits on.

---

## Why there's another pass

The spine labels come from **CTSpine1K**, and QC found genuine errors in that original ground truth: vertebrae split into two masks, one label covering two vertebral bodies, and levels in the scan that were never labelled. We're correcting the source data. **Some earlier submissions were re-opened** under the stricter QC; if one comes back to you it will say exactly what to fix.

---

## The convention

### 1. Anchor at the lumbosacral junction and count **up**
Identify **S1 / the sacrum**, count up the lumbar vertebrae (L5, L4, L3, L2, L1); the next vertebra above L1 is **T12**. Use the radiologist's existing labels as a **starting point to reconcile — not as ground truth** — since they're sometimes inconsistent at this junction.

Normal anatomy has **five** free lumbar bodies, but **LSTV changes the count** and that is the anatomy, not an error: a **sacralization** has **four** free lumbar (L5 is fused into the sacrum), and a **lumbarization** has **six**. So do **not** force five — if the sacrum clearly extends farther rostrally than a normal S1 and the bottom lumbar is fused into it, that is a sacralization and the case is genuinely **L1–L4**. Keep it L1–L4.

### 2. A rib belongs to the vertebra the count places it on — it does **not** rename it
- If the count says the rib-bearing vertebra is **T12**, it's a **T12 rib** (full or stump — length doesn't matter).
- If the count says it's **L1**, it stays **L1 with a lumbar rib** — **do not promote it to T12.**
- Force-promoting a rib-bearing vertebra is exactly what miscounts the spine and manufactures false sacralization.

### 3. Ribless ⇒ lumbar; but a real L1 lumbar rib is a real variant
Lumbar vertebrae are normally rib-free, so **a vertebra with no rib is lumbar.** A genuine **L1 lumbar rib** does occur, though — label it per rule 2 (keep it L1), don't absorb it into T12.

### 4. Default: keep the existing labels — fix only the class-mixing
Your job on almost every QC block is **not** to renumber the spine. It's to clean up a **local** labeling error and leave everything else alone:

- **Keep the existing labels.** They were assigned S1-up and usually already encode the correct level identity. **Do not shift the whole column up or down** (a "+1 shift") to force a canonical count.
- **Fix only genuine class-mixing** — two different level labels bleeding into one vertebral body, or an adjacent-level / left–right swap. Make each body carry one consistent label so the sequence is monotonic. That, by itself, clears most blocks.
- A QC message can name a level that isn't in the scan — e.g. *"X order: T10 (id 17) sits at/above T9 (id 16) → out of sequence"* when T9 isn't visible. That means a **stray piece is still carrying a T9 label**; find it, merge/relabel it into the correct level, and the block clears. **Fix, not flag.**

### 5. Fix the unambiguous structural errors (the QC enforces these)
- **One vertebra = one label = one connected mask.** If a vertebra is split into disconnected pieces, **merge it.** *Exception:* a vertebra **clipped by the top/bottom of the scan** is genuinely in two pieces (FOV truncation) — leave it; the QC exempts it.
- **A segmentation blob that straddles two bodies** — one *connected mask* spanning two adjacent vertebrae — **separate them** along the disc, keeping each body's existing number.
- **Numbering must be consecutive and ascending** — no gaps, no repeats. **Change as little as possible**: repair the sequence, don't re-derive the whole spine.
- **BUT — a genuine duplicate vertebra is unsolvable → flag it.** If one *level number* is spread across **two distinct vertebral bodies** (e.g. **"L2" on two separate bodies**), there is no clean one-to-one label↔body mapping, and you cannot tell from this FOV whether it's a lumbarization or the whole column should shift up by one. **Do not guess and do not renumber** — flag it (see below).

### 6. Annotate every vertebra visible in the FOV
Segment each vertebral body you can clearly see, continuing the consecutive numbering upward. If one is partly cut off at the scan edge, label what's actually there.

---

## What to do with the radiologist's existing label

**Keep it** — unless it conflicts with the **S1-anchored count** (rule 1) or with **rib evidence** (rule 2). If it conflicts, don't silently overwrite it and don't force it to fit: **flag the case** (below). The radiologist's labels are a prior to reconcile, and the junction is exactly where they're least reliable.

---

## Flag — don't guess

Some cases can't be resolved from these images. **Flag and move on** instead of forcing a label:
- a **genuine duplicate vertebra** — one level number spread across two distinct bodies (e.g. "L2" on two bodies) — so you can't tell **lumbarization** from a **+1 shift** of the whole column;
- the count and the ribs **disagree** (e.g. a rib on what counting says is L1, and you're unsure);
- the **lumbosacral junction is ambiguous** (possible sacralization or lumbarization), so the upward count is uncertain;
- the radiologist's label **conflicts** with the S1-anchored count or with rib evidence;
- you spot a **rib** problem while in the spine Space.

For the duplicate-vertebra / count-shift case, use this exact reason:

```bash
python -m reviewtool flag "lumbarization vs shift up +1"
```

Run it **while holding the case.** It goes to the radiologist's queue, comes off the student queue, and releases your claim. Then `python -m reviewtool next`. Other ambiguities can use their own short reason, e.g. `python -m reviewtool flag "junction ambiguous — possible sacralization"`.

**The honest limitation:** with only a bottom anchor and a limited FOV, a sacralized or lumbarized junction throws off the entire upward count, and we sometimes can't tell a T12 stump rib from an L1 lumbar rib. When that's the case, we **flag rather than guess** — that's the correct, rigorous answer.

---

## Example — case 22

![sagittal example](example_22.png)

One lumbar mask covers **two** vertebral bodies — separate them (two vertebrae can't share a label). Then renumber by counting up from S1: five lumbar (L5→L1), and the body above L1 is T12. Ribs attach to whatever level the count lands on; they don't rename it. If the junction or the count is ambiguous, flag it.

---

## Example — sacralization (four lumbar, stump rib on T12)

A case comes in labeled **L1–L4 with a stump rib on T12**, and it's tempting to relabel it **L1–L5 with the rib on L1**. **Don't.** Look at the sacrum: it extends farther rostrally than a normal S1, and the bottom lumbar is **fused into it** — that's why there are only four free lumbar bodies. This is a **sacralization**, and it's the LSTV phenotype itself, not a numbering error.

- **Keep it L1–L4.** Do not shift the column up to manufacture a free L5.
- **The stump rib stays on T12** (length doesn't matter — a T12 rib can be short).
- Fix any class-mixing (e.g. two labels sharing one body) and submit.

The only time this kind of case gets flagged instead is when it's a **genuine duplicate vertebra** — one number across two bodies — where you can't tell lumbarization from a +1 shift. Then: `python -m reviewtool flag "lumbarization vs shift up +1"`.

---

## Rules & commands

- **Stay in the spine here.** Only correct vertebrae / sacrum / pelvis. Rib problems → flag, don't fix them here.
- **Ribs are protected automatically** — you can only edit the spine/pelvis labels.
- **The QC gates only on facts:** one connected mask per label, no label covering two bodies, consecutive + ascending numbering; FOV-truncated vertebrae exempt. Level identity is your judgment (count from S1) — flag it when unsure; the QC won't second-guess your count.

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
