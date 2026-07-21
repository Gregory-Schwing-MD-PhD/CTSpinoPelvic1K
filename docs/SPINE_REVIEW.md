# CTSpinoPelvic1K — Spine Extension Task (last stretch!)

This is a **small, final** task: **21 cases** where thoracic vertebrae are clearly **in the scan's field of view but were never labeled**. Your job is to **continue the numbering upward** and label them. That's it.

You already know the tools from the rib task — this is the same workflow pointed at a new Space.

---

## What you're doing

Some scans have ribs (or clearly-visible vertebral bodies) **above the topmost labeled vertebra**. The spine label just stops early. You **add the missing thoracic vertebrae**, numbering **up** from the most rostral (highest) vertebra that's already labeled.

**Example:** the spine is labeled up to **T10**, but you can see **T9, T8, T7** vertebral bodies above it in the CT. You label them **T9, then T8, then T7** — counting upward, one level at a time.

---

## The rules (short + important)

1. **Only ADD thoracic vertebrae (T1–T12).** Number them upward from the existing top.
2. **Do NOT change anything already labeled** — vertebrae, ribs, pelvis are the radiologist's ground truth. (The server automatically keeps *only* your new additions and restores everything else, so you can't break anything — but please don't try to "fix" existing labels here.)
3. **Count carefully, one level at a time.** Use the ribs as a guide where present. If you're unsure how high to go, label what you're confident about and stop.
4. Two people review each case independently; disagreements go to an adjudicator — so just do your honest best read.

---

## Setup (once)

```bash
hf auth login                       # your HuggingFace account
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-spine.hf.space
```

## Reviewing a case

```bash
python -m reviewtool next           # claims a case, opens CT + label in ITK-SNAP
# -> add the missing thoracic vertebrae (number upward), then SAVE and QUIT ITK-SNAP
python -m reviewtool resume         # submits; then run `next` for the following case
```

- `python -m reviewtool skip` — if a case is unclear or you'd rather leave it.
- `python -m reviewtool mystats` — see your counts.

## Switching back to ribs

Point the tool back at the rib Space anytime:
```bash
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-ribs.hf.space
```

---

## Please also wrap up your rib work

If you have **open rib reviews or adjudications**, please finish them (`reviewtool resume`, and for adjudicators `reviewtool adjudicate`). **The dataset is almost complete** — these last pieces are what get us to the finish line. Thank you!
