# Annotation guide — Ribs (`TASK=ribs`)

**Difficulty: easier** · paints ids **26–49** · Space `…/CTSpinoPelvic1K-review-ribs`

Read [README.md](README.md) first (one-time setup, IRR, label-space rules).

## Copy-paste to start
After the one-time setup in [README.md](README.md) (clone + `pip install` + `hf auth login`):
```bash
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-ribs.hf.space
python -m reviewtool next      # claims a case + opens ITK-SNAP; annotate ribs, save & close to submit
python -m reviewtool next      # ...repeat for each case
python -m reviewtool status    # your progress
```

## Goal
Segment each **visible rib**, painted at its **correct number and side**, onto the
v3 base label. Ribs reuse the v3 reserved ids:

| side | ids | name |
|---|---|---|
| left | 26–37 | `rib_left_1` … `rib_left_12` |
| right | 38–49 | `rib_right_1` … `rib_right_12` |

## The numbering rule (objective — do not guess)
Rib number = the number of the **GT thoracic vertebra its head articulates with**
(the costovertebral joint). The v3 thoracic GT is shown as grey context:
ids 13–25 = T1–T13, so **id 13→T1 … 24→T12, 25→T13**.

- A rib whose head joins the vertebra labelled **24 (T12)** → paint it
  `rib_left_12` (37) / `rib_right_12` (49).
- Always read the number off the vertebra; never count "down from the top" on a
  FOV-limited scan.

## What to paint
- Only ribs **in the field of view** — most spinopelvic scans show only the lower
  thoracic ribs. Paint the visible portion of each; partial ribs are fine.
- Paint the **bony rib** (cortex + medulla), from the costovertebral joint laterally
  as far as it's in view.

## AI-assist
Seed each rib with **nnInteractive** — a scribble along the rib generates a mask —
then correct the boundaries. (We are **not** using TotalSegmentator rib seeds.) The
key manual step is **numbering**: give each rib the number of the GT thoracic
vertebra its head articulates with (below).

## The LSTV/TLTV flag (why this matters)
If the vertebra labelled **T12 (24) has no rib**, do **not** invent one — leave it
blank and add a `reviewtool` note "T12 no rib". A GT-T12 without a rib (or an L1
*with* a rib) is a **thoracolumbar transitional** signal; these flagged cases feed
the LSTV/TLTV analysis. Your job is to record what's there, accurately.

## Reference images
- **Rib count → numbering (LSTV context):**
  [Radiopaedia — Lumbosacral transitional vertebra](https://radiopaedia.org/articles/lumbosacral-transitional-vertebra)
  — see *Case 8 (11 rib pairs and L5 sacralization)* for how rib count drives level numbering.
- **Worked example (add one):** place an annotated CT screenshot at
  `figs/ribs_annotated_example.png` and embed it here — see [figs/README.md](figs/README.md).

## QC checklist before submit
- [ ] Each painted rib is one connected structure on the correct **side**.
- [ ] The **number** matches the GT thoracic vertebra at its costovertebral joint.
- [ ] No rib voxels overlap a vertebra/femur/hip id (overlay only on background).
- [ ] Ribs out of FOV left unpainted; any "missing expected rib" noted.
