# Annotation guide — Lumbosacral nerve roots (`TASK=ls_nerve`)

**Difficulty: hard** · paints ids **53–58** · Space `…/CTSpinoPelvic1K-review-nerve`

Read [README.md](README.md) first. This is the hardest task — read the limits.

## Copy-paste to start
After the one-time setup in [README.md](README.md) (clone + `pip install` + `hf auth login`):
```bash
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-nerve.hf.space
python -m reviewtool next      # claims a case + opens ITK-SNAP; annotate roots, save & close to submit
python -m reviewtool next      # ...repeat for each case
python -m reviewtool status    # your progress
```

## Goal
Segment the **exiting nerve roots** L4, L5, S1 **bilaterally** onto the v3 base:

| id | root | id | root |
|---|---|---|---|
| 53 | `nerve_L4_left` | 54 | `nerve_L4_right` |
| 55 | `nerve_L5_left` | 56 | `nerve_L5_right` |
| 57 | `nerve_S1_left` | 58 | `nerve_S1_right` |

Level reference: the GT vertebrae **L4 (4), L5 (5), S1 (7)** are grey context. The
exiting root at a foramen is named for the vertebra **above** that foramen (the L4
root exits the L4–L5 foramen, etc.).

## Two payloads (why we do this on CT)
1. **Kambin's triangle / foraminal mapping** — the exiting root is the hypotenuse of
   Kambin's triangle (clinical motivation: Tabarestani 2023; CT feasibility: Fan 2019).
2. **LSTV neural enumeration** — at the lateral sacrum the **L5 root does not split
   proximally and is ~2× the caliber of the L4 peroneal branch**; counting similar-
   caliber roots distinguishes 4- vs 5- vs 6-lumbar-segment anatomy.

## How to identify on non-contrast CT (and the honest limits)
- The root is visible **within foraminal/epidural fat** — the fat is the contrast.
  Trace from where it leaves the thecal sac, **through the foramen, as far lateral
  as it stays visible** in fat.
- **Do paint:** the proximal/foraminal root and dorsal root ganglion where seen in fat.
- **Do NOT fabricate:** the intrathecal segment or fine distal branches that CT can't
  resolve. A shorter, *correct* root beats a long guess.
- **Caliber matters** for enumeration — segment the root's true thickness; note if a
  root looks doubled or **conjoined** (paint it, flag it; conjoined roots shrink
  Kambin's triangle and confound counting).

## AI-assist
Use **nnInteractive** scribbles to seed each root, then correct. Place a scribble in
the foraminal fat on the root, generate, prune over-segmentation into vessels/fat.
(See the configured nnInteractive server in the project notes.)

## Reference standard (mandatory for this task)
- Where the patient has paired **MRI**, check your CT roots against it.
- All disagreements go to the **expert adjudicator** (this task's IRR will be lower
  than ribs — that's expected; the adjudication + MRI spot-checks are the safeguard).

## Reference images
- **Nerve-morphology enumeration at the lateral sacrum** (the L5-vs-L4 caliber/split
  rule): [Radiopaedia — Lumbosacral transitional vertebra](https://radiopaedia.org/articles/lumbosacral-transitional-vertebra)
  — see *Case 9 (left S1 partial lumbarization)* for the axial nerve appearance.
- **CT feasibility + Kambin's geometry:** the SPINECT (Fan 2019) and Tabarestani
  2023 figures are in the project PDFs — use them to calibrate what a root looks
  like in foraminal fat before you start.
- **Worked example (add one):** place an annotated CT screenshot at
  `figs/ls_nerve_annotated_example.png` and embed it here — see [figs/README.md](figs/README.md).

## QC checklist before submit
- [ ] Correct **level** (read off the GT vertebra) and **side**.
- [ ] Root traced only where genuinely visible in fat; no intrathecal guessing.
- [ ] Caliber looks anatomic; conjoined/duplicated roots painted **and noted**.
- [ ] No overlap with bone ids (overlay on background/fat only).
