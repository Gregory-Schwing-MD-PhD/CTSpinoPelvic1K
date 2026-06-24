# Annotation guide — Iliolumbar ligament (`TASK=iliolumbar`)

**Difficulty: moderate** · paints ids **58/59** · Space `…/CTSpinoPelvic1K-review-ili`

Read [README.md](README.md) first.

## Copy-paste to start
After the one-time setup in [README.md](README.md) (clone + `pip install` + `hf auth login`):
```bash
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-ili.hf.space
python -m reviewtool next      # claims a case + opens ITK-SNAP; annotate ligament, save & close to submit
python -m reviewtool next      # ...repeat for each case
python -m reviewtool status    # your progress
```

## Goal
Segment the **iliolumbar ligament** bilaterally onto the v3 base:

| id | structure |
|---|---|
| 58 | `iliolumbar_left` |
| 59 | `iliolumbar_right` |

## Anatomy (what you're tracing)
A strong band running from the **transverse process of L5** (its origin in **>96%**
of people) **postero-laterally to the iliac crest / posterior iliac wing**. It
stabilises L5 on the sacrum.

- The GT **L5 (id 24)** transverse process is your origin landmark (grey context).
- Trace the band from the **L5 TP tip** to its **iliac insertion**; do both sides.
- It is often **ossified** in DISH / seronegative spondyloarthropathy — when ossified
  it follows bone density and is easy to see; when not, it's a soft-tissue band
  between the TP and ilium (lower contrast — trace conservatively).

## Why it matters (the LSTV cross-check)
Because it almost always arises from the **last lumbar (L5) transverse process**, the
iliolumbar ligament is a **vertebral-level landmark** — a third, independent anchor
(alongside ribs and the L5 nerve) for confirming L5 in transitional anatomy.
**Caveat:** its reliability is *questioned in LSTV* cases, so it's a corroborating
signal, not the sole authority. If the ligament appears to arise from a level other
than the GT L5, paint what you see and **note it** — that discordance is the useful
signal.

## AI-assist
Small structure — usually fastest to paint by hand, or seed with **nnInteractive**
(a scribble along the band) and tidy. Ossified ligaments can be thresholded from the
CT then trimmed to the band.

## Reference images
- **Anatomy + CT/MRI appearance:**
  [Radiopaedia — Iliolumbar ligament](https://radiopaedia.org/articles/iliolumbar-ligament)
  — see *Case 1 (iliolumbar ligament on CT)* and *Case 3 (ossified iliolumbar ligaments)*.
- **LSTV labelling context:**
  [Radiopaedia — Lumbosacral transitional vertebra](https://radiopaedia.org/articles/lumbosacral-transitional-vertebra)
  (*Case 4: iliolumbar ligaments on CT*) for how it's used as a level landmark.
- **Worked example (add one):** place an annotated CT screenshot at
  `figs/iliolumbar_annotated_example.png` and embed it here — see [figs/README.md](figs/README.md).

## QC checklist before submit
- [ ] Band **originates at the L5 (GT id 24) transverse process** (or note the level
      it actually arises from).
- [ ] **Inserts on the ilium** (crest / posterior wing); both sides done.
- [ ] Ossified portions included; soft-tissue band traced conservatively, not into
      adjacent muscle.
- [ ] No overlap with bone ids (overlay only).
