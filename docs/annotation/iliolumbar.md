# Annotation guide — Iliolumbar ligament (`TASK=iliolumbar`)

**Difficulty: moderate** · paints ids **51/52** · Space `…/CTSpinoPelvic1K-review-ili`

Read [README.md](README.md) first.

## Goal
Segment the **iliolumbar ligament** bilaterally onto the v3 base:

| id | structure |
|---|---|
| 51 | `iliolumbar_left` |
| 52 | `iliolumbar_right` |

## Anatomy (what you're tracing)
A strong band running from the **transverse process of L5** (its origin in **>96%**
of people) **postero-laterally to the iliac crest / posterior iliac wing**. It
stabilises L5 on the sacrum.

- The GT **L5 (id 5)** transverse process is your origin landmark (grey context).
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

## QC checklist before submit
- [ ] Band **originates at the L5 (GT id 5) transverse process** (or note the level
      it actually arises from).
- [ ] **Inserts on the ilium** (crest / posterior wing); both sides done.
- [ ] Ossified portions included; soft-tissue band traced conservatively, not into
      adjacent muscle.
- [ ] No overlap with bone ids (overlay only).
