# Why we annotate the last rib-bearing vertebra (the counting anchor)

This document is the standalone scientific/clinical justification for adding the
**last rib-bearing vertebra** and its **rib** to the CTSpinoPelvic1K masks. It
holds on its own merits, independent of any specific paper's acceptance.

## What we are doing

On every CT we add two structures to the segmentation — `last_rib_vertebra` (11)
and `rib` (12) — alongside the existing L1–L6, sacrum, and hips. Students also
correct two pre-existing label defects while they are in each case:
**class-mixing** (a single vertebra labeled with two numbers) and
**partially-colored vertebrae** (a body only partly labeled).

## The problem it solves

Automated spine segmenters mislabel vertebrae on anyone whose spine is not
"standard" — the 5–35% of people with a lumbosacral transitional vertebra (LSTV:
an extra sixth lumbar, or a fifth fused into the sacrum). The model counts up
from the sacrum, assumes the usual five lumbar bodies, and when there are six (or
four) it **shifts every label by one level**: the body that is anatomically L6 is
called L5, L5 becomes L4, and so on. In surgical planning this output is
indistinguishable from a wrong-level plan — the exact failure this project
targets. This is the **caudal level-shift cascade**: top-of-spine is anchored
roughly correctly, but labels drift caudally as the count discrepancy compounds,
because the disambiguating signal — *vertebra count from a known anchor* — is
**non-local** and not reliably present in a single network field of view.

## Why the rib anchor fixes it

You cannot tell L5 from L6 by looking at one vertebra; the only reliable way to
number the spine is to **count from a fixed landmark.** The last rib-bearing
vertebra is that landmark — the vertebra directly below it is L1, and you count
down to the sacrum. Three properties make it the right anchor:

1. **In field of view.** It is almost always captured at the top of a
   lumbosacral scan, so it gives the model a top reference it can actually *see*
   — closer and more reliably present than the cervico-thoracic boundary a
   counting network would otherwise need.
2. **Relational, not absolute.** It is "the last rib-bearing vertebra," a
   verifiable relative identity — never an absolute number (T12/T13 is unknowable
   and irrelevant in a lumbosacral FOV).
3. **Invariant to the variant anatomy.** Whether the patient has 4, 5, or 6
   lumbar bodies, counting from the rib gives the correct label every time. It
   replaces a brittle "assume five" *absolute positional prior* — which the
   level-shift proves fails on LSTV — with a robust "count from the rib"
   *relative prior* that is invariant to the extra/missing vertebra that breaks
   the absolute one.

## The payoff

This is the piece that lets a single end-to-end model correctly label
**L1–L4, L5, and L6 across all spine variants** — normal, lumbarization, and
sacralization — instead of silently shifting labels on exactly the patients where
the stakes are highest. It supplies the missing **in-FOV global count reference**
in a single forward pass, the cleanest test of whether a segmentation network can
resolve the level shift end-to-end without a downstream sequence predictor.

It is fast, AI-assisted work (a couple of minutes per case). See
`docs/RIB_ANCHOR_REVIEW_GUIDE.md` and `docs/STUDENT_ANNOTATION_PROTOCOL.md`.
