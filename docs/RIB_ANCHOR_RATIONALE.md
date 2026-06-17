# Why we anchor the lumbar count — rostral (T12) and caudal (S1)

This document is the standalone scientific/clinical justification for bracketing
the lumbar count with two fixed landmarks in the CTSpinoPelvic1K masks: the
**last rib-bearing vertebra** (rostral) and **S1** (caudal). It holds on its own
merits, independent of any specific paper's acceptance.

## What we are doing

We anchor the lumbar count from **both ends**:

- **Rostral — the last rib-bearing vertebra (T12).** Not a stored class: it is
  simply the last thoracic vertebra already present in the native vertebral column
  (class 31 in the legacy scheme). Earlier drafts reserved separate
  `last_rib_vertebra`/`rib` classes (11/12); those were retired. The anchor is free
  from the native labels. (Rib-cage *segmentation* — distinct from this rib-bearing
  *anchor* — is **deferred to a future v4**; FOV-limited spinopelvic scans can't be
  rib-numbered reliably, so v3 reserves but does not populate rib ids 26–49.)
- **Caudal — S1.** The first sacral body, segmented in v3 as its own class,
  `(GT sacrum) ∩ (TS vertebrae_S1)` (see *The caudal anchor: S1* below).

(Independently of the anchors, label-defect cleanup — class-mixing, a single
vertebra labelled with two numbers, and partially-coloured bodies — is handled
in QC.)

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

## The caudal anchor: S1

The rostral anchor plus "count down to the sacrum" already brackets the lumbar
column — but the sacrum is a single fused mass, so its *top* (where the count
terminates) is not sharply marked. **S1 makes the bottom bracket explicit:** it is
the first sacral body, immediately below L5 (or L6), so the count runs from the
last rib-bearing vertebra down to S1 with both ends pinned. Two reasons to segment
it as its own class:

1. **It tightens the count.** A sharp caudal landmark (S1) plus the rostral anchor
   leaves no ambiguity about where the lumbar run starts and stops — the same
   relational logic as the rib anchor, applied at the other end.
2. **It is the sacral-endplate landmark.** The S1 superior endplate defines
   **sacral slope and pelvic incidence**, the core spinopelvic parameters, so S1
   doubles as the geometric primitive for automated PI / PT / SS measurement.

S1 is derived **GT-bounded**: only the part of the radiologist sacrum that
TotalSegmentator identifies as the S1 body is relabelled, so the sacrum's outer
boundary stays ground truth and TS only decides the internal S1/sacrum split. On
transitional anatomy this is safe by construction — in lumbarization the mobile
body is L6 (class 6), never the GT sacrum, so the carve cannot touch it; in
sacralization the fused mass stays radiologist-bounded. S1 is a **landmark** class
(the S1/S2 edge is an intrinsically fuzzy fused boundary), not a precise-volume one.

> **Released-v3 status:** the S1 carve is **on by default** (`--no_carve_s1` to
> disable). It only subdivides the GT sacrum in place — the sacrum's outer boundary
> always stays radiologist GT — and supplies the S1 superior-endplate landmark for
> PI / SS.

## The payoff

This is the piece that lets a single end-to-end model correctly label
**L1–L4, L5, and L6 across all spine variants** — normal, lumbarization, and
sacralization — instead of silently shifting labels on exactly the patients where
the stakes are highest. It supplies the missing **in-FOV global count references** — a
rib-bearing top (T12) and an S1 bottom — in a single forward pass, the cleanest
test of whether a segmentation network can resolve the level shift end-to-end
without a downstream sequence predictor.

The femurs and the S1 carve are produced automatically by TotalSegmentator (see
`scripts/build_v3_totalseg.py`), GT-bounded so they never overwrite radiologist
labels, while the rostral T12 anchor is free from the native vertebral column. The
**rib cage is deferred to v4** (ids 26–49 reserved but unpopulated in released v3).
Label-defect cleanup remains light, AI-assisted review — see
`docs/RIB_ANCHOR_REVIEW_GUIDE.md` and `docs/STUDENT_ANNOTATION_PROTOCOL.md`.
