# Detecting transitional anatomy by segmentation — an architecture

Working notes toward the CNS talk. The claim: one-shot semantic labelling of L1–L6 cannot
work for transitional anatomy, and the reason is structural rather than a matter of more
data or a better backbone.

---

## Why the current pipeline fails, and it is not a training problem

The pipeline segments 10 classes: L1–L6, sacrum, hips. That asks a network to output the
*identity* of each vertebra. Three things make that unlearnable in the cases that matter:

**Identity is not a local property.** L5 and L6 are the same bone with different names.
Which one you are looking at depends on how many vertebrae lie above it — information
that may be entirely outside the field of view. A per-voxel segmentation loss is local;
counting is global. The loss cannot express the constraint that makes the answer
decidable.

**The rare class is rare in exactly the way that defeats a segmentation loss.** L6 appears
in roughly 2% of cases. A network minimising Dice over the whole dataset is correct
almost always by never predicting it, and oversampling fights the symptom rather than
the cause.

**The training labels encode a convention, not an observation.** When an annotator writes
"L5, sacralised" on one scan and "L6" on a morphologically identical one, the network is
being trained on the disagreement between two radiologists rather than on anatomy. It
will learn to reproduce whichever convention is commoner.

This is the documented failure mode, not a hunch: the VerSe challenge reported that
methods perform well on normal anatomy and fail on variants not frequently present in
training, and it named transitional vertebrae specifically.

---

## The principle: separate what is perceptual from what is combinatorial

A radiologist does not recognise "L5" — they find an anchor, count, and apply a naming
convention. The parts are different in kind:

| step | kind | who should do it |
|---|---|---|
| is this bone a vertebra, a rib, a sacrum? | perceptual, local | the network |
| where does one vertebra end and the next begin? | perceptual, local | the network |
| does a rib articulate with this vertebra? | geometric | deterministic code |
| how many rib-free vertebrae are there? | counting | deterministic code |
| is that one L5 or L6? | convention | stated, not learned |
| is this transitional? | logic over the count | deterministic code |

Ask the network only for the first two rows. Everything below them is code we have already
written and validated against 802 cases.

---

## Proposed architecture

### Stage 1 — semantic segmentation (nnU-Net), convention-free

Classes contain no vertebral identity at all:

```
1  vertebra                 (body + posterior elements, any level)
2  intervertebral space     (the separator — see below)
3  rib, left
4  rib, right
5  sacrum
6  hip, left
7  hip, right
8  femur, left / right
```

**Why an explicit intervertebral-space class.** nnU-Net is semantic, not instance, and
"vertebra" alone gives one connected blob wherever bodies touch — which is precisely
where counting happens. Predicting the disc space as its own class turns instance
separation into a segmentation task the network can do locally, and connected components
of `vertebra ∧ ¬space` are then individual vertebrae. This is cheaper and more robust
than a separate instance head, and it degrades gracefully: a missed disc merges two
vertebrae, which the counting stage can detect as an outlier in height.

**Why ribs are left and right but not numbered.** Side is locally decidable; number is
not. Rib number has exactly the same problem as vertebral identity and must not be asked
of the network.

### Stage 2 — instance separation

Connected components on `vertebra ∧ ¬space`, ordered by position along the spine axis.
No learning, no thresholds beyond a minimum volume.

**The separator has to cut three joints, not one, and this is where the design nearly
failed.** Adjacent vertebrae meet at the disc *and at both facet joints*. A rule that
fills the intervertebral gap leaves the column connected through the facets, and connected
components then return the whole lumbar spine as one object — measured, not supposed: on
case 0001 the first implementation gave **2 components for 10 vertebrae**.

Worse, in these labels the two bodies frequently touch with **no background voxel between
them at all**, because the tools that produced the source segment bone and do not segment
joints. Where there is no gap, a gap rule has nothing to fill. The separator therefore has
to carve a thin sheet out of what the labels call bone, along the surface where two
vertebrae abut. That is the honest reading — a real joint occupies space the labels give
to one side or the other — but it means the disc class is *not* purely background, and
anyone reimplementing this will get 2 components and a plausible-looking render if they
assume otherwise.

A second attempt is worth recording because it is the kind of idea that looks strictly
better and is not. Requiring each candidate voxel to lie geometrically *between* its two
nearest surface points removes a small collar of voxels that wrap around the sides of the
bodies. On a synthetic phantom it is exactly right. On real anatomy the endplates are
concave, so the two nearest points are seldom collinear with the voxel between them: it
rejected almost the whole seam, left **2262 fragments** on one case, and still did not cut
the column. The phantom passed all three versions. Only the component count on real labels
distinguished them.

The lesson generalises past this pipeline, and it is the one worth saying out loud in the
talk: *the test has to be the thing the stage is for.* Dice on the disc class would have
looked respectable in all three versions. "How many vertebrae came out?" separated them
immediately.

### Stage 3 — rib-to-vertebra association

**This code already exists and is validated**: `scripts/qc_rib_vertebra_incidence.py`
and the anchor-and-increment logic in `scripts/anchor_and_increment_ribs.py`. Across
11,548 ribs in the existing release it leaves 2 unresolved associations, both field-of-view
truncations. Reusing it means the association step arrives pre-validated rather than
needing its own study.

### Stage 4 — counting, naming, flagging

Deterministic, and the only place a convention appears:

- Count rib-free vertebrae between the lowest rib-bearing vertebra and the sacrum.
- **5** → typical. **4 or 6** → transitional; flag it.
- Report the count-free description always; report a level name only if asked, with the
  convention named in the output.

Transitional anatomy is then *detected*, not predicted — it is a property of a count, and
a count has an audit trail.

---

## The three cases that break a naive version

**Unilateral ribs.** A transitional vertebra may carry a rib on one side only. A binary
rib-bearing flag cannot express that, which is why side is segmented separately and
rib-bearing status is computed *per side* in stage 3. Unilateral is then a first-class
output, and it maps onto the Castellvi a/b distinction.

**The lumbar rib.** A rib on a lumbar body is the same object as a hypoplastic twelfth
rib under a different count. Any rule that says "rib-bearing means thoracic" gets this
wrong. The dataset already gives lumbar ribs their own class for this reason, and the
counting stage must treat a small rib on the first rib-free candidate as *ambiguous* and
report both counts rather than choose.

**Fusion and bridging.** DISH or a bony bridge merges two vertebrae into one component.
The disc-space class helps but will not always save it. Detect it as an outlier in
component height rather than letting it silently reduce the count by one — a silent
off-by-one here is exactly the failure the whole design exists to prevent.

---

## What to measure, and against what

The benchmark should not be Dice. Dice on vertebrae is high and uninformative; the
question is whether the *count* is right.

1. **Count accuracy** — the fraction of cases where rib-free count matches ground truth,
   reported separately for typical and transitional anatomy. The gap between those two
   numbers is the result.
2. **Level-naming accuracy under a stated convention**, for comparison with prior work
   that reports it.
3. **Transitional detection** — sensitivity and specificity for flagging 4 or 6.
4. **The failure mode when it fails** — off-by-one, or something else. An off-by-one in
   the counting stage is traceable to a specific missed rib or merged vertebra, which a
   one-shot model can never tell you.

The comparator is the existing 10-class pipeline on the same frozen splits. If the
two-stage design is right, the two should be similar on typical anatomy and separate on
transitional — and that separation is the talk.

---

## Prior art this rests on

- **VerSe** (Sekuboyina et al., *Medical Image Analysis* 2021) — the benchmark, and the
  source of the finding that transitional anatomy is the dominant labelling failure.
  Two-stage locate-then-label approaches outperformed direct multi-class segmentation.
- **Payer et al.** — heatmap localisation followed by labelling, rather than one-shot
  identity.
- **Sekuboyina et al.**, *Radiology: AI* 2020 — incorporating prior knowledge of spine
  anatomy into labelling, adversarially. Same underlying observation: the anatomy carries
  a constraint the loss does not.

The departure here is that the constraint is not learned at all. It is arithmetic, it runs
outside the network, and it is auditable.

---

## Status and next step

**Not launched.** Grid space is currently consumed by the 802-case ablation merge, and
this needs a fresh preprocessed dataset. What can be done before space frees:

1. ~~Write the label-remapping script — v5 labels to the convention-free scheme above.~~
   **Done**: `--countfree` in `tools/convert_hf_to_nnunet.py`. The lookup table was the
   easy half; the derived class took three attempts against real anatomy.
2. ~~Decide whether the disc-space class is derived geometrically or annotated.~~
   **Done, and the inspection was worth it.** Derived, by a distance rule plus an
   abutment cut — see Stage 2 above and `_derive_disc_space()` in
   `tools/convert_hf_to_nnunet.py` in the nnU-Net repo, with synthetic tests in
   `tests/test_countfree_disc.py` and the functional check in
   `tools/check_countfree_disc.py`. The functional check is the one that matters: it asks
   whether the component count equals the number of vertebrae the source labels say are
   there, which is the only question stage 2 exists to answer.
3. Keep the existing splits. The comparison is only meaningful on identical folds.
