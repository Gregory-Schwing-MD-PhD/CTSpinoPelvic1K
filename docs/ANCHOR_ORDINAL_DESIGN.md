# Two anchors, two counts, and the disagreement between them

A design for single-pass LSTV/TLTV labelling. The argument is that the hard part is not the
architecture, it is what you ask the network to predict — and that a change of target turns
the rare-class problem into an abundant-data problem.

---

## The problem with predicting level names

Predicting `L5` directly has a defect that no amount of capacity fixes: **L6 appears in
about 2% of cases and the model is rewarded for never predicting it.** Worse, a model that
learns normal anatomy well will call every lowest lumbar vertebra L5, because the prior is
right roughly 97% of the time. This is the documented VerSe failure mode, and it is a
property of the *target*, not of the backbone.

Oversampling fights the symptom. The target itself has to change.

---

## The formulation

There are two anatomical anchors and they bracket the count from opposite ends:

- **the sacrum**, unmistakable, enormous, locally decidable;
- **the lowest rib-bearing vertebra**, equally local — a rib either articulates or it does
  not.

So predict, densely, for every vertebra voxel, **two ordinal fields**:

| head | target | learned from |
|---|---|---|
| `n_up` | how many vertebrae this one is *above the sacrum* | every case |
| `n_down` | how many it is *below the lowest rib-bearing vertebra* | every case |

A level name is then a deterministic function of the pair, and is never predicted by the
network at all.

### Why this is the whole idea

In typical anatomy the two counts agree: the vertebra that is 1 above the sacrum is also 4
below the lowest rib-bearing one, and `n_up + n_down` is constant down the column.

**In transitional anatomy they disagree, and the sign of the disagreement names the
phenotype.** A sacralised lowest lumbar segment shortens the interval from below, so `n_up`
runs one short against `n_down`. A lumbar rib shortens it from above, so `n_down` runs one
short against `n_up`. Lumbarisation runs the other way.

That is exactly the thing Greg described — *use the variation in the numbers, against two
anchors, to number sacralisation and lumbarisation correctly* — expressed as something a
dense network can be trained on.

### And this is why the rare-class problem dissolves

**Both counting functions are learned from the 97% of cases that are normal.** The model
never has to memorise 16 lumbar-rib examples in order to recognise one. It has to learn to
count from the sacrum and to count from the ribs, and it has ~800 cases of each. The
anomaly is then detected as a **violation of consistency between two things it learned
well**, not as a rare category it saw a handful of times.

This is the single strongest argument for the design and it should be the centre of the
talk. It converts anomaly *classification* into anomaly *detection by disagreement*, which
is a fundamentally better-posed learning problem when positives are scarce.

It also gives an uncertainty for free: `|n_up + n_down − constant|` is a per-case scalar
that is near zero in normal anatomy and near one at a transition. That is a **calibrated
flag**, not a softmax probability over a class that barely exists.

---

## The objection that breaks the naive version: the anchors move too

Two anchors are not enough, and the reason is the whole point of this dataset. **Every
anchor named so far is perturbed by exactly the variants it is meant to detect.**

- A **lumbar rib** moves the "lowest rib-bearing vertebra" down one.
- An **aplastic twelfth rib** moves it up one.
- A **sacralised** lowest lumbar segment moves the top of the sacrum up one.
- A **lumbarised** first sacral segment moves it down one.

And the failure is worse than it first appears: a lumbar rib and a sacralisation both reduce
`n_up + n_down` by one. **The sum cannot distinguish them.** A design resting on the sum
would flag both correctly and name both wrongly, half the time.

### The fix: choose anchors that fail to DIFFERENT variants

Anchors defined by *morphology* do not move when the *feature* moves. Four are available,
and what matters is that each variant family perturbs a different subset:

| anchor | how it is found | normal |
|---|---|---|
| **A. lowest costal-facet vertebra** | costal facets on the vertebral BODY — thoracic morphology | T12 |
| **B. lowest rib-bearing vertebra** | a rib articulates | T12 |
| **C. superior extent of the sacroiliac articulation** | the ilium's auricular surface | S1 |
| **D. sacral promontory / S1 endplate** | the sacral base | S1 |

Which move, under which variant:

| variant | A facets | B rib | C SI joint | D promontory |
|---|---|---|---|---|
| typical | T12 | T12 | S1 | S1 |
| **lumbar rib** | **T12** | **L1 ↓** | S1 | S1 |
| **aplastic/hypoplastic T12 rib** | **T12** | **T11 ↑** | S1 | S1 |
| **sacralisation** (Castellvi II) | T12 | T12 | **S1** | **↑ one** |
| **sacralisation** (Castellvi III–IV) | T12 | T12 | **↑ may move** | **↑ one** |
| **lumbarisation** | T12 | T12 | **↓ may move** | **↓ one** |

Three things fall out, and they are the design:

1. **A is the control for B.** A rib variant perturbs B and leaves A fixed, because an
   aplastic twelfth rib *leaves its costal facet behind* and a lumbar rib grows on a
   vertebra that never had one. So `A − B` is zero in typical anatomy, +1 for a lumbar rib,
   −1 for an aplastic twelfth. **The sign names the rib variant.**

2. **C is the control for D.** A Castellvi II sacralisation moves the promontory without
   moving the true sacroiliac articulation, because the auricular surface belongs to the
   *ilium* and is not relinquished when a lumbar segment fuses above it. So `C − D` is zero
   typically and non-zero at a sacral transition. In Castellvi III–IV the neo-articulation
   can drag C as well, which is itself informative: **C and D moving together is a marker of
   a higher Castellvi grade than D moving alone.**

3. **The two families are orthogonal.** Rib variants perturb {A, B} and leave {C, D}; sacral
   variants perturb {C, D} and leave {A, B}. So a case carrying *both* — a lumbar rib and a
   sacralisation, which this corpus contains — is decomposable, where a single-anchor design
   would see one net shift and be unable to say what caused it.

This is what makes the design robust to the variants rather than defeated by them: **no
anchor is individually reliable, and that is fine, because the pattern of which anchors
disagree is a signature and the signature is what identifies the variant.**

### A fifth anchor that does not survive contact with the data

Sacral foramina are the textbook sacrum-intrinsic anchor: four pairs is a normal sacrum,
five means it absorbed a lumbar segment. It would be an ideal anchor C, because it is a
property of the sacrum itself and needs no reference to the ilium.

It does not work here, and the corpus says so plainly. Counting foramina per side across
802 records gives, for cases labelled **normal**: 2 in 211 cases, 3 in 324, 4 in 163, 5 in
49, and a scatter from 1 to 10. The mode is three, not four, and the distribution overlaps
the labelled sacralisations (mode 3) and lumbarisations (mode 2) completely.

The measurement is dominated by field of view — many sacra are truncated inferiorly in an
abdominal CT, so the lower foramina are simply absent — and the automated count inherits
that. **An anchor whose distribution in normal anatomy overlaps the variants it should
detect is not an anchor**, and it is listed here so the next person does not spend a week
rediscovering it. Anchors C and D as defined above avoid this because both are read from the
sacral base and the iliac articulation, which sit at the top of the sacrum and are in the
field of view whenever the lumbar spine is.

### What it costs

C requires the sacroiliac articulation to be locatable, which needs the ilium in the
receptive field — reinforcing the patch problem below rather than easing it. And A requires
the network to see a costal facet, a feature a few millimetres across, which is the
argument against the coarse in-plane spacing that would let one patch cover the pelvis.

Both constraints point the same way: **the cascade**, where the low-resolution stage locates
C and D against the whole pelvis and the full-resolution stage resolves A at native detail.

---

## What each head can actually see, and why the anchors have to be in the patch

An ordinal count is not local. `n_up` at L1 requires the sacrum to be in the receptive
field, four vertebrae away — about **265 mm from T11 to the S1 endplate**, measured on this
corpus. This is where the patch geometry from `COUNTING_PRIOR_ART.md` becomes a design
constraint rather than a tuning knob:

- a 112 × 112 × 320 patch at 1 mm covers the whole column in **4.01 Mvox**, cheaper than
  nnU-Net's default 192³ at 7.08;
- the pelvis needs 309 mm of width at p95, which that patch cannot span;
- so either the cascade (low-res stage sees everything and hands its prediction to the
  full-res stage as a channel) or an anisotropic 160 × 160 × 320 at 2 mm in-plane — which
  blurs the 2–4 mm costotransverse joint, the very feature that separates a hypoplastic
  twelfth rib from a lumbar rib.

**The ordinal heads and the morphology classes want different patches.** That is the real
architectural tension in this problem, and naming it is more useful than pretending one
configuration serves both.

---

## Where the morphology comes in

The rib anchor fails when the twelfth rib is aplastic: there *is* no lowest rib, so
`n_down` has no origin. This is precisely the case Greg wants detected, and it is where
morphology carries the answer rather than counting.

Thoracic and lumbar vertebrae are **different bones**, and every discriminating feature is
patch-local:

- **costal facets (demifacets) on the body** — thoracic only. This is the decisive one: a
  T12 body carries a costal facet whether or not a rib grew from it. *An aplastic twelfth
  rib leaves its facet behind.*
- **the costotransverse joint** — a rib meets the process across a joint space; a
  transverse process is continuous with the pedicle.
- **process morphology** — thoracic processes short, thick, club-shaped, posterolateral;
  lumbar long, flat, blade-like, with mammillary and accessory processes.

So the third head is a per-vertebra **thoracic/lumbar type** prediction, learned from
morphology, and it repairs the rib anchor when the rib is missing. A vertebra with costal
facets and no rib is a T12 with an aplastic rib; a vertebra with no facets and a small rib
is L1 with a lumbar rib. **The vertebra names the rib, not the other way round.**

That is the answer to "how would it detect aplastic/hypoplastic T12 ribs" and it is
genuinely learnable, because the facet is present in every thoracic vertebra in the
training set — ~800 cases of it, not 16.

---

## Heads, losses, and what to weight

| head | type | loss | purpose |
|---|---|---|---|
| semantic | per-voxel classes | Dice + CE (nnU-Net default) | the segmentation |
| `n_up` | dense ordinal regression | L1 or ordinal-CE on a small integer range | count from the sacrum |
| `n_down` | dense ordinal regression | same | count from the ribs |
| type | per-voxel binary (region) | BCE + Dice | thoracic vs lumbar morphology |
| rib-bearing | per-voxel binary (region) | BCE + Dice | already built, `--rib_regions` |

Ordinal regression rather than plain regression: the targets are small integers with a
natural order, and rank-consistent formulations (CORAL/CORN-style, or simply a set of
cumulative binary "is it at least k above the sacrum" heads) respect that order where an L2
loss does not.

**Consistency loss.** The strongest supervisory signal is free: penalise
`|n_up + n_down − K|` where K is the case's own constant. In normal anatomy this is a
self-supervised constraint that holds on every case, so it can be applied to *all* the data,
including cases with no transitional label at all.

---

## Architecture, given four H200s

Four H200s at 141 GB each is enough to stop treating memory as the binding constraint,
which changes what is worth trying.

**Backbone.** `Primus` / `PrimusV2` (Wald, Roy, Isensee et al., TMLR 2025, arXiv
2503.01835), implemented inside nnU-Net. Their finding is the relevant one: most 3D
"Transformers" over-rely on convolutional blocks so heavily that *performance is unaffected
by removing the Transformer*. Primus is Transformer-centric with high-resolution tokens and
improved positional embeddings, and PrimusV2 matches ResEnc-L and MedNeXt across nine
datasets.

This matters here more than it does in general. *nnU-Net Revisited* (Isensee et al., MICCAI
2024) found CNNs beating Transformers under controlled comparison — but on organ
segmentation, where long-range ordering is not the task. Here it is the entire task. The
positional embeddings that Primus specifically improves are the mechanism by which a
network can know *which* vertebra it is looking at.

**Ordering of arms**, each with a reason:

1. **ResEnc-L, tall narrow patch (112 × 112 × 320), semantic heads only.** The honest
   baseline. Establishes what the column-spanning receptive field alone buys.
2. **The same plus the two ordinal heads and the consistency loss.** Isolates the
   contribution of the formulation from the contribution of the architecture. This is the
   comparison the talk turns on.
3. **`3d_cascade_fullres`.** Global context for the counting heads, native resolution for
   the morphology. The principled resolution of the patch tension.
4. **PrimusV2 on the winner.** The attention arm, with the argument above for why the
   generic anti-Transformer result does not transfer.
5. **The staged count-free pipeline** as the comparator that already exists.

Run 1 and 2 as a matched pair on the same folds. The delta between them is the paper.

---

## What to measure

Dice is not the endpoint and reporting it alone would hide the result.

1. **Rib-free count accuracy, reported separately for typical and transitional anatomy.**
   A model that is 97% accurate overall and 0% on transitional cases is the null result
   dressed up, and only the split reveals it.
2. **Anchor-agreement calibration.** Does `|n_up + n_down − K|` actually separate
   transitional from typical? An ROC on that scalar is the cleanest single figure the
   design can produce.
3. **Aplastic-twelfth-rib recall specifically**, since it is the case where the rib anchor
   fails and morphology has to carry it alone.
4. **The failure mode when it fails.** Off-by-one traceable to a specific vertebra, or a
   silent misnumbering. The whole argument for two anchors is that disagreement is visible.

---

## Prior art this rests on

**Set prediction with variable cardinality.** DETR and Mask2Former solve "an unknown number
of instances" with a fixed set of queries and **Hungarian bipartite matching** between
predictions and ground truth — the general form of the problem here. Mask2Former combines
cross-entropy, BCE and Dice weighted by matched and unmatched queries. Worth knowing as the
alternative formulation: queries as vertebra instances, matched by position. It is a
heavier lift than the ordinal heads and buys instance separation the disc-space class
already provides.

**Anatomy-conditioned queries.** *Anatomy-guided Pathology Segmentation* (MICCAI 2024)
decodes a joint feature space into query representations for **anatomy**, then interleaves
them into a pathology decoder — structurally what "attention toward the anchors" means, done
in a query-based transformer.

**SpatialConfiguration-Net** (Payer et al.) won VerSe-2020 with local appearance multiplied
by the *global joint configuration* of all vertebrae, after a U-Net centreline heatmap
localises the spine. The canonical statement that identity needs global configuration, and
the strongest baseline in this exact domain.

**Btrfly Net with an adversarial spine prior** (Sekuboyina et al.) learns the anatomical
constraint adversarially rather than coding it. The direct comparator for a learned-prior
approach.

**Graph optimisation with an anatomic consistency cycle** (Meng et al. 2022) — localisation,
segmentation and identification solved jointly under a self-consistency constraint. The
closest existing relative of the two-anchor consistency idea, in graph form rather than as
a dense loss.

**Message passing and recurrent shape-basis labelling** (Yang et al., arXiv 1705.05998) —
explicit sequence models over the vertebral chain.

**CoordConv** (Liu et al. 2018) and **Kayhan & van Gemert** (CVPR 2020), the latter showing
CNNs already leak absolute position through zero padding, so the translation invariance is
a leaky abstraction and a network may already be counting by a cue you cannot audit.

**LSTV specifically.** Automated LSTV detection on plain radiographs has been attempted with
ResNet-50, DINOv2 and CLIP backbones, with ResNet-50 best — a 2D classification framing,
and a low bar that a 3D segmentation-native method should clear comfortably. The clinical
literature is consistent that the failure mode matters: inconsistent LSTV identification
drives wrong-level surgery, and *assuming the lowest lumbar level is L5 is named as the
specific error*. That assumption is exactly what two anchors remove.

---

## Evidence from this corpus that both heads are needed

The rib-free count against the source transitional label, on all 802 records:

| rib-free count | labelled normal | labelled transitional |
|---:|---:|---:|
| 4 | 21 | 8 |
| 5 | 736 | **7** |
| 6 | 9 | 18 |

Two things follow, and both are design constraints rather than curiosities.

**A count of 4 or 6 is enriched but not decisive.** Twenty-one records with four rib-free
vertebrae carry no transitional label. Some of those are genuine misses in the source, which
is what `screen_missed_castellvi.py` exists to surface; some are counts that arose for
reasons that are not transitional anatomy. Either way a model that flags every non-five
count would be wrong more often than right.

**Seven labelled transitional cases have a perfectly normal count of five.** These are
morphological transitions — a Castellvi grade with no change in the number of rib-free
vertebrae — and the counting heads *cannot see them by construction*. Only the morphology
head can. That is direct evidence from this corpus that the thoracic/lumbar type prediction
is not a redundant extra: it is the only mechanism that reaches a fifth of the labelled
positives.

The converse also holds — the 8 transitional cases at count 4 and 18 at count 6 are where
the anchors do the work and morphology alone would be ambiguous. **Neither head suffices;
the design needs both, and the corpus says so numerically.**

---

## The honest limits

**L5 versus L6 remains undecidable from morphology.** They are the same bone under two
counts. The two-anchor design does not solve this — it makes the ambiguity *explicit and
measurable* by producing two counts that disagree, instead of silently picking the commoner
name. That is a better outcome than a confident wrong answer, and it should be presented as
the result rather than apologised for.

**16 lumbar-rib cases is still 16 cases** for anything that has to be learned specifically
about them. The claim is that most of what matters is not learned from them: the counting
functions come from the normal cases, and the costal facet comes from every thoracic
vertebra in the corpus.

**The consistency loss assumes K is knowable per case**, which it is in training and is not
at inference. At inference the disagreement is the output, not a constraint — the loss
shapes the two heads to be mutually consistent where the anatomy is typical, and their
divergence is then informative.
