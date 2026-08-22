# Getting a CNN to enumerate a variable number of things

Notes toward the LSTV segmentation model. The question: a convolution is
translation-*equivariant* by construction, so a feature that identifies "a vertebra" fires
identically wherever the vertebra is. Counting requires knowing *which* one it is, which is
a fact about position in a sequence. That mismatch is the whole difficulty, and it is not
solved by more data or a bigger backbone.

The literature answers it in four families. They are not exclusive and the interesting
designs combine them.

---

## Family 1 — break the invariance directly

**CoordConv** (Liu et al., NeurIPS 2018, *An intriguing failing of convolutional neural
networks and the CoordConv solution*). Append coordinate channels to the input so the
network can read absolute position. Trivially cheap and it dissolves tasks that plain
convolution provably cannot do.

**Kayhan & van Gemert** (CVPR 2020, *On Translation Invariance in CNNs: Convolutional
Layers Can Exploit Absolute Spatial Location*) is the important refinement: CNNs already
leak absolute position through **zero padding at the boundary**, and a one-layer FCN with
global max pooling can classify patch location. So the invariance is a leaky abstraction —
which means a network may already be counting by a cue you did not intend and cannot audit.
That is an argument for supplying position *explicitly* rather than hoping it is absent.

**Directly usable here, and cheapest thing on this page.** nnU-Net takes multi-channel
input natively. A second channel carrying a normalised coordinate — or better, a
*sacrum-relative* one, since the sacrum is the anatomical anchor the count starts from —
costs a `channel_names` entry and a preprocessing step, no architecture change:

```json
"channel_names": {"0": "CT", "1": "z_from_sacrum"}
```

The caveat is real: a coordinate channel invites the network to learn "the fifth vertebra
up is L1", which is the convention we are trying *not* to bake in. Sacrum-relative distance
in millimetres is safer than an index, because it is a continuous physical measurement
rather than an ordinal.

---

## Family 2 — local appearance times global configuration

**SpatialConfiguration-Net** (Payer et al.) is the canonical answer in this exact domain
and the backbone of strong VerSe entries. Two branches: one learns local appearance, the
other learns the joint spatial configuration of all landmarks, and their **product** is the
prediction. A vertebra that looks right but sits in the wrong place is suppressed. This is
the cleanest formal statement of "identity is not local" in the literature.

**Btrfly Net with an energy-based adversarial local spine prior** (Sekuboyina et al.,
arXiv 1804.01307, and *Radiology: AI* 2020). Labels vertebrae from two orthogonal maximum
intensity projections, with an adversary trained to reject anatomically impossible label
sequences. The prior is *learned* rather than coded — the opposite design choice from ours,
and the honest comparison for the talk.

**Message passing with sparsity regularisation** (Yang et al., arXiv 1705.05998) and the
**recurrent image-to-image network with a shape basis** (same group). Both add an explicit
sequence model over the vertebra chain on top of a per-voxel network.

**Graph optimisation with an anatomic consistency cycle** (Meng et al. 2022) —
localisation, segmentation, and identification solved jointly, with a consistency
constraint that a labelling must be self-consistent as a graph.

The pattern across all of these: **the CNN proposes locally, and a second, non-convolutional
mechanism disposes globally.** Ours is the same shape with the second mechanism written by
hand instead of learned.

---

## Family 3 — do not ask the network to count

What `docs/SEGMENTATION_ARCHITECTURE.md` argues for. The network says what is bone and
where one vertebra ends; counting, naming and flagging happen in deterministic code
downstream, where they have an audit trail.

The advantage over families 1 and 2 is not accuracy, it is **failure mode**. An off-by-one
from a learned prior is untraceable; an off-by-one from our counting stage points at a
specific missed rib or merged vertebra.

Relevant from the counting literature outside medicine: **Cheng et al.** (CVPR 2022,
*Rethinking Spatial Invariance of Convolutional Networks for Object Counting*) find that
density-regression counting improves when pixel-level spatial invariance is *relaxed*.
Their conclusion runs the same direction: strict invariance is the obstacle.

---

## Family 4 — reformulate the label so the thing you need is local

**This is the rib-bearing regions idea and it is a real contribution, not a variant of the
others.** Vertebral identity is not local. But *rib-bearing status* is: the rib is in the
image, articulating with the body, and a per-voxel loss can see it. So the one piece of the
counting problem that genuinely can be learned gets learned, and it happens to be the piece
the count is anchored on — the lowest rib-bearing vertebra is the top bracket of the
interval that defines the phenotype.

Expressed as nnU-Net **regions** rather than exclusive classes, because the two facts are
nested (rib-bearing ⊂ vertebra). Under a softmax, a vertebra the network is certain about
but whose rib status is genuinely ambiguous gets a diluted answer on *both*. Under
independent sigmoids it can say "certainly a vertebra, 0.5 that it bears a rib" — which is
the true state of belief at a transitional level, and exactly what a downstream counting
stage should consume.

**What it does not solve, stated plainly.** The hard case is a lumbar rib against a large
transverse process, and that *is* the Castellvi question. Reorganising the label does not
dissolve the ambiguity. What changes is that the model can express it as a calibrated
probability instead of being forced to resolve it, and a count assembled from probabilities
can carry its own uncertainty. That is the argument for expecting it to win, and it is a
claim to be measured.

Implemented: `--rib_regions` in `tools/convert_hf_to_nnunet.py`.

---

## nnU-Net-specific knobs worth using

**ResEnc presets.** `nnUNetResEncUNetMPlans` / `LPlans` / `XLPlans`. From *nnU-Net
Revisited* (Isensee et al., MICCAI 2024, arXiv 2404.09556): under controlled comparison,
CNN-based nnU-Net variants (vanilla, ResEnc, MedNeXt) **outperformed Transformer- and
Mamba-based networks**, and many published claims of beating nnU-Net did not survive
scrutiny. The practical reading: use ResEnc-L as the baseline and treat "we replaced the
backbone with a transformer" as a claim needing evidence, not a default.

**Region-based training.** Now wired up. Note the two gotchas that cost a run if missed:
`sort_keys=False` when writing `dataset.json` (nnU-Net reads the labels dict *in order*,
and alphabetical sorting would put `rib_bearing` before `vertebra` and paint the
encompassing region over the substructure), and the ignore label must be the highest
integer and absent from `regions_class_order`.

**Multi-channel input** for the coordinate channel above. No custom trainer needed.

**Custom trainer** (`nnUNetTrainer` subclass) if a counting head or an auxiliary loss on
rib-free count is wanted later. That is where family 2 would be added.

---

## What I would measure

Not Dice. Dice on vertebrae is high and uninformative. The comparison that decides this:

1. **rib-free count accuracy**, reported separately for typical and transitional anatomy —
   the gap between those two numbers is the result;
2. **calibration of the rib-bearing probability** at transitional levels, since the whole
   argument for regions is that the uncertainty becomes usable;
3. **the failure mode when it fails** — off-by-one traceable to a specific vertebra, or not.

Arms worth running, in increasing cost: exclusive count-free classes (built) → rib-bearing
regions (built) → regions plus a sacrum-relative coordinate channel → ResEnc-L on the best
of those.

---

## Sources

- Liu et al., *An intriguing failing of convolutional neural networks and the CoordConv
  solution*, NeurIPS 2018
- [Kayhan & van Gemert, *On Translation Invariance in CNNs*, CVPR 2020](https://jvgemert.github.io/pub/kayhanCVPR20translationInvarianceCNN.pdf)
- [Sekuboyina et al., *Btrfly Net: Vertebrae Labelling with Energy-based Adversarial Learning of Local Spine Prior*](https://arxiv.org/pdf/1804.01307)
- [Yang et al., *Automatic Vertebra Labeling in Large-Scale 3D CT using Deep Image-to-Image Network with Message Passing and Sparsity Regularization*](https://arxiv.org/pdf/1705.05998)
- [Cheng et al., *Rethinking Spatial Invariance of Convolutional Networks for Object Counting*, CVPR 2022](https://arxiv.org/pdf/2206.05253)
- [Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*, MICCAI 2024](https://papers.miccai.org/miccai-2024/562-Paper2847.html)
- [Meng et al., *Vertebrae localization, segmentation and identification using a graph optimization and an anatomic consistency cycle*](https://www.researchgate.net/publication/363735732)
- Payer et al., SpatialConfiguration-Net — integrating local appearance with the joint
  spatial configuration of landmarks
