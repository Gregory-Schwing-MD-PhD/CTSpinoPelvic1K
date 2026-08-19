# Lab journal — CTSpinoPelvic1K transitional anatomy

Newest entry first. Each entry records what changed, what was decided and why, what the
numbers were at that moment, and what is still owed. Negative results are recorded with
the same weight as positive ones — several of the decisions below were made *because* a
method failed, and the failure is the reason.

---

## 2026-08-19 — the naming decision, and what survives it

### The decision that reframes the project

Transitional vertebrae cannot be named from these scans, and the project has stopped
trying.

**Sacralization and lumbarization are not two anatomies.** They are one anatomy under two
counts: a transitional vertebra at the lumbosacral junction is "sacralized L5" if five
lumbar vertebrae are counted above it and "lumbarized S1" if six are. The bone is
identical; only the count differs. The same holds at the other end — a short rib-bearing
vertebra at the thoracolumbar junction is "a lumbar rib on L1" or "a hypoplastic 12th rib
on T12" depending entirely on what the vertebra above it is called.

Naming therefore requires an unambiguous count from a fixed landmark. Counting up from the
sacrum is circular when the sacrum is the structure in question; counting down requires
C2. Neither is available in a spine- or abdomen-limited field of view.

This was confirmed independently by radiology mentorship (G. Kilic), who raised two
objections that turn out to be the same objection: that characterizing transitional
vertebrae properly needs thorax-abdomen-pelvis CT so counting can start at T1, and that
rudimentary or absent 12th ribs make T12 unidentifiable. The second is the sharper point,
because it means even a full T1-down count can fail at the thoracolumbar junction. That
argues for count-free phenotypes generally, not merely as a concession to a small field of
view.

**What the manuscript reports instead** — measures that require no level assignment:

- transitional morphology at the lumbosacral junction (Castellvi-style lateral span and
  gap to the ala, per side; the asymmetry is the phenotype)
- rib-length ratio at the thoracolumbar junction (rib 12 against rib 11)
- the number of non-rib-bearing vertebrae **between** the lowest rib and the sacrum — an
  *interval* count rather than an absolute level assignment, valid whenever both endpoints
  are in the field

The headline result never depended on naming, which is why it survives intact.

### The labelling convention

The label format forces a name onto every structure, so "unknown" is not expressible and
some convention must be adopted. The one in force:

> **The inherited CTSpine1K vertebra labels are the reference. A rib is named for the
> vertebra it articulates with. A rib on a lumbar vertebra is a lumbar rib.**

Stated as a convention, not a truth claim. It is explicit, uniform and documented, which
the alternative was not: TotalSegmentator numbered ribs from the bottom in 13-rib spines
and from the top elsewhere, and that inconsistency is what produced 109 mismatched ribs.

Consistency was chosen over per-case correctness deliberately. Thirteen cases share one
morphology; labelling twelve of them one way and one another way teaches a model noise.

### Results as they stand

**802 densified cases · 33 carrying an LSTV label.**

*The main finding.* Thoracolumbar and lumbosacral border variants co-occur:

| | typical LS | variant LS |
|---|---|---|
| **typical TL** | 667 | 25 |
| **variant TL** | 60 | 44 |

44 observed against 9.0 expected under independence; **odds ratio 19.6** over 796 cases
with complete measurements. This is an association between two morphological findings and
required no level naming to state.

*A bimodal 12th rib.* Rib 12 length as a fraction of rib 11 splits into two modes — a main
population near 0.68 and a distinct hypoplastic one near 0.32, with 67 cases below 0.30.
Two distributions rather than one with a tail, which is the shape a discrete developmental
variant should have.

*LSTV cases avoid five free lumbar vertebrae.* The cohort piles at five (734 cases);
labelled LSTV cases sit at four (35) and six (26).

*Which measures carry the phenotype* (standardised difference against the LSTV label):
`has_l6_label` 4.7 · `n_non_rib_bearing` 0.8 · `rib12_len_left_mm` 0.55. The last is a
thoracolumbar measurement separating a lumbosacral label — the co-occurrence appearing a
second time, in a measure that knows nothing about the sacrum.

### Rib numbering, finished

| | before | after |
|---|---|---|
| ribs on the wrong vertebra | 109 | 19 |
| misnumbered cases | 16 | 2 |
| lumbar ribs mislabelled "rib 12" | 14 | 1 |
| the `+1` offset family | 34 | 1 |

Cause: TotalSegmentator numbered these cages from the bottom, so a 13th rib on L1 consumed
the id the T12 rib needed. Giving it its own class (`rib_left_lumbar` / `rib_right_lumbar`,
ids 74/75 — present in `label_scheme.py` but never applied until now) freed the sequence,
after which each rib could be named for the vertebra it articulates with.

Three of the 19 residual offsets are permanent by design: a floating 12th rib whose head
sits nearer T11 while rib 11 is present and correct. The label is right and the proximity
metric is fooled. That is a sentence in the methods, not a defect.

### Negative results

**Sacral foramina counting does not work, at any parameter setting.** Four pairs is a
normal sacrum, five means L5 was assimilated, three means S1 failed to fuse — so a single
count would have separated both directions of transition without trusting any vertebra
label. Two hand-tuned implementations failed, so the whole parameter space was mapped
instead: 93 cases (all 33 LSTV-labelled plus 60 normals) against 81 settings of projection
slab, morphological closing, minimum hole area and CT darkness threshold.

Best setting reached 31–39% per-case accuracy (always answering "4" scores 17%), and the
group distributions overlap almost completely — normals span 2–8, sacralizations 3–6,
lumbarizations 2–6. Normals do not even peak at four; their mode is three. No cut in that
space sorts the groups.

Recorded because tuning alone would have found a 38.7% setting and called it a screen. The
sweep is what made the claim falsifiable.

**Castellvi proxies are flat** (|d| < 0.2). Either a true negative or a bad proxy, and the
current data cannot distinguish the two.

### Data defects found

- **Vertebra label speckle.** Case 0344's L5 label is shattered into 112 connected
  components, the largest holding 99.3% of the voxels. Small fragments of L3, L4 and L5
  sit at z≈660, up among ribs 6 and 7, which is what produced the impossible
  "left rib 6 articulates with L3" flag. The ribs there are clean; the vertebra labels are
  not. A dataset-wide scan for this is outstanding.
- **16 cases have no thoracic vertebra labelled at all** — `0033 0068 0167 0241 0344 0357
  0383 0389 0409 0424 0428 0696 0720 0730 1004 1106`. Under the naming convention their
  ribs are simply unnameable. Three of them were on the residual-defect list for exactly
  this reason.
- **6 cases where no rib matches any vertebra** — `0068 0167 0409 0424 0696 0816`.
- Thoracic coverage is thin throughout: the modal case has 4 thoracic bodies labelled and
  only 34 of 802 have 7 or more. This is why rib-bearing-vertebra counts were unusable as
  a variant criterion.

### Required to complete the manuscript

**Blocking — can change what is written**

1. **Test the FOV-truncation confound on rib lengths.** The rib 12/11 ratio drives both the
   bimodality and the co-occurrence odds ratio, and a rib clipped by the scan edge is short
   for reasons unrelated to hypoplasia. The per-rib truncation flag exists in
   `qc_rib_vertebra_incidence` and is not carried into the morphometrics. Re-run the
   bimodality restricted to cases where ribs 11 and 12 are both untruncated. If the second
   mode survives it is anatomy; if it collapses, the headline was framing.

**Statistics**

2. Fisher's exact test and a confidence interval on the odds ratio of 19.6.
3. Sensitivity analysis over the variant thresholds (`rib12/11 < 0.30`, `lowest rib-bearing
   ≠ T12`, `n_non_rib_bearing ≠ 5`) — the OR depends on them and the stability should be
   shown, not assumed.
4. FDR adjustment on the 14-measure effect ranking, or explicit framing as exploratory.
5. Stratification by sex, age, scanner/kernel and acquisition `config`.

**Validation**

6. Expert re-read of a stratified sample (~50 cases spanning the four quadrants of the 2×2),
   blind to the automated call.
7. Resolve what `lstv_agreement` actually encodes across all 802. In the cases inspected it
   is `False` for every non-normal label and `True` for every normal one, which may reflect
   genuine difficulty or an artifact of how the field was populated. It changes what the 33
   LSTV labels mean.

**Data and release**

8. Add a `count_ambiguous` flag wherever `lstv_pelvic` and `lstv_vertebral` disagree or
   `lstv_confusion_zone` is set — recording the label as convention rather than truth, in
   the data, where a downstream user will see it. This is also the count-ambiguous category
   the manuscript needs.
9. Thoracic vertebra annotation for the 16 zero-thoracic cases (student task; the numbering
   must count up from L1, not follow an AI segmenter's own numbering, or the off-by-one
   returns).
10. Rewrite the figure's language to count-free phrasing — it still says "Sacralization,
    disentangled" and labels axes in naming terms.
11. Ship v5: choose the HuggingFace namespace, document classes 74/75 in
    `dataset_labels.json`, `dataset_interface.py` and the README (they are live and
    downstream code will silently drop them), confirm the 5-fold splits still hold, rotate
    the `anonymous-mlhc` token.
12. Literature: LSTV numbering error and wrong-level surgery is an established topic; the
    count-dependence argument above should be cited rather than re-derived.

### Where the label sources come from

Worth recording because it caused an hour of confusion. `lstv_label` inherits from
`lstv_pelvic`, which is parsed from the **CTPelvic1K mask filename** — that dataset's
authors encoded the variant in the filename itself
(`..._0006_4_260_sacralization_mask_4label.nii.gz`). It is a human annotation from the
source dataset. `prov_pelvis: pseudo` describes how the pelvic *segmentation mask* was
produced in this pipeline and is unrelated; conflating the two led to briefly and wrongly
dismissing the label as pseudo-derived.

`lstv_vertebral` comes from the spine side and is count-based. The two disagree precisely
where the count is ambiguous — which is the population of interest, so the disagreement is
signal rather than corruption.

### Next study

A retrospective analysis using trauma CT (thorax-abdomen-pelvis), where counting can begin
at T1 and levels can be named. The surgery department's trauma database is the intended
source.
