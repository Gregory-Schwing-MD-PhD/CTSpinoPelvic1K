# What each measurement is, how it is computed, and what it is checked against

Every derived measure in this release follows the same contract: an automated method with
a named antecedent, and a published value it has to agree with. A measure that disagrees
is withheld with the reason recorded, not shipped with a caveat.

This document exists because "the number looks plausible" is not a check. Several of the
measures below produced perfectly plausible millimetres while measuring the wrong
structure, and only the comparison against prior art caught them.

---

## The rule, and what it has actually caught

| measure | first answer | why it was wrong | published |
|---|---|---|---|
| pelvic incidence | 154.6° | endplate fitted by top-decile-by-z | 54.7 ± 10.6 |
| sacral slope | 82.9° | reported the complement of the angle | 41.0 ± 8.4 |
| L1 trabecular HU | 264 | axis assumptions on a P,I,R volume | ~155 |
| lumbar disc height | 4–6 mm | measured rim to rim across concave endplates | 8–12 |
| transverse-process gap | 3.4 mm | measured the L5–S1 **facet joint** | — |
| disc above the lowest segment | 1.7–4.9 mm | measured a **facet joint** | 8–12 |
| femoral head diameter | 34–37 mm in men | sphere fitted to head **+ greater trochanter** | 48–52 |
| DISH prevalence | 0.5 / 48.9 / 48.5 / 23.3 % | withheld — see below | 3.8–27 |

The pattern in the middle three is one mistake wearing three costumes: **the minimum
distance between two bones is almost never the measurement you wanted.** Articulating
bones approach at several sites and a minimum finds the tightest, which is usually a joint
nobody asked about. A distance needs its site named — a region of one bone, a region of
the other — before it means anything anatomical.

---

## Spinopelvic

**Pelvic incidence, sacral slope, pelvic tilt.** Sacral endplate fitted as a surface;
femoral head centres from the femur labels. Checked against Vialle et al. 2005 (*JBJS*),
an asymptomatic standing series of 260: PI 54.7 ± 10.6, SS 41.0 ± 8.4, PT 13 ± 6.

This cohort is **supine**, so SS reads 4.1° low and PT 4.3° high. That is one fact and not
two: `PI = PT + SS` holds by construction, so with PI fixed a fall in SS must appear as an
equal rise in PT. The offsets matching in magnitude is evidence the geometry is internally
consistent.

**Lumbar lordosis, PI − LL mismatch.** LL is measured only where the arc reaches the top
of the lumbar segment, because a field of view clipping the upper lumbar spine returns a
smaller angle than the patient has and that error would land in the mismatch. PI − LL
sitting near zero across an unoperated cohort is the strongest available check on the two
angles: they come from entirely different structures and still agree.

---

## Level-by-level geometry

**Endplate width, body height, canal width, transverse-process span.** Published lumbar
norms: endplate ~41.8 mm at L1 rising caudally; anterior body height 29.9–34.5 mm L1–L5;
canal transverse diameter ~22.0 mm at L1 to ~26.5 at L5; TP span ~68 mm at L1.

**Vertebral wedging.** Anterior and posterior body height, each as the tallest column of
its half of a 10 mm mid-sagittal slab. The maximum rather than the mean, because a
biconcave endplate makes the middle of a body shorter than either wall, and published
anterior height is measured at the anterior **cortex**.

Graded against **Genant's semiquantitative method** (Genant et al., *JBMR* 1993): grade 1
is 20–25% height loss, grade 2 is 25–40%, grade 3 is over 40%. A ratio below 0.80 is
grade 1 or worse.

*On comparing prevalence.* Published vertebral-fracture prevalence of 5–10% in this age
band is a **whole-spine** figure, and most osteoporotic fractures sit at the thoracolumbar
junction. A prevalence computed over L1–L5 alone is therefore not comparable to it, and it
read low: 1.0%. T12 is labelled in 767 of 802 cases and T11 in most, so the junction is
measured rather than the gap explained away, and the data then say plainly where the
wedging is:

| level | n | median ratio | below 0.80 |
|---|---:|---:|---:|
| T11 | 732 | 0.943 | **11** |
| T12 | 743 | 0.944 | **5** |
| L1 | 739 | 0.971 | 3 |
| L2 | 722 | 1.000 | 0 |
| L3 | 758 | 1.026 | 0 |
| L4 | 771 | 1.054 | 2 |
| L5 | 744 | 1.056 | 2 |

Sixteen of the 23 affected vertebrae are at T11–T12 and the mid-lumbar spine has none at
all, which is the distribution the fracture literature describes. Per-person Genant grade
1+ rises from 1.0% to **2.9%**. It remains under the published band, and the two reasons
are visible rather than assumed: published series usually include T7–T10, which these
fields of view do not reach, and this is an asymptomatic screening cohort rather than an
osteoporosis series.

Note also that the median ratio at T11 and T12 sits *below* 1.0 by construction — thoracic
vertebrae are wedged anteriorly, which is what makes a kyphosis. Genant's criterion is a
within-vertebra height loss and so is unaffected, but a reader comparing level medians
should not read 0.943 at T11 as pathology.

The other per-level measures here are validated against *lumbar* norms, which do not
transfer to thoracic levels; their T11 and T12 columns are written but are not meant to be
used.

*A guard, and it took three attempts and a picture to get right.* Everything here rests on
the anterior wall of the canal correctly dividing body from posterior elements. That wall
is found as the largest filled hole in an axial slice, and when the largest hole is a
trabecular void near the front of the body rather than the spinal canal, the wall lands
almost at the anterior margin — and what survives the cut is a thin anterior **sliver** of
the vertebra. A sliver tapers, so its tallest column at the back exceeds its tallest column
at the front, and the ratio of the two reads 0.27 to 0.54: the exact signature of a severe
wedge, on a vertebra with nothing wrong with it.

Two wrong diagnoses preceded the right one, both reasoned from the numbers alone. First
that posterior elements were inflating the posterior height — false; those cases' posterior
heights sit right on their neighbours'. Then that these were genuine grade 3 fractures,
which do occur at about a percent in this age band. Rendering the mask against the CT
settled it in one image.

Worse, the first *guard* was actively harmful: it asked whether each half's tallest column
sat near that half's outer wall, which in a genuinely wedged vertebra it does not. It
rejected 29% of all levels and took Genant grade 1+ prevalence from 3.2% to 0.3%. **A guard
that correlates with the finding is worse than no guard.**

What ships tests the mask itself rather than any number derived from it: a vertebral body
is a substantial part of a vertebra's front-to-back extent, not a rind on it. Anything
under 12 mm deep, or under 30% of the vertebra's depth, is the cut having failed. That
withholds 1.2% of levels, takes the minimum ratio from 0.273 to 0.592, and takes the
distribution's skew from −2.68 to +0.46 and its excess kurtosis from 13.0 to 4.4. It also
halved the endplate widths falling below a physiological floor, which were the same failure
seen through a different measurement.

---

## Opportunistic screening

**L1 trabecular attenuation.** Eroded core of the vertebral body, so cortex and
osteophytes cannot enter the region. Threshold from **Pickhardt et al.** (*Ann Intern Med*
2013): 110 HU as a screen for osteoporosis on non-contrast abdominal CT, which is the
whole premise of measuring it on a scan taken for the colon.

**Femoral neck attenuation.** Same eroded-core method at the site the fracture that
matters actually happens. Reported as a distribution with **no cut-point drawn**, because
the hip has no validated equivalent of the L1 threshold.

*Open, and deliberately not patched.* Thirteen of 800 cases return 700 to 1368 HU, which is
cortical bone or metal and not trabecular anything. The envelope check (§2.0 of the release
checklist) flags them and they are left in the CSV. The likely explanation is a hip
prosthesis that carries no hardware label — the hardware classes are declared and
unpopulated — and it is cheap to test by looking at whether the whole femur is bright on
those cases or only the neck region. Clamping the value to a plausible ceiling would hide
the thing worth knowing, which is that some of these femora may not be bone.

---

## Proximal femur

**Head diameter.** A sphere fit — the standard construct in hip morphometry and in
femoroacetabular-impingement work, where the head is treated as spherical and the
departure from it is the finding.

Seeded on **acetabular contact** rather than on a height percentile: the acetabulum wraps
the head and nothing else, so femur voxels near the hip bone are head surface by
construction. Refined by **least-trimmed-squares** — fit, discard the worst residuals by
rank, refit — because the opening fit need not be right and a band drawn around a wrong
sphere keeps the points that put it there. Fits are rejected unless the residual RMS says
the cloud was actually spherical.

**Hip axis length.** Faulkner et al. (*JBMR* 1993): along the neck axis from the greater
trochanter to the inner pelvic brim. It predicts hip fracture **independently of bone
mineral density**, which is why it is extracted rather than inferred.

**Neck-shaft angle.** Published 125–135°; below 120 coxa vara, above 135 coxa valga. The
sex difference is small and inconsistent in the literature and is small here too — a
measure that agrees with prior art about what does *not* differ is evidence the pipeline
is not manufacturing differences.

---

## Degeneration

**Disc height.** Per-column, then the median. Endplates are concave, so the lowest voxel
of the upper body anywhere in a region is its **rim** projecting into the space;
rim-to-rim measures the narrowest part rather than the height. Column-wise gives the
height a radiologist reads off a mid-sagittal slice, and moved the numbers from 4–6 mm to
8.8–10.4 against a published 8–12.

The column bundle is centred on the **interface** — the inferior surface of the upper bone
— not on the two masks pooled. Two adjacent vertebrae are nearly coaxial so pooling works
for them; a lumbar vertebra and the sacrum are not, and pooling dragged the bundle off the
lumbosacral disc entirely.

**Vacuum phenomenon.** Voxels below −150 HU inside the disc space. Gas sits near −1000 HU
where a disc is near +50, and there is no normal variant that puts air inside a disc, so
this is among the least ambiguous measurements here.

**DISH — withheld.** Resnick's criteria require flowing ossification bridging four
contiguous vertebrae. Four detectors were built; the fourth landed at 23.3% against a
published 3.8–27% and was rejected anyway, because prevalence fell with age and DISH only
accumulates. Full reasoning in the header of `scripts/extract_degenerative.py`.

---

## Transitional anatomy

**Castellvi's classification** (Castellvi et al., *Spine* 1984) grades the lowest lumbar
transverse process against the sacral ala: Type I is an enlarged process of at least
19 mm, II a pseudo-articulation, III bony fusion, IV mixed; a and b are unilateral and
bilateral.

So the quantity is the **craniocaudal height** of the process, plus its distance to the
ala. The height was not being measured at all until it was added; it separates cleanly
(18.4 mm median in unlabelled cases against 25.6 in labelled ones). The gap is measured
from the process **tip**, identified as the lateral extreme, because the whole lateral
third of the vertebra contains the inferior articular process and returns the facet cleft.

**The dataset contains no Castellvi grades.** The 33 labelled cases carry an LSTV label —
sacralization, lumbarization — which describes a *count* where Castellvi describes a
*morphology*. The two are not interchangeable. Grading those cases is task A1 in
`COLLABORATOR_TASKS.md` and blocks anything that claims to be a Castellvi screen.

**Screening for unrecorded cases** follows the positive-unlabelled framing of Elkan and
Noto (2008) and Bekker and Davis (2020): unlabelled is not negative, so the output is a
ranking and a re-read queue rather than a classification. Ranked by **Fisher's linear
discriminant**, not by distance to the positive centroid — the latter put known positives
at median rank 708 of 767, because distance to a centroid measures typicality and a rare
phenotype is by construction not typical.

---

## Statistics and display

**Proportions** carry Wilson score intervals, which stay inside the unit interval at the
single-digit counts the transitional subgroups have. **Medians** carry the notched-boxplot
interval, ±1.58·IQR/√n, which assumes nothing about shape.

**Densities** use Silverman's rule with a robust scale — the smaller of the SD and
IQR/1.34 — because on bimodal data the plain SD is inflated by the separation between the
modes and returns a bandwidth wide enough to merge them. The rule is trimmed 15% at large
n so the genuinely bimodal rib-ratio panel keeps its two modes, and **reverts to Silverman
below n = 100 and widens below 50**: undersmoothing is only earned by sample size, and at
n = 30 it draws whatever bumps the sample contained.

A **bandwidth floor at the measurement's own resolution**, estimated from the modal gap
between adjacent distinct values. Several of these measures are a voxel count times a
spacing and can only land on a comb; a kernel narrower than that comb draws the teeth, and
the teeth are the grid rather than the anatomy.

Every curve carries a pointwise 95% band, Var[f(x)] ≈ f(x)·R(K)/(nh) in closed form. It is
what separates a real second mode from a wobble, and it is why the bone-density ridges for
the oldest decade are not presented as bimodal.

**Age is drawn as bars, not a curve.** 48.9% of recorded ages end in a zero against 10%
expected; the Whipple-type index over 51–80 is 227, where the UN scale calls anything above
175 "very rough". The spike at 89 is the HIPAA Safe Harbor ceiling, which requires ages
above 89 to be aggregated. Roughly half these ages are rounded to the decade and which half
cannot be told. Whipple (1919) and Myers (1940) are the standard instruments for exactly
this question in census work.

The consequence was checked rather than assumed: age heaping attenuates a slope against
age by about 5%, because the true spread (SD 9.6 yr) is wide next to a rounding of at most
ten years applied to half the sample. Age-slope conclusions survive. It is reported anyway,
because a null result on a noisy predictor is exactly what attenuation can manufacture.
