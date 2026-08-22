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

*A guard, added because the measure failed loudly enough to notice.* Everything here rests
on the anterior wall of the canal correctly dividing body from posterior elements. When
that detection fails, pedicles and articular processes stay in the mask and do two things
at once — they are taller than the body, inflating the posterior height, and they push the
halfway point backwards so the "anterior" half lands on the biconcave middle, deflating
it. Both errors drive the ratio down together and the result is indistinguishable from a
severe compression fracture. The tell is *where* each maximum sits: the anterior cortex is
at the front of the mask and the posterior wall at the back, so a maximum far from its own
outer wall means the mask is not a vertebral body. Those levels are withheld and counted.

---

## Opportunistic screening

**L1 trabecular attenuation.** Eroded core of the vertebral body, so cortex and
osteophytes cannot enter the region. Threshold from **Pickhardt et al.** (*Ann Intern Med*
2013): 110 HU as a screen for osteoporosis on non-contrast abdominal CT, which is the
whole premise of measuring it on a scan taken for the colon.

**Femoral neck attenuation.** Same eroded-core method at the site the fracture that
matters actually happens. Reported as a distribution with **no cut-point drawn**, because
the hip has no validated equivalent of the L1 threshold.

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
