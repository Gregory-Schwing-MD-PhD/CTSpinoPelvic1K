# The spinopelvic estimator, and four things that were wrong with it

This note records how the release's pelvic incidence, sacral slope, pelvic tilt and lumbar
lordosis are computed, and — because each was found by a different kind of check — what was
wrong before. The short version is that the numbers in the manuscript before 2026-09-10 were
computed from a different label set than the one shipped, by code with three independent
defects, and that none of the defects announced itself.

## What the release actually differs by

Every `morphometrics/*.csv` was originally computed from `data/v5_final`. The released
labels are the newer set. Regenerating everything against them shows the difference is
confined to one place: `level_gradients.csv` comes back **bit-for-bit identical**, every
value and every n, so the vertebrae and the canal are untouched. **S1 is carved as its own
label (29) in the release and was not in v5_final**, and that single change moved three
measurements, because three estimators reached for the sacrum when S1 was absent.

## 1. The label scheme was assumed rather than detected

`ostk.labels` held one map, the release uses another. "S1" resolved to id 7, which in a
released volume is **C7**; "femur_left" to id 11, which is **T4**. Nothing raised. Every
case returned a QC flag saying S1 had too few voxels while S1 sat in the volume with two
hundred thousand, so it read as a segmentation problem rather than a lookup one.

Fixed: `ostk.labels.labels_for(volume)` detects the scheme from pelvis voxel mass under each
hypothesis and **raises** when the two are close rather than guessing. Threaded through
every path that reads a volume, once per case. Tests in `ostk/tests/test_labels.py`.

## 2. The S1 endplate plane was fitted two different wrong ways

**Without an S1 carve** (v5_final), the plate was fitted to the *whole sacrum*, which the
alae flatten: the normal comes back near vertical, the slope near zero, and pelvic incidence
too low — but stable-looking. Median PI 54.7, 3.4% anatomically impossible.

**With the S1 carve** (the release), the same code switched branch and fitted the S1 label —
and latched onto the near-vertical anterior face of the **promontory** instead of the
superior plate. Normal z-component 0.03–0.44, i.e. a sacral slope near 88°, on **9.4%** of
records. Neither definition is right, so this was never a choice between them.

The fix is ostk's, and it needed a third correction of its own. `fit_endplate(method=
"corner")` builds the normal from the anterior-to-posterior cortical corner line — correct
on a vertebral body, wrong on the sacral base, where the promontory is a rounded lip and the
posterior rim blends into the canal. Sweeping its `drop_post` parameter over one case gives
sacral slopes of 37.2, 37.2, 5.2, 3.8, 7.8, 7.2°: the parameter decides the answer, not the
anatomy. The **surface** fit (TLS with MAD trimming) is stable, and `S1` and `sacrum` now use
it while vertebrae keep the corner method.

Only the plate's *orientation* comes from the surface fit. The point PI and PT are measured
**from** is still the corner midpoint, because that is the operational definition PI's
published norms were calibrated against, and it lies on the same rim line either way.

## 3. Two wrong numbers that summed correctly

The corner fit on S1 under-read **sacral slope by about 9°** and **lumbar lordosis — which
shares the S1 plate — by about 12°**, on every case, while PI stayed close enough to look
reasonable and the geometric identity PI = SS + PT held throughout.

This is worth stating plainly: **an identity check cannot catch a consistent error in a
shared landmark.** Both quantities moved together and the sum was preserved. It was found
only by comparing against an outside reference — the release's own independent extraction
code (SS 35.8°) and published automated supine CT (36.5°, Veilleux et al., *JBJS Am*
2020;102:e130, n=200) — against ostk's 24.8°. Every check in this note that actually caught
something compared against something outside the estimator.

## 4. Flags were dropped for exactly the cases that needed them

`spinopelvic_summary_from_label` attached a parameter's QC flags only when that parameter
came back `None`. But a geometry that violates the identity, or returns a pelvic incidence no
pelvis has, still produces a finite number — so the cases the guards exist to catch arrived
unflagged and indistinguishable from good ones.

## What the toolkit does now

PI and SS are computed from their own geometry — plate normal against the radius, normal
against the vertical — and PT is signed by construction as PI − SS. The check that remains
is PT's *magnitude*, measured independently from the radius against the vertical, against
that derived value; the PI = SS + PT identity is now true by construction and tests
nothing, which is the honest description of what it was mostly doing anyway. The sagittal
plane is derived from the two femoral head centres rather than assumed from the scanner
axes. Each femoral head centre is a
least-squares sphere fitted to its articular surface, seeded from the acetabular interface
and grown through the neck, which is the method with the best published repeatability
(0.20 mm inter-operator, Renault et al., *J Biomech* 2018;80:171; 0.5 mm hip-centre
agreement in STAPLE, Modenese & Renault, *J Biomech* 2021;116:110186). PI is taken to the
**midpoint** of the bicoxofemoral axis, which is the published definition (Legaye et al.,
*Eur Spine J* 1998;7:99) and what every 3-D CT implementation uses.

Plausibility gates, set at roughly published mean ± 4–5 SD so an unusual patient survives
them, are in `ostk/metrics.py` with their sources. A case failing a gate is reported with
its reason, never as a bare number.

None of that is sufficient, as the next section shows. The toolkit remains the better
*description* of how these measurements should be made; it is not yet the better
*measurement*, and the release reports the one that measures better.

## Which estimator the release actually reports, and why it is not the elaborate one

Measured head-to-head over the same 300 records, **the release's own extraction code beats
the toolkit's more principled estimator**, and the released numbers are therefore the
extraction code's:

| | median | IQR | outside a plausible range |
|---|---|---|---|
| PI, extraction code | **49.2** | 39.4–58.2 | **6.3%** |
| PI, toolkit | 45.8 | 25.5–54.6 | 16.0% |
| SS, extraction code | **35.9** | 28.1–49.9 | 3.3% |
| SS, toolkit | 32.3 | 25.0–40.0 | 2.7% |

Published supine CT is PI 47.1–52.1 and SS 36.5, so the extraction code is closer on both.

The toolkit's failure is worth naming precisely, because it does not look like a failure.
Its S1 surface fit is **bimodal**: on the full 802, 28.3% of cases land on a steeper
surface and return a pelvic incidence clustered near 18.6° with a sacral slope near 42°,
while the rest land on a flatter one and return PI near 50° with SS near 29°. A population
does not have two pelvic incidences. Both groups individually look defensible — the failing
group's sacral slope is actually *closer* to the published value than the good group's —
which is why the mean of the two looked reasonable and why nothing internal caught it.

Those 28.3% arrive carrying `identity_violation`, and the arithmetic in every one of them
is `PI = |SS − PT|` rather than `PI = SS + PT`. That signature has an innocent explanation
— an anteverted pelvis, where the true tilt is negative and an unsigned angle discards the
sign — and taking that explanation at face value would have been wrong. Looking at the
distribution instead settles it: the negative values cluster tightly at −20 to −30° with
PI near 18°, which is not anteversion, it is a second surface.

Pelvic tilt is signed now regardless, because that is independently correct: a negative
tilt is an anteverted pelvis and the SRS-Schwab PT modifier is defined on the signed value.

## What the release does instead

Gate on the **geometry, not the answer**. A sacral plate whose fitted normal lies more than
60° off the cranial axis is not a plate — it is the anterior face of the promontory — so the
case is reported missing with a reason (`s1_plate_rejected`, with the measured tilt in
`s1_plate_tilt_deg`). Rejecting a pelvic incidence for being an unlikely *number* would
discard unusual patients along with broken fits; rejecting a plate that cannot be a plate
discards only the latter, whatever it implies. The bound is deliberately loose: a sacral
slope of 60° is steep but real, 88° is not.

## Reference values, and why the modality matters

The cohort is supine CT. CT reads lower than standing radiography in the same subjects — 53°
against 56° of pelvic incidence (Lee & Liu, *Eur Spine J* 2022;31:241) — so comparing against
radiographic norms alone builds in an offset that has nothing to do with this dataset.

| | supine CT | standing XR |
|---|---|---|
| PI | 47.1 ± 10.0 (Vrtovec 2012, n=370); 44.97 ± 8.52 (Chen 2019, n=320); 52.05 (Veilleux 2020, n=200) | 55 ± 10.6 (Vialle 2005, n=300) |
| SS | 36.49 (Veilleux 2020) | 41 ± 8.4 (Vialle 2005) |
| PT | 15.60 (Veilleux 2020) | 13 ± 6 (Vialle 2005) |

## 5. Sacral slope read low, and the tidy explanation was wrong again

Against Veilleux's automated supine CT (n=200) this cohort reads **sacral slope 3.4 lower,
pelvic tilt 4.0 higher, pelvic incidence 0.5 higher**. That is not three independent
discrepancies. It is the exact signature of a rotated vertical reference: a rotation of
delta moves SS by -delta and PT by +delta and leaves PI untouched, which Ohashi et al.
(*Spine Surg Relat Res* 2024;8:61) write out as `aSS = SS - APPA` and `aPT = PT + APPA`.

So the obvious reading was that Veilleux references the **anterior pelvic plane** and this
code references the scanner axis. The evidence for that part is decent: Veilleux's abstract
lists ASIS and pubic tubercles as landmarks, which have no role in a scanner-axis slope; the
companion paper on the same 200 subjects (Higgins et al., *JBJS Am* 2014;96:1776) says the
frame is "an automatically identified anterior pelvic plane reference frame"; and Veilleux
reports a sacropubic angle, which is APP-referenced by definition. Both conventions are
published — Hatem et al. (*Orthop J Sports Med* 2025, n=3,695 supine CT) align to the APP,
Xu et al. (*Sci Rep* 2024;14:21453) use the image z axis, as here.

**It still does not work, and the reason is the sign.** Recumbency rotates the pelvis
*anteriorly*, which RAISES sacral slope and lowers tilt: +0.9 degrees in 211 patients
(Banitalebi, *Clin Spine Surg* 2026;39:E104), +3.9 in 15 volunteers (Chevillotte, *OTSR*
2018;104:565), +7.1 in 24 (Hasegawa, below). Supine APP tilt is anteriorly directed in most
reported cohorts (+5.1 median in 422 pre-THA hips, Uemura; +8.6 to +12.2 in Jenkinson), so
an APP-framed slope should land *below* a scanner-framed one, not above. To explain this gap
by frame alone you need APPA ~ -3.4 degrees, which contradicts those series.

**What does explain it is the cohort.** Veilleux's 200 were asymptomatic subjects imaged for
non-musculoskeletal reasons. This is an abdominopelvic CT population: older, symptomatic, on
a table with knee bolsters. Hasegawa et al. (*BMC Musculoskelet Disord* 2018;19:437), the
one published supine-CT series in an older symptomatic cohort, reports **PI 53.4, SS 34.1,
PT 19.2** against **52.6, 33.1, 19.6** here -- within a degree on all three, from a
scanner-axis vertical. Kiapour's 9,721-CT study finds obesity alone worth about 2 degrees of
slope, and tilt rises with age.

Figure 5 now draws both series as a band rather than one line through one of them, and
Table II carries both columns. The manuscript also said twice that recumbency lowers sacral
slope; it does not, and that is corrected.

**The decisive experiment is cheap and has not been done.** Fit the APP from the hip labels
already in the release -- ASIS and pubic tubercles are extremal points on them -- measure
APPA over all 802, and report `aSS = SS - APPA` alongside the raw values. If aSS lands at
36-38 the whole gap was convention; if it lands near 29 this cohort is genuinely retroverted
on the table and the numbers are right as they stand. Anatomical sacral slope is also the
better parameter to report regardless: it is posture-independent, it does not use the
femoral head centre, and its inter-rater ICC is 0.856 against pelvic incidence's 0.653
(Suzuki et al., *J Orthop Surg* 2020;28).

## Still open

- ~~128 records return PI 1-17 with a normal plate and normal femoral heads.~~ **SOLVED
  2026-09-11.** The plate tilt gate measures only the angle from the cranial axis, which is
  blind to the DIRECTION of the tilt. `_endplate` orients its normal by the superior
  component alone (`n if n[2] >= 0 else -n`), so a plane fitted to the wrong surface comes
  back leaning POSTERIORLY at a perfectly normal angle from vertical and sails through.
  39 of 39 records returning PI <= 17 had a posteriorly leaning normal; 39 of 39 returning
  PI > 35 had an anteriorly leaning one -- a clean separation on a quantity nothing
  downstream was looking at.

  Rendering the fit shows what it lands on, and it implicates the S1 carve. Where the carve
  produces a compact wedge under L5 the selected voxels lie on the superior end-plate and
  the normal points up and forward. Where it fails -- an irregular blob, or on some records
  an "S1" spanning most of the sacrum -- the plane is fitted to the VENTRAL SURFACE OF THE
  SACRUM, a long antero-inferior ramp, and pelvic incidence collapses. Case 0097 reads
  PI 2.8 for exactly this reason. It also explains why two label sets that differ only in
  the S1 carve (Dice 0.301 on that label) give completely different spinopelvic numbers for
  the same patient.

  Gated on the geometry: a normal with a non-positive anterior component is not an S1
  superior end-plate. 140 plates rejected for it, and afterwards **no record returns a
  pelvic incidence below 22.4 or a pelvic tilt below -15**, so the figure's ad-hoc
  "tilt >= -15" filter -- which was removing these by their answer -- is gone. n goes from
  575 to 563.

  Still not fixed is the carve itself. A substructure label gives the sacral plate directly
  and is the planned revision.
- **Eight records measure PI from a prosthetic femoral head.** Sphere-fitting a prosthesis
  is not validated anywhere I could find; the published fallbacks avoid the femoral head
  entirely (anatomical sacral slope, Imai et al., *J Orthop Surg Res* 2019;14:126; sacral
  incidence to pubis, Takahashi et al., *BMC Musculoskelet Disord* 2021;22:214) or use the
  acetabular rim (Veilleux 2020). The inter-femoral-head-separation gate will catch a gross
  failure but not a subtle one.
- **LSTV shifts PI by 10–15°** depending on which endplate is called S1 (*J Neurosurg Spine*
  2017;26:45 reports 61° against 48° on one case; Müller et al., *J Anat* 2024,
  doi:10.1111/joa.13985). This cohort is LSTV-stratified, so that is a stratified sensitivity
  analysis worth doing and not yet done.
- **No published head-to-head exists** comparing plane-fitting estimators for the S1
  endplate, nor femoral-head-centre estimators, nor any open-source PI/SS/PT-from-CT
  implementation. All three are gaps this dataset and toolkit could fill.
- **The anterior pelvic plane is not measured here**, so anatomical sacral slope and
  anatomical pelvic tilt cannot be reported and the frame question above stays open. The
  landmarks are present in the released labels; the code is not written.

---

# Pedicle width: which code the figure uses, and why not the toolkit's

Figure 6(c) plots the extraction script's transverse pedicle width, the mean of the two
sides. Against the thirteen published series in `morphometrics/pedicle_width_references.csv`
it tracks them closely except at L5:

| level | ours | published mean |
|---|---|---|
| T11 | 8.60 | 9.75 |
| T12 | 8.10 | 8.70 |
| L1 | 6.90 | 8.35 |
| L2 | 6.90 | 8.72 |
| L3 | 9.00 | 10.24 |
| L4 | 12.20 | 12.56 |
| **L5** | **20.70** | **16.20** |

L5 sits inside the published range (11.8–21.6, thirteen series) but above eleven of the
thirteen. This is the transverse-process merge: at L5 the processes arise far enough
forward to join the body and the ala, and the widest slice in the front third of the canal
catches them. The figure shows the disagreement rather than hiding it, and the caption
claims no agreement.

**The toolkit's implementation measures L5 better, and was not substituted anyway.** Called
directly on released volumes it returns L1 6.62 mm and L5 **15.10 mm** — and 15.10 is nearer
the published 16.20 than the extraction script's 20.70. So where it runs, it is the better
measurement of the two at the level that matters.

Two things stopped it being used here, and one of them I first got wrong:

- **It is expensive.** The signed-distance resample to 0.35 mm costs roughly 30–60 s per
  level; `ostk morph` over *two* cases exceeded a 280-second timeout. A full 802-record run
  is hours, not minutes.
- **The 2026-09-10 shard run produced pedicle columns for two level-instances out of about
  5,600** — while each shard completed in 52–90 minutes, far less than 100 cases × 7 levels
  of this code would need. The columns were never computed rather than computed and lost.
  *Why* is not established. I first recorded this as a broken integration and as the `frame`
  argument; both were wrong — `pedicle_widths` never reads `frame`, and `level_morphometry`
  returns pedicle widths correctly when called today. The likeliest reading is that those
  shards ran an earlier state of the package.

**So Figure 6(c) keeps the extraction script's values, and the L5 point stays high.** The
honest summary is that a better L5 measurement exists, has been demonstrated on individual
cases, and has not been shown to survive a full run — which is not a basis for putting it in
a published figure. Substituting it is the obvious next piece of work.

