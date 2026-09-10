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

This is worth stating plainly: an identity check cannot catch a consistent error in a shared
landmark. It was found only by comparing against an outside reference — the release's own
independent extraction code (SS 35.8°) and published automated supine CT (36.5°, Veilleux et
al., *JBJS Am* 2020;102:e130, n=200) — against ostk's 24.8°.

## 4. Flags were dropped for exactly the cases that needed them

`spinopelvic_summary_from_label` attached a parameter's QC flags only when that parameter
came back `None`. But a geometry that violates the identity, or returns a pelvic incidence no
pelvis has, still produces a finite number — so the cases the guards exist to catch arrived
unflagged and indistinguishable from good ones.

## What the estimator does now

PI, SS and PT are computed **independently** — plate normal against the radius, normal
against the vertical, radius against the vertical — so `|PI − (SS + PT)|` is a real
per-case check rather than a tautology. The sagittal plane is derived from the two femoral
head centres rather than assumed from the scanner axes. Each femoral head centre is a
least-squares sphere fitted to its articular surface, seeded from the acetabular interface
and grown through the neck, which is the method with the best published repeatability
(0.20 mm inter-operator, Renault et al., *J Biomech* 2018;80:171; 0.5 mm hip-centre
agreement in STAPLE, Modenese & Renault, *J Biomech* 2021;116:110186). PI is taken to the
**midpoint** of the bicoxofemoral axis, which is the published definition (Legaye et al.,
*Eur Spine J* 1998;7:99) and what every 3-D CT implementation uses.

Plausibility gates, set at roughly published mean ± 4–5 SD so an unusual patient survives
them, are in `ostk/metrics.py` with their sources. A case failing the identity or a gate is
reported as missing **with its reason**, never as a number.

## Reference values, and why the modality matters

The cohort is supine CT. CT reads lower than standing radiography in the same subjects — 53°
against 56° of pelvic incidence (Lee & Liu, *Eur Spine J* 2022;31:241) — so comparing against
radiographic norms alone builds in an offset that has nothing to do with this dataset.

| | supine CT | standing XR |
|---|---|---|
| PI | 47.1 ± 10.0 (Vrtovec 2012, n=370); 44.97 ± 8.52 (Chen 2019, n=320); 52.05 (Veilleux 2020, n=200) | 55 ± 10.6 (Vialle 2005, n=300) |
| SS | 36.49 (Veilleux 2020) | 41 ± 8.4 (Vialle 2005) |
| PT | 15.60 (Veilleux 2020) | 13 ± 6 (Vialle 2005) |

## Still open

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
