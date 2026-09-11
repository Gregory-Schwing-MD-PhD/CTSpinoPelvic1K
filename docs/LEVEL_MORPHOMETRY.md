# Canal depth and end-plate width, and why both were wrong

Two of the three measures in Figure 6 of the dataset article were measured by code that
produced finite, plausible-looking numbers from the wrong place. Neither announced itself.
Both were found the same way every defect in this project has been found: by comparing
against something outside the computation.

This note records what was wrong, what the published methods literature says, what the
code does now, and what is still not settled.

---

## How they were caught

Not by an internal check. The distributions passed every plausibility gate the extractors
carry, and the medians sat close to the published means at every level. What failed was a
comparison of **spread** against the living-cohort CT literature:

| measure | this cohort's SD | widest SD in any published living cohort | verdict |
|---|---|---|---|
| canal depth, L1 | ~2.3 | 2.91 (Masharawi, M) | fine |
| **canal depth, L3** | **~3.9** | 2.45 (Griffith, n=540 M) | **1.6× above anything published** |
| canal depth, L5 | ~3.6 | 3.76 (Griffith, M) | fine — the L5 canal really is variable |
| end-plate width, L1 | ~4.6 | 3.09 (Yadav, M) | mildly wide |
| **end-plate width, L5** | **~8.7** | 5.54 (Yadav, M) | **1.6× above anything published** |

And then by a prevalence argument, which is the stronger of the two. The Framingham CT
cohort puts congenital absolute lumbar stenosis at **2.62%** (95% CI 0.86–6.00). This
cohort had **157 of 736 records under 11 mm at L3 alone — 21%**. The smallest midsagittal
canal diameter anywhere in 1,000 level-measurements of living adults is **11.70 mm**
(Čizmić et al., *Acta Inform Med* 2023;31:200), and Aly & Amin (*Orthopedics*
2013;36:e229, n=300) give a floor of 11.07 mm across all levels. A 5th percentile of
7.0 mm at L3 is not a cohort; it is a method.

The same argument at the other end for end-plate width: Bonczar et al. (*Surg Radiol Anat*
2024;46:2097) pool L5 EPWu at **48.81 mm** over 1,481 individuals — and put the L5
**transverse-process span at 85.91 mm**. Ninety-five of 760 L5 records read over 60 mm.
They were walking from the one number toward the other.

---

## 1. Canal depth: three defects, all pushing the same way

The measurement was the total anteroposterior extent of the enclosed hole, on the **first**
axial slice in the middle third of the vertebra whose bony ring happened to close.

**The slice was whichever one closed first, not an anatomical plane.** Every normative
series fixes the plane at the pedicle. Maeder et al. (*Diagnostics* 2023;13:734, n=1,050)
say why in their own words: measurements are made "in a plane perpendicular to the
longitudinal axis of the spine and **at the vertebral pedicle levels** … to limit the
influence of degenerative changes on CSA measurements, which typically occur at the
intervertebral disc and facet joint levels." Cook & Baker (*Int J Spine Surg*
2021;15:1072, n=196) measure at the "pediculolaminar level". Taking the first closing
slice is close to taking a minimum over ~20 noisy slices, and a minimum manufactures a low
tail.

**The ring need not close in an axial plane at all.** A lumbar vertebra in a supine scan of
a lordotic adult is tilted, and an oblique section through the canal is filled in by volume
averaging. Eubanks, Cann & Brant-Zawadzki (*Radiology* 1985;157:243) showed this in phantom
and stated it plainly: sections cut at an angle to the canal's transverse plane "do not
always overestimate its diameter, as previously suggested, but **can make it appear
artifactually stenotic**." That explains the shape of the failure exactly — level-dependent
(L1 near-axial and clean, L3–L5 progressively tilted and wrecked), sporadic rather than
patient-clustered (it depends on where the reconstruction grid lands on that vertebra), and
one-sided low. A modern pipeline puts the cost at **22% of axis-aligned measurement lines
invalid** versus an orientation-aware one (Nixon et al., *BMC Med Imaging* 2026;26:134).

**The max extent is not the midsagittal diameter.** Every normative series measures a
midsagittal chord — Verbiest's criterion, Čizmić's "mediosagittal diameter", Aly & Amin's
"midsagittal diameter", Cook & Baker on the midsagittal reformat, and Panjabi's SCD. Aly &
Amin also report that the canal runs "circular in the upper lumbar vertebrae to triangular
in the mid-lumbar to **trefoil in the lower lumbar, especially at L5**", so at L4/L5 the
widest anteroposterior chord of the region runs through a lateral recess. That is a
high-side bias *and* a shape-dependent variance source, at exactly the levels where shape
varies most.

### What the code does now, and what it bought

Over all 802 records the rewrite took the count reading below 11 mm — under any canal
diameter reported in a living adult — from **473 to 12**, and the per-level SD at T11–L4
from 1.55–4.37 to **1.55–2.04**, inside the published consensus of 1.5–2.5. L3 alone went
from 157 implausible records to 1. The L5 median moved from 21.0 to **17.4**, which sits
inside every published series (Griffith 18.1–19.7, Čizmić 17.8, Cook & Baker 15.6) where
21.0 was above all of them — that is the trefoil max-extent bias going away. It costs
about 5% of levels, rejected by the continuity gate and reported missing rather than
guessed.

`scripts/extract_surgical_morphometrics.py`, `_canal_column` and `_canal_ap`:

1. Per axial section, **close the ring before looking for the hole** — a 2 mm-radius disc,
   the in-plane equivalent of Banik, Rangayyan & Boag's tubular element (*J Digit Imaging*
   2010;23:301). A lamina one voxel short of meeting otherwise leaves the canal open to
   the background and no hole is found at all. The closing is used only to *find* the
   hole; the hole is taken against the original mask, so the canal is not shrunk.
2. Require the section's hole to exceed **18 mm²**, not 20 voxels. Twenty voxels is
   2.5 mm² at 0.35 mm pitch.
3. Keep the sections whose hole centroid lies within **8 mm of the column's robust axis**,
   and take the longest contiguous run, bridging a single dropped section. This is
   Díaz-Parra, Arana & Moratal's 3-D connectivity and centroid interpolation (*EMBC*
   2014:5514) in a form that survives thick slices — requiring literal 3-D voxel adjacency
   was implemented first and cost 15% of the levels, because on 3–5 mm sections the canal
   steps several millimetres between neighbours.
4. Take the **central half of that closed-ring column as the pedicle slab** — the column is
   the pediculolaminar extent by construction, since it is where bone surrounds the canal
   on all four sides.
5. Read the chord on a **±1.5 mm midsagittal strip**, as the run *containing* the canal
   centroid so a second enclosed pocket cannot be spliced on, and take the **median over
   the slab**.

ostk's `canal_dimensions` carries the same change, and already had the one thing the
extraction script does not: it measures in the vertebra's own frame rather than the
scanner's, so the obliquity defect above does not arise there. Its `canal_mask` now closes
the ring too.

---

## 2. End-plate width: the shoulders the canal cut leaves behind

The body is isolated by cutting at the anterior wall of the canal. At L5, and often L4,
that keeps the vertebral body **and two lateral shoulders** — the roots of the transverse
processes and pedicles, which at these levels arise far enough forward to fall in *front*
of the canal wall. The dome alone is 50–60 mm across; dome plus shoulders is 70–86 mm.

Three separate attempts to remove them failed, and it is worth recording why, because each
looked right on paper:

- **Erosion in voxels** (what shipped). The physical radius is whatever the pitch happens
  to be — 0.7 mm at 0.35 mm spacing, which cannot snap an isthmus several millimetres
  thick. It also had a fallback that abandoned the erosion when the core kept less than a
  third of the slice, which is exactly the L5 case.
- **Erosion in millimetres, progressive, stopping at the first radius that separates the
  slice.** The shoulder isthmus is as thick as the end-plate rim the same erosion removes,
  so every radius that snaps the one eats the other. Measured, it made L5 *worse*.
- **A depth-per-column rule** — keep columns whose anteroposterior extent exceeds a
  fraction of the deepest. The shoulders are deep, being pedicle root and not process
  wing, and the body's own lateral edges taper, so the threshold trimmed the measurement
  before it trimmed the contaminant. At every fraction tried it lost 2–3 mm at L1–L4 while
  gaining only 2–4 mm at L5, and it collapsed two L4 records to 26 mm.

**What settled it was rendering the mask.** The shoulders occupy the posterior fringe of
what is kept and nothing else; the body occupies all of it. So the fourth attempt dropped
the **posterior quarter of the body's anteroposterior span** and took the width as the
**90th percentile of the per-row transverse extents** of what is left — Mastmeyer's
geometric constraint, cutting the body out with a shape derived from the anatomy.

**It was run over all 802 and it lost too.** Scored against the pooled published mean per
level, the widest SD published in any living cohort, and the count of records outside
plausible bounds:

| rule | total \|median − published\| | SD in excess of published | implausible records |
|---|---|---|---|
| **shipped (erode, whole slab)** | 7.2 mm | 9.82 | 173 |
| anterior band | **6.5 mm** | 13.59 | 230 |
| anterior band + area gate | 16.6 mm | **5.85** | **161** |

The anterior band has the best medians and the worst spread. Its slab runs past the
end-plate into slices where a tilted body has collapsed to an oblique corner — `ztop` is
the 80th percentile of the body's extent along the **scanner's** z, not the vertebra's —
and measuring the width of a corner put **60 L5 and 37 L4 records newly below plausible**.
Gating those sections out by area fixes the tail and gives the gain straight back, because
the shoulders live in exactly the sections the gate keeps; its medians then run 1.2 to
4.6 mm high at every level.

No rule dominates, so **the one that shipped stays**, and `endplate_width_*` is unchanged —
verified bit-for-bit identical on all 741/773/764 records at L1, L4 and L5.

**The experiment did produce one real finding.** The area-gated rule reads a *lower*
section of the body and cuts the L5 SD from **9.07 to 6.16**. So L5's spread is not only
transverse-process contamination: the superior slab is itself unstable. That is exactly
Museyko and Engelke's result (*Bone* 2009;44:429) — superior sub-VOI precision errors are
50–100% worse than mid sub-VOI ones — and it points at the two fixes that would actually
work, both listed below.

## Still open

- **The obliquity correction is only in ostk.** The extraction script, which is what the
  release reports, still slices on the scanner axes. The pedicle slab and the midsagittal
  chord remove most of the artefact, but resampling each vertebra into its own frame is
  the published fix (Maeder does it by hand via MPR in 1,050 patients) and is not done.
- **The rib bases were hardcoded and are now read from the shipped scheme.** Rib n on a
  side is `base + n`; the right block's base is 45 where the left reserves 12 slots and 46
  where it reserves 13. Two directories that both look like "the labels" differ by exactly
  that, with voxel-identical vertebrae and pelvis, and the constant was right for one and
  wrong for the other. It changed `rib12_to_crest_mm` in **237 of 802 records**, worst
  36 mm. Resolved per worker from `dataset_labels.json`, never in `main()` — a value
  computed there reaches forked workers and not spawned ones.
- **SPINEPS already emits a corpus label.** `scripts/spineps_ct_pipeline.py` produces the
  subregion semantic mask with the vertebral corpus as its own class (Möller et al., *Eur
  Radiol* 2025, avg. Dice 0.918–0.931). Taking end-plate width from that label would make
  the shoulder problem disappear by construction rather than by a geometric rule. Not
  done, and worth doing.
- **No published variance decomposition exists** for either measure — only ICCs, from
  which the observer share is derivable as 1 − ICC: 0.83–0.88 for canal CSA over three
  observers in 1,050 patients (Maeder), 0.902 interrater for canal and body dimensions
  (Cook & Baker). So published manual measurement of these quantities already carries
  roughly 10–17% of its variance as measurement noise, which is the right yardstick for
  how much of this cohort's excess is method.
- **No published automatic-failure-detection rule exists** for either measure. Every
  automated paper validates with MAE or ICC against readers; none publishes a per-case
  reject rule. The gates here are constructed over cited numbers, not taken from a source.
- **Griffith et al., *Eur Radiol* 2022;32:6688** (n=505 abdominopelvic CT, mean age 57) is
  the closest-matched published cohort in existence — canal CSA, width and depth plus
  vertebral body CSA, L1–L5 — and is paywalled with no PMC copy. Its per-level SDs would
  be the best available comparator. Worth an interlibrary request.
