# Ways to help on CTSpinoPelvic1K

A menu, not a queue. Each item says what it is, what it needs from you, roughly how long
it takes, and what you get out of it. Pick whatever suits the time you actually have —
several of these are genuinely a weekend, and two of them are first-author papers.

The dataset: 802 abdominal CT records from a colorectal-cancer screening cohort (median
age 59, 393 women / 345 men), with every vertebra, rib, hip and femur segmented. The
scans were taken to look for polyps. Everything below is something else those images
already contain.

---

## A. Clinical / anatomical — no coding required

### A1. Castellvi grading of the transitional cases
**What:** 33 cases carry a transitional-vertebra label; each needs a Castellvi grade
(I–IV, a/b) read from the images. Only 5 currently have a second independent read.
**You need:** to be comfortable with the Castellvi scheme (one afternoon to learn — it is
four types about how far a transverse process reaches and whether it articulates or
fuses). ITK-SNAP, which we will set up for you.
**Time:** ~4 hours for a full pass.
**Why it matters:** inter-rater agreement on a rare phenotype is publishable in its own
right, and every derived measure that touches transitional status depends on these.

### A2. Second read on the deferred and disputed cases
**What:** a short list of cases where the automated checks and the source labels disagree,
plus `docs/DEFERRED_CASES.md`. Each needs a human to look and decide.
**Time:** ~2 hours.
**Why it matters:** these are the cases that decide whether a QC number is honest.

### A3. Hand-annotate the instrumented case (`0068`)
**What:** one spine with an interbody cage. Dense metal leaves *no image gradient* where
the boundary between the two vertebral bodies belongs, so no segmentation tool can find
it — it has to be inferred anatomically and drawn.
**You need:** ITK-SNAP and patience.
**Time:** 2–4 hours.
**Why it matters:** it is currently the only thing blocking the hardware classes from
being populated, and hardware bridging an interspace reads as fusion to every distance
measurement in the dataset.

---

## B. Analysis — Python, but the pipeline already exists

Every measure below follows the same pattern: a script reads the labels, writes a CSV,
and checks its own medians against published values. You would be adding one script and
one panel, with four existing ones to copy from.

### B1. Sacroiliac joint measurement
**What:** SI joint width and the degree of ankylosis, from the sacrum and hip labels.
**Prior art:** SI abnormalities are reported in a majority of DISH patients (AJR 2017),
and we already detect DISH by Resnick's criteria — so this closes a loop that is
currently open.
**Difficulty:** moderate. The joint space is the gap between two labels we already have.
**Output:** a panel, and a real finding if SI change tracks the DISH cases.

### B2. Vertebral rotation / coronal alignment
**What:** axial rotation of each lumbar vertebra, and coronal Cobb angle.
**Prior art:** established scoliosis morphometry; degenerative lumbar scoliosis is common
in this age band and nobody has measured it here.
**Difficulty:** moderate — the vertebral body's principal axes give rotation directly.
**Output:** prevalence of degenerative curvature in a screening population.

### B3. Spinal canal cross-sectional AREA rather than diameter
**What:** we measure canal width and AP diameter; area is the better stenosis metric and
we already isolate the canal per slice.
**Prior art:** dural sac cross-sectional area is the standard measure; <100 mm² is severe
stenosis by Schizas.
**Difficulty:** easy — the canal mask exists, it needs summing rather than measuring.
**Output:** stenosis prevalence in an asymptomatic cohort, which is a genuinely
interesting number.

### B4. Rib cage geometry
**What:** thoracic index, rib angles, chest wall dimensions from the 24 rib labels.
**Difficulty:** easy to moderate.
**Output:** normative rib geometry, of which surprisingly little exists.

---

## C. Machine learning

### C1. Train a level-numbering model and find where it breaks
**What:** the dataset exists because tools mis-number lumbar levels when a transitional
vertebra is present. Train a segmenter on the frozen splits and measure error *as a
function of transitional status*.
**You need:** nnU-Net or similar, and a GPU (we have cluster access).
**Time:** a few weeks including training.
**Why it matters:** this is the benchmark the dataset was built to support, and nobody
has run it.

### C2. Predict bone density from geometry alone
**What:** we measure trabecular attenuation AND full 3-D shape. Can shape predict
density? If it can, every segmentation-only dataset becomes an osteoporosis screening set.
**Difficulty:** self-contained ML project with a clean target.

---

## D. Writing

### D1. Second author on the dataset paper (Medical Physics, dataset article)
Drafted, needs figures finalised, references completed and critical reading.

### D2. **First author on the clinical measurements paper**
Target: *Spine Surgery and Related Research* — no article processing charge, PubMed and
Scopus indexed, about 15 weeks from submission to publication.

The analysis is done. The findings are already there:

- **The iliac crest reaches L4 in 87% of typical spines but only 46% of those with four
  rib-free vertebrae.** Surgeons palpate that crest to find L4–5. The landmark fails
  hardest in exactly the patients whose levels are already ambiguous to count, so the two
  errors compound rather than cancel.
- **And the direction of that error is predictable.** Four rib-free vertebrae happens two
  ways — a rib on a lumbar vertebra, or a lowest lumbar segment fused into the sacrum —
  and the two fail in opposite directions: with a lumbar rib the crest reads caudal in 7
  of 8 misses, without one it reads cranial in 6 of 7 (Fisher exact p = 0.010). Pooling
  the two routes is what made this look like directionless noise. It rests on 15 misplaced
  cases and needs someone to attack it.
- **Bone density crosses over by sex**: women start denser in their fifties and end lower
  by their seventies; 31% of women in the oldest decade fall below the osteoporosis
  threshold against 10% of men — in scans nobody ordered for bone.
- **Pelvic incidence does not move with age** (−0.006°/yr) while the pelvis retroverts
  around it.

What it needs: a literature introduction, a methods section written from the scripts, and
someone to argue with me about the interpretation. That is a first-author paper on work
that is already finished.

---

## Getting started

1. Read `README.md` (the dataset card) and `docs/RELEASE_CHECKLIST.md`.
2. For annotation work: ITK-SNAP plus the label descriptor at
   `data/itksnap_v5_labels.txt` — colours are chosen so adjacent levels never look alike.
3. For analysis work: copy `scripts/extract_level_gradients.py`. It is the cleanest
   template — read labels, measure, write CSV, check the medians against published values.

**The one rule that matters more than any other here:** every measure checks itself
against a published value, and anything that fails is withheld with the reason recorded
rather than shipped. Four measures are currently withheld for exactly that reason. That
discipline is why the rest can be trusted, and it is not negotiable — if your number
disagrees with the literature, the number is probably wrong, and finding out why is the
actual work.
