# The screen recovers the subtle grades and misses the gross ones

All 33 LSTV cases carry a radiologist Castellvi grade. Joining them to record ids
(`join_castellvi_grades.py`, matched through the released manifest's own token field)
lets `screen_missed_castellvi.py` be scored against the thing it is named after for the
first time. Two results came out of that, and the second one was the opposite of what I
expected.

## The join buys no accuracy

Every graded case is an LSTV-labelled case, so the positive set is the same 33 and every
leave-one-out number is unmoved: median rank 199 of 760, 18% in the top 25. The join makes
the name honest. It does not make the screen better.

## Recovery by grade, which is the figure that matters

Recomputed after the transverse-process measurement was corrected. The first version of
this table was built on heights contaminated by segmentation speckle -- 169 values across
the cohort were overstated by more than 5 mm, several by more than 25 mm -- so the numbers
below replace it rather than supplementing it.

| grade | n | median LOO rank | in top 100 |
|-------|---|-----------------|------------|
| IIb   | 4 | **4**   | 4/4 |
| IIIa  | 3 | **25**  | 3/3 |
| IIa   | 2 | 58      | 2/2 |
| Ib    | 2 | 82      | 1/2 |
| IV    | 4 | 378     | 2/4 |
| IIIb  | 18| **442** | 3/18 |

Fixing the measurement nearly doubled the screen's recovery overall: 11 of 33 in the top 25
against 6 of 33 before.

## The dividing line is not the grade number

The first reading of this table was "I/II is recovered, III/IV is not". The corrected
numbers say something more specific, and it is the mechanism rather than the ordinal:

**IIIa is recovered at rank 25. IIIb sits at 442.** Both are grade III -- both are bony
fusion -- and they differ only in that IIIa is unilateral and IIIb bilateral. In a IIIa one
transverse process is fused into the ala and labelled sacrum, and the other is still a free
process on the vertebra, so there remains something to measure. In a IIIb both are gone.

That is the actual boundary: not II against III, but whether any unfused process survives
on the vertebra to be measured. It predicts the rest of the table -- I, IIa, IIb and IIIa
all retain at least one free process and all sit in the top 100; IIIb and IV, where fusion
is bilateral or mixed, do not.

## What was tested and came back negative

If fusion moves a transverse process into the sacrum label, the sacrum should be bigger.
The coarse sacral features do not show it — median z against the unlabelled cohort is
−0.16 for `sacrum_width_mm` and +0.11 for `sacrum_height_mm`. So the claim "fusion widens
the sacrum" is **not supported** and is not being made. A whole-bone bounding width is far
too coarse to see a few millimetres of absorbed process.

## The measurement that would settle it

Left–right asymmetry of the upper sacral ala at S1, not global width. Castellvi IIIa is
unilateral and should be asymmetric there; IIIb is bilateral and should show a symmetric
but taller upper ala. That feature does not exist yet and needs volume-level work.

Until it does, the honest statement is: **this queue is a screen for Castellvi I and II.**
That is a narrower claim than the filename implies and a more useful one than the filename
delivers, because I/II is exactly the grade a vertebra count cannot reach — seven of the 33
graded cases have a perfectly normal count of five, and grade IIIb occurs at counts of
four, five and six alike.

---

## A correction, and the footgun behind it

An earlier version of this note said the join had been confirmed because "802 tokens map to
802 records, so the zero-padded guess happened to be right." That is false. **Zero-padding
the token reproduces the record id for one record in 802** — token 149 is record 0208. A
padded join is not a rough version of the right answer; it is 32 of the 33 grades landing
on the wrong 32 patients, in a file of exactly the right shape and length.

The join is correct because it goes through the manifest's own `token` field. It was never
correct for the reason I gave.

The mechanism that made this hard to see is worth recording, because it nearly destroyed
the result. `join_castellvi_grades.py` used to fall back to padding when the manifest was
absent, print a warning, and **write the file anyway**. Re-running it on a machine without
the manifest therefore replaced a correct `castellvi_grades.csv` with a corrupt one, and
nothing downstream could tell. The per-grade numbers above survived only because they were
computed before that overwrite and re-verified after the manifest was fetched — both runs
agree exactly, which is what confirms them.

The fallback is now removed. Without a manifest the script exits non-zero and writes
nothing, and it also refuses to write if any graded token has no record.

## Two other manifest fields that are declared and wrong

Found while checking the above, and the same failure mode as the null `castellvi_type`:

- **`has_l6`** is true for exactly one record — token 500, labelled `normal` — while all 14
  `LUMBARIZATION` cases have it false. Lumbarization is the phenotype that produces an L6,
  so this field is inverted, stale, or was never populated. L6 support is a headline claim
  and cannot rest on it.
- **`n_lumbar_labels`** is 0 for 795 of 802 records, including every LSTV case. It is not a
  count of lumbar labels.

Neither is used by anything in this repo, which is why they survived. Both need either
populating from the label volumes or removing from the schema; a declared field that is
wrong is worse than an absent one, which is the lesson `castellvi_type` already taught.

---

## Closing the blind spot: measure the sacrum, not the vertebra

The finding above says where the evidence went. If a transverse process fuses to the ala
and the fused mass is labelled sacrum, then no vertebra-side measurement can see it, and
the sacrum-side one should.

`measure_sacral_ala.py` computes, per side:

    ala_rise = (most cranial sacral voxel in the lateral third)
             - (most cranial sacral voxel in the midline band)

A normal sacrum is highest at the midline promontory and its alae slope away and downward,
so this sits at or below zero. Bone fused into an ala from a process above it raises the
lateral side above the midline.

On the twelve cases inspected by hand the laterality falls out of the measurement rather
than being imposed on it:

| case | grade | left | right | asymmetry |
|------|-------|------|-------|-----------|
| 0208 | IIIa (unilateral) | 14.4 | 7.2 | **7.2** |
| 0234 | IIIb (bilateral)  | 11.2 | 11.2 | **0.0** |
| 0005 | IV                | 10.4 | 10.4 | 0.0 |
| 0244 | Ib                | 14.4 | 12.8 | 1.6 |
| 0268 | IIIb              | 0.8  | −1.6 | 2.4 |
| ungraded (7 cases) | — | −8.0 to 7.2 | | |

IIIa is unilateral and reads asymmetric; IIIb is bilateral and reads symmetric with both
sides raised. That is the $a$/$b$ distinction appearing in a quantity that was not built
from it. 0268, also IIIb, is a miss.

Twelve cases is a direction, not a validation, and the honest caveats are the same as
everywhere else here: nothing is fitted, because 25 grade III/IV records can check that a
measurement points the right way and cannot train anything; and a raised lateral sacrum is
also what a large osteophyte, an ossified iliolumbar ligament, or a mis-assigned voxel at
the sacroiliac joint looks like. Every case it ranks is a request for a radiologist.

### The first version of that measurement failed, and the twelve-case check is why I believed it

Run over all 802 volumes, `ala_rise` does not separate anything:

| group | n | median ala_rise_max |
|-------|---|---------------------|
| I/II | 8 | 5.2 mm |
| III/IV | 25 | 7.2 mm |
| ungraded | 768 | **8.0 mm** |

III/IV is *lower* than the ungraded cohort. As a ranking it is slightly worse than useless:
**AUC 0.450** for III/IV against ungraded, and 0.437 for I/II — both below chance.

The twelve-case smoke test looked convincing and was not evidence. Five of those twelve were
graded cases and the seven ungraded ones happened to sit low, which is exactly what a
twelve-case sample of a distribution with median 8.0 and p90 13.6 will do a fair fraction of
the time. Reading the IIIa/IIIb laterality out of it was reading a pattern into noise.

**Why it fails, and it is the same error twice.** The topmost bone in the lateral sacrum is
not the ala. It is the **S1 superior articular process**, which projects cranially and
posteriorly in every normal sacrum. So `ala_rise` measured that process in all 802 cases,
which is why the ungraded median is a healthy 8 mm rather than the near-zero the reasoning
predicted.

This is the same mistake as the original `tp_gap`, which measured the minimum distance from
the lateral vertebra to the sacrum and returned the width of the L5–S1 facet joint in
everybody. Both times the number was computed correctly and described a different structure.
The general form is now stated in the extractor's header and deserves restating: *the
extreme point of a region is almost never the landmark you wanted, because some other
structure is usually more extreme.*

**The refinement being tested.** The sacral ala proper is *anterior* to the superior
articular processes, so restricting each lateral band to the anterior half of the sacrum's
own depth should exclude them. `ala_rise_wholedepth_max_mm` is retained alongside so the
failure stays visible in the released measurements rather than being quietly replaced.
