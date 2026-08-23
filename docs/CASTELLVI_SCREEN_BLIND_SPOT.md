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

| grade | n | median LOO rank | in top 100 |
|-------|---|-----------------|------------|
| Ib    | 2 | 157 | 1/2 |
| IIa   | 2 | 40  | 2/2 |
| IIb   | 4 | **7**   | 4/4 |
| IIIa  | 3 | 38  | 2/3 |
| IIIb  | 18| **401** | 4/18 |
| IV    | 4 | 293 | 2/4 |

Grouped: **I/II median rank 32, III/IV median rank 305.**

I predicted the reverse. Castellvi I and II are an enlarged or articulating transverse
process — morphology and nothing else, invisible to any count — and III/IV is gross bony
fusion, so I assumed fusion would be the easy case. It is the blind spot.

## Why, and it is a measurement artifact rather than biology

The feature medians say it plainly:

| group | n | tp_height_max | tp_gap L / R | ll_span |
|-------|---|---------------|--------------|---------|
| I/II  | 8 | **32.0 mm** | 5.2 / 3.6 | 99.2 |
| III/IV| 25| 20.8 mm | 7.9 / 8.2 | 95.7 |
| unlabelled | 769 | 18.4 mm | 8.5 / 8.8 | 92.5 |

A Castellvi III measures almost exactly like an ordinary case. I/II separates hugely.

The per-case rows make the mechanism visible: the IIIb cases with a rib-free count of four
(0094, 0151, 0156, 0158, 0760, 0787, 1031) carry *small* transverse processes — 13.6 to
18.4 mm — and *large* gaps, 8 to 19 mm. That is backwards for a fused vertebra, and it is
backwards for a reason. **When the transverse process is fused to the ala, the fused mass
is labelled sacrum.** The script then measures whatever free vertebra is left over: a short
process, and a wide gap to a sacrum that has already absorbed the bone which bridged it.
The screen is not failing on III/IV; it is measuring the wrong object, on the wrong side of
a label boundary.

This is the same class of error as the original `tp_gap` bug, where measuring from the
whole lateral third returned the width of the L5–S1 facet joint in everybody. In both cases
the number was computed correctly and described something else.

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
