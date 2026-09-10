# Why this cohort's vertebral body height sits ~6 mm above Panjabi

Overlaying Panjabi 1992 on the level atlas showed posterior body height 5.5 to 7.5 mm
above the published value at every lumbar level, while canal depth, canal width and
end-plate width agreed within a millimetre. A one-sided disagreement on one measure is
either a real property of the cohort or a defect in how we measure, and the two demand
opposite responses, so it was worth settling rather than arguing about.

Four candidate causes, each tested. Three are excluded.

## 1. The estimator — EXCLUDED

The released field is the tallest column over the posterior half of a 10 mm mid-sagittal
slab. A maximum over a region is greater than or equal to a reading at a landmark by
construction, so this was the first suspect.

`height_probe.py` measures four estimators on the same masks: that maximum, the 90th
percentile and the median of the columns inside the posterior 20% of the body, and the
single column at the posterior-most edge. Over 80 cases:

| level | max over half | wall p90 | wall median | single wall column | Panjabi |
|---|---|---|---|---|---|
| L1 | 30.4 | 30.4 | 29.6 | 28.0 | 23.8 |
| L2 | 30.4 | 30.4 | 29.6 | 28.8 | 24.3 |
| L3 | 30.4 | 30.4 | 29.0 | 28.8 | 23.8 |
| L4 | 29.6 | 29.0 | 28.8 | 28.8 | 24.1 |
| L5 | 30.4 | 28.0 | 26.8 | 26.4 | 22.9 |

Every variant lands within about two millimetres of the others and all of them sit four
or more above Panjabi. **The choice of estimator is not the cause.**

A first version of this probe appeared to show the opposite, and was wrong: it measured
the whole vertebra without carving the body off the posterior elements, so its
"posterior 20%" band landed on the spinous process, which happens to be about 23 mm.

## 2. Tilt, measuring along the scanner instead of the bone — EXCLUDED as the main cause

A straight-up extent is the true height divided by the cosine of the tilt. That is worth
6 to 15% at L5 but only a few tenths of a millimetre at L1 to L3, where the body is
nearly horizontal and the gap is still 6.6 mm. `ostk.morphometry` removes the term
properly by measuring corner to corner in the vertebra's own frame; on spot checks it
reads 0.5 to 2 mm below the old estimator, which is the size of correction expected.
Real, worth having, far too small to explain the gap.

## 3. The millimetre scale — EXCLUDED

If voxel-to-millimetre conversion were wrong, every length would be wrong in the same
proportion. Lumbar disc heights in this cohort run 8.8 to 10.4 mm against a published
norm of roughly 8 to 12. The scale is right.

## 4. Label dilation at the end-plates — EXCLUDED

The one candidate an overlay cannot settle: a mask a voxel or two proud of the cortex
adds height at both ends and is invisible at any window. `edge_probe.py` steps across the
label boundary at the posterior body and reads the mean CT profile against signed
distance from the edge, over 40 cases. Step 0 is the last voxel inside the label:

| level | plate | -2 | -1 | **0** | +1 | +2 |
|---|---|---|---|---|---|---|
| L1 | superior | 304 | 429 | **363** | 154 | 91 |
| L3 | superior | 322 | 454 | **366** | 148 | 92 |
| L3 | inferior | 440 | 579 | **415** | 153 | 88 |
| L5 | inferior | 458 | 525 | **375** | 161 | 82 |

The cortical peak sits one voxel inside the edge, the last voxel inside the label is
still bone at 360 to 445 HU, and the first voxel outside has already fallen to disc at
about 150. Where bone ends relative to the label edge: **0.00 mm at every level, both
plates.** The mask ends where the bone ends.

## What is left

The population, and possibly what Panjabi's VBHp names. Twelve dried cadaveric spines
against 802 living patients aged 50 and over is a real difference, and the direction is
the one observed: cadaveric specimens from elderly donors have lost height.

This matters for the manuscript beyond bookkeeping. It is the paper's own argument
landing on its most quotable number: the value a surgeon looks up, from a dozen
specimens, sits six millimetres below what 802 living patients measure. Publishing the
overlay therefore needs a sentence about definition and population, not a bare dashed
line.

## Reproducing

    python scripts/height_probe.py <labels> 80      # estimator comparison
    python scripts/edge_probe.py <ct> <labels> 40   # label edge against CT intensity
    python scripts/render_height.py <ct> <labels> <out> 0001 0003 0004 0016
