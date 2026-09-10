# Two implementations, measured against the same reference

`ostk.morphometry` was written to remove two biases in the release's extraction scripts:
measuring along the scanner's axes instead of the bone's, and taking an extremum over a
region instead of a reading at a landmark. Both are real. The question this note answers
is whether replacing the scripts with it would make the released numbers better, and the
answer is **not uniformly**, so the figure was not switched.

Both were run over the same 802 records and compared against Panjabi 1992, median against
their mean, in millimetres.

| measure | level | scripts | ostk | Panjabi | scripts − P | ostk − P | closer |
|---|---|---|---|---|---|---|---|
| posterior body height | L1 | 30.4 | 29.0 | 23.8 | +6.6 | +5.2 | **ostk** |
| | L3 | 30.4 | 28.5 | 23.8 | +6.6 | +4.7 | **ostk** |
| | L5 | 30.4 | 25.2 | 22.9 | +7.5 | **+2.3** | **ostk** |
| end-plate width | L1 | 40.7 | 43.1 | 41.2 | −0.5 | +1.9 | scripts |
| | L3 | 44.1 | 46.6 | 44.1 | 0.0 | +2.5 | scripts |
| | L5 | 51.0 | 55.6 | 47.3 | +3.7 | +8.3 | scripts |
| canal width | L1 | 23.9 | 25.0 | 23.7 | +0.2 | +1.3 | scripts |
| | L3 | 25.0 | 27.4 | 24.3 | +0.7 | +3.1 | scripts |
| | L5 | 30.0 | 35.2 | 27.1 | +2.9 | +8.1 | scripts |
| canal depth | L1 | 18.6 | 18.2 | 19.0 | −0.4 | −0.8 | scripts |
| | L3 | 17.1 | 18.2 | 17.5 | −0.4 | +0.7 | scripts |
| | L5 | 19.9 | 22.1 | 19.7 | +0.2 | +2.4 | scripts |

## What this says

**The frame does exactly what it was built for, and only that.** Posterior body height is
the measure the two biases actually attack, and ostk beats the scripts at every level. At
L5, where the body is most tilted and the extremum has most room to run away, the error
falls from +7.5 to +2.3 mm. Two thirds of that gap was method, and the frame removed it.

**The width measures went the other way, and the pattern says why.** Every one of them
agrees at L1 and drifts high toward L5. That is the signature of including structure that
is not the body: at L5 the transverse processes arise far enough forward to merge with the
body and the ala, and the extraction scripts have an erosion step tuned over several
iterations specifically to cut them off. ostk's end-plate band has no equivalent. The
scripts are not more principled here; they have simply been fitted to a failure mode ostk
has not met yet.

**So the figure keeps the scripts' values,** and the manuscript says only what is true:
the dimensions come from the release's own code, and ostk is released separately as a
reusable implementation that measures in the vertebra's frame. Claiming the figure was
produced by ostk would have been false, and swapping it would have traded a 4 mm
improvement on one panel for an 5 mm regression on two others.

## The residual body-height gap is not method

Even ostk reads +2.3 to +5.2 above Panjabi. That residual is the cohort, established
separately in `BODY_HEIGHT_INVESTIGATION.md`: every estimator agrees within about 2 mm of
every other, the millimetre scale checks out against disc heights, and the label edge sits
0.00 mm from where the CT says bone ends. Twelve dried cadaveric spines against 802 living
patients over fifty.

## Pedicle width is not here, and should not be

Four principled attempts, each failing for its own identifiable reason, are recorded in
`ostk/morphometry.py`. It is behind an opt-in flag with a test asserting it cannot arrive
by accident. Isolating a pedicle from a whole-vertebra mask is a real problem and not a
detail.

## The data

`morphometrics/ostk_level_morphometry.csv`, one row per (case, level) over all 802
records: VBHa, VBHp, EPWu, EPWl, EPDu, EPDl, SCW, SCD, and the end-plate fit residual that
gates each row. Reproduce with `python -m ostk morph --labels <dir> --out morph.csv`.
