# Held back for the next paper: the multi-series reference comparison

The dataset article draws **only Panjabi** on Figure 6, and only on endplate width, canal
width and depth, and pedicle width. That is deliberate. Everything below works, is
committed, and is not used there.

## What exists

`morphometrics/level_references.csv` (355 rows) and
`morphometrics/pedicle_width_references.csv` (65 rows), built by
`morphometrics/make_reference_table.py` from per-series literals so a level cannot inherit
another study's error bar. Between them they carry five to eighteen independent series for
each of seven quantities, with the cohort, modality, dispersion type, sex stratification,
primary-or-secondary status and known definitional divergence recorded per row.

`paper/mpda/make_levelatlas_fig.py` still contains the whole drawing path: `SERIES_STYLE`
with a unique Okabe-Ito colour and dash per series, an assert that refuses duplicates, and
`draw_reference_series`. It is gated by one constant:

    DRAW_SERIES = {"Panjabi 1992", "Panjabi"}   # set to None to draw them all

## Why it is a different paper

The published series disagree with each other by 21 to 69% of their own median:

| measure | series | spread |
|---|---|---|
| anterior body height | 7 | 21-27% |
| posterior body height | 8 | 23-29% |
| endplate width | 10 | 24-30% |
| canal width | 18 | 27-40% |
| canal depth | 18 | 41-54% |
| disc height | 8 | 43% |
| pedicle width | 13 | 61-69% |

That is a finding about the literature, not about this dataset, and it deserves its own
argument rather than a legend. Some of the spread is real population difference, some is
definitional (endosteal against outer cortical differs by about 5 mm at L5; endplate depth
as maximum contour extent runs 4 mm above the same name as a midsagittal chord; disc height
from a recumbent radiograph runs 6 mm above the same disc on CT), and some is measurement
plane (an axial pedicle reading overestimates one perpendicular to the axis by about 12%).
Separating those three is the paper.

## Two panels deferred with it

**Body height.** The cohort-versus-method question is settled to the point of knowing it is
NOT method (see `BODY_HEIGHT_INVESTIGATION.md`: every estimator agrees within 2 mm, the
millimetre scale checks against disc heights, and the label edge sits 0.00 mm from where
the CT says bone ends), but the residual offset against cadaveric series needs the
population argument made properly before it is drawn.

**Disc height.** No cadaveric counterpart worth the panel, and the method spread between
published series is larger than the biology.

## Two things worth carrying into it

- A widely cited "Kunkel prediction equation" is not an independent series. It is a
  polynomial regression fitted to Panjabi's same twelve spines.
- The Bonczar meta-analysis includes Panjabi among its eighteen inputs, so plotting both
  is not two independent references.
