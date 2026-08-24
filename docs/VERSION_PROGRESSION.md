# Did each release actually improve on the last?

Identical checks, run against every version, so the version history is a measurement rather
than a list of dates. Produced by `scripts/qc_version_progression.py`; raw table in
`qc_final/version_progression.csv`.

**Read the version names carefully.** `v5pre` is the *released* export
(`data/hf_export_v5/labels`) and `v5` is the *working copy* on the grid (`data/v5_final`).
They are not a before and after. Where they differ, the released one is what users have.

| metric | v2 | v3 | v5pre (released) | v5 (working) |
|---|---|---|---|---|
| cases | 802 | 802 | 802 | 802 |
| stray identifiers | 0 | 0 | 0 | 0 |
| **fragmented labels** | 366 | **1146** | 378 | 377 |
| cases carrying a fragment | 328 | 463 | 330 | 329 |
| ribs present | 0% | 99.9% | 99.9% | 99.9% |
| femurs present | 0% | 99.9% | 99.9% | 99.9% |
| S1 present | 0% | 99.8% | 99.8% | 99.8% |
| lumbar-rib records | 0 | 0 | **15** | **16** |
| ribs checked | 0 | 6254 | 6375 | 6372 |
| **rib offset** | — | **10.83%** | 7.94% | **7.69%** |
| count coherent | — | 100% | 100% | 100% |

## What it shows

**The speckle cleanup is real and this is the measurement of it.** v3 introduced ribs and
brought 1146 fragmented labels across 463 cases with them — nearly every case gained one.
By v5 that is 377 across 329, back to roughly the pre-rib baseline of 366 while now carrying
ribs. A fragmented label is either scan truncation, which is legitimate, or speckle, which is
not; the fall is the speckle being removed.

**Rib numbering improved monotonically**: 10.83% offset at v3, 7.94% at v5pre, 7.69% at v5.

**Ribs, femurs and S1 all arrive at v3** — 0% to 99.9% in one step.

## Two cautions about these numbers

**The rib-offset percentages are not comparable to the release figure.** The release QC
(`qc_rib_vertebra_incidence.py`) reports 2 residual offsets in 5,749 *evaluable* ribs, or
0.035%. This check reports 7.69% on 6,372 ribs. They are different denominators, not a
contradiction: this check calls any rib within 40 mm of a vertebra "checked", including ribs
whose own expected vertebra lies outside the field of view, which the release QC correctly
excludes as not evaluable. **Use these percentages for comparison across versions and never
as an absolute quality claim.**

**The sidedness column is from the pooled check and undercounts.** It reports 3 failures in
v5 where there are 4: it pools ribs, hips and femurs into one centroid per side, and hips and
femurs outweigh the ribs, so a large correctly-sided structure masks a smaller transposed
one. 0790 is the record it misses. `check_release_invariants.py` now compares each sided
pair separately, and all four transposed records (0027, 0107, 0935, 0790) have been
corrected — after this table was computed, so the column describes the pre-fix release.

## The 0787 divergence, confirmed independently

`v5pre` has 15 lumbar-rib records and `v5` has 16. That difference is a single case, 0787,
which carries the lumbar-rib class in the grid working copy and does not in the released
export. This table was not built to find that, and finding it anyway is the useful kind of
corroboration. The paper follows the release and says 15.

One of the two is wrong and it is not decidable from here — 0787 is a sacralisation graded
Castellvi IIIb with only T11 through L4 in the field of view, and is the single most
ambiguous thoracolumbar junction in the corpus. It is also, separately, one of the two
records carrying a residual rib–vertebra offset.
