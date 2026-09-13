# Rib QC: pseudolabels (v4) against the release, same code on both

Run 2026-09-13 on the grid (job 40220664). The v4 pseudolabels number right ribs 46-57 and
carry no rib 13; the release numbers them 47-59 (rib 13 inserted) and lumbar ribs 60/61.
`remap_v4.py` rewrites v4 into the release scheme first, so `scripts/review_anatomy_qc.py`
(rib_label_mixing, rib_spine_gap), the fragmentation count in `rib_qc_stages.py`, and
`scripts/qc_rib_vertebra_incidence.py` read both stages identically.

| Check | v4 pseudolabels | Release |
|---|---|---|
| Two numbers on one bone (records / bones) | 151 / 584 | 0 / 0 |
| Rib detached from the spine (records / ribs) | 37 / 40 | 0 / 0 |
| Rib in two or more substantial pieces (records / ribs) | 153 / 510 | 10 / 13 |
| Records failing any of the three gates | 166 of 802 | 10 of 802 |
| Rib offset from its vertebra (misnumbered cases / offset ribs) | 24 / 170 | 0 / 2 |
| No vertebra within reach (ribs) | 38 | 11 |
| Rib on a lumbar body (thoracic-numbered rib articulating with a lumbar vertebra: ribs / cases) | 21 / 14 | 0 / 0 (in the release such ribs carry the lumbar-rib ids 60/61) |
| Ribs evaluable for the incidence check / total | 5,598 / 11,578 | 5,764 / 11,560 |

"Substantial piece": a second connected component of at least 50 voxels and at least 15% of
the largest. `review_anatomy_qc.rib_vertebra_match` is not used here because it does not
account for ribs cut by the scan edge; the incidence script does.

Files: `gates_v4_vs_release.json`, `incidence_v4.json`, `incidence_release.json`,
`misnumbered_*.csv`, and the two scripts.
