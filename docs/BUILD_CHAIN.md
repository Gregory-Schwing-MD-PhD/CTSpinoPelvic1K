# CTSpinoPelvic1K build archive — how v10 is regenerated from the published sources

This archive holds every input to the v10 release that no script can regenerate, so that
the released label volumes (Zenodo 10.5281/zenodo.22642578, concept 10.5281/zenodo.22139642)
can be rebuilt from the public source datasets plus the code in the repository
`github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K` at tag `v10-build-archive`.

Three kinds of thing went into the release. **Public sources** (TCIA imaging, the CTSpine1K
and CTPelvic1K annotations) are cited, not copied, except for the two small label sets
that pin the exact files used. **Model outputs** (TotalSegmentator, the Möller rib network,
the spinopelvic-seg pseudolabels) are not bit-reproducible across software versions and
GPUs, so the volumes they produced are archived rather than re-run. **Human work** (student
rib and thoracic corrections, radiologist hardware reads, hand-corrected records, Castellvi
grades) is archived in full. Everything after the last human touch is a deterministic
script, and the archive carries a check that proves the final hops regenerate the
published volumes voxel for voxel.

## Sources, cited

| source | where | used for |
|---|---|---|
| TCIA CT COLONOGRAPHY | doi 10.7937/K9/TCIA.2015.NWTESAY1; series listed by SeriesInstanceUID in `manifest.json` of every release | the 802 CT volumes (`reconstruct_ct.py` in the release rebuilds them) |
| CTSpine1K, COLONOG fold | Hugging Face `alexanderdann/CTSpine1K` (gated) | vertebral annotations, C1–L6 and sacrum |
| CTPelvic1K dataset 2 masks | Zenodo record 4588403 | sacrum and hip annotations |
| Möller rib network weights | Zenodo 10.5281/zenodo.14850928 | binary rib mask used to complete the v4 ribs |
| TotalSegmentator 2.4.0 | `docker/Dockerfile.totalsegmentator` | femora, S1 carve, rib numbering, thoracic column in v3 |
| spinopelvic-seg checkpoints | Hugging Face `OpenSpineConsortium/spinopelvic-seg-checkpoints` | out-of-fold pseudolabels for the partially annotated records (v2) |
| containers | `ctspinopelvic1k.sif`, `ctspinopelvic1k-ts.sif`, built from `docker/` | every stage below |

## The chain

Each row names the script in the repository, what it consumed, and which archive component
carries the input that cannot be regenerated.

| step | script | input | output | archived in |
|---|---|---|---|---|
| 1 download | `slurm/download_raw.sh` | the three sources | `data/tcia`, `data/ctspine1k`, `data/ctpelvic1k` | `01_sources` (labels and masks only) |
| 2 patient database | `scripts/build_db.py` | source file names, DICOM headers | `patient_db.json` | `02_placement` |
| 3 placement | `scripts/place_fused_masks.py`, `scripts/apply_manual_flips.py` with `configs/flip_list.json` | 2 + the CTs | `data/placed/{spine,pelvic,pelvic_propagated}`, placed manifests | `02_placement` |
| 4 export and pseudolabel (v1–v2) | `scripts/export_hf.py`, `scripts/pseudolabel.py` | 3 + spinopelvic-seg out-of-fold predictions | v2 tree | superseded by 5 |
| 5 thoracic column, femora, S1 (v3) | `scripts/build_v3_totalseg.py` (`slurm/ship_v3.sh`) | v2 + TotalSegmentator | v3 tree | superseded by 6 |
| 6 ribs (v4) | `scripts/build_v4_ribs.py` (`slurm/ship_v4.sh`) | v3 + Möller rib mask + TotalSegmentator rib numbering | v4 labels, `rib_worklist.json` | `03_v4_base` (the model-output base every human correction was made on) |
| 7 student review | review Spaces in `review_service/`; ledgers reviews-ribs (169 cases), reviews-spine (113), reviews-triaged (106) | v4 | per-slot corrected labels and a finalised label per case | `04_reviews` (handles pseudonymised) |
| 8 merge (v5) | `scripts/build_final_dataset.py` (`slurm/build_v5.sh`), then `slurm/finalize_v5.sh` = `strip_vertebra_speckle.py --apply`, `lumbar_rib_class_v5.py`, `fix_label_breaks.py`, `fix_rib_offsets.py` | v4 + 7 | `data/v5_final` | `05_v5` (the merged, cleaned tree and the per-record fixes: 0068 thoracic relabel, 0816 rebuild, detached-piece and L6 audits) |
| 9 hardware and laterality (v6) | `scripts/build_v6.py` (`slurm/build_v6.sh`), `scripts/relabel_hips_by_midline.py`, `scripts/fix_hip_sidedness.py` | v5 + radiologist-read hardware labels (11 records) + 22 hip corrections | v6 tree | `06_v6_inputs` |
| 10 S1 re-carve (v7) | `scripts/recarve_s1_all.py` (`slurm/recarve_s1_all.sh`) | v6 | `data/s1_recarve/labels` = the v7 volumes | `07_v7` |
| 11 metadata (v8) | `scripts/apply_castellvi_consensus.py`, `zenodo/new_version.py` | v7 + `docs/castellvi_consensus.csv` | manifest with two-reader Castellvi grades; soft-tissue names retired | in git |
| 12 renumber (v9) | `scripts/renumber_labels.py --map v9` | v7 volumes | contiguous 0–66 | in git |
| 13 renumber (v10) | `scripts/renumber_labels.py --map v10`, `zenodo/finalize_v9.py` | v9 | thirteen ribs per side, lumbar ribs 60–61, hardware 62–68 | in git; published |

`CHAIN_CHECK_v7_to_v10.txt` is the output of `scripts/compare_label_trees.py` after
running steps 12 and 13 on the archived v7 volumes and comparing against the published v10
`labels.zip`. It states how many of the 802 volumes are voxel-identical.

## What is in each component

| file | contents | size |
|---|---|---|
| `01_sources.tar.zst` | CTSpine1K COLONOG labels and metadata; CTPelvic1K dataset 2 masks and metadata | ~0.8 GB |
| `02_placement.tar.zst` | `patient_db.json`, placed spine and pelvic masks on the chosen TCIA series, both placed manifests, `configs/flip_list.json` | ~4.5 GB |
| `03_v4_base.tar.zst` | the 802 v4 labels (TotalSegmentator + Möller ribs on the v3 base), v4 manifest and label dictionary, rib worklist | ~1.9 GB |
| `04_reviews.tar.zst` | the three review ledgers with every slot label and finalised label; reviewer handles replaced by annotator_NN | ~0.5 GB |
| `05_v5.tar.zst` | `v5_final` (802), the 0068 and 0816 corrections, detached-piece and L6 audit files, speckle report, QC tables | ~1.9 GB |
| `06_v6_inputs.tar.zst` | radiologist-read hardware labels (before and after dust removal), hardware proposals with renders, the 22 pre-correction hip labels, v6 manifest | ~0.4 GB |
| `07_v7.tar.zst` | the v7 volumes (S1 re-carved) with their QC table, v7 manifest, label dictionary, checksums, README, KNOWN_ISSUES | ~1.3 GB |
| `08_grid_worktree.tar.zst` | the HPC checkout's uncommitted diff and untracked helper scripts at archive time, including `recarve_s1_all.py` | small |
| `09_nnunet_meta.tar.zst` | nnU-Net plans, fingerprint and fold splits of the training runs whose preprocessed data was deleted | small |

Label identifiers inside components 03–07 follow the scheme of their era (ribs 34–57,
soft tissue 58–73 reserved, lumbar ribs 74–75, hardware 76–82, ignore 255). Step 12 maps
them to the current scheme; `scripts/label_scheme.py` carries both tables.

## What was deliberately not archived, and why

- CT volumes (180 GB): TCIA holds them; `reconstruct_ct.py` in every release rebuilds the
  exact resampled volume, and the Hugging Face mirror `OpenSpineConsortium/CTSpinoPelvic1K`
  carries them ready-made.
- CTSpine1K volumes (151 GB) and CTPelvic1K images (29 GB): public, and only their labels
  entered the build.
- nnU-Net raw and preprocessed trees (1.1 TB): deterministic from the labels via
  `tools/convert_hf_to_nnunet.py` in `spinesurg-ct-nnunet`.
- Trained checkpoints: on Hugging Face under `OpenSpineConsortium`.
- Intermediate export trees v2, v3, v5, v6 and the earlier Zenodo staging folders: each is
  either superseded by a later archived tree or published.
