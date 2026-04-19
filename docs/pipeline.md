# CTSpinoPelvic1K — Pipeline

This document walks through the three SLURM stages in detail.  Each stage is
idempotent: re-running it with existing outputs will either skip or resume
(and nothing gets clobbered unless you set `FORCE_PLACEMENT=1`).

```
┌───────────────────┐   ┌────────────────────┐   ┌─────────────────────┐
│  Stage 1          │   │  Stage 2           │   │  Stage 3            │
│  download_raw     │ → │  create_dataset    │ → │  export_dataset     │
└───────────────────┘   └────────────────────┘   └─────────────────────┘
  TCIA / Spine /         build_db.py +              split + NPZ export
  Pelvic downloads       place_fused_masks.py       + Hugging Face push
                         (+ QC figures)
```

## Stage 1 — `download_raw`

Fetches three source datasets in parallel (each toggleable with
`RUN_TCIA=1`, `RUN_SPINE=1`, `RUN_PELVIC=1`):

- **TCIA CT COLONOGRAPHY** via `tcia_utils.downloadSeries`.  Two scopes:
  - `TCIA_SCOPE=all`      — all ~3,451 CT series (~350 GB)
  - `TCIA_SCOPE=filtered` — only patients annotated in CTPelvic1K COLONOG (~1,194)
- **CTSpine1K (COLONOG subset)** via `huggingface_hub.snapshot_download`.
  Requires `HF_TOKEN` because the COLONOG fold is gated.
- **CTPelvic1K dataset2** via Zenodo.

Output tree (under `${DATA_ROOT}`):
```
data/
├── tcia/
│   ├── {series_uid_1}/*.dcm
│   ├── {series_uid_2}/*.dcm
│   └── ...
├── ctspine1k/
│   ├── rawdata/labels/COLONOG/*_seg.nii.gz
│   └── rawdata/volumes/COLONOG/*.nii.gz
└── ctpelvic1k/
    └── masks/CTPelvic1K_dataset2_mask_mappingback/dataset2_*.nii.gz
```

Partial downloads are resumable — the script checks filesystem state on
every run and skips series/files that are already complete.

## Stage 2 — `create_dataset`

Two steps, both implemented as standalone Python:

### Step A — `build_db.py`

Builds `data/patient_db.json`, the canonical patient-centred database.  For every
COLONOG / CTC patient it records:

- All TCIA series (SeriesInstanceUIDs, study groupings, reconstructed spatial
  metadata, scout/CT classification) grouped by DICOM PatientID.
- Every spine seg file owned by that patient (identified by filename-encoded UID).
- Every pelvic mask file owned by that patient (identified by filename-encoded UID).
- For each mask: a ranked list of TCIA series candidates with per-candidate
  confidence + reasons.  Assignment is patient-local — a pelvic mask can only be
  matched to series belonging to the same patient, eliminating the class of
  cross-patient mis-assignment errors that plagued affine-based matchers.
- LSTV labels from both sources (vertebral counting for spine, filename
  qualifiers for pelvic) plus cross-check agreement.

### Step B — `place_fused_masks.py`

Consumes `patient_db.json` and, for each patient, finds the TCIA series that
maximises bone coverage (HU > 200) when each mask is resampled into it:

1. **dcm2niix** converts every candidate series DICOM folder → NIfTI.
2. **Spine placement**: world-space affine nearest-neighbour resample onto each
   candidate, tracking the best `bone_pct`.  Fallback chain: phase
   cross-correlation (8 axis-flip combinations) → CTSpine1K native affine
   anchor (last resort).
3. **Pelvic placement**: bone z-profile cross-correlation (CTPelvic1K masks
   are cropped subvolumes, so bone profile matching is the right primitive).
4. **PCA-based IS-ordering check** on every placed spine seg to flag
   vertebral-label inversions (robust to lumbar lordosis — tolerance 8 mm).
5. Writes placed masks in PIR orientation alongside a per-case JSON sidecar
   with `series_uid`, `bone_pct`, `vox`, `method`, `IS_ok`.

Final `data/placed/placed_manifest.json` contains every patient's winning
series + LSTV fields + match_type (fused / separate / spine_only / pelvic_only).

### Step C — QC figures (optional)

`visualize_qc.py` reads the manifest and produces per-case PNG figures,
routing vertebral-ordering failures to `data/qc/per_case/is_fail/` for fast
triage.

## Stage 3 — `export_dataset`

`export_hf.py` turns the placed manifest into Hugging Face-ready artefacts:

- **10-class label remap**.  CTSpine1K VerSe labels 20–25 → {1..6}, CTPelvic1K
  labels 1/2/3 → {7,8,9} (sacrum / left hip / right hip), with the CTPelvic1K
  sacrum taking priority over CTSpine1K 26.
- **Per-case .npz** containing `ct` (int16, Z,Y,X), `label` (uint8, Z,Y,X),
  `affine` (float32, 4×4), and a JSON `meta` string with token, series_uid,
  LSTV fields, bone_pct, and the label scheme.
- **Stratified splits** (70/15/15) keyed on (lstv_class × match_type) so every
  split contains the rare LSTV classes.
- **README.md + splits_summary.json** describing the dataset.
- Optional `--push` to a Hugging Face Hub repo (requires `HF_TOKEN`).

## Idempotency and forcing re-runs

- Stage 1 is resumable — rerunning only downloads missing files.
- Stage 2's `build_db.py` caches the TCIA index (`.tcia_patient_index.json`)
  and the mask parsing results (`.spine_mask_cache.pkl`, `.pelvic_mask_cache.pkl`).
  Pass `--rebuild_tcia_index` / `--rebuild_mask_cache` to force a full rescan.
- Stage 2's `place_fused_masks.py` skips placements whose output file already
  exists.  Set `FORCE_PLACEMENT=1` to delete all cached placed files and
  re-run from scratch.
- Stage 3 overwrites the target output directory — re-running is safe.


---

## Stage 4 — TotalSegmentator zero-shot benchmark

**Entry point:** `slurm/benchmark_totalseg.sh` (via `make benchmark-totalseg`)

**Goal:** establish a publication-grade zero-shot baseline against which downstream LSTV-aware models are evaluated, and characterise the L5 / sacrum / L6 confusion that motivates the dataset.

**Inputs:**
- `data/hf_export/` (from Stage 3)
- `containers/ctspinopelvic1k-ts.sif` (via `make hpc-pull`)
- TotalSegmentator model weights, cached under `$HOME/totalseg_weights`

**What runs:**

```
scripts/benchmark_totalseg.py
    --dataset_dir data/hf_export
    --split       all_fused
    --window_mm   40.0
    --device      gpu
```

Note: `--fast` is deliberately NOT used. Full TS precision is required for publication numbers.

The benchmark:

1. Loads `CTSpinoPelvic1K` via `scripts/dataset_interface.py`, filters to the `all_fused` split (every fused-config record), and for each case:
2. Runs `totalsegmentator(..., task="total", ml=True, roi_subset=[...L1-L5, sacrum, hips])` with a per-case prediction cache at `ts_preds/{token}_{config}/segmentation.nii.gz`
3. Resamples TS output to the GT label grid via nearest-neighbour `SimpleITK.ResampleImageFilter`
4. Remaps TS IDs to the unified 10-class scheme (`{31→1 (L1), 30→2 (L2), 29→3 (L3), 28→4 (L4), 27→5 (L5), 25→7 (sacrum), 77→8 (L hip), 78→9 (R hip)}`; TS has no L6).
5. Computes per-class Dice + HD95 and the junction-analysis block (40 mm axial window around L5/S1):
   - `junction_dice` for L5 / L6 / sacrum
   - `labelling_error_rate` (voxels in the junction window where TS label ≠ GT label)
   - `l5_called_sacrum_rate`  (the main LSTV confusion signal)
   - `s1_called_l5_rate`
   - `ts_assigned_l6` flag (always False — structural inability)

**Outputs** (`results/totalseg_bench_{jobid}/`):

```
paper_tables.txt              Table 5 (human-readable)
benchmark_results.json        Per-case + aggregate dump
benchmark_summary.json        Aggregate only
benchmark_per_case.csv        Flat CSV for plotting
ts_preds/{token}_{config}/    Cached TS predictions (reused on rerun)
```

Aggregation subgroups: `all`, `fused_only`, `spine_only`, `pelvic_native`, `normal`, `any_lstv`, `sacralization`, `lumbarization`.

**Complementary figure pipeline:**

```
# GT LSTV panel (4 panels: Normal / Sac-morph / Sac-count / Lumbarization)
make render-lstv-gt

# TS LSTV panel — same 4 tokens, TS predictions rendered in the same color scheme
TS_PRED_DIR=results/totalseg_bench_{jobid}/ts_preds make render-lstv-ts
```

Running both produces `LSTV_panel_4x1_gt.jpg` and `LSTV_panel_4x1_ts.jpg` — the side-by-side comparison illustrating TS's morphological misclassification and its structural inability to emit L6.
