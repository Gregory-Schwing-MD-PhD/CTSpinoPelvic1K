# SPINEPS (CT) benchmark + Möller rib measurements

Everything needed to run the Hendrik-code CT toolchain over CTSpinoPelvic1K and score it
against our labels. One container, three stages.

| stage | upstream | what it gives us |
|---|---|---|
| spine | [spineps](https://github.com/Hendrik-code/spineps) | vertebra **instance** mask (VerSe ids), spine **semantic** (subregion) mask, centroids |
| ribs | Möller binary rib nnU-Net ([Zenodo 10.5281/zenodo.14850928](https://doi.org/10.5281/zenodo.14850928)) | binary rib mask |
| rib measurement | [rib-segmentation](https://github.com/Hendrik-code/rib-segmentation) | ribs assigned to their vertebra, rib **length**, stump-rib **features** |

Stage 3 needs stages 1 and 2 as input — that is why they live in one image.

## The "new mode of inference"

SPINEPS used to be MR-only (`t2w` / `t1w` / `vibe`). It now ships **modality-specific CT
models**, so the CT run is not "the T2w model pointed at a CT":

```
spineps sample -i sub-x_ct.nii.gz \
    -model_semantic ct -model_instance ct_instance -model_labeling ct_labeling \
    -ignore_bids_filter -ignore_inference_compatibility -non4
```

Weights come from the GitHub releases (`ct.zip` + `CT_instance.zip` @ v1.4.2,
`ct_labeling.zip` @ v1.4.0). `-model_labeling` defaults to `t2w_labeling`, so it **must** be
passed explicitly for CT. `-non4` skips N4 bias correction, an MR step.

## Build

```bash
# local workstation
DOCKERHUB_USER=gregoryschwingmdphd SPINEPS_ONLY=1 ./scripts/docker_push.sh

# pin upstream instead of tracking main:
DOCKERHUB_USER=... SPINEPS_ONLY=1 SPINEPS_REF=<sha> RIBSEG_REF=<sha> ./scripts/docker_push.sh
```

The SPINEPS CT weights are **baked into the image** (`docker/fetch_spineps_weights.py`, run at
build time and the build fails if they don't land) because compute nodes have no outbound
internet and SPINEPS auto-downloads on first use. Resolved upstream refs are recorded in
`/opt/ctspinopelvic1k/BUILD_INFO.txt` inside the image.

The Möller rib weights are **not** baked — they're already on the grid at
`models/moller_ribseg/ribseg_model_weights/` (the flattened nnU-Net dir `v4_ribs.sh` uses) and
get bind-mounted.

```bash
# HPC
DOCKERHUB_USER=gregoryschwingmdphd SPINEPS_ONLY=1 bash scripts/hpc_pull.sh
```

## Run

```bash
N_SHARDS_OVERRIDE=8 sbatch slurm/spineps_bench.sh          # GPU array, resumable
GT_DIR=$DATA_DIR/hf_export_v5 OUT_DIR=$DATA_DIR/spineps_bench \
    N_SHARDS_OVERRIDE=8 sbatch slurm/spineps_bench.sh
RIB_MODEL='' sbatch slurm/spineps_bench.sh                 # SPINEPS only, no ribs
```

Output tree (`scripts/spineps_ct_pipeline.py`):

```
<out>/spineps/<case>_seg-vert_msk.nii.gz     vertebra instances, native SPINEPS ids
<out>/spineps/<case>_seg-spine_msk.nii.gz    subregion semantic
<out>/spineps/<case>_ctd.json                centroids / POI
<out>/ribs/<case>_ribmask.nii.gz             Möller binary ribs
<out>/rib_measurements_shard<k>.csv          one row per (case, vertebra, side)
<out>/rib_features/<case>.json               full feature dicts
<out>/_done/<case>.json                      resume marker + timings
```

Predictions stay in **SPINEPS' own label space** on disk. Mapping into `label_scheme.py` ids
happens only at scoring time, so a re-score never needs a re-run.

## Score

```bash
singularity exec containers/ctspinopelvic1k-spineps.sif \
  python3 /workspace/scripts/benchmark_spineps.py \
    --pred_dir $DATA_DIR/spineps_bench/spineps \
    --gt_dir   $DATA_DIR/hf_export_v5/labels \
    --out_dir  results/spineps_bench \
    --splits_file $DATA_DIR/hf_export_v5/splits_5fold.json \
    --rib_csv "$DATA_DIR/spineps_bench/rib_measurements_shard*.csv"
```

Two distinct numbers, and the second is the one that matters here:

* **Dice** — per-vertebra overlap against the *same id* in GT.
* **Identification rate** — does the GT vertebra's best-overlapping predicted id *equal* its
  GT id (the standard VerSe metric).

A method can segment every vertebra beautifully and name them all one level off. On an LSTV
cohort that is the failure mode, so the report also gives the **offset histogram**
(`pred_id − gt_id`) and counts cases with a *consistent whole-spine shift* — a counting error,
not a segmentation error — broken down by LSTV phenotype when `--splits_file` is passed.

Scoring is restricted to labels **present in GT**: our thoracic GT is FOV-limited, so a
vertebra SPINEPS finds above our labelled extent is neither credited nor penalised.

Rib *detection* is scored separately by `scripts/benchmark_moller.py` (binary Dice / precision /
recall vs corrected rib GT) — the rib section of `benchmark_spineps.py` summarises the
*measurements* (lengths, stump-rib flags) and cross-tabs the `sr` flag against our lumbar-rib
class 74/75 as an exploratory look, not a scored metric: `sr` is a morphology call on any rib,
while 74/75 means "rib on a lumbar vertebra". They overlap; they are not the same label.

## Gotchas

* SPINEPS wants BIDS-ish filenames. The driver stages each CT as `sub-<alnum>_ct.nii.gz` in a
  private dir and globs the derivatives folder afterwards, rather than predicting output
  filenames (upstream has renamed them before). Mask roles are resolved by filename first,
  then by content — only the semantic mask carries subregion Locations 41–50.
* `numpy` is pinned to 1.26 in the image: antspyx and nnunetv2 both float it and the 2.x ABI
  break silently poisons the nnU-Net predictor.
* The rib-segmentation repo is a plain module tree, not a pip package — `/opt/ribseg` has to be
  on `PYTHONPATH` (the image and the SLURM script both set it).
