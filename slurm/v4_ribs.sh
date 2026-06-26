#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_v4_ribs
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/v4_ribs_%A_%a.out
#SBATCH --error=logs/v4_ribs_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=msa1
# =============================================================================
# v4_ribs — v4 = v3 + Möller binary rib nnU-Net, numbered by our v3 thoracic
# (relabel_ribs) and overlaid on v3. Sharded GPU --array like v3_totalseg.sh; resumable.
# Reuses the TS container (has nnunetv2) — no TPTBox/SPINEPS needed.
#
# ONE-TIME: download ribseg_model_weights.zip (Zenodo 10.5281/zenodo.14850928 ->
# record 14864106) and unzip; it expands to a FLATTENED nnU-Net model dir
# (ribseg_model_weights/ with dataset.json, plans.json, fold_0/1/2/). Point MOLLER_MODEL
# at that dir. No Dataset-id needed — the nnU-Net Python API reads the folder directly.
#
#   N_SHARDS_OVERRIDE=8 sbatch slurm/v4_ribs.sh
#   resubmit a subset:  N_SHARDS_OVERRIDE=8 sbatch --array=1 slurm/v4_ribs.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"
V4_DIR="${V4_DIR:-${DATA_DIR}/hf_export_v4}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
MOLLER_MODEL="${MOLLER_MODEL:-${PROJECT_ROOT}/models/moller_ribseg/ribseg_model_weights}"  # flattened model dir
MOLLER_FOLDS="${MOLLER_FOLDS:-0}"                   # "0" (fast) or "0,1,2" (ensemble)
MOLLER_CHECKPOINT="${MOLLER_CHECKPOINT:-checkpoint_final.pth}"
RESUME="${RESUME:-1}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

[[ -f "${NNUNET_SIF}" ]]              || { echo "ERROR: container missing ${NNUNET_SIF}"; exit 1; }
[[ -f "${V3_DIR}/manifest.json" ]]   || { echo "ERROR: no v3 tree at ${V3_DIR}"; exit 1; }
[[ -f "${MOLLER_MODEL}/plans.json" ]] || { echo "ERROR: no nnU-Net model at ${MOLLER_MODEL} (need dataset.json/plans.json/fold_*); unzip the Zenodo weights"; exit 1; }
mkdir -p "${LOGS_DIR}" "${V4_DIR}/labels"

NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NODE_SCRATCH}/container_tmp"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

# shard 0 mirrors the shared v3 tree into v4 (CTs + manifest + label legend); every
# shard writes only its own v4 labels (= v3 label with ribs replaced).
if [[ "${SHARD_ID}" == "0" ]]; then
    mkdir -p "${V4_DIR}/ct"
    cp -an "${V3_DIR}/ct/." "${V4_DIR}/ct/" 2>/dev/null || true
    for f in manifest.json dataset_labels.json splits_5fold.json README.md; do
        [[ -f "${V3_DIR}/${f}" ]] && cp -an "${V3_DIR}/${f}" "${V4_DIR}/${f}" 2>/dev/null || true
    done
fi

echo "[v4_ribs] shard ${SHARD_ID}/${N_SHARDS}  v3=${V3_DIR} -> v4=${V4_DIR}  model=${MOLLER_MODEL} folds=${MOLLER_FOLDS}  $(date)"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${MOLLER_MODEL}:${MOLLER_MODEL},${HOST_CONTAINER_TMP}:/tmp"
CENV="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ARGS=( --v3_dir  "/data/$(realpath --relative-to="${DATA_DIR}" "${V3_DIR}")"
       --out_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${V4_DIR}")"
       --model_folder "${MOLLER_MODEL}" --folds "${MOLLER_FOLDS}" --checkpoint "${MOLLER_CHECKPOINT}"
       --device cuda --shard_id "${SHARD_ID}" --n_shards "${N_SHARDS}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )

stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${NNUNET_SIF}" python3 -u /workspace/scripts/build_v4_ribs.py "${ARGS[@]}"

echo "[v4_ribs] shard ${SHARD_ID} done -> ${V4_DIR}  $(date)"
