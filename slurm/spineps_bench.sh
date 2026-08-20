#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_spineps
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/spineps_%A_%a.out
#SBATCH --error=logs/spineps_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=msa1
# =============================================================================
# spineps_bench — SPINEPS (CT mode) + Möller ribs + rib measurements over the cohort.
# Sharded GPU --array like v4_ribs.sh; resumable (per-case markers in <out>/_done).
#
# Container: containers/ctspinopelvic1k-spineps.sif
#   docker build -f docker/Dockerfile.spineps -t <user>/ctspinopelvic1k-spineps:latest .
#   docker push  <user>/ctspinopelvic1k-spineps:latest
#   DOCKERHUB_USER=<user> SPINEPS_ONLY=1 bash scripts/hpc_pull.sh
# The SPINEPS CT weights are BAKED into the image (compute nodes have no internet).
#
# Möller rib weights are NOT baked — same dir v4_ribs.sh already uses:
#   models/moller_ribseg/ribseg_model_weights  (Zenodo 10.5281/zenodo.14850928)
# Set RIB_MODEL='' to run SPINEPS only (no ribs, no rib measurements).
#
#   N_SHARDS_OVERRIDE=8 sbatch slurm/spineps_bench.sh
#   resubmit a subset:  N_SHARDS_OVERRIDE=8 sbatch --array=3 slurm/spineps_bench.sh
#
# Scoring (CPU, after all shards land):
#   singularity exec containers/ctspinopelvic1k-spineps.sif \
#       python3 /workspace/scripts/benchmark_spineps.py \
#           --pred_dir <OUT>/spineps --gt_dir <GT>/labels --out_dir results/spineps_bench \
#           --splits_file <GT>/splits_5fold.json --rib_csv '<OUT>/rib_measurements_shard*.csv'
# =============================================================================
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

GT_DIR="${GT_DIR:-${DATA_DIR}/hf_export_v5}"                 # CTs + labels to benchmark against
OUT_DIR="${OUT_DIR:-${DATA_DIR}/spineps_bench}"
SPINEPS_SIF="${SPINEPS_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-spineps.sif}"
RIB_MODEL="${RIB_MODEL:-${PROJECT_ROOT}/models/moller_ribseg/ribseg_model_weights}"
RIB_FOLDS="${RIB_FOLDS:-0}"                                   # "0" (fast) or "0,1,2" (ensemble)
RIB_CHECKPOINT="${RIB_CHECKPOINT:-checkpoint_final.pth}"
MODEL_SEMANTIC="${MODEL_SEMANTIC:-ct}"
MODEL_INSTANCE="${MODEL_INSTANCE:-ct_instance}"
MODEL_LABELING="${MODEL_LABELING:-ct_labeling}"
CALC_ORIENTATION="${CALC_ORIENTATION:-0}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-0}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

[[ -f "${SPINEPS_SIF}" ]] || { echo "ERROR: container missing ${SPINEPS_SIF}"; exit 1; }
[[ -d "${GT_DIR}/ct" ]]   || { echo "ERROR: no CTs at ${GT_DIR}/ct"; exit 1; }
if [[ -n "${RIB_MODEL}" && ! -f "${RIB_MODEL}/plans.json" ]]; then
    echo "ERROR: no nnU-Net rib model at ${RIB_MODEL} (need dataset.json/plans.json/fold_*);"
    echo "       unzip the Zenodo weights, or set RIB_MODEL='' to skip ribs."; exit 1
fi
mkdir -p "${LOGS_DIR}" "${OUT_DIR}"

NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NODE_SCRATCH}/container_tmp"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

echo "[spineps] shard ${SHARD_ID}/${N_SHARDS}  gt=${GT_DIR} -> out=${OUT_DIR}"
echo "[spineps] models: semantic=${MODEL_SEMANTIC} instance=${MODEL_INSTANCE} labeling=${MODEL_LABELING}"
echo "[spineps] ribs:   ${RIB_MODEL:-<disabled>} folds=${RIB_FOLDS}  $(date)"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${HOST_CONTAINER_TMP}:/tmp"
[[ -n "${RIB_MODEL}" ]] && BINDS="${BINDS},${RIB_MODEL}:${RIB_MODEL}"
# PYTHONPATH keeps /opt/ribseg (the rib-segmentation clone) importable alongside our scripts.
CENV="PYTHONPATH=/opt/ribseg:/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ARGS=( --ct_dir  "/data/$(realpath --relative-to="${DATA_DIR}" "${GT_DIR}")/ct"
       --out_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${OUT_DIR}")"
       --model_semantic "${MODEL_SEMANTIC}" --model_instance "${MODEL_INSTANCE}"
       --model_labeling "${MODEL_LABELING}"
       --device cuda --shard_id "${SHARD_ID}" --n_shards "${N_SHARDS}" )
[[ -n "${RIB_MODEL}" ]]         && ARGS+=( --rib_model "${RIB_MODEL}" --folds "${RIB_FOLDS}"
                                           --checkpoint "${RIB_CHECKPOINT}" )
[[ "${CALC_ORIENTATION}" == "1" ]] && ARGS+=( --calc_orientation )
[[ "${RESUME}" == "0" ]]        && ARGS+=( --no_resume )
[[ "${LIMIT}" != "0" ]]         && ARGS+=( --limit "${LIMIT}" )

stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${SPINEPS_SIF}" python3 -u /workspace/scripts/spineps_ct_pipeline.py "${ARGS[@]}"

echo "[spineps] shard ${SHARD_ID} done -> ${OUT_DIR}  $(date)"
