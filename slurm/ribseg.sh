#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_ribseg
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/ribseg_%A_%a.out
#SBATCH --error=logs/ribseg_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=msa1
# =============================================================================
# ribseg — relabel the rib cage with RibSeg v2 (PointNet++), REPLACING the
# TotalSegmentator ribs in the v3 tree (TS mis-numbers the lower ribs). Runs AFTER
# ship_v3 (needs a built v3 tree). Sharded like v3_totalseg.sh: an --array of GPU
# tasks each doing a disjoint 1/N of the cases (index %% N in ribseg_relabel.py).
#
# RibSeg is pure-Python PointNet++ (torch + nibabel) -> it runs in the EXISTING
# TotalSegmentator container; no new image. Clone the repo + drop the weights and
# bind it in:
#   git clone https://github.com/HINTLab/RibSeg  third_party/RibSeg
#   # place pretrained weights at: third_party/RibSeg/log/<LOG_DIR>/checkpoints/best_model.pth
#
# Resubmit a failed subset (pin the original shard count):
#   N_SHARDS_OVERRIDE=8 sbatch --array=3,5 slurm/ribseg.sh
#
# Options (env):
#   V3_DIR       v3 source tree     (default: data/hf_export_v3)
#   V4_DIR       v4 output tree     (default: data/hf_export_v4)
#   RIBSEG_DIR   cloned RibSeg repo (default: third_party/RibSeg)
#   RIBSEG_MODEL models/<name>      (default: pointnet2_part_seg_msg)  [VERIFY]
#   RIBSEG_LOG   checkpoint dir under log/ (default: c2_a)             [VERIFY]
#   NNUNET_SIF   CUDA container     (default: containers/ctspinopelvic1k-ts.sif)
#   RIBSEG_LIMIT cap cases (debug)  (default: 0 = all)
#   RESUME       1 = continue from markers (default)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"
V4_DIR="${V4_DIR:-${DATA_DIR}/hf_export_v4}"
RIBSEG_DIR="${RIBSEG_DIR:-${PROJECT_ROOT}/third_party/RibSeg}"
RIBSEG_MODEL="${RIBSEG_MODEL:-pointnet2_part_seg_msg}"
RIBSEG_LOG="${RIBSEG_LOG:-c2_a}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
RIBSEG_LIMIT="${RIBSEG_LIMIT:-0}"
RESUME="${RESUME:-1}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

[[ -f "${NNUNET_SIF}" ]]               || { echo "ERROR: container missing at ${NNUNET_SIF}"; exit 1; }
[[ -f "${V3_DIR}/manifest.json" ]]     || { echo "ERROR: no v3 tree at ${V3_DIR} (run ship_v3 first)"; exit 1; }
[[ -d "${RIBSEG_DIR}/models" ]]        || { echo "ERROR: RibSeg repo not at ${RIBSEG_DIR} (git clone HINTLab/RibSeg)"; exit 1; }
[[ -f "${RIBSEG_DIR}/log/${RIBSEG_LOG}/checkpoints/best_model.pth" ]] || {
    echo "ERROR: RibSeg weights missing at ${RIBSEG_DIR}/log/${RIBSEG_LOG}/checkpoints/best_model.pth"
    echo "       (download the pretrained checkpoint — see OPEN ITEM [A])"; exit 1; }
mkdir -p "${LOGS_DIR}" "${V4_DIR}/labels"

# Node-local scratch for the singularity sandbox (same policy as v3_totalseg.sh).
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NODE_SCRATCH}/container_tmp"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

# One-time mirror of the v3 CTs + manifest into v4 (shard 0 only) so v4 is a complete
# tree; every shard writes only its own labels. (RibSeg replaces ribs; everything else
# is carried verbatim from v3.)
if [[ "${SHARD_ID}" == "0" ]]; then
    echo "[ribseg] shard 0: mirroring v3 CTs + manifest -> ${V4_DIR}"
    mkdir -p "${V4_DIR}/ct"
    cp -an "${V3_DIR}/ct/." "${V4_DIR}/ct/" 2>/dev/null || true
    for f in manifest.json dataset_labels.json splits_5fold.json README.md; do
        [[ -f "${V3_DIR}/${f}" ]] && cp -an "${V3_DIR}/${f}" "${V4_DIR}/${f}" 2>/dev/null || true
    done
fi

echo "======================================================================"
echo " RibSeg v2 rib relabel  (resume=${RESUME})"
echo "   Shard      : ${SHARD_ID} / ${N_SHARDS}"
echo "   v3 source  : ${V3_DIR}"
echo "   v4 out     : ${V4_DIR}"
echo "   RibSeg     : ${RIBSEG_DIR}  model=${RIBSEG_MODEL}  log=${RIBSEG_LOG}"
echo "   container  : ${NNUNET_SIF}"
echo "   GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
echo "   Started    : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${RIBSEG_DIR}:/opt/ribseg,${HOST_CONTAINER_TMP}:/tmp"
CENV="PYTHONPATH=/workspace/scripts:/opt/ribseg:/workspace,PYTHONUNBUFFERED=1"
CENV+=",PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ARGS=( --v3_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${V3_DIR}")"
       --out_dir    "/data/$(realpath --relative-to="${DATA_DIR}" "${V4_DIR}")"
       --ribseg_dir /opt/ribseg
       --ribseg_model "${RIBSEG_MODEL}"
       --log_dir    "${RIBSEG_LOG}"
       --device     cuda
       --shard_id   "${SHARD_ID}" --n_shards "${N_SHARDS}" )
[[ "${RIBSEG_LIMIT}" != "0" ]] && ARGS+=( --limit "${RIBSEG_LIMIT}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )

stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${NNUNET_SIF}" python3 -u /workspace/scripts/ribseg_relabel.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " RibSeg done at $(date)  ->  ${V4_DIR}"
echo "======================================================================"
