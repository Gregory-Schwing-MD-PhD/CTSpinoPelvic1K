#!/usr/bin/env bash
#SBATCH --job-name=ctsp_resegment_thoracic
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/resegment_thoracic_%j.out
#SBATCH --error=logs/resegment_thoracic_%j.err
# =============================================================================
# resegment_thoracic.sh — one case, one GPU, the thoracic levels that a scan
# images and its label does not carry.
#
#   CASES=0068 sbatch slurm/resegment_thoracic.sh
#   CASES="0068 1234" sbatch slurm/resegment_thoracic.sh
#
# Writes a PROPOSAL to ${OUT_DIR} and never touches ${LABEL_DIR}. See
# scripts/resegment_thoracic.py for why TotalSegmentator's vertebra NAMES are
# thrown away and the level identity is anchored on the labelled L1 below.
#
# Everything below the launcher line is lifted from slurm/v3_totalseg.sh, which
# is where the container binds, the node-local scratch policy and the TS weight
# paths were worked out. The one thing that differs is scale: a single case for
# an hour rather than 802 across an eight-way array.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

CASES="${CASES:?set CASES=0068 (space separated for more than one)}"
CT_DIR="${CT_DIR:-${DATA_DIR}/hf_export_v5/ct}"
LABEL_DIR="${LABEL_DIR:-${DATA_DIR}/v5_final}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/thoracic_fix}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"
HIGHEST="${HIGHEST:-T8}"

[[ -f "${NNUNET_SIF}" ]] || { echo "ERROR: TS container missing at ${NNUNET_SIF}"; exit 1; }
mkdir -p "${LOGS_DIR}" "${OUT_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

# NODE-LOCAL /tmp, not NFS. TS writes hundreds of MB of temp NIfTIs per case and the
# NFS-bound container /tmp was what got wiped out from under the original v3 run.
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NODE_SCRATCH}/container_tmp"
export XDG_RUNTIME_DIR="${NODE_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

echo "======================================================================"
echo " resegment thoracic   cases: ${CASES}"
echo "   Job  : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
echo "   CT   : ${CT_DIR}"
echo "   v5   : ${LABEL_DIR}   (read only)"
echo "   out  : ${OUT_DIR}     (proposals)"
echo "   cap  : nothing named above ${HIGHEST}"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${HOST_CONTAINER_TMP}:/tmp"
BINDS+=",${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
CENV="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"
CENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS},TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CENV+=",HOME=${TOTALSEG_CONFIG_DIR},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REL() { realpath --relative-to="${DATA_DIR}" "$1"; }

for CASE in ${CASES}; do
  echo ""
  echo "--- ${CASE} ------------------------------------------------------------"
  CT="${CT_DIR}/${CASE}_ct.nii.gz"
  LB="${LABEL_DIR}/${CASE}_label.nii.gz"
  [[ -f "${CT}" ]] || { echo "  ! no CT at ${CT}"; continue; }
  [[ -f "${LB}" ]] || { echo "  ! no label at ${LB}"; continue; }
  stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
      "${NNUNET_SIF}" python3 -u /workspace/scripts/resegment_thoracic.py \
      --case "${CASE}" \
      --ct    "/data/$(REL "${CT}")" \
      --label "/data/$(REL "${LB}")" \
      --out   "/data/$(REL "${OUT_DIR}")" \
      --device gpu --highest "${HIGHEST}"
done

echo ""
echo "======================================================================"
echo " done $(date) -> ${OUT_DIR}   (PROPOSALS; v5 untouched)"
echo "======================================================================"
