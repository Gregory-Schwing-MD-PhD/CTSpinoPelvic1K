#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_check_spine_labels
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/check_spine_labels_%j.out
#SBATCH --error=logs/check_spine_labels_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# check_spine_labels — is the rib anchor already in the CTSpine1K GT?
# Scans the placed spine masks (raw VerSe labels, before export_hf drops the
# thoracic ones) and reports how often T12=19 (the anchor, "vertebra above L1"),
# T13=28 (supernumerary, already adjudicated) and L6=25 are present. See
# scripts/check_spine_labels.py.
#
# Options (env):
#   SPINE_DIR   placed spine masks   (default: data/placed/spine)
#   WORKERS     reader threads       (default: 16)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

SPINE_DIR="${SPINE_DIR:-${PLACED_DIR}/spine}"
WORKERS="${WORKERS:-16}"

mkdir -p "${LOGS_DIR}"
[[ -f "${SIF_PATH}" ]]  || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -d "${SPINE_DIR}" ]] || { echo "ERROR: no spine-mask dir at ${SPINE_DIR}"; exit 1; }

# Bind DATA_DIR -> /data and reference SPINE_DIR by its path relative to it.
C_SPINE="/data/$(realpath --relative-to="${DATA_DIR}" "${SPINE_DIR}")"

echo "======================================================================"
echo " check_spine_labels — anchor-in-GT check"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   spine_dir : ${SPINE_DIR}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

# shellcheck disable=SC2086
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/check_spine_labels.py \
        --spine_dir "${C_SPINE}" --workers "${WORKERS}"

echo ""
echo "======================================================================"
echo " check_spine_labels done at $(date)"
echo "======================================================================"
