#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_verify_seriesuid
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_seriesuid_%j.out
#SBATCH --error=logs/verify_seriesuid_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# verify_mask_seriesuid — empirically confirm the CTSpine1K / CTPelvic1K masks
# carry NO SeriesInstanceUID (only a PatientID), so they cannot be trivially
# matched to a TCIA DICOM series. See scripts/verify_mask_seriesuid.py.
#
# Options (env):
#   TCIA_DIR    raw COLONOG download  (default: data/tcia)
#   SPINE_DIR   CTSpine1K masks       (default: data/ctspine1k)
#   PELVIC_DIR  CTPelvic1K masks      (default: data/ctpelvic1k)
#   SAMPLE      cap masks per source  (default: 0 = all)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

TCIA_DIR="${TCIA_DIR:-${DATA_DIR}/tcia}"
SPINE_DIR="${SPINE_DIR:-${CTSPINE1K_DIR:-${DATA_DIR}/ctspine1k}}"
PELVIC_DIR="${PELVIC_DIR:-${CTPELVIC1K_DIR:-${DATA_DIR}/ctpelvic1k}}"
SAMPLE="${SAMPLE:-0}"

mkdir -p "${LOGS_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -d "${TCIA_DIR}" ]] || { echo "ERROR: no TCIA dir at ${TCIA_DIR}"; exit 1; }

C_TCIA="/data/$(realpath --relative-to="${DATA_DIR}" "${TCIA_DIR}")"
C_SPINE="/data/$(realpath --relative-to="${DATA_DIR}" "${SPINE_DIR}")"
C_PELVIC="/data/$(realpath --relative-to="${DATA_DIR}" "${PELVIC_DIR}")"

ARGS=( --tcia_dir "${C_TCIA}" --sample "${SAMPLE}" )
[[ -d "${SPINE_DIR}"  ]] && ARGS+=( --spine_dir  "${C_SPINE}" )  || echo "NOTE: no spine dir at ${SPINE_DIR}"
[[ -d "${PELVIC_DIR}" ]] && ARGS+=( --pelvic_dir "${C_PELVIC}" ) || echo "NOTE: no pelvic dir at ${PELVIC_DIR}"

echo "======================================================================"
echo " verify_mask_seriesuid — masks lack a SeriesInstanceUID?"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   tcia      : ${TCIA_DIR}"
echo "   spine     : ${SPINE_DIR}"
echo "   pelvic    : ${PELVIC_DIR}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

# shellcheck disable=SC2086
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/verify_mask_seriesuid.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " verify_mask_seriesuid done at $(date)"
echo "======================================================================"
