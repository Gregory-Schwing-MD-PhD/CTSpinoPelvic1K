#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_check_scouts
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/check_scouts_%j.out
#SBATCH --error=logs/check_scouts_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# check_scouts — are the COLONOG scout/topogram series on disk, and do they
# reach high enough to count ribs? Two questions (see scripts/check_scouts.py):
#   (a) recoverability — filters the cached TCIA index to is_scout records and
#       stats each series_dir; reports coverage joined to the manifest tokens.
#   (b) craniocaudal extent — for a sample, the scout's SI extent (mm) + z-range
#       vs the matched axial CT (does it reach above the abdomen, toward ribs);
#       DUMP_PNG renders them to eyeball the rib cage.
#
# COLONOG-only: ctspine1k/ctpelvic1k are derived NIfTI with no DICOM scouts.
#
# Options (env):
#   TCIA_DIR        raw COLONOG download   (default: data/tcia)
#   MANIFEST_DIR    dir holding manifest.json (default: data/hf_export_v2)
#   SAMPLE          # scouts to inspect    (default: 12)
#   DUMP_PNG        1=render sampled scouts to data/scout_samples (default: 1)
#   WORKERS         index threads          (default: 16)
#   FORCE_REBUILD   1=rebuild the TCIA index cache (default: 0)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

TCIA_DIR="${TCIA_DIR:-${DATA_DIR}/tcia}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_DIR}/hf_export_v2}"
SAMPLE="${SAMPLE:-12}"
DUMP_PNG="${DUMP_PNG:-1}"
WORKERS="${WORKERS:-16}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "${LOGS_DIR}"
[[ -f "${SIF_PATH}" ]]        || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -d "${TCIA_DIR}" ]]        || { echo "ERROR: no TCIA dir at ${TCIA_DIR}"; exit 1; }

# Container paths: DATA_DIR is bound to /data, so reference basenames under it.
C_TCIA="/data/$(basename "${TCIA_DIR}")"
C_MANIFEST="/data/$(basename "${MANIFEST_DIR}")/manifest.json"
C_DUMP="/data/scout_samples"

ARGS=( --tcia_dir "${C_TCIA}" --sample "${SAMPLE}" --workers "${WORKERS}" )
[[ -f "${MANIFEST_DIR}/manifest.json" ]] && ARGS+=( --manifest "${C_MANIFEST}" ) \
    || echo "NOTE: no manifest at ${MANIFEST_DIR}/manifest.json — coverage report skipped."
[[ "${DUMP_PNG}" == "1" ]]    && ARGS+=( --dump_png "${C_DUMP}" )
[[ "${FORCE_REBUILD}" == "1" ]] && ARGS+=( --force_rebuild )

echo "======================================================================"
echo " check_scouts — scout recoverability + craniocaudal extent"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   tcia_dir  : ${TCIA_DIR}"
echo "   manifest  : ${MANIFEST_DIR}/manifest.json"
echo "   sample    : ${SAMPLE}   dump_png=${DUMP_PNG}   force_rebuild=${FORCE_REBUILD}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

# shellcheck disable=SC2086
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/check_scouts.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " check_scouts done at $(date)"
[[ "${DUMP_PNG}" == "1" ]] && echo "   sampled scout PNGs -> ${DATA_DIR}/scout_samples/"
echo "======================================================================"
