#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_export_crops
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/export_crops_%j.out
#SBATCH --error=logs/export_crops_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# export_crops — cut small ROI crops of the QC-flagged cases for fast review.
# Reads the full CTs once (here on the cluster, where the data is local) and
# writes a few-MB CT+mask crop per flagged case. See scripts/export_review_crops.py
#
# Options (env):
#   QC_MASTER_CSV  merged QC worklist     (default: data/qc_master.csv)
#   PSEUDO_OUT_DIR pseudo tree            (default: data/hf_export_v2)
#   CROPS_OUT_DIR  crops output dir       (default: data/review_crops)
#   CROP_PAD       voxel pad around bbox  (default: 8)
#   CROPS_CLEAN    1 = wipe CROPS_OUT_DIR first, so it holds ONLY the current
#                  worklist (no stale crops from a previous run) (default: 0)
#   QC_LIMIT       cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

QC_MASTER_CSV="${QC_MASTER_CSV:-${DATA_DIR}/qc_master.csv}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
CROPS_OUT_DIR="${CROPS_OUT_DIR:-${DATA_DIR}/review_crops}"
CROP_PAD="${CROP_PAD:-8}"
CROPS_CLEAN="${CROPS_CLEAN:-0}"
QC_WORKERS="${QC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
QC_LIMIT="${QC_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${CROPS_OUT_DIR}"
if [[ "${CROPS_CLEAN}" != "0" ]]; then
    echo "CROPS_CLEAN=1 -> wiping ${CROPS_OUT_DIR} before export (fresh worklist only)"
    rm -rf "${CROPS_OUT_DIR:?}"/* 2>/dev/null || true
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
if [[ ! -f "${QC_MASTER_CSV}" ]]; then
    echo "ERROR: QC worklist not found: ${QC_MASTER_CSV} (run merge_qc.py first)"; exit 1
fi
if [[ ! -f "${PSEUDO_OUT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${PSEUDO_OUT_DIR}"; exit 1
fi

echo "======================================================================"
echo " export_crops — ROI crops of QC-flagged cases"
echo "   Job ID   : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   worklist : ${QC_MASTER_CSV}"
echo "   tree     : ${PSEUDO_OUT_DIR}"
echo "   crops    : ${CROPS_OUT_DIR}   (pad=${CROP_PAD})"
echo "   Started  : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

EXTRA=""
[[ "${QC_LIMIT}" != "0" ]] && EXTRA="--limit ${QC_LIMIT}"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/export_review_crops.py \
        --qc_csv  "/data/$(basename "${QC_MASTER_CSV}")" \
        --tree    "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out     "/data/$(basename "${CROPS_OUT_DIR}")" \
        --pad     "${CROP_PAD}" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""
echo "======================================================================"
echo " export_crops done at $(date)"
echo "   crops -> ${CROPS_OUT_DIR}   (sync this dir to a laptop to review off-cluster)"
echo "   review: python -m reviewtool fix-list ${QC_MASTER_CSV} \\"
echo "             --tree ${PSEUDO_OUT_DIR} --crops ${CROPS_OUT_DIR}"
echo "======================================================================"
