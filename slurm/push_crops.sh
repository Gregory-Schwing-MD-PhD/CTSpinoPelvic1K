#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_push_crops
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/push_crops_%j.out
#SBATCH --error=logs/push_crops_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# push_crops — upload review crops (+ crops_index.json) to the v2 dataset repo
# under crops/, so the review Space triages to those cases and serves the small
# crops. Network upload (light resources). See scripts/push_crops.py
#
# Options (env):
#   HF_TOKEN       write token for the dataset repo   (REQUIRED)
#   HF_REPO_ID     dataset repo, e.g. gregoryschwingmdphd/CTSpinoPelvic1K  (REQUIRED)
#   HF_REVISION    branch/revision                    (default: v2)
#   CROPS_OUT_DIR  local crops dir                    (default: data/review_crops)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REVISION="${HF_REVISION:-v2}"
CROPS_OUT_DIR="${CROPS_OUT_DIR:-${DATA_DIR}/review_crops}"

mkdir -p "${LOGS_DIR}"

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: HF_TOKEN is required (write token). Run:"
    echo "  HF_TOKEN=hf_xxx HF_REPO_ID=org/Name make push-crops"; exit 1
fi
if [[ -z "${HF_REPO_ID}" ]]; then
    echo "ERROR: HF_REPO_ID is required (e.g. gregoryschwingmdphd/CTSpinoPelvic1K)"; exit 1
fi
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
if [[ ! -f "${CROPS_OUT_DIR}/crops_index.json" ]]; then
    echo "ERROR: no crops_index.json in ${CROPS_OUT_DIR}"
    echo "       Run 'make export-crops' first (regenerates crops + the index)."; exit 1
fi

echo "======================================================================"
echo " push_crops — upload crops to the v2 dataset"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   crops     : ${CROPS_OUT_DIR}"
echo "   repo      : ${HF_REPO_ID}  @ ${HF_REVISION}  (under crops/)"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,HF_TOKEN=${HF_TOKEN}"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/push_crops.py \
        --crops    "/data/$(basename "${CROPS_OUT_DIR}")" \
        --repo_id  "${HF_REPO_ID}" \
        --revision "${HF_REVISION}"

echo ""
echo "======================================================================"
echo " push_crops done at $(date)"
echo "   crops are now at ${HF_REPO_ID}@${HF_REVISION}:crops/"
echo "   restart the review Space to triage to the flagged worklist."
echo "======================================================================"
