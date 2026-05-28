#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_seg_compare
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/seg_compare_%j.out
#SBATCH --error=logs/seg_compare_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# seg_compare — quantify model-vs-intensity segmentation disagreement.
#
# Reads v1 (manual), v2 (model pseudo), refined (intensity) trees; per scoped
# case in the pseudo region only, computes per-class Dice + voxel volumes +
# average symmetric surface distance (ASSD). Writes a per-case CSV and prints
# an aggregate summary. CPU-only, no network.
#
# Use the aggregate to size the case for retraining: low Dice / high ASSD /
# low vol_ratio (intensity/model) in a class = where the model bled hardest.
#
# Options (env):
#   HF_EXPORT_DIR    v1 manual tree     (default: data/hf_export)
#   PSEUDO_OUT_DIR   v2 model tree      (default: data/hf_export_v2)
#   REFINE_OUT_DIR   refined tree       (default: data/hf_export_v2_refined)
#   COMPARE_CSV      output CSV         (default: data/seg_compare.csv)
#   COMPARE_WORKERS  parallel procs     (default: $SLURM_CPUS_PER_TASK)
#   COMPARE_NO_ASSD  1 = skip ASSD for speed (Dice + volumes only)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REFINE_OUT_DIR="${REFINE_OUT_DIR:-${DATA_DIR}/hf_export_v2_refined}"
COMPARE_CSV="${COMPARE_CSV:-${DATA_DIR}/seg_compare.csv}"
COMPARE_WORKERS="${COMPARE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
COMPARE_NO_ASSD="${COMPARE_NO_ASSD:-0}"

mkdir -p "${LOGS_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
for d in "${HF_EXPORT_DIR}" "${PSEUDO_OUT_DIR}" "${REFINE_OUT_DIR}"; do
    if [[ ! -f "${d}/manifest.json" ]]; then
        echo "ERROR: no manifest.json in ${d}"; exit 1
    fi
done

echo "======================================================================"
echo " seg_compare (model vs intensity)"
echo "   Job ID     : ${SLURM_JOB_ID:-local}"
echo "   Node       : $(hostname)"
echo "   v1 manual  : ${HF_EXPORT_DIR}"
echo "   model v2   : ${PSEUDO_OUT_DIR}"
echo "   intensity  : ${REFINE_OUT_DIR}"
echo "   out csv    : ${COMPARE_CSV}"
echo "   workers    : ${COMPARE_WORKERS}"
echo "   no_assd    : ${COMPARE_NO_ASSD}"
echo "   Started    : $(date)"
echo "======================================================================"

EXTRA=""
[[ "${COMPARE_NO_ASSD}" == "1" ]] && EXTRA="--no_assd"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

stdbuf -oL -eL singularity exec \
    --env "PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1" \
    --bind "${BINDS}" --pwd /workspace \
    "${SIF_PATH}" \
    python3 -u /workspace/scripts/seg_compare.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --model       "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --intensity   "/data/$(basename "${REFINE_OUT_DIR}")" \
        --out_csv     "/data/$(basename "${COMPARE_CSV}")" \
        --workers     "${COMPARE_WORKERS}" \
        ${EXTRA}

echo ""
echo "======================================================================"
echo " seg_compare done at $(date)"
echo "   csv: ${COMPARE_CSV}"
echo "======================================================================"
