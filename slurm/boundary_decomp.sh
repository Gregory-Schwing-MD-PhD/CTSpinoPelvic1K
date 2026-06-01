#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_boundary_decomp
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/boundary_decomp_%j.out
#SBATCH --error=logs/boundary_decomp_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# boundary_decomp — split the pseudolabel-vs-GT Dice gap into irreducible
# boundary noise vs fixable interior error, on the fused complete-GT cases.
# Needs the cached fused predictions (run pseudolabel --predict_fused first).
# See scripts/boundary_decomp.py
#
# Options (env):
#   HF_EXPORT_DIR  manual/GT tree   (default: data/hf_export)
#   PRED_DIR       cached preds     (default: data/hf_export_v2_work/preds)
#   MODELS_CONFIG  remap json       (default: configs/pseudolabel_models.json)
#   DECOMP_CSV     out CSV          (default: data/boundary_decomp.csv)
#   DECOMP_K       surface band vox (default: 1)
#   QC_LIMIT       cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PRED_DIR="${PRED_DIR:-${DATA_DIR}/hf_export_v2_work/preds}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
DECOMP_CSV="${DECOMP_CSV:-${DATA_DIR}/boundary_decomp.csv}"
DECOMP_K="${DECOMP_K:-1}"
QC_WORKERS="${QC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
QC_LIMIT="${QC_LIMIT:-0}"

mkdir -p "${LOGS_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_EXPORT_DIR}"; exit 1
fi
if [[ ! -d "${PRED_DIR}" ]]; then
    echo "ERROR: pred cache not at ${PRED_DIR}"
    echo "       Run 'make pseudolabel PREDICT_FUSED=1' so fused preds are cached."
    exit 1
fi

echo "======================================================================"
echo " boundary_decomp — boundary vs interior error on fused GT"
echo "   Job ID   : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   GT       : ${HF_EXPORT_DIR}"
echo "   preds    : ${PRED_DIR}"
echo "   k (band) : ${DECOMP_K}"
echo "   Started  : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

EXTRA=""
[[ "${QC_LIMIT}" != "0" ]] && EXTRA="--limit ${QC_LIMIT}"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/boundary_decomp.py \
        --manual_from   "/data/$(basename "${HF_EXPORT_DIR}")" \
        --preds_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${PRED_DIR}")" \
        --models_config "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
        --out_csv       "/data/$(basename "${DECOMP_CSV}")" \
        --k             "${DECOMP_K}" \
        --workers       "${QC_WORKERS}" ${EXTRA}

echo ""
echo "======================================================================"
echo " boundary_decomp done at $(date)   csv -> ${DECOMP_CSV}"
echo "======================================================================"
