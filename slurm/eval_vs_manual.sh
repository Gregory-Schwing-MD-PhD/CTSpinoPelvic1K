#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_eval_vs_manual
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_vs_manual_%j.out
#SBATCH --error=logs/eval_vs_manual_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# eval_vs_manual — quantify model accuracy vs MANUAL ground truth.
#
# For each scoped case, recover the cached raw model prediction (from
# pseudolabel's work dir) and compare it to the v1 manual annotation on the
# MANUAL side only (where ground truth is real). Per-class Dice + volumes +
# ASSD; CSV + aggregate summary. CPU-only. See scripts/eval_vs_manual.py.
#
# Options (env):
#   HF_EXPORT_DIR    v1 manual tree                    (default: data/hf_export)
#   PRED_DIR         pseudolabel preds cache           (default: data/hf_export_v2_work/preds)
#   MODELS_CONFIG    label_remap config                (default: configs/pseudolabel_models.json)
#   EVAL_CSV         output CSV                        (default: data/eval_vs_manual.csv)
#   EVAL_WORKERS     parallel procs                    (default: $SLURM_CPUS_PER_TASK)
#   EVAL_NO_ASSD     1 = skip ASSD (Dice + volumes only)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PRED_DIR="${PRED_DIR:-${DATA_DIR}/hf_export_v2_work/preds}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
EVAL_CSV="${EVAL_CSV:-${DATA_DIR}/eval_vs_manual.csv}"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
EVAL_NO_ASSD="${EVAL_NO_ASSD:-0}"

mkdir -p "${LOGS_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_EXPORT_DIR}"; exit 1
fi
if [[ ! -d "${PRED_DIR}" ]]; then
    echo "ERROR: pred cache not found at ${PRED_DIR}"
    echo "       Run pseudolabel first (its work dir keeps the raw preds)."
    exit 1
fi

echo "======================================================================"
echo " eval_vs_manual (model vs MANUAL ground truth)"
echo "   Job ID    : ${SLURM_JOB_ID:-local}"
echo "   Node      : $(hostname)"
echo "   v1 manual : ${HF_EXPORT_DIR}"
echo "   preds     : ${PRED_DIR}"
echo "   remap     : ${MODELS_CONFIG}"
echo "   out csv   : ${EVAL_CSV}"
echo "   workers   : ${EVAL_WORKERS}"
echo "   no_assd   : ${EVAL_NO_ASSD}"
echo "   Started   : $(date)"
echo "======================================================================"

EXTRA=""
[[ "${EVAL_NO_ASSD}" == "1" ]] && EXTRA="--no_assd"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

stdbuf -oL -eL singularity exec \
    --env "PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1" \
    --bind "${BINDS}" --pwd /workspace \
    "${SIF_PATH}" \
    python3 -u /workspace/scripts/eval_vs_manual.py \
        --manual_from   "/data/$(basename "${HF_EXPORT_DIR}")" \
        --preds_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${PRED_DIR}")" \
        --models_config "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
        --out_csv       "/data/$(basename "${EVAL_CSV}")" \
        --workers       "${EVAL_WORKERS}" \
        ${EXTRA}

echo ""
echo "======================================================================"
echo " eval_vs_manual done at $(date)"
echo "   csv: ${EVAL_CSV}"
echo "======================================================================"
