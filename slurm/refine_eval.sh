#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_refine_eval
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/refine_eval_%j.out
#SBATCH --error=logs/refine_eval_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# refine_eval — three stages in ONE job:
#
#   1. intensity_refine     pseudo region of v2 -> refined tree
#   2. seg_compare          model (v2) vs intensity (refined)
#   3. eval_vs_manual       raw model AND intensity-refined model vs MANUAL
#                           ground truth on the scoped manual side
#
# All on one node allocation, one log file. Same container is started three
# times (cheap on already-converted sandbox). See scripts/intensity_refine.py,
# seg_compare.py, eval_vs_manual.py for per-stage details.
#
# Resources mirror slurm/intensity_refine.sh (which dominates wall time).
# All knobs honored from env; defaults match the per-stage scripts.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# Per-stage defaults — fall through to the per-stage scripts' own defaults.
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REFINE_OUT_DIR="${REFINE_OUT_DIR:-${DATA_DIR}/hf_export_v2_refined}"
PRED_DIR="${PRED_DIR:-${DATA_DIR}/hf_export_v2_work/preds}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
COMPARE_CSV="${COMPARE_CSV:-${DATA_DIR}/seg_compare.csv}"
EVAL_CSV="${EVAL_CSV:-${DATA_DIR}/eval_vs_manual.csv}"

REFINE_MODE="${REFINE_MODE:-clip}"
REFINE_GROW="${REFINE_GROW:-3}"
REFINE_PCTL="${REFINE_PCTL:-10}"
REFINE_ERODE="${REFINE_ERODE:-1}"
REFINE_FILL="${REFINE_FILL:-1}"
REFINE_LIMIT="${REFINE_LIMIT:-0}"
REFINE_WORKERS="${REFINE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
EVAL_NO_ASSD="${EVAL_NO_ASSD:-0}"
COMPARE_NO_ASSD="${COMPARE_NO_ASSD:-0}"

mkdir -p "${LOGS_DIR}" "${REFINE_OUT_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
for d in "${HF_EXPORT_DIR}" "${PSEUDO_OUT_DIR}"; do
    if [[ ! -f "${d}/manifest.json" ]]; then
        echo "ERROR: no manifest.json in ${d}"; exit 1
    fi
done
if [[ ! -d "${PRED_DIR}" ]]; then
    echo "WARNING: pred cache not at ${PRED_DIR} — stage 3 (eval_vs_manual) will exit 1"
fi

echo "======================================================================"
echo " refine+eval — three stages in ONE job"
echo "   Job ID    : ${SLURM_JOB_ID:-local}"
echo "   Node      : $(hostname)"
echo "   v1 manual : ${HF_EXPORT_DIR}"
echo "   v2 pseudo : ${PSEUDO_OUT_DIR}"
echo "   refined   : ${REFINE_OUT_DIR}"
echo "   preds     : ${PRED_DIR}"
echo "   mode=${REFINE_MODE} grow=${REFINE_GROW} pctl=${REFINE_PCTL} erode=${REFINE_ERODE} fill=${REFINE_FILL}"
echo "   workers   : ${REFINE_WORKERS}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"
ENV_VARS="PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1"

EXTRA_REFINE=""
[[ "${REFINE_FILL}" == "0" ]]    && EXTRA_REFINE="${EXTRA_REFINE} --no_fill_holes"
[[ "${REFINE_LIMIT}" != "0" ]]   && EXTRA_REFINE="${EXTRA_REFINE} --limit ${REFINE_LIMIT}"

EXTRA_COMPARE=""
[[ "${COMPARE_NO_ASSD}" == "1" ]] && EXTRA_COMPARE="--no_assd"

EXTRA_EVAL=""
[[ "${EVAL_NO_ASSD}" == "1" ]]    && EXTRA_EVAL="--no_assd"
[[ "${REFINE_FILL}" == "0" ]]     && EXTRA_EVAL="${EXTRA_EVAL} --no_refine_fill"

echo ""; echo "================== STAGE 1: intensity_refine =================="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/intensity_refine.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --in          "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out         "/data/$(basename "${REFINE_OUT_DIR}")" \
        --mode        "${REFINE_MODE}" \
        --grow_iters  "${REFINE_GROW}" \
        --percentile  "${REFINE_PCTL}" \
        --erode_iter  "${REFINE_ERODE}" \
        --workers     "${REFINE_WORKERS}" \
        ${EXTRA_REFINE}

echo ""; echo "================== STAGE 2: seg_compare ======================"; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/seg_compare.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --model       "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --intensity   "/data/$(basename "${REFINE_OUT_DIR}")" \
        --out_csv     "/data/$(basename "${COMPARE_CSV}")" \
        --workers     "${REFINE_WORKERS}" \
        ${EXTRA_COMPARE}

echo ""; echo "================== STAGE 3: eval_vs_manual ==================="; echo ""
if [[ -d "${PRED_DIR}" ]]; then
    stdbuf -oL -eL singularity exec \
        --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
        python3 -u /workspace/scripts/eval_vs_manual.py \
            --manual_from   "/data/$(basename "${HF_EXPORT_DIR}")" \
            --preds_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${PRED_DIR}")" \
            --models_config "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
            --out_csv       "/data/$(basename "${EVAL_CSV}")" \
            --workers       "${REFINE_WORKERS}" \
            --refine_mode   "${REFINE_MODE}" \
            --refine_grow   "${REFINE_GROW}" \
            --refine_pctl   "${REFINE_PCTL}" \
            --refine_erode  "${REFINE_ERODE}" \
            ${EXTRA_EVAL}
else
    echo "Skipping STAGE 3: ${PRED_DIR} not found (no cached preds to compare)."
fi

echo ""
echo "======================================================================"
echo " refine+eval done at $(date)"
echo "   refined tree : ${REFINE_OUT_DIR}"
echo "   compare csv  : ${COMPARE_CSV}"
echo "   eval csv     : ${EVAL_CSV}"
echo "======================================================================"
