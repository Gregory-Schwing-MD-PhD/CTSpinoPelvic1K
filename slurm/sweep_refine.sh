#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_sweep_refine
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_refine_%j.out
#SBATCH --error=logs/sweep_refine_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# sweep_refine — find best (pctl, grow) on the manual side, then build the
# refined tree with those params and run the standard compare + eval reports.
# One job, one log, one push-ready tree.
#
#   Stage 1  sweep_refine.py        -> picks best (pctl, grow), writes JSON
#   Stage 2  intensity_refine.py    -> writes data/hf_export_v2_refined
#                                      with the best params (force overwrite)
#   Stage 3  seg_compare.py         -> model vs intensity-refined
#   Stage 4  eval_vs_manual.py      -> raw + refined vs MANUAL with best params
#
# Final state: REFINE_OUT_DIR contains the refined tree built from the BEST
# params and is ready to push to HF.
#
# Options (env):
#   PCTL_SWEEP    comma list (default: 5,10,15,20,30)
#   GROW_SWEEP    comma list (default: 0,1,2)
#   plus the usual HF_EXPORT_DIR / PSEUDO_OUT_DIR / REFINE_OUT_DIR / PRED_DIR /
#   MODELS_CONFIG / *_CSV / BEST_JSON / REFINE_WORKERS
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REFINE_OUT_DIR="${REFINE_OUT_DIR:-${DATA_DIR}/hf_export_v2_refined}"
PRED_DIR="${PRED_DIR:-${DATA_DIR}/hf_export_v2_work/preds}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
SWEEP_CSV="${SWEEP_CSV:-${DATA_DIR}/sweep_refine.csv}"
BEST_JSON="${BEST_JSON:-${DATA_DIR}/best_refine_params.json}"
COMPARE_CSV="${COMPARE_CSV:-${DATA_DIR}/seg_compare.csv}"
EVAL_CSV="${EVAL_CSV:-${DATA_DIR}/eval_vs_manual.csv}"
PCTL_SWEEP="${PCTL_SWEEP:-5,10,15,20,30}"
GROW_SWEEP="${GROW_SWEEP:-0,1,2}"
WORKERS="${REFINE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "${LOGS_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
for d in "${HF_EXPORT_DIR}" "${PSEUDO_OUT_DIR}"; do
    if [[ ! -f "${d}/manifest.json" ]]; then
        echo "ERROR: no manifest.json in ${d}"; exit 1
    fi
done
if [[ ! -d "${PRED_DIR}" ]]; then
    echo "ERROR: pred cache not at ${PRED_DIR} (needed for the sweep)"
    exit 1
fi

echo "======================================================================"
echo " sweep + refine + compare + eval — one job"
echo "   Job ID    : ${SLURM_JOB_ID:-local}"
echo "   Node      : $(hostname)"
echo "   sweep     : pctl ∈ {${PCTL_SWEEP}}  grow ∈ {${GROW_SWEEP}}"
echo "   workers   : ${WORKERS}"
echo "   refined   : ${REFINE_OUT_DIR}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"
ENV_VARS="PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1"

echo ""; echo "================== STAGE 1: sweep =================="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/sweep_refine.py \
        --manual_from   "/data/$(basename "${HF_EXPORT_DIR}")" \
        --preds_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${PRED_DIR}")" \
        --models_config "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
        --pctl_list     "${PCTL_SWEEP}" \
        --grow_list     "${GROW_SWEEP}" \
        --out_csv       "/data/$(basename "${SWEEP_CSV}")" \
        --best_out      "/data/$(basename "${BEST_JSON}")" \
        --workers       "${WORKERS}"

# Read best params from the JSON on the host (singularity exec returned).
BEST_PCTL=$(python3 -c "import json,sys; print(json.load(open('${BEST_JSON}'))['best_pctl'])")
BEST_GROW=$(python3 -c "import json,sys; print(json.load(open('${BEST_JSON}'))['best_grow'])")
echo ""
echo " ==> chosen by sweep: pctl=${BEST_PCTL}  grow=${BEST_GROW}"
echo ""

# Clean refined tree so the new params take effect (intensity_refine's resume
# would otherwise skip cases that have any output, regardless of the params
# they were built with).
rm -rf "${REFINE_OUT_DIR}"

echo ""; echo "============= STAGE 2: intensity_refine (best params) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/intensity_refine.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --in          "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out         "/data/$(basename "${REFINE_OUT_DIR}")" \
        --mode        clip \
        --grow_iters  "${BEST_GROW}" \
        --percentile  "${BEST_PCTL}" \
        --erode_iter  1 \
        --workers     "${WORKERS}"

echo ""; echo "================== STAGE 3: seg_compare =================="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/seg_compare.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --model       "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --intensity   "/data/$(basename "${REFINE_OUT_DIR}")" \
        --out_csv     "/data/$(basename "${COMPARE_CSV}")" \
        --workers     "${WORKERS}"

echo ""; echo "============= STAGE 4: eval_vs_manual (best params) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/eval_vs_manual.py \
        --manual_from   "/data/$(basename "${HF_EXPORT_DIR}")" \
        --preds_dir     "/data/$(realpath --relative-to="${DATA_DIR}" "${PRED_DIR}")" \
        --models_config "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
        --out_csv       "/data/$(basename "${EVAL_CSV}")" \
        --workers       "${WORKERS}" \
        --refine_mode   clip \
        --refine_grow   "${BEST_GROW}" \
        --refine_pctl   "${BEST_PCTL}"

echo ""
echo "======================================================================"
echo " sweep_refine done at $(date)"
echo "   best params (read me) : ${BEST_JSON}"
echo "   sweep csv             : ${SWEEP_CSV}"
echo "   refined tree          : ${REFINE_OUT_DIR}    <- push this"
echo "   compare csv           : ${COMPARE_CSV}"
echo "   eval csv              : ${EVAL_CSV}"
echo ""
echo " Push (additive — only changed masks upload, CTs dedup-skip):"
echo "   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \\"
echo "     HF_EXPORT_DIR=${REFINE_OUT_DIR} make hf-push"
echo "======================================================================"
