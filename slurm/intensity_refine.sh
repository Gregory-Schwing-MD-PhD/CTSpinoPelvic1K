#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_intensity_refine
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/intensity_refine_%j.out
#SBATCH --error=logs/intensity_refine_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Intensity refinement (Stage 3.6) — CPU post-step on a pseudo-labelled tree.
#
# Re-segments the PSEUDO-filled region with CT-intensity bone segmentation,
# calibrated per-case from the manual annotation and connected-component-gated
# by the model prediction. Reads v1 (manual) + v2 (pseudo) and writes a NEW
# refined tree; the pseudolabel output is left untouched. No GPU, no network.
# See scripts/intensity_refine.py.
#
# Publish the refined tree to the v2 BRANCH (replaces the model-mask v2):
#   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 WIPE_REMOTE=1 \
#     HF_EXPORT_DIR=$(pwd)/data/hf_export_v2_refined make hf-push
#
# Options (env):
#   HF_EXPORT_DIR   v1 manual tree   (default: data/hf_export)
#   PSEUDO_OUT_DIR  v2 pseudo tree   (default: data/hf_export_v2)
#   REFINE_OUT_DIR  refined out tree (default: data/hf_export_v2_refined)
#   REFINE_PCTL     manual-HU percentile threshold (default 10)
#   REFINE_ERODE    manual erosion (vox) before HU sampling (default 1)
#   REFINE_FILL     1 = hole-fill marrow (default), 0 = leave hollow
#   REFINE_LIMIT    cap cases (debug);  REFINE_DRY_RUN=1 = plan only
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REFINE_OUT_DIR="${REFINE_OUT_DIR:-${DATA_DIR}/hf_export_v2_refined}"
REFINE_MODE="${REFINE_MODE:-clip}"
REFINE_PCTL="${REFINE_PCTL:-10}"
REFINE_ERODE="${REFINE_ERODE:-1}"
REFINE_FILL="${REFINE_FILL:-1}"
REFINE_LIMIT="${REFINE_LIMIT:-0}"
REFINE_DRY_RUN="${REFINE_DRY_RUN:-0}"

mkdir -p "${LOGS_DIR}" "${REFINE_OUT_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
if [[ ! -f "${PSEUDO_OUT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${PSEUDO_OUT_DIR} (the v2 pseudo tree)."
    echo "       Run 'make pseudolabel' first."; exit 1
fi
if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_EXPORT_DIR} (the v1 manual tree)."; exit 1
fi

echo "======================================================================"
echo " Intensity refinement (v2 pseudo -> refined)"
echo "   Job ID      : ${SLURM_JOB_ID:-local}"
echo "   Node        : $(hostname)"
echo "   v1 manual   : ${HF_EXPORT_DIR}"
echo "   v2 pseudo   : ${PSEUDO_OUT_DIR}"
echo "   refined out : ${REFINE_OUT_DIR}"
echo "   mode        : ${REFINE_MODE}   (clip = subtractive; resegment = can grow)"
echo "   percentile  : ${REFINE_PCTL}   erode: ${REFINE_ERODE}   fill_holes: ${REFINE_FILL}"
echo "   DRY_RUN     : ${REFINE_DRY_RUN}"
echo "   Started     : $(date)"
echo "======================================================================"

EXTRA=""
[[ "${REFINE_FILL}" == "0" ]]      && EXTRA="${EXTRA} --no_fill_holes"
[[ "${REFINE_LIMIT}" != "0" ]]     && EXTRA="${EXTRA} --limit ${REFINE_LIMIT}"
[[ "${REFINE_DRY_RUN}" == "1" ]]   && EXTRA="${EXTRA} --dry_run"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

stdbuf -oL -eL singularity exec \
    --env "PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1" \
    --bind "${BINDS}" --pwd /workspace \
    "${SIF_PATH}" \
    python3 -u /workspace/scripts/intensity_refine.py \
        --manual_from "/data/$(basename "${HF_EXPORT_DIR}")" \
        --in          "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out         "/data/$(basename "${REFINE_OUT_DIR}")" \
        --mode        "${REFINE_MODE}" \
        --percentile  "${REFINE_PCTL}" \
        --erode_iter  "${REFINE_ERODE}" \
        ${EXTRA}

echo ""
echo "======================================================================"
echo " Intensity refinement done at $(date)"
echo "   refined tree: ${REFINE_OUT_DIR}"
echo ""
echo " Publish to the v2 BRANCH (replaces the model-mask v2):"
echo "   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 WIPE_REMOTE=1 \\"
echo "     HF_EXPORT_DIR=${REFINE_OUT_DIR} make hf-push"
echo "======================================================================"
