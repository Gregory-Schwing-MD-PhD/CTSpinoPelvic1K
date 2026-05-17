#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pseudolabel
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/pseudolabel_%j.out
#SBATCH --error=logs/pseudolabel_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Pseudo-label completion — builds a FULL v2 tree from a staged v1 export.
#
# Completes spine_only / pelvic_native (separate-mode) records by filling the
# MISSING region with a 5-fold nnU-Net ensemble. Manual voxels are never
# overwritten. fused cases pass through unchanged. See scripts/pseudolabel.py.
#
# This NEVER touches HuggingFace. Publishing the result is a separate,
# explicit step (the v2 goes to a BRANCH so the reviewed main URL is safe):
#   make hf-push HF_REPO_ID=org/Name HF_REVISION=v2 \
#       HF_EXPORT_DIR=$(pwd)/data/hf_export_v2     # + HF_TOKEN=hf_xxx
#
# Prereqs:
#   * `make hf-stage` already produced data/hf_export/ (the v1 tree).
#   * configs/pseudolabel_models.json has the relevant model(s) enabled
#     with final checkpoint identity + label_remap (model is in flux —
#     disabled models are skipped, their records left partial, not faked).
#   * An nnU-Net inference runtime. nnU-Net is NOT in the project container,
#     so point NNUNET_SIF at a container that has nnU-Net v2 + CUDA torch,
#     and NNUNET_RESULTS at the trained nnUNet_results root.
#
# Usage:
#   NNUNET_SIF=/path/nnunet.sif NNUNET_RESULTS=/path/nnUNet_results \
#       sbatch slurm/pseudolabel.sh
#   DRY_RUN=1 ... sbatch slurm/pseudolabel.sh      # plan only, no inference
#
# Options (env overrides):
#   HF_EXPORT_DIR   v1 source tree   (default: data/hf_export)
#   PSEUDO_OUT_DIR  v2 output tree   (default: data/hf_export_v2)
#   MODELS_CONFIG   (default: configs/pseudolabel_models.json)
#   PSEUDO_LIMIT=N  cap pseudo-filled records (debug)
#   DRY_RUN=1       copy v1->v2 verbatim, log plan, run no inference
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${nnUNet_results:-}}"
DRY_RUN="${DRY_RUN:-0}"
PSEUDO_LIMIT="${PSEUDO_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${PSEUDO_OUT_DIR}"

echo "======================================================================"
echo " Pseudo-label completion (v2 tree)"
echo "   Job ID        : ${SLURM_JOB_ID:-local}"
echo "   Node          : $(hostname)"
echo "   GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "   v1 source     : ${HF_EXPORT_DIR}"
echo "   v2 out        : ${PSEUDO_OUT_DIR}"
echo "   Models config : ${MODELS_CONFIG}"
echo "   nnUNet_results: ${NNUNET_RESULTS:-<unset>}"
echo "   DRY_RUN       : ${DRY_RUN}"
echo "   Started       : $(date)"
echo "======================================================================"

if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_EXPORT_DIR}."
    echo "       Run 'make hf-stage' first — pseudolabel never re-exports."
    exit 1
fi

EXTRA_ARGS=""
PSEUDO_DEVICE="cuda"
if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --dry_run"
    PSEUDO_DEVICE="cpu"
fi
[[ "${PSEUDO_LIMIT}" != "0" ]] && EXTRA_ARGS="${EXTRA_ARGS} --limit ${PSEUDO_LIMIT}"

PPATH="/workspace/scripts:/workspace/src:/workspace"
BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
[[ -n "${NNUNET_RESULTS}" ]] && BINDS="${BINDS},${NNUNET_RESULTS}:${NNUNET_RESULTS}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${NNUNET_SIF:-}" || ! -f "${NNUNET_SIF}" ]]; then
        echo "ERROR: NNUNET_SIF not set / not found. A real run needs an"
        echo "       nnU-Net v2 + CUDA container (the project .sif lacks it)."
        echo "       Re-submit with NNUNET_SIF=/path/nnunet.sif, or DRY_RUN=1."
        exit 1
    fi
    if [[ -z "${NNUNET_RESULTS}" ]]; then
        echo "ERROR: NNUNET_RESULTS (or \$nnUNet_results) is required for a"
        echo "       real run. Re-submit with NNUNET_RESULTS=/path, or DRY_RUN=1."
        exit 1
    fi
fi

_run() {
    # DRY_RUN needs no GPU/nnU-Net: use the project container. A real run
    # uses the caller-supplied nnU-Net+CUDA container with --nv.
    if [[ "${DRY_RUN}" == "1" ]]; then
        singularity exec \
            --env "PYTHONPATH=${PPATH}" --bind "${BINDS}" --pwd /workspace \
            "${SIF_PATH}" "$@"
    else
        singularity exec --nv \
            --env "PYTHONPATH=${PPATH},nnUNet_results=${NNUNET_RESULTS}" \
            --bind "${BINDS}" --pwd /workspace \
            "${NNUNET_SIF}" "$@"
    fi
}

_run python3 /workspace/scripts/pseudolabel.py \
    --hf_export      "/data/$(basename "${HF_EXPORT_DIR}")" \
    --out            "/data/$(basename "${PSEUDO_OUT_DIR}")" \
    --models_config  "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
    --nnunet_results "${NNUNET_RESULTS}" \
    --device         "${PSEUDO_DEVICE}" \
    ${EXTRA_ARGS}

echo ""
echo "======================================================================"
echo " Pseudo-label done at $(date)"
echo "   v2 tree: ${PSEUDO_OUT_DIR}"
echo ""
echo " Publish to a v2 BRANCH (main / review URL untouched):"
echo "   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \\"
echo "     HF_EXPORT_DIR=${PSEUDO_OUT_DIR} make hf-push"
echo "======================================================================"
