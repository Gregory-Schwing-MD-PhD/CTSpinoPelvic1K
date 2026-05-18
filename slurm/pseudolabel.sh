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
#   NNUNET_SIF=/path/spinopelvic.sif sbatch slurm/pseudolabel.sh
#   DRY_RUN=1 sbatch slurm/pseudolabel.sh          # plan only, no inference
#
# The 5-fold Dataset803 checkpoints are DOWNLOADED automatically from
# HuggingFace (configs/pseudolabel_models.json) into NNUNET_RESULTS. The
# nnU-Net container is the spinopelvic-seg one (containers/spinopelvic.sif);
# it ships nnunetv2 + huggingface_hub. Inference is OUT-OF-FOLD: each
# training case is predicted only by the fold that held it out.
#
# Options (env overrides):
#   HF_EXPORT_DIR   v1 source tree   (default: data/hf_export)
#   PSEUDO_OUT_DIR  v2 output tree   (default: data/hf_export_v2)
#   MODELS_CONFIG   (default: configs/pseudolabel_models.json)
#   NNUNET_RESULTS  checkpoint download dir (default: <root>/nnunet/results)
#   NNUNET_SIF      nnU-Net+CUDA container (required for a real run)
#   SKIP_DOWNLOAD=1 reuse already-downloaded checkpoints
#   PSEUDO_LIMIT=N  cap pseudo-filled records (debug)
#   DRY_RUN=1       copy v1->v2 verbatim, log per-case held-out fold,
#                   run no download/inference
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${nnUNet_results:-${PROJECT_ROOT}/nnunet/results}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PSEUDO_LIMIT="${PSEUDO_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${PSEUDO_OUT_DIR}" "${NNUNET_RESULTS}"

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
[[ "${SKIP_DOWNLOAD}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --skip_download"
[[ "${PSEUDO_LIMIT}" != "0" ]] && EXTRA_ARGS="${EXTRA_ARGS} --limit ${PSEUDO_LIMIT}"

PPATH="/workspace/scripts:/workspace/src:/workspace"
# nnUNet_results is bound at the SAME host path so the container writes
# downloaded checkpoints back to NFS (persists across re-submits).
BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${NNUNET_RESULTS}:${NNUNET_RESULTS}"
CENV="PYTHONPATH=${PPATH},nnUNet_results=${NNUNET_RESULTS}"
[[ -n "${HF_TOKEN:-}" ]] && CENV="${CENV},HF_TOKEN=${HF_TOKEN}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${NNUNET_SIF:-}" || ! -f "${NNUNET_SIF:-}" ]]; then
        echo "ERROR: NNUNET_SIF not set / not found. A real run needs the"
        echo "       spinopelvic-seg nnU-Net+CUDA container (the project .sif"
        echo "       lacks nnunetv2). Re-submit with"
        echo "       NNUNET_SIF=/path/spinopelvic.sif, or DRY_RUN=1."
        exit 1
    fi
fi

_run() {
    # DRY_RUN needs no GPU/nnU-Net (only huggingface_hub/nibabel, which the
    # project .sif has): use the project container. A real run uses the
    # caller-supplied nnU-Net+CUDA container with --nv.
    if [[ "${DRY_RUN}" == "1" ]]; then
        singularity exec \
            --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
            "${SIF_PATH}" "$@"
    else
        singularity exec --nv \
            --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
            "${NNUNET_SIF}" "$@"
    fi
}

_run python3 /workspace/scripts/pseudolabel.py \
    --hf_export      "/data/$(basename "${HF_EXPORT_DIR}")" \
    --out            "/data/$(basename "${PSEUDO_OUT_DIR}")" \
    --models_config  "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
    --splits         "/data/$(basename "${HF_EXPORT_DIR}")/splits_5fold.json" \
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
