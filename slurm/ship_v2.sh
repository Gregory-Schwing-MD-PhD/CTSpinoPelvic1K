#!/usr/bin/env bash
# =============================================================================
# ship_v2.sh — build + push the v2 release in one shot.
#
# v2 = the LSTV-segmenter training artifact: fused + spine_only ONLY (every case
# has a radiologist spine + T12 anchor), with the spine_only pelves pseudolabelled.
# pelvic_native is EXCLUDED (real pelvis but pseudo spine — held out for the
# pelvis-pseudolabel Dice). REUSES the cached preds in hf_export_v2_work (no GPU
# inference) — that dir is protected, never deleted.
#
# Run as a LAUNCHER (bash, not sbatch). It clears stale labels, then submits a
# 3-job chain with dependencies, threading your token through:
#   (1) export filtered base + anchor   [CPU]
#   (2) pseudolabel — fill spine_only pelves, reuse cached preds   [GPU]
#   (3) push the v2 tree -> <repo>@v2    [CPU]
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/ship_v2.sh
#
# DRY_RUN=1 plans the pseudolabel step (no inference). Optional env: HF_WORKERS,
# HF_PRIVATE, NNUNET_RESULTS, MODELS_CONFIG, MANIFEST_FILE.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
source configs/default.env

: "${HF_TOKEN:?paste your token -> HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/ship_v2.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${PROJECT_ROOT}/nnunet/results}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"          # filtered base (scratch)
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"     # the v2 tree we push
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SKIP_QC="${SKIP_QC:-0}"
NO_PIR="${NO_PIR:-0}"
DRY_RUN="${DRY_RUN:-0}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ "${DRY_RUN}" == "1" || -f "${NNUNET_SIF}" ]] || {
    echo "ERROR: nnUNet container missing at ${NNUNET_SIF} (needed unless DRY_RUN=1)."
    echo "       set NNUNET_SIF=/path/to/ctspinopelvic1k-ts.sif"; exit 1; }

# Guards: force fresh base labels (anchor) AND fresh v2 labels (so the pelvis
# fill is re-applied onto the anchored base). NEVER touch *_work — those are the
# cached preds the pseudolabel step reuses to avoid a GPU inference run.
echo "[ship_v2] clearing stale base labels  ${HF_EXPORT_DIR}/{labels,qc,manifest.json}"
rm -rf "${HF_EXPORT_DIR}"/labels "${HF_EXPORT_DIR}"/qc "${HF_EXPORT_DIR}"/manifest.json
echo "[ship_v2] clearing stale v2 labels    ${PSEUDO_OUT_DIR}/{labels,qc,manifest.json}  (KEEPING ${PSEUDO_OUT_DIR}_work)"
rm -rf "${PSEUDO_OUT_DIR}"/labels "${PSEUDO_OUT_DIR}"/qc "${PSEUDO_OUT_DIR}"/manifest.json

# (1) export the filtered base (fused + spine_only) WITH the anchor, NO push.
# INCLUDE_CONFIGS holds a comma, which sbatch --export would mis-split — so put
# it in the ENVIRONMENT and let --export=ALL carry it through intact.
export INCLUDE_CONFIGS="fused,spine_only"
echo "[ship_v2] (1/3) export filtered base (${INCLUDE_CONFIGS} + anchor) [CPU]"
J1=$(sbatch --parsable \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=0,SKIP_EXPORT=0,SKIP_QC=${SKIP_QC},NO_PIR=${NO_PIR},HF_REPO_ID=,HF_EXPORT_DIR=${HF_EXPORT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

# (2) pseudolabel — fill the spine_only pelves, reusing cached preds [GPU].
echo "[ship_v2] (2/3) pseudolabel (reuse cached preds, DRY_RUN=${DRY_RUN}) [GPU]  after ${J1}"
J2=$(sbatch --parsable --dependency=afterok:${J1} \
  --export=ALL,SIF_PATH=${SIF_PATH},NNUNET_SIF=${NNUNET_SIF},NNUNET_RESULTS=${NNUNET_RESULTS},HF_EXPORT_DIR=${HF_EXPORT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},MODELS_CONFIG=${MODELS_CONFIG},DRY_RUN=${DRY_RUN},HF_TOKEN=${HF_TOKEN} \
  slurm/pseudolabel.sh)

# (3) push the v2 tree (export step skipped — it already exists from step 2) [CPU].
echo "[ship_v2] (3/3) push ${PSEUDO_OUT_DIR} -> ${HF_REPO_ID}@v2 [CPU]  after ${J2}"
J3=$(sbatch --parsable --dependency=afterok:${J2} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=0,HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v2,HF_EXPORT_DIR=${PSEUDO_OUT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "[ship_v2] submitted chain:  export ${J1}  ->  pseudolabel ${J2}  ->  push ${J3}"
echo "[ship_v2]   monitor:  tail -f logs/*${J1}* logs/*${J2}* logs/*${J3}*"
echo "[ship_v2]   NOTE: step 1 rebuilds data/hf_export as fused+spine_only only."
echo "[ship_v2]         Run ship_v1.sh (or eval_vs_manual) BEFORE this if you need the all-configs base."
