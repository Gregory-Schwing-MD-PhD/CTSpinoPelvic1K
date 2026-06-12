#!/usr/bin/env bash
# =============================================================================
# ship_v2.sh — build + push BOTH releases in one shot (v1 on the way, then v2).
#
# v1 = the partial-annotation base (ALL configs + T12 anchor): the input that
#      trained the pseudolabeller. Built and pushed @v1 in step 1.
# v2 = the LSTV-segmenter training artifact: fused + spine_only ONLY (every case
#      has a radiologist spine + T12 anchor), spine_only pelves pseudolabelled,
#      pelvic_native dropped at the pseudolabel step (held out for the pelvis
#      Dice). REUSES the cached preds in hf_export_v2_work (no GPU inference) —
#      that dir is protected, never deleted.
#
# So a single run ships v1 AND v2. SKIP_BASE=1 reuses an existing base and skips
# the v1 build+push (just refresh v2).
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
# This launcher runs on the login node (no SLURM_JOB_ID); default.env references
# it for the singularity tmpdir. Give it a placeholder so `set -u` + source is
# happy — the actual sbatch jobs get real SLURM_JOB_IDs.
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
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
# WIPE=1 (default): clear each target branch's files on HF before pushing it
# (v1 in step 1, v2 in step 3), so no stale files survive. Set WIPE=0 to skip.
WIPE="${WIPE:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ "${DRY_RUN}" == "1" || -f "${NNUNET_SIF}" ]] || {
    echo "ERROR: nnUNet container missing at ${NNUNET_SIF} (needed unless DRY_RUN=1)."
    echo "       set NNUNET_SIF=/path/to/ctspinopelvic1k-ts.sif"; exit 1; }

# Guards: force fresh v2 labels (so the pelvis fill is re-applied onto the
# anchored base) and, unless SKIP_BASE=1, fresh base labels too. NEVER touch
# *_work — those are the cached preds the pseudolabel step reuses (no GPU run).
SKIP_BASE="${SKIP_BASE:-0}"
echo "[ship_v2] clearing stale v2 labels    ${PSEUDO_OUT_DIR}/{labels,qc,manifest.json}  (KEEPING ${PSEUDO_OUT_DIR}_work)"
rm -rf "${PSEUDO_OUT_DIR}"/labels "${PSEUDO_OUT_DIR}"/qc "${PSEUDO_OUT_DIR}"/manifest.json

# (1) Build the v1 BASE = ALL configs + anchor (NOT filtered). v2 is derived
# from this base; the fused+spine_only filter is applied at the pseudolabel step
# (2), not by re-exporting a filtered tree. Skip with SKIP_BASE=1 if you just ran
# ship_v1 and data/hf_export already holds the anchored all-configs base.
DEP=""
if [[ "${SKIP_BASE}" == "1" ]]; then
    echo "[ship_v2] (1/3) SKIP_BASE=1 — reusing existing all-configs base at ${HF_EXPORT_DIR}"
    [[ -f "${HF_EXPORT_DIR}/manifest.json" ]] || { echo "ERROR: no base at ${HF_EXPORT_DIR}; run ship_v1 first or unset SKIP_BASE"; exit 1; }
else
    echo "[ship_v2] clearing stale base labels  ${HF_EXPORT_DIR}/{labels,qc,manifest.json}"
    rm -rf "${HF_EXPORT_DIR}"/labels "${HF_EXPORT_DIR}"/qc "${HF_EXPORT_DIR}"/manifest.json
    echo "[ship_v2] (1/3) export the v1 base (ALL configs + anchor) + PUSH @v1 [CPU]"
    J1=$(sbatch --parsable \
      --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=0,SKIP_QC=${SKIP_QC},NO_PIR=${NO_PIR},WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v1,HF_EXPORT_DIR=${HF_EXPORT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
      slurm/export_dataset.sh)
    DEP="--dependency=afterok:${J1}"
fi

# Now set the filter — AFTER the base export was submitted, so the base stays
# ALL configs and only the pseudolabel step (which snapshots this env) filters.
export INCLUDE_CONFIGS="fused,spine_only"

# (2) pseudolabel — fill spine_only pelves, reuse cached preds, DROP pelvic_native [GPU].
echo "[ship_v2] (2/3) pseudolabel: fill pelves + keep ${INCLUDE_CONFIGS} (DRY_RUN=${DRY_RUN}) [GPU]  ${DEP:-no dep}"
J2=$(sbatch --parsable ${DEP} \
  --export=ALL,SIF_PATH=${SIF_PATH},NNUNET_SIF=${NNUNET_SIF},NNUNET_RESULTS=${NNUNET_RESULTS},HF_EXPORT_DIR=${HF_EXPORT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},MODELS_CONFIG=${MODELS_CONFIG},DRY_RUN=${DRY_RUN},HF_TOKEN=${HF_TOKEN} \
  slurm/pseudolabel.sh)

# (3) push the v2 tree (export step skipped — it already exists from step 2) [CPU].
echo "[ship_v2] (3/3) push ${PSEUDO_OUT_DIR} -> ${HF_REPO_ID}@v2 [CPU]  after ${J2}"
J3=$(sbatch --parsable --dependency=afterok:${J2} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v2,HF_EXPORT_DIR=${PSEUDO_OUT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "[ship_v2] submitted:  v1 build+push ${J1:-<skipped>}  ->  pseudolabel ${J2}  ->  v2 push ${J3}"
echo "[ship_v2]   monitor:  tail -f logs/*${J1:-}* logs/*${J2}* logs/*${J3}*"
echo "[ship_v2]   one run ships BOTH: v1 = all-configs partial base (step 1),"
echo "[ship_v2]   v2 = derived by DROPPING pelvic_native at the pseudolabel step."
