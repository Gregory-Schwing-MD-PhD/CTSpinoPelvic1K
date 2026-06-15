#!/usr/bin/env bash
# =============================================================================
# launch_from_qc.sh — RESTART the release pipeline from the QC stage onward,
# reusing the v2 labels a completed pseudolabel run already wrote. Use this when
# pseudolabel is done and you only need: QC triage -> push v2 -> build+push v3
# (skips the expensive base export + GPU pseudolabel).
#
# It does NOT touch your running jobs. If you also want it to cancel the in-flight
# pipeline jobs (pseudolabel / qc / export / v3_totalseg) for this user first, pass
# CANCEL=1 explicitly. Otherwise cancel whatever you want by hand, then run this.
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/launch_from_qc.sh
#
# Env: CANCEL=0 (default; set 1 to cancel in-flight pipeline jobs first),
#      SHIP_V3=1 (chain v3), WIPE=1, HF_WORKERS, HF_PRIVATE, PSEUDO_OUT_DIR,
#      SBATCH_QOS/SBATCH_EXTRA.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

: "${HF_TOKEN:?paste your token -> HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/launch_from_qc.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
WIPE="${WIPE:-1}"
SHIP_V3="${SHIP_V3:-1}"
CANCEL="${CANCEL:-0}"    # default OFF — never cancels your jobs unless you ask (CANCEL=1)
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"

# Same QOS/extras passthrough as ship_v2.
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"
SB="${SB} ${SBATCH_EXTRA:-}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ -f "${PSEUDO_OUT_DIR}/manifest.json" ]] || {
    echo "ERROR: no v2 tree at ${PSEUDO_OUT_DIR} — run pseudolabel first (this entrypoint"
    echo "       reuses an existing v2; it does NOT regenerate the labels)."; exit 1; }

# --- Cancel the in-flight pipeline jobs (scoped to this pipeline's job names) ----
if [[ "${CANCEL}" == "1" ]]; then
    echo "[from_qc] cancelling in-flight pipeline jobs for ${USER} ..."
    for jn in ctspinopelvic1k_pseudolabel ctspinopelvic1k_qc_pseudo_pelvis \
              ctspinopelvic1k_export_dataset ctspinopelvic1k_v3_totalseg; do
        scancel -u "${USER}" --name="${jn}" 2>/dev/null && echo "   scancel --name=${jn}" || true
    done
    sleep 2     # let the controller register the cancellations before resubmitting
fi

# --- (1) QC triage — reads the EXISTING v2 tree, no dependency. Non-fatal. -------
echo "[from_qc] (1) qc_pseudo_pelvis triage -> ${PSEUDO_OUT_DIR}/qc [CPU]"
JQC=$(sbatch --parsable ${SB} \
  --export=ALL,SIF_PATH=${SIF_PATH},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR} \
  slurm/qc_pseudo_pelvis.sh)

# --- (2) push v2 — gated on QC so the triage CSV ships inside the tree. ----------
echo "[from_qc] (2) push ${PSEUDO_OUT_DIR} -> ${HF_REPO_ID}@v2 [CPU]  after ${JQC}"
J_PUSH=$(sbatch --parsable ${SB} --dependency=afterok:${JQC} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v2,HF_EXPORT_DIR=${PSEUDO_OUT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

# --- (3) chain v3 (ribs + push) after the v2 push. ------------------------------
if [[ "${SHIP_V3}" == "1" ]]; then
    echo "[from_qc] (3) chaining v3 after v2 push ${J_PUSH}"
    # NOTE: do NOT pass SBATCH_QOS/SBATCH_EXTRA here. Forcing an EMPTY SBATCH_QOS into
    # the env makes sbatch override the rib job's `#SBATCH -q gpu` with nothing -> no
    # QOS -> "No partition specified". They're inherited naturally if you set them.
    EXTRA_DEP="${J_PUSH}" HF_TOKEN="${HF_TOKEN}" HF_REPO_ID="${HF_REPO_ID}" \
        SIF_PATH="${SIF_PATH}" NNUNET_SIF="${NNUNET_SIF}" WIPE="${WIPE}" \
        HF_WORKERS="${HF_WORKERS}" HF_PRIVATE="${HF_PRIVATE}" \
        MANIFEST_FILE="${MANIFEST_FILE}" \
        bash slurm/ship_v3.sh
fi

echo "[from_qc] submitted:"
echo "[from_qc]   qc triage : ${JQC}"
echo "[from_qc]   v2 push   : ${J_PUSH}   (completeness-gated)"
echo "[from_qc]   v3        : $([[ "${SHIP_V3}" == "1" ]] && echo 'chained after v2 push' || echo '<skipped>')"
echo "[from_qc]   monitor   : squeue -u ${USER}"