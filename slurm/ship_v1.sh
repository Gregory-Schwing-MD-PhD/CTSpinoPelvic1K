#!/usr/bin/env bash
# =============================================================================
# ship_v1.sh — build + push the v1 release in one shot.
#
# v1 = the EXACT partial-annotation artifact the model trained on (ALL configs:
# fused + spine_only + pelvic_native, ignore protocol intact) + the T12 anchor
# class. No pseudolabel: pelvic_native keeps `ignore` on the spine (never faked).
#
# Run as a LAUNCHER (bash, not sbatch). It clears stale labels so the anchor
# regenerates, then submits the canonical export+push job (CPU) with your HF
# token threaded in. Paste the token at submit time:
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K bash slurm/ship_v1.sh
#
# Optional env: HF_EXPORT_DIR (default data/hf_export), SIF_PATH, HF_WORKERS,
#               HF_PRIVATE=1, SKIP_QC=1, MANIFEST_FILE.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
source configs/default.env

: "${HF_TOKEN:?paste your token -> HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/ship_v1.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SKIP_QC="${SKIP_QC:-0}"
NO_PIR="${NO_PIR:-0}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing at ${SIF_PATH}"; exit 1; }

# Guard: force fresh labels so the T12 anchor (class 11) is regenerated. Keeps
# ct/ (CTs are unchanged) so this is a fast re-export, not a re-copy.
echo "[ship_v1] clearing stale export labels in ${HF_EXPORT_DIR} (keeping ct/) ..."
rm -rf "${HF_EXPORT_DIR}"/labels "${HF_EXPORT_DIR}"/qc "${HF_EXPORT_DIR}"/manifest.json

echo "[ship_v1] submitting export(ALL configs + anchor) + push -> ${HF_REPO_ID}@v1"
JID=$(sbatch --parsable \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=0,SKIP_QC=${SKIP_QC},NO_PIR=${NO_PIR},WIPE_REMOTE=0,HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v1,HF_EXPORT_DIR=${HF_EXPORT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "[ship_v1] submitted job ${JID}"
echo "[ship_v1]   monitor:  tail -f logs/*${JID}*"
echo "[ship_v1]   note: this rebuilds data/hf_export with ALL configs (the v1 base)."
