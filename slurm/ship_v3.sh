#!/usr/bin/env bash
# =============================================================================
# ship_v3.sh — build + push v3 = v2 + anatomically-numbered ribs.
#
#   (1) v3_ribs  — TotalSegmentator ribs, re-numbered from GT thoracic vertebrae,
#                  merged onto the v2 labels (background only).            [GPU]
#   (2) push     — the v3 tree -> <repo>@v3.                              [CPU]
#
# Standalone:
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K bash slurm/ship_v3.sh
# Chained after v2 (launch_all.sh sets EXTRA_DEP to the v2 push job):
#   EXTRA_DEP=<jobid> bash slurm/ship_v3.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

: "${HF_TOKEN:?HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/ship_v3.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
V2_DIR="${V2_DIR:-${DATA_DIR}/hf_export_v2}"
V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"
SPINE_DIR="${SPINE_DIR:-${DATA_DIR}/placed/spine}"
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
WIPE="${WIPE:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"
SB="${SB} ${SBATCH_EXTRA:-}"
# Cancel a chained job immediately if its dependency can never be satisfied (upstream
# exited non-zero) instead of leaving it PENDING for hours. Harmless on dep-less jobs.
SB="${SB} --kill-on-invalid-dep=yes"

# EXTRA_DEP chains the rib job AFTER the v2 push (so v3 reads a finished v2 tree).
RIB_DEP=""; [[ -n "${EXTRA_DEP:-}" ]] && RIB_DEP="--dependency=afterok:${EXTRA_DEP}"

# This cluster's `gpu` QOS has no default partition, so the GPU rib job needs an
# explicit --partition (the CPU push does not). Set GPU_PARTITION=<name> — find it
# with:  sinfo -o '%P %G' | grep -i gpu   (or sacct -j <a working GPU job> -o Partition).
RIB_SB="${SB}"
[[ -n "${GPU_PARTITION:-}" ]] && RIB_SB="${RIB_SB} --partition=${GPU_PARTITION}"

echo "[ship_v3] (1) v3 ribs (TS ribs + GT-thoracic renumber, merge onto v2) [GPU]  ${RIB_DEP:-no dep}"
JR=$(sbatch --parsable ${RIB_SB} ${RIB_DEP} \
  --export=ALL,NNUNET_SIF=${NNUNET_SIF},V2_DIR=${V2_DIR},V3_DIR=${V3_DIR},SPINE_DIR=${SPINE_DIR} \
  slurm/v3_ribs.sh)

echo "[ship_v3] (2) push ${V3_DIR} -> ${HF_REPO_ID}@v3 [CPU]  after ${JR}"
JP=$(sbatch --parsable ${SB} --dependency=afterok:${JR} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v3,HF_EXPORT_DIR=${V3_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "V3_PUSH_JOB=${JP}"
echo "[ship_v3] submitted:  ribs=${JR}  push=${JP}"
echo "[ship_v3]   monitor:  tail -f logs/*${JR}* logs/*${JP}*"
