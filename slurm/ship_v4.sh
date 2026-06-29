#!/usr/bin/env bash
# =============================================================================
# ship_v4.sh — build + push v4 = v3 + Möller-segmented, our-numbered ribs.
#   (1) v4_ribs   — sharded GPU array: binary rib nnU-Net + relabel_ribs -> overlay [GPU]
#   (2) push      — the v4 tree -> <repo>@v4                                        [CPU]
#
# v4 keeps the entire VerSe-native v3 (spine/pelvis/femurs/S1) and replaces the ribs
# with the Möller-segmented, T12-anchored numbered ribs (ids 34-57). Runs AFTER v3.
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/ship_v4.sh
#
# Prereqs: v3 already built/pushed; Möller weights unzipped at $MOLLER_MODEL (default
# models/moller_ribseg/ribseg_model_weights — the dir with dataset.json/plans.json/fold_*).
# No Dataset-id needed (nnU-Net Python API reads the folder). Toggles: V4_SHARDS (8),
# V4_CONCURRENT (8), MOLLER_FOLDS ("0" fast / "0,1,2" ensemble), SYNC_MAIN=1 to move @main.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

: "${HF_TOKEN:?HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/ship_v4.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
MOLLER_MODEL="${MOLLER_MODEL:-${PROJECT_ROOT}/models/moller_ribseg/ribseg_model_weights}"
V4_DIR="${V4_DIR:-${DATA_DIR}/hf_export_v4}"
V4_SHARDS="${V4_SHARDS:-8}"; V4_CONCURRENT="${V4_CONCURRENT:-8}"
HF_WORKERS="${HF_WORKERS:-8}"; HF_PRIVATE="${HF_PRIVATE:-0}"; WIPE="${WIPE:-1}"
SYNC_MAIN="${SYNC_MAIN:-0}"     # v4 is new/experimental -> don't move @main by default
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"; SB="${SB} ${SBATCH_EXTRA:-}"
RIB_DEP=""; [[ -n "${EXTRA_DEP:-}" ]] && RIB_DEP="--dependency=afterok:${EXTRA_DEP}"

[[ -f "${MOLLER_MODEL}/plans.json" ]] || { echo "ERROR: no Möller model at ${MOLLER_MODEL} (unzip ribseg_model_weights.zip; expects dataset.json/plans.json/fold_*)"; exit 1; }

echo "[ship_v4] (1) v4 ribs — ${V4_SHARDS}-way array %${V4_CONCURRENT} [GPU]  model=${MOLLER_MODEL}  ${RIB_DEP:-no dep}"
JR=$(sbatch --parsable ${SB} ${RIB_DEP} \
  --array=0-$((V4_SHARDS - 1))%${V4_CONCURRENT} \
  --export=ALL,NNUNET_SIF=${NNUNET_SIF},V4_DIR=${V4_DIR},MOLLER_MODEL=${MOLLER_MODEL},MOLLER_FOLDS=${MOLLER_FOLDS:-0},MOLLER_CHECKPOINT=${MOLLER_CHECKPOINT:-checkpoint_final.pth},RESUME=${RESUME:-1},N_SHARDS_OVERRIDE=${V4_SHARDS} \
  slurm/v4_ribs.sh)

# (2) rib-connection QC + correction worklist — cooked in so it runs automatically after
# every build (no manual rerun). Reads <V4_DIR>/_v4ribs_done/*.json (local, no GPU/network),
# now including the false-positive-filter stats (fp_drop). qc_v4_ribs.sh self-selects -q
# primary (CPU), so don't pass ${SB} here. Informational — does NOT gate the push.
echo "[ship_v4] (2) rib-connection QC + worklist [CPU]  after all ${V4_SHARDS} shards of ${JR}"
JQ=$(sbatch --parsable --dependency=afterok:${JR} \
  --export=ALL,NNUNET_SIF=${NNUNET_SIF},V4_DIR=${V4_DIR},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v4 \
  slurm/qc_v4_ribs.sh)
echo "V4_QC_JOB=${JQ}"

echo "[ship_v4] (3) push ${V4_DIR} -> ${HF_REPO_ID}@v4 [CPU]  after all ${V4_SHARDS} shards of ${JR}"
JP=$(sbatch --parsable ${SB} --dependency=afterok:${JR} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v4,HF_EXPORT_DIR=${V4_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)
echo "V4_PUSH_JOB=${JP}"

JM=""
if [[ "${SYNC_MAIN}" == "1" ]]; then
  JM=$(sbatch --parsable ${SB} --dependency=afterok:${JP} \
    --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=0,HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=main,HF_EXPORT_DIR=${V4_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
    slurm/export_dataset.sh)
  echo "MAIN_PROMOTE_JOB=${JM}"
fi
echo "[ship_v4] submitted:  ribs=${JR}  qc=${JQ}  push=${JP}${JM:+  main=${JM}}"
echo "[ship_v4]   monitor:  tail -f logs/*${JR}* logs/qc_v4_ribs_${JQ}* logs/*${JP}*"
