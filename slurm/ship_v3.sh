#!/usr/bin/env bash
# =============================================================================
# ship_v3.sh — build + push v3 = v2 + a TotalSegmentator pass (bone).
#
#   (1) v3_totalseg  — femurs + S1 carve (+ GT thoracic) merged onto the v2
#                  labels (GT boundaries never overwritten).               [GPU]
#   (2) push     — the v3 tree -> <repo>@v3.                              [CPU]
#   (3) promote  — re-push the SAME tree -> <repo>@main so main tracks v3
#                  (afterok the v3 push; SYNC_MAIN=0 to skip).             [CPU]
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
SYNC_MAIN="${SYNC_MAIN:-1}"     # 1 = after the v3 push, also push the SAME tree to
                                # @main so main tracks v3 (set 0 to leave main alone)
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"
SB="${SB} ${SBATCH_EXTRA:-}"

# EXTRA_DEP chains the rib job AFTER the v2 push (so v3 reads a finished v2 tree).
RIB_DEP=""; [[ -n "${EXTRA_DEP:-}" ]] && RIB_DEP="--dependency=afterok:${EXTRA_DEP}"

# Shard the TS pass across an --array of GPU tasks (like benchmark_totalseg.sh): each
# does a disjoint 1/N of the cases, so the run finishes ~N× faster and a short shard
# is far less exposed to the NFS/scratch events that orphaned the old single 48h job.
# N_SHARDS_OVERRIDE pins the count so the per-shard case split is stable on resubmit.
V3_SHARDS="${V3_SHARDS:-8}"            # number of shards (array tasks)
V3_CONCURRENT="${V3_CONCURRENT:-8}"   # max simultaneously running (GPUs permitting)

echo "[ship_v3] (1) v3 TotalSegmentator — ${V3_SHARDS}-way array, %${V3_CONCURRENT} concurrent [GPU]  ${RIB_DEP:-no dep}"
JR=$(sbatch --parsable ${SB} ${RIB_DEP} \
  --array=0-$((V3_SHARDS - 1))%${V3_CONCURRENT} \
  --export=ALL,NNUNET_SIF=${NNUNET_SIF},V2_DIR=${V2_DIR},V3_DIR=${V3_DIR},SPINE_DIR=${SPINE_DIR},RESUME=${RESUME:-1},N_SHARDS_OVERRIDE=${V3_SHARDS} \
  slurm/v3_totalseg.sh)

# afterok on the ARRAY job id waits for EVERY shard (incl. shard 0's mirror) to finish ok.
echo "[ship_v3] (2) push ${V3_DIR} -> ${HF_REPO_ID}@v3 [CPU]  after all ${V3_SHARDS} shards of ${JR}"
JP=$(sbatch --parsable ${SB} --dependency=afterok:${JR} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v3,HF_EXPORT_DIR=${V3_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "V3_PUSH_JOB=${JP}"

# (3) Promote main -> v3: re-push the SAME v3 tree to @main, ONLY after the v3
# push succeeds (afterok:${JP}). ship_v3 itself never touches main, so without
# this main would be left at whatever it pointed to before (e.g. the prior,
# possibly-broken commit) while v3 moved ahead. We re-use the proven push path
# (export_hf.py upload_large_folder) with HF_REVISION=main; the LFS blobs already
# exist on the remote from the v3 push, so HF dedupes them by hash and this just
# writes a commit on main referencing the same files -> main becomes
# content-identical to v3. Additive (WIPE_REMOTE=0): the v3 filename schema is
# stable (NNNN_*), so re-pushing overwrites every file in place with no orphans,
# and main (the default branch / live review URL) never passes through an empty
# state. Set SYNC_MAIN=0 to skip.
JM=""
if [[ "${SYNC_MAIN}" == "1" ]]; then
  echo "[ship_v3] (3) promote ${HF_REPO_ID}@main -> v3 content (re-push ${V3_DIR}) [CPU]  after ${JP}"
  JM=$(sbatch --parsable ${SB} --dependency=afterok:${JP} \
    --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=0,HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=main,HF_EXPORT_DIR=${V3_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
    slurm/export_dataset.sh)
  echo "MAIN_PROMOTE_JOB=${JM}"
fi

echo "[ship_v3] submitted:  ribs=${JR}  push=${JP}${JM:+  main=${JM}}"
echo "[ship_v3]   monitor:  tail -f logs/*${JR}* logs/*${JP}*${JM:+ logs/*${JM}*}"
