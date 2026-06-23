#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_v3_totalseg
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/v3_totalseg_%A_%a.out
#SBATCH --error=logs/v3_totalseg_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=msa1
# =============================================================================
# SHARDED like benchmark_totalseg.sh: an --array of GPU tasks each processing a
# disjoint 1/N of the released cases (split by index %% N in build_v3_totalseg.py).
# Short per-shard wall (12h) + parallel = far less exposure to the NFS/scratch
# events that orphaned the old single 48h job, and it finishes much faster. Still
# RESUMABLE (per-case markers) so a wall-hit/preemption continues on resubmit.
# Shard 0 owns the one-time v2->v3 mirror (CTs + manifest); every shard writes only
# its own labels. ship_v3.sh sets the array size + N_SHARDS_OVERRIDE.
#
# Resubmit a failed subset (pin the ORIGINAL shard count, as in benchmark):
#   N_SHARDS_OVERRIDE=8 sbatch --array=3,5 slurm/v3_totalseg.sh
#
# v3 TotalSegmentator — derive the v3 tree from v2 with one TS pass per case:
# GT-vertebra-matched ribs + femurs + an S1 carve out of the GT sacrum (bone only).
#
# Ribs are emitted only where a GT thoracic vertebra backs them; femurs are added
# directly; S1 = (GT sacrum) ∩ (TS vertebrae_S1). All land on background / relabel
# the sacrum in place — GT boundaries are never overwritten.
# See scripts/build_v3_totalseg.py.
#
# Options (env):
#   V2_DIR     v2 source tree     (default: data/hf_export_v2)
#   V3_DIR     v3 output tree     (default: data/hf_export_v3)
#   SPINE_DIR  placed VerSe spine masks (rib/S1 anchors)  (default: data/placed/spine)
#   NNUNET_SIF TS+CUDA container  (default: containers/ctspinopelvic1k-ts.sif)
#   V3_LIMIT   cap cases (debug)  (default: 0 = all)
#   RESUME     1 = continue from .totalseg_done markers; 0 = full rebuild
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

V2_DIR="${V2_DIR:-${DATA_DIR}/hf_export_v2}"
V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"
SPINE_DIR="${SPINE_DIR:-${DATA_DIR}/placed/spine}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"
V3_LIMIT="${V3_LIMIT:-0}"
RESUME="${RESUME:-1}"          # 1 = continue from .totalseg_done markers (default)

# Shard identity. N_SHARDS_OVERRIDE pins the ORIGINAL shard count so a partial
# resubmit (e.g. --array=3,5) keeps the SAME case split — without it,
# SLURM_ARRAY_TASK_COUNT would be the count of the resubmitted subset and re-shard
# the cases. (Same lesson as benchmark_totalseg.sh.)
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

[[ -f "${NNUNET_SIF}" ]] || { echo "ERROR: TS container missing at ${NNUNET_SIF}"; exit 1; }
[[ -f "${V2_DIR}/manifest.json" ]] || { echo "ERROR: no v2 tree at ${V2_DIR} (run ship_v2 first)"; exit 1; }
mkdir -p "${LOGS_DIR}" "${V3_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

# Scratch policy: EVERYTHING heavy goes on NODE-LOCAL /tmp, not NFS. The container
# /tmp (where TS/nnUNet write hundreds of MB of temp NIfTIs per case) was bound to
# NFS, which is slow and was getting wiped out from under the run (deleting the
# per-case temp AND, via getcwd, cascading FileNotFoundError onto every remaining
# case -> bone-less labels). Node-local /tmp is fast and private to this job.
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
NFS_SCRATCH="${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NODE_SCRATCH}/container_tmp"     # node-local, NOT NFS
export XDG_RUNTIME_DIR="${NODE_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"
# Clean up only on a CLEAN exit. Do NOT rm on TERM/INT: a preemption/requeue
# signal must not delete scratch out from under a still-running child (that was a
# way the run could lose its /tmp mid-case). SLURM's epilog reclaims node /tmp.
trap 'rm -rf "${NODE_SCRATCH}" "${NFS_SCRATCH}" 2>/dev/null || true' EXIT

# Preflight: fail loud & early if scratch is too tight — instead of dying ~100
# cases in when the NFS-bound container /tmp fills (the original v3 failure mode).
# Same thresholds as pseudolabel.sh / benchmark_totalseg.sh. build_v3_totalseg.py
# now also purges TS temp files per case, so /tmp stays bounded during the run.
_free_gib() {
    local kb
    kb=$(df -k --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ')
    echo $(( ${kb:-0} / 1024 / 1024 ))
}
# Everything (sandbox unpack + container /tmp) is on node-local /tmp now, so only
# that needs checking: ~15 GiB sandbox + per-case TS temp (bounded by per-case
# cleanup, a couple GiB). Require 25 GiB headroom.
NODE_FREE_GIB=$(_free_gib "/tmp")
if [[ "${NODE_FREE_GIB}" -lt 25 ]]; then
    echo "ERROR: node /tmp has only ${NODE_FREE_GIB} GiB free; need 25 for the" >&2
    echo "       singularity sandbox + per-case TS temp. Likely too many concurrent" >&2
    echo "       jobs on $(hostname); re-submit, optionally with --exclude=$(hostname)." >&2
    exit 1
fi

echo "======================================================================"
echo " v3 TotalSegmentator  (resume=${RESUME})"
echo "   Shard     : ${SHARD_ID} / ${N_SHARDS}  $([[ -n "${N_SHARDS_OVERRIDE:-}" ]] && echo '(N pinned via N_SHARDS_OVERRIDE)')"
echo "   Job       : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}  Task: ${SLURM_ARRAY_TASK_ID:-0}   Node: $(hostname)"
echo "   GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
echo "   v2 source : ${V2_DIR}"
echo "   v3 out    : ${V3_DIR}"
echo "   numbering : TotalSegmentator native (rib_left/right_1..12)"
echo "   TS SIF    : ${NNUNET_SIF}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${HOST_CONTAINER_TMP}:/tmp"
BINDS+=",${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
CENV="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"
CENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS},TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CENV+=",HOME=${TOTALSEG_CONFIG_DIR},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ARGS=( --v2_dir   "/data/$(realpath --relative-to="${DATA_DIR}" "${V2_DIR}")"
       --v3_dir   "/data/$(realpath --relative-to="${DATA_DIR}" "${V3_DIR}")"
       --spine_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${SPINE_DIR}")"
       --device gpu
       --shard_id "${SHARD_ID}" --n_shards "${N_SHARDS}" )
[[ "${V3_LIMIT}" != "0" ]] && ARGS+=( --limit "${V3_LIMIT}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )

stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${NNUNET_SIF}" python3 -u /workspace/scripts/build_v3_totalseg.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " v3 TotalSegmentator done at $(date)  ->  ${V3_DIR}  (+ totalseg_qc.csv)"
echo "======================================================================"
