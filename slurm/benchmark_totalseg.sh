#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_ts_bench
#SBATCH -q gpu
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ts_bench_%A_%a.out
#SBATCH --error=logs/ts_bench_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 4 — TotalSegmentator zero-shot benchmark on CTSpinoPelvic1K (sharded)
#
# SCRATCH POLICY (split for speed + safety)
# -----------------------------------------
#   Sandbox unpack  (~10 GB, read-mostly hot path)  → node-local /tmp
#   Container /tmp  (nnU-Net runtime scratch)       → project NFS
#   XDG runtime     (basically empty)               → project NFS
#
# RESUME SEMANTICS
# ----------------
#   sbatch slurm/benchmark_totalseg.sh                         # full 8-way
#   N_SHARDS_OVERRIDE=8 SHARED_BASE=<old_dir> \
#     sbatch --array=4,5,6 slurm/benchmark_totalseg.sh         # retry subset
#
#   N_SHARDS_OVERRIDE is mandatory when resubmitting a subset because
#   SLURM sets SLURM_ARRAY_TASK_COUNT to the count of CURRENTLY submitted
#   tasks, not the original array size. Without the override, a partial
#   resubmit writes to a different directory (shard_4_of_3 instead of
#   shard_4_of_8) and resharding picks different tokens → resume breaks.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"

# ── Shard identity ───────────────────────────────────────────────────────────
# N_SHARDS_OVERRIDE pins the total shard count so partial resubmits land in
# the SAME directory as the original full submission. Without this, a
# partial resubmit (e.g. --array=4,5,6) would set N_SHARDS=3 from
# SLURM_ARRAY_TASK_COUNT, breaking both the output path AND the token
# sharding modulo.
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

SHARED_BASE="${SHARED_BASE:-${PROJECT_ROOT}/results/totalseg_bench_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
OUT_DIR="${SHARED_BASE}/shard_${SHARD_ID}_of_${N_SHARDS}"
PRED_DIR="${SHARED_BASE}/ts_preds"

# ── Split scratch: sandbox on node /tmp, runtime on NFS ──────────────────────
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
NFS_SCRATCH="${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-$$}"
mkdir -p "${NODE_SCRATCH}" "${NFS_SCRATCH}"

export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NFS_SCRATCH}/container_tmp"
export XDG_RUNTIME_DIR="${NFS_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"

export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${NXF_SINGULARITY_CACHEDIR}"

trap 'rm -rf "${NODE_SCRATCH}" "${NFS_SCRATCH}" 2>/dev/null || true' EXIT TERM INT

# ── Prechecks ────────────────────────────────────────────────────────────────
_free_gib() {
    local kb
    kb=$(df -k --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ')
    echo $(( ${kb:-0} / 1024 / 1024 ))
}

NODE_FREE_GIB=$(_free_gib "${NODE_SCRATCH}")
NFS_FREE_GIB=$(_free_gib "${NFS_SCRATCH}")

if [[ "${NODE_FREE_GIB}" -lt 15 ]]; then
    echo "ERROR: node /tmp ${NODE_SCRATCH} has only ${NODE_FREE_GIB} GiB free." >&2
    echo "       Need 15 GiB for the singularity sandbox unpack." >&2
    echo "       Likely cause: too many concurrent jobs on $(hostname)." >&2
    exit 1
fi

if [[ "${NFS_FREE_GIB}" -lt 30 ]]; then
    echo "ERROR: project NFS ${NFS_SCRATCH} has only ${NFS_FREE_GIB} GiB free." >&2
    echo "       Need 30 GiB. Free up space under ${PROJECT_ROOT}." >&2
    exit 1
fi

# ── Conda / env setup ────────────────────────────────────────────────────────
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
unset SINGULARITYENV_HOME
which singularity
export NXF_SINGULARITY_HOME_MOUNT=true

export TOTALSEG_WEIGHTS_PATH="${TOTALSEG_WEIGHTS}"
mkdir -p logs "${OUT_DIR}" "${PRED_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

BINDS="${PROJECT_ROOT}:/workspace"
BINDS+=",${OUT_DIR}:/results"
BINDS+=",${PRED_DIR}:/pred_cache"
BINDS+=",${DATASET_DIR}:/dataset"
BINDS+=",${HOST_CONTAINER_TMP}:/tmp"
BINDS+=",${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS}"
BINDS+=",${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"

PPATH="/workspace/scripts:/workspace"

CONTAINER_ENV="PYTHONPATH=${PPATH}"
CONTAINER_ENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS}"
CONTAINER_ENV+=",TOTALSEG_CONFIG_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",HOME=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}"
CONTAINER_ENV+=",OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"

_run() {
    singularity exec --nv \
        --env "${CONTAINER_ENV}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

# ── Compute this shard's token list (uses pinned N_SHARDS) ───────────────────
TOKENS=$(python3 - <<PY
import json, sys
from pathlib import Path

manifest_path = Path("${DATASET_DIR}") / "manifest.json"
manifest = json.loads(manifest_path.read_text())
if isinstance(manifest, dict) and "records" in manifest:
    manifest = manifest["records"]

all_tokens = sorted({str(r["token"]) for r in manifest if r.get("token") is not None})
shard_id = int("${SHARD_ID}")
n_shards = int("${N_SHARDS}")
my_tokens = [t for i, t in enumerate(all_tokens) if i % n_shards == shard_id]
print(",".join(my_tokens), end="")
sys.stderr.write(f"shard {shard_id}/{n_shards}: {len(my_tokens)} of {len(all_tokens)} tokens\n")
PY
)

NUM_TOKENS=$(echo -n "$TOKENS" | tr ',' '\n' | grep -c .)

echo "======================================================================"
echo " benchmark_totalseg  [H200]  shard ${SHARD_ID} / ${N_SHARDS}"
echo " Array Job  : ${SLURM_ARRAY_JOB_ID:-?}  Task: ${SLURM_ARRAY_TASK_ID:-?}"
echo " N_SHARDS   : ${N_SHARDS}  $([[ -n "${N_SHARDS_OVERRIDE:-}" ]] && echo '(pinned via N_SHARDS_OVERRIDE)' || echo '(from SLURM_ARRAY_TASK_COUNT)')"
echo " Node       : $(hostname)"
echo " GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo " Dataset    : ${DATASET_DIR}"
echo " SIF        : ${SIF_PATH}"
echo " Shared     : ${SHARED_BASE}"
echo " Output     : ${OUT_DIR}"
echo " Pred cache : ${PRED_DIR}  (shared across all shards)"
echo " Sandbox    : ${SINGULARITY_TMPDIR}  (node /tmp, ${NODE_FREE_GIB} GiB free)"
echo " Ctr /tmp   : ${HOST_CONTAINER_TMP}  (NFS, ${NFS_FREE_GIB} GiB free)"
echo " TS cfg     : ${TOTALSEG_CONFIG_DIR}"
echo " TS wts     : ${TOTALSEG_WEIGHTS}"
echo " Tokens     : ${NUM_TOKENS} (this shard)"
echo " Scope      : whole dataset (all configs — zero-shot, no split filter)"
echo " Mode       : FULL precision (no --fast)"
echo " Resume     : auto-retry of any prior ok=false records"
echo " Started    : $(date)"
echo "======================================================================"

# ── Sanity: warn loudly if OUT_DIR has no JSONL but SHARED_BASE is set ───────
# This is the symptom of the N_SHARDS mismatch bug. If the user explicitly
# passed SHARED_BASE expecting to resume, but OUT_DIR is empty, something
# is wrong — almost certainly N_SHARDS mismatch.
if [[ -n "${SHARED_BASE_USER_SET:-}" || -n "${SHARED_BASE:-}" ]] && \
   [[ ! -f "${OUT_DIR}/per_case_partial.jsonl" ]] && \
   [[ "${SHARED_BASE}" != "${PROJECT_ROOT}/results/totalseg_bench_${SLURM_ARRAY_JOB_ID:-x}" ]]; then
    echo " WARNING: SHARED_BASE points at an existing run but ${OUT_DIR}"
    echo "          has no per_case_partial.jsonl. Either this shard is new"
    echo "          to the run, or N_SHARDS doesn't match the original"
    echo "          submission (set N_SHARDS_OVERRIDE=<original_count>)."
    echo "          Other shard dirs in ${SHARED_BASE}:"
    ls -d "${SHARED_BASE}"/shard_*_of_* 2>/dev/null | sed 's/^/            /'
fi

if [[ "${NUM_TOKENS}" -eq 0 ]]; then
    echo "Shard ${SHARD_ID}/${N_SHARDS} has no tokens to process. Exiting cleanly."
    exit 0
fi

_run python scripts/benchmark_totalseg.py \
    --dataset_dir /dataset \
    --out_dir     /results \
    --pred_dir    /pred_cache \
    --config      all \
    --tokens      "${TOKENS}" \
    --window_mm   40.0 \
    --device      gpu

echo ""
echo "======================================================================"
echo " PAPER TABLES (this shard only — for full results, run merge script)"
echo "======================================================================"
[[ -f "${OUT_DIR}/paper_tables.txt" ]] && cat "${OUT_DIR}/paper_tables.txt"

echo ""
echo " Shard output: ${OUT_DIR}"
echo " Shared base:  ${SHARED_BASE}"
echo " Completed:    $(date)"
