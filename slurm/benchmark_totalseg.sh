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
# Runs benchmark_totalseg.py across the ENTIRE dataset — zero-shot inference
# doesn't care about train/val/test splits. Subgroup analysis (LSTV class,
# match_type, config) is applied at aggregation time.
#
# NO --fast flag. Full TS precision for publication-quality numbers.
#
# Sharding (added Apr 2026)
# -------------------------
# This script is a SLURM job array. Each task in the array runs on one
# H200 and benchmarks a round-robin slice of the patient tokens, with
# the slice computed from manifest.json on the host (cheap, deterministic).
# Per-shard metric logs live under shard_K_of_N/, isolated so writes
# never contend. The TS prediction cache (ts_preds/) is SHARED across
# shards: whichever shard runs a given (token, config) first deposits
# the .nii.gz, and any other shard hitting the same key reuses it.
#
# Resubmit semantics:
#   sbatch slurm/benchmark_totalseg.sh                 # all 8 shards
#   sbatch --array=3-3 slurm/benchmark_totalseg.sh     # retry just shard 3
#   sbatch --array=0-7%4 slurm/benchmark_totalseg.sh   # 8 shards, 4 at a time
#
# Resuming a partial run after a transient failure:
#   SHARED_BASE=<old_array_jobid_dir> sbatch slurm/benchmark_totalseg.sh
#   The benchmark python now AUTO-RETRIES failed cases (ok=false records
#   in per_case_partial.jsonl don't block retry; only ok=true does).
#
# After all shards finish:
#   sbatch slurm/merge_benchmark_shards.sh
#
# Container-writability note (revised Apr 2026)
# ---------------------------------------------
# TotalSegmentator + nnU-Net write per-process scratch directories under
# /tmp/nnunet_tmp_*. These need to be visible across all processes inside
# the container (parent + multiprocessing workers).
#
#   Old approach (BROKEN with multi-CPU jobs):
#     --writable-tmpfs gave the container a per-PROCESS tmpfs overlay.
#     Workers spawned via multiprocessing.spawn got their own fresh
#     tmpfs that DIDN'T see the parent's /tmp writes, producing
#     [Errno 2] No such file or directory on /tmp/nnunet_tmp_<rand>.
#
#   Current approach (works with multi-CPU jobs):
#     Bind the host's per-job ${SINGULARITY_TMPDIR} into the container's
#     /tmp. All processes inside share one real filesystem. The per-job
#     directory is created with mkdir -p above and cleaned up by the
#     trap on exit, so it's still job-isolated and ephemeral.
#
# TS config dir / HOME write is handled separately by mitigation #2:
# TOTALSEG_CONFIG_DIR + TOTALSEG_HOME_DIR + HOME env vars all pointing at
# a host-writable directory bind-mounted into the container.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"

# TS config / cache location: host-writable, bind-mounted into the container
# so any TS version writing to $HOME, TOTALSEG_CONFIG_DIR, or
# TOTALSEG_HOME_DIR will land here and persist across jobs.
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"

# ── Shard identity ───────────────────────────────────────────────────────────
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

# Shared base for ALL shards in a given submission. Keyed off
# SLURM_ARRAY_JOB_ID so every task in an array agrees on the base.
SHARED_BASE="${SHARED_BASE:-${PROJECT_ROOT}/results/totalseg_bench_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
OUT_DIR="${SHARED_BASE}/shard_${SHARD_ID}_of_${N_SHARDS}"
PRED_DIR="${SHARED_BASE}/ts_preds"

# ── Singularity runtime dirs ─────────────────────────────────────────────────
# SINGULARITY_TMPDIR doubles as the host source for the container's /tmp
# bind mount below.  Per-job (keyed by SLURM_JOB_ID, not array-job-id) so
# concurrent shards each get their own isolated /tmp inside the container.
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

export TOTALSEG_WEIGHTS_PATH="${TOTALSEG_WEIGHTS}"
mkdir -p logs "${OUT_DIR}" "${PRED_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

# Scrub host LD_LIBRARY_PATH etc. so the container's libs win
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# Container binds:
#   /workspace     <- project root
#   /results       <- this shard's OUT_DIR (isolated per-shard JSONL log)
#   /pred_cache    <- shared TS prediction cache across all shards
#   /dataset       <- HF-export dataset
#   /tmp           <- host-bound per-job tmpdir (so multiprocessing workers
#                     see the same /tmp as the parent for nnU-Net scratch
#                     files; replaces the broken --writable-tmpfs approach)
#   TS weights and config dirs use their host paths inside the container too
BINDS="${PROJECT_ROOT}:/workspace,${OUT_DIR}:/results,${PRED_DIR}:/pred_cache,${DATASET_DIR}:/dataset,${SINGULARITY_TMPDIR}:/tmp,${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
PPATH="/workspace/scripts:/workspace"

CONTAINER_ENV="PYTHONPATH=${PPATH}"
CONTAINER_ENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS}"
CONTAINER_ENV+=",TOTALSEG_CONFIG_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",HOME=${TOTALSEG_CONFIG_DIR}"
# Quiet NumExpr's "detected 128 cores, enforcing safe limit of 16" notice
# by setting an explicit ceiling matching --cpus-per-task.
CONTAINER_ENV+=",NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}"
CONTAINER_ENV+=",OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"

_run() {
    # NOTE: --writable-tmpfs is INTENTIONALLY OMITTED. See the header
    # comment for why; replacing it with a host /tmp bind in the BINDS
    # string above gives all processes inside the container a coherent
    # /tmp filesystem, which nnU-Net's multiprocessing requires.
    singularity exec --nv \
        --env "${CONTAINER_ENV}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

# ── Compute this shard's token list ──────────────────────────────────────────
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
echo " Array Job : ${SLURM_ARRAY_JOB_ID:-?}  Task: ${SLURM_ARRAY_TASK_ID:-?}"
echo " Node      : $(hostname)"
echo " GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo " Dataset   : ${DATASET_DIR}"
echo " SIF       : ${SIF_PATH}"
echo " Shared    : ${SHARED_BASE}"
echo " Output    : ${OUT_DIR}"
echo " Pred cache: ${PRED_DIR}  (shared across all shards)"
echo " Host /tmp : ${SINGULARITY_TMPDIR}  (bound to container's /tmp)"
echo " TS cfg    : ${TOTALSEG_CONFIG_DIR}"
echo " TS wts    : ${TOTALSEG_WEIGHTS}"
echo " Tokens    : ${NUM_TOKENS} (this shard)"
echo " Scope     : whole dataset (all configs — zero-shot, no split filter)"
echo " Mode      : FULL precision (no --fast)"
echo " Resume    : auto-retry of any prior ok=false records"
echo " Started   : $(date)"
echo "======================================================================"

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
