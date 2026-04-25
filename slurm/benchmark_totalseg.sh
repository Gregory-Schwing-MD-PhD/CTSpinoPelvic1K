#!/bin/bash
# =============================================================================
# slurm/benchmark_totalseg.sh — Sharded TotalSegmentator benchmark
# =============================================================================
#
# Submits a SLURM job array. Each task in the array runs benchmark_totalseg.py
# on a subset of tokens (round-robin assignment by token index modulo
# N_SHARDS). Predictions are shared across all shards via a single
# ts_preds_shared/ directory; per-case metric logs are per-shard.
#
# Usage
# -----
# Default: 8 shards
#   sbatch slurm/benchmark_totalseg.sh
#
# Smoke test on shard 0 only:
#   sbatch --array=0-0 slurm/benchmark_totalseg.sh
#
# More parallelism (16 shards) — note this requires 16 free GPUs:
#   sbatch --array=0-15 slurm/benchmark_totalseg.sh
#
# Retry just shard 3 after it timed out / crashed (cache + per-shard JSONL
# both make this resumable; surface metrics for already-done cases are
# read straight from per_case_partial.jsonl):
#   sbatch --array=3-3 slurm/benchmark_totalseg.sh
#
# Override paths via environment:
#   DATASET_DIR=/scratch/myhf  sbatch slurm/benchmark_totalseg.sh
#   RESULTS_BASE=$HOME/CTSpinoPelvic1K/results/ts_v2  sbatch slurm/benchmark_totalseg.sh
#
# Watching progress across all shards (run from login node):
#   for d in $HOME/CTSpinoPelvic1K/results/totalseg_bench/shard_*/; do
#       n=$(wc -l < "$d/per_case_partial.jsonl" 2>/dev/null || echo 0)
#       printf "%-60s %5d cases done\n" "$(basename $d)" "$n"
#   done
#
# After all shards finish:
#   sbatch slurm/merge_benchmark_shards.sh
#
# =============================================================================

#SBATCH --job-name=ts_bench
#SBATCH --array=0-7%8                          # 8 shards, all concurrent (use %N to throttle)
#SBATCH --partition=gpu                        # adjust for your cluster
#SBATCH --gres=gpu:1                           # 1 GPU per shard
#SBATCH --cpus-per-task=8                      # for kd-tree query workers + dataloading
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ts_bench_%A_%a.out
#SBATCH --error=logs/ts_bench_%A_%a.err

set -euo pipefail

# ── Configurable knobs ───────────────────────────────────────────────────────
# Shard identity. SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_COUNT are set
# automatically when this script is launched via `sbatch --array=...`.
# When running this script directly outside SLURM (rare), default to a
# single non-sharded run.
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

# Repo + data layout
REPO_ROOT="${REPO_ROOT:-$HOME/CTSpinoPelvic1K}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/data/hf_export}"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/results/totalseg_bench}"

# Shared TS prediction cache. All shards read/write this directory --
# whoever computes a prediction first deposits it here, and any later
# shard hitting the same (token, config) reuses it.
SHARED_PRED_DIR="${SHARED_PRED_DIR:-${RESULTS_BASE}_shared/ts_preds}"

# Per-shard output directory. Each shard's per_case_partial.jsonl lives
# here, isolated so writes never contend.
SHARD_OUT_DIR="${RESULTS_BASE}/shard_${SHARD_ID}_of_${N_SHARDS}"

# Optional: limit which configs to benchmark (default: all three).
TS_CONFIG="${TS_CONFIG:-all}"

# Surface metrics: leave on by default (HD95 + ASSD + MSD).
# Pass SKIP_SURFACE=1 to skip them all (much faster, Dice + junction only).
SKIP_SURFACE_FLAG=""
if [[ "${SKIP_SURFACE:-0}" == "1" ]]; then
    SKIP_SURFACE_FLAG="--skip_surface"
fi

# Force recomputation of metrics even for cases already in per_case_partial.jsonl
# (predictions are still cached on disk).
FORCE_FLAG=""
if [[ "${FORCE_RECOMPUTE_METRICS:-0}" == "1" ]]; then
    FORCE_FLAG="--force_recompute_metrics"
fi

# ── Environment ──────────────────────────────────────────────────────────────
mkdir -p "$SHARED_PRED_DIR" "$SHARD_OUT_DIR" "$REPO_ROOT/logs"

# Activate conda env. Adjust 'spinesurg' to whatever your env is called.
# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate spinesurg

cd "$REPO_ROOT"

echo "================================================================"
echo "  TS benchmark shard $SHARD_ID of $N_SHARDS"
echo "  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  date=$(date -Iseconds)"
echo "  DATASET_DIR     = $DATASET_DIR"
echo "  SHARED_PRED_DIR = $SHARED_PRED_DIR"
echo "  SHARD_OUT_DIR   = $SHARD_OUT_DIR"
echo "  TS_CONFIG       = $TS_CONFIG"
echo "  SKIP_SURFACE    = ${SKIP_SURFACE:-0}"
echo "  FORCE_RECOMPUTE = ${FORCE_RECOMPUTE_METRICS:-0}"
echo "================================================================"

# ── Compute this shard's token list ──────────────────────────────────────────
# Round-robin assignment by token-string sort order. We use the manifest
# as the token universe (rather than a directory listing) to skip cases
# that failed export. Output is a comma-separated list passed to
# benchmark_totalseg.py via --tokens.
TOKENS=$(python - <<PY
import json
import sys
from pathlib import Path

manifest_path = Path("$DATASET_DIR") / "manifest.json"
manifest = json.loads(manifest_path.read_text())
if isinstance(manifest, dict) and "records" in manifest:
    manifest = manifest["records"]

# Unique tokens, sorted for deterministic shard assignment
all_tokens = sorted({str(r["token"]) for r in manifest if r.get("token") is not None})
n_total = len(all_tokens)

shard_id = int("$SHARD_ID")
n_shards = int("$N_SHARDS")
my_tokens = [t for i, t in enumerate(all_tokens) if i % n_shards == shard_id]

print(",".join(my_tokens), end="")
print(f"  // {len(my_tokens)} of {n_total} tokens", file=sys.stderr)
PY
)

# Strip the trailing comment that the python script wrote to stderr (it's
# already on stderr; the stdout capture above is just the comma list).
NUM_TOKENS=$(echo -n "$TOKENS" | tr ',' '\n' | grep -c .)

if [[ "$NUM_TOKENS" -eq 0 ]]; then
    echo "Shard $SHARD_ID/$N_SHARDS has no tokens. Exiting cleanly."
    exit 0
fi

echo "Shard $SHARD_ID/$N_SHARDS will benchmark $NUM_TOKENS tokens."

# ── Run the benchmark ────────────────────────────────────────────────────────
# Note: --pred_dir is the SHARED cache, --out_dir is THIS shard's output.
# All shards reading/writing the same prediction cache is safe: TS
# inference writes to its (token, config) subdir, and the patched
# benchmark only WRITES if the file isn't already there.
python scripts/benchmark_totalseg.py \
    --dataset_dir "$DATASET_DIR" \
    --config      "$TS_CONFIG" \
    --tokens      "$TOKENS" \
    --device      gpu \
    --out_dir     "$SHARD_OUT_DIR" \
    --pred_dir    "$SHARED_PRED_DIR" \
    $SKIP_SURFACE_FLAG \
    $FORCE_FLAG

echo "Shard $SHARD_ID/$N_SHARDS complete."
