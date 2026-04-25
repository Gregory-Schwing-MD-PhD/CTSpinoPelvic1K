#!/bin/bash
# =============================================================================
# slurm/merge_benchmark_shards.sh — Aggregate sharded TS benchmark results
# =============================================================================
#
# Runs scripts/merge_benchmark_shards.py on a CPU node. This is a fast
# JSONL parse + aggregation pass, no inference or surface metric
# recomputation, so a small CPU allocation is plenty.
#
# Usage
# -----
#   sbatch slurm/merge_benchmark_shards.sh
#
# After all `slurm/benchmark_totalseg.sh` array tasks complete, submit
# this. It walks results/totalseg_bench/shard_*/per_case_partial.jsonl,
# dedupes by (token, config), runs aggregate() + table formatters, and
# writes results to results/totalseg_bench/_merged/.
#
# Override paths via environment (mirrors benchmark_totalseg.sh):
#   RESULTS_BASE=$HOME/.../totalseg_bench  sbatch slurm/merge_benchmark_shards.sh
#
# =============================================================================

#SBATCH --job-name=ts_merge
#SBATCH --partition=standard               # adjust for your cluster (CPU partition)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ts_merge_%j.out
#SBATCH --error=logs/ts_merge_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CTSpinoPelvic1K}"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/results/totalseg_bench}"
OUT_DIR="${OUT_DIR:-$RESULTS_BASE/_merged}"

# Activate conda env. Adjust to your env name.
# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate spinesurg

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"

echo "================================================================"
echo "  TS benchmark merge"
echo "  host=$(hostname)  date=$(date -Iseconds)"
echo "  RESULTS_BASE = $RESULTS_BASE"
echo "  OUT_DIR      = $OUT_DIR"
echo "================================================================"

python scripts/merge_benchmark_shards.py \
    --results_base "$RESULTS_BASE" \
    --out_dir      "$OUT_DIR"

echo "Merge complete. See $OUT_DIR/paper_tables.txt"
