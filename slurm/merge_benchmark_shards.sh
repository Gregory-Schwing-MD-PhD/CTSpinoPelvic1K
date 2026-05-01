#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_ts_merge
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ts_merge_%j.out
#SBATCH --error=logs/ts_merge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 4b — Merge sharded TotalSegmentator benchmark results
#
# Walks <RESULTS_BASE>/shard_*/per_case_partial.jsonl, deduplicates by
# (token, config), runs the aggregation + table formatters from
# benchmark_totalseg.py, and writes unified outputs to <OUT_DIR>.
#
# This is pure JSONL parsing + dict aggregation — no inference, no GPU,
# no Singularity container needed. The host conda env that already has
# numpy is sufficient.
#
# Apr 2026 update — 4-way LSTV subgroup:
#   The merge step now passes --splits_file to the Python merger, which
#   joins each per-case record against splits_5fold.json:token_info to
#   resolve the LSTV subgroup. This produces Tables 5 and 6 with the
#   4-way breakdown (lumb / sacr / ambiguous / normal) instead of the
#   legacy 3-way (lumb / sacr / normal).
#
# Usage
# -----
#   # Default: auto-detect most recent results/totalseg_bench_* directory
#   sbatch slurm/merge_benchmark_shards.sh
#
#   # Or point at a specific run:
#   RESULTS_BASE=$HOME/CTSpinoPelvic1K/results/totalseg_bench_35793471 \
#       sbatch slurm/merge_benchmark_shards.sh
#
#   # Or write to a new output dir (e.g. to keep the legacy 3-way merge
#   # for comparison while you re-merge with 4-way):
#   OUT_DIR=$HOME/CTSpinoPelvic1K/results/totalseg_bench_35793471/_merged_4way \
#       sbatch slurm/merge_benchmark_shards.sh
#
#   # Override the splits file (defaults to data/hf_export/splits_5fold.json):
#   SPLITS_FILE=/path/to/some_other_splits.json \
#       sbatch slurm/merge_benchmark_shards.sh
#
#   # Force legacy 3-way (no ambiguous row) by pointing at /dev/null:
#   SPLITS_FILE=/dev/null sbatch slurm/merge_benchmark_shards.sh
#
# Outputs (under <OUT_DIR>, default <RESULTS_BASE>/_merged):
#   paper_tables.txt           Table 5 (Dice) + Table 6 (surface metrics)
#                              now includes Ambiguous (Castellvi II/III) row
#   benchmark_summary.json     aggregated subgroup means/stds
#   benchmark_results.json     summary + every per-case record
#   benchmark_per_case.csv     per-case CSV with `lstv_subgroup` column
#   per_case_partial.jsonl     deduplicated combined log
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# ── Resolve RESULTS_BASE ─────────────────────────────────────────────────────
# If not given, pick the most recently modified totalseg_bench_* directory
# under results/. This makes "submit and forget" work after a benchmark run.
if [[ -z "${RESULTS_BASE:-}" ]]; then
    RESULTS_BASE=$(ls -dt "${PROJECT_ROOT}/results/"totalseg_bench_* 2>/dev/null | head -1 || true)
    if [[ -z "${RESULTS_BASE}" ]]; then
        echo "ERROR: no totalseg_bench_* directories found under ${PROJECT_ROOT}/results/" >&2
        echo "       Run the sharded benchmark first, or pass RESULTS_BASE explicitly:" >&2
        echo "       RESULTS_BASE=/path/to/totalseg_bench_<jobid> sbatch $0" >&2
        exit 1
    fi
    echo "Auto-detected RESULTS_BASE=${RESULTS_BASE}"
fi

OUT_DIR="${OUT_DIR:-${RESULTS_BASE}/_merged}"

# ── Splits file for 4-way LSTV subgroup binning ──────────────────────────────
# splits_5fold.json (schema v5+) provides token_info[token].lstv_subtype
# with values 'lumb' | 'sacr' | 'ambiguous' | 'normal'. The Python merger
# uses this to group cases by LSTV subtype in the final tables.
#
# If the file doesn't exist, the merger falls back to the per-record
# lstv_label string (legacy 3-way: lumb / sacr / normal). Set
# SPLITS_FILE=/dev/null to force this fallback explicitly.
SPLITS_FILE="${SPLITS_FILE:-${PROJECT_ROOT}/data/hf_export/splits_5fold.json}"

# ── Environment setup (mirrors benchmark_totalseg.sh pattern) ────────────────
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# Scrub host library paths so the conda env's libs win cleanly.
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# scripts/ dir on PYTHONPATH so merge_benchmark_shards.py can `import
# benchmark_totalseg` to reuse its aggregate() / format_table5() helpers.
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUT_DIR}"

cd "${PROJECT_ROOT}"

echo "======================================================================"
echo " benchmark merge"
echo " Job          : ${SLURM_JOB_ID:-local}"
echo " Node         : $(hostname)"
echo " Project root : ${PROJECT_ROOT}"
echo " RESULTS_BASE : ${RESULTS_BASE}"
echo " OUT_DIR      : ${OUT_DIR}"
echo " SPLITS_FILE  : ${SPLITS_FILE}"
if [[ -f "${SPLITS_FILE}" ]]; then
    echo " (splits file present -> 4-way LSTV subgroup with 'ambiguous' row)"
else
    echo " (splits file MISSING -> legacy 3-way fallback, no ambiguous row)"
fi
echo " Python       : $(which python)"
echo " Started      : $(date)"
echo "======================================================================"

# Quick sanity: count input shard files before running
n_shards=$(ls -d "${RESULTS_BASE}"/shard_* 2>/dev/null | wc -l)
n_jsonl=$(find "${RESULTS_BASE}" -maxdepth 2 -name per_case_partial.jsonl 2>/dev/null | wc -l)
echo " Found ${n_shards} shard directories, ${n_jsonl} per_case_partial.jsonl files"
if [[ "${n_jsonl}" -eq 0 ]]; then
    echo "ERROR: no per_case_partial.jsonl files found under ${RESULTS_BASE}" >&2
    exit 1
fi
echo

python scripts/merge_benchmark_shards.py \
    --results_base "${RESULTS_BASE}" \
    --out_dir      "${OUT_DIR}" \
    --splits_file  "${SPLITS_FILE}"

echo
echo "======================================================================"
echo " MERGED PAPER TABLES"
echo "======================================================================"
[[ -f "${OUT_DIR}/paper_tables.txt" ]] && cat "${OUT_DIR}/paper_tables.txt"

echo
echo " Merged outputs: ${OUT_DIR}"
echo " Completed:      $(date)"
