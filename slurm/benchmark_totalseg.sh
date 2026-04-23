#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_ts_bench
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ts_bench_%j.out
#SBATCH --error=logs/ts_bench_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 4 — TotalSegmentator zero-shot benchmark on CTSpinoPelvic1K
#
# Runs benchmark_totalseg.py across the ENTIRE dataset — zero-shot inference
# doesn't care about train/val/test splits. Subgroup analysis (LSTV class,
# match_type, config) is applied at aggregation time.
#
# NO --fast flag. Full TS precision for publication-quality numbers.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/totalseg_bench_${SLURM_JOB_ID:-local}}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"


# ── Singularity runtime dirs ─────────────────────────────────────────────────
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
mkdir -p logs "${OUT_DIR}" "${TOTALSEG_WEIGHTS}"

# Scrub host LD_LIBRARY_PATH etc. so the container's libs win
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

BINDS="${PROJECT_ROOT}:/workspace,${OUT_DIR}:/results,${DATASET_DIR}:/dataset,${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS}"
PPATH="/workspace/scripts:/workspace"

_run() {
    singularity exec --nv \
        --env "PYTHONPATH=${PPATH},TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

echo "======================================================================"
echo " benchmark_totalseg  [H200]"
echo " Job       : ${SLURM_JOB_ID:-local}"
echo " Node      : $(hostname)"
echo " GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo " Dataset   : ${DATASET_DIR}"
echo " SIF       : ${SIF_PATH}"
echo " Output    : ${OUT_DIR}"
echo " Scope     : whole dataset (all configs — zero-shot, no split filter)"
echo " Mode      : FULL precision (no --fast)"
echo " Started   : $(date)"
echo "======================================================================"

_run python scripts/benchmark_totalseg.py \
    --dataset_dir /dataset \
    --out_dir     /results \
    --pred_dir    /results/ts_preds \
    --config      all \
    --window_mm   40.0 \
    --device      gpu

echo ""
echo "======================================================================"
echo " PAPER TABLES"
echo "======================================================================"
[[ -f "${OUT_DIR}/paper_tables.txt" ]] && cat "${OUT_DIR}/paper_tables.txt"

echo ""
echo " Output: ${OUT_DIR}"
echo " Completed: $(date)"
