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
#
# Container-writability note
# --------------------------
# TotalSegmentator writes a config file at startup (/opt/totalseg/config.json
# or similar) to cache license state, GPU flags, and nnU-Net paths. Inside a
# SIF, /opt is read-only and the write raises [Errno 30] Read-only file
# system, killing inference on the first case.
#
# Two independent mitigations are applied; either one alone would suffice,
# but layering them protects against TS version drift inside the SIF:
#
#   1.  --writable-tmpfs: gives the container a per-session tmpfs overlay,
#       so any write inside the container succeeds.  Discarded on exit.
#   2.  TOTALSEG_CONFIG_DIR + TOTALSEG_HOME_DIR + HOME env vars pointing at
#       a host-writable path that is bind-mounted into the container. Newer
#       TS versions honor these; older versions fall back to $HOME, which
#       also resolves to the writable location.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/totalseg_bench_${SLURM_JOB_ID:-local}}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"

# TS config / cache location: host-writable, bind-mounted into the container
# so any TS version writing to $HOME, TOTALSEG_CONFIG_DIR, or
# TOTALSEG_HOME_DIR will land here and persist across jobs.
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"

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
mkdir -p logs "${OUT_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

# Scrub host LD_LIBRARY_PATH etc. so the container's libs win
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

BINDS="${PROJECT_ROOT}:/workspace,${OUT_DIR}:/results,${DATASET_DIR}:/dataset,${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
PPATH="/workspace/scripts:/workspace"

# Comma-joined env list for --env. Keep on one logical line (bash doesn't mind
# the concatenation-via-adjacent-strings pattern inside double quotes).
CONTAINER_ENV="PYTHONPATH=${PPATH}"
CONTAINER_ENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS}"
CONTAINER_ENV+=",TOTALSEG_CONFIG_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CONTAINER_ENV+=",HOME=${TOTALSEG_CONFIG_DIR}"

_run() {
    # --writable-tmpfs gives the container a per-session writable overlay, so
    # TS's fallback write to /opt/totalseg/config.json (or wherever) succeeds
    # even on TS versions that don't honor TOTALSEG_CONFIG_DIR.
    singularity exec --nv --writable-tmpfs \
        --env "${CONTAINER_ENV}" \
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
echo " TS cfg    : ${TOTALSEG_CONFIG_DIR}"
echo " TS wts    : ${TOTALSEG_WEIGHTS}"
echo " Scope     : whole dataset (all configs — zero-shot, no split filter)"
echo " Mode      : FULL precision (no --fast)"
echo " Writable  : --writable-tmpfs (TS config dir also bind-mounted)"
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
