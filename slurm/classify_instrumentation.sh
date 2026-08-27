#!/usr/bin/env bash
#SBATCH --job-name=ctsp_verdict
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/verdict_%j.out
#SBATCH --error=logs/verdict_%j.err
# =============================================================================
# classify_instrumentation.sh — implant or artefact, per case.
#
#   sbatch slurm/classify_instrumentation.sh
#
# A SCRIPT, NOT --wrap. Passing a multi-argument python command through
# sbatch --wrap over ssh from WSL has now failed twice in this project the same
# way: the layered quoting splits the command and bash tries to `export` the
# flags. A file has one layer of quoting and no such failure mode.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs qc_hardware
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/classify_instrumentation.py \
        --proposals /data/hardware_fix \
        --ct /data/hf_export_v5/ct \
        --labels /data/v5_final \
        --out qc_hardware/verdicts.csv
