#!/usr/bin/env bash
#SBATCH --job-name=ctsp_v5drift
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v5drift_%j.out
#SBATCH --error=logs/v5drift_%j.err
# Why do data/v5_final and data/hf_export_v5/labels disagree, and over how many cases?
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs qc_final
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/diagnose_v5_drift.py \
        --a /data/v5_final --b /data/hf_export_v5/labels \
        --out qc_final/v5_drift.csv
