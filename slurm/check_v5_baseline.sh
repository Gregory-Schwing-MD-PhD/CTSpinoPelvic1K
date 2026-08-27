#!/usr/bin/env bash
#SBATCH --job-name=ctsp_v5base
#SBATCH -q primary
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v5base_%j.out
#SBATCH --error=logs/v5base_%j.err
# Do the two copies of v5 agree? build_v6 read data/v5_final; verify compared against
# data/hf_export_v5/labels. If those differ, "1153 changed" is about v5, not about v6.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
export SINGULARITY_TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}/s"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "/tmp/${USER}_${SLURM_JOB_ID:-$$}" 2>/dev/null || true' EXIT
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/w/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/check_v5_baseline.py
