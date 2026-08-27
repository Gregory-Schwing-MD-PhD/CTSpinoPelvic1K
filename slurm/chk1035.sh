#!/usr/bin/env bash
#SBATCH --job-name=ctsp_chk1035
#SBATCH -q primary
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/chk1035_%j.out
#SBATCH --error=logs/chk1035_%j.err
# Did the hip laterality problem on 1035 exist before the hardware work, or did I cause it?
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
export SINGULARITY_TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}/s"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "/tmp/${USER}_${SLURM_JOB_ID:-$$}" 2>/dev/null || true' EXIT
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONUNBUFFERED=1 containers/ctspinopelvic1k.sif \
    python3 -u /w/scripts/chk1035.py
