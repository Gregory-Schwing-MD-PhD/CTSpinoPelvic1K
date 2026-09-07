#!/usr/bin/env bash
#SBATCH --job-name=ctsp_qc_cons
#SBATCH -q primary
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qc_cons_%j.out
#SBATCH --error=logs/qc_cons_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
export SINGULARITY_TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}/s"
mkdir -p "${SINGULARITY_TMPDIR}" logs qc_hardware
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 containers/ctspinopelvic1k.sif \
    python3 -u /w/scripts/qc_hardware_consolidation.py \
        --labels /data/hardware_final --original /data/v5_final \
        --out qc_hardware/consolidation.csv
