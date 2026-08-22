#!/usr/bin/env bash
# Opportunistic measures: what these scans say beyond the reason they were taken.
#
# Needs the CT volumes, not just the labels, so it is heavier than the label-only passes
# -- every case loads a few hundred megabytes of image.
#SBATCH --job-name=ctsp_femur
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=logs/femur_%j.out
#SBATCH --error=logs/femur_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

singularity exec --bind "$(pwd)":/w --pwd /w \
  --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/extract_proximal_femur.py \
    --labels data/v5_final --ct data/hf_export/ct \
    --manifest data/hf_export/manifest.json \
    --workers "${SLURM_CPUS_PER_TASK:-12}" --out morphometrics
echo "=== done ==="
