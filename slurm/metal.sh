#!/usr/bin/env bash
#SBATCH --job-name=ctsp_metal
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/metal_%j.out
#SBATCH --error=logs/metal_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "$SIF"   python3 scripts/detect_metal.py --labels data/v5_final --ct data/hf_export/ct     --workers 16 --out morphometrics
