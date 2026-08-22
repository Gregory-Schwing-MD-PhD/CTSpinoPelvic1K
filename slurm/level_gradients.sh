#!/usr/bin/env bash
#SBATCH --job-name=ctsp_levelgrad
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/levelgrad_%j.out
#SBATCH --error=logs/levelgrad_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w \
  --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/extract_level_gradients.py --labels data/v5_final \
    --manifest data/hf_export/manifest.json --workers 12 --out morphometrics
echo "=== done ==="
