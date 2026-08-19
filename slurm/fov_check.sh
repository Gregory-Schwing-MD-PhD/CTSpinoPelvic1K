#!/usr/bin/env bash
#SBATCH --job-name=ctsp_fov
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fov_%j.out
#SBATCH --error=logs/fov_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/render_fov_check.py --labels data/v5_final --ct data/hf_export_v4/ct \
    --workers 8 --out fov_check \
    --cases 0033,0068,0167,0241,0344,0357,0383,0389,0409,0424,0428,0696,0720,0730,1004,1106,0816
echo "=== done ==="
