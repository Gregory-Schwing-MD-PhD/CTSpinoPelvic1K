#!/usr/bin/env bash
#SBATCH --job-name=ctsp_turntable
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/turntable_%j.out
#SBATCH --error=logs/turntable_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/render_turntable.py --labels data/v5_final \
    --cases 0431,0033,0631,1035,0004 --descriptor data/itksnap_v5_labels.txt \
    --frames 16 --out gallery_stills
echo "=== done ==="