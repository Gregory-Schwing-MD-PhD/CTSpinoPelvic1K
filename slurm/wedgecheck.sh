#!/usr/bin/env bash
#SBATCH --job-name=ctsp_wedge
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/wedge_%j.out
#SBATCH --error=logs/wedge_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "$SIF"   python3 scripts/render_wedge_check.py --labels data/v5_final --ct data/hf_export/ct     --cases 0468:L4,0568:L3,0851:L2,0972:L3,0213:L2,0913:L3 --out wedge_renders
