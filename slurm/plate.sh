#!/usr/bin/env bash
#SBATCH --job-name=ctsp_plate
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plate_%j.out
#SBATCH --error=logs/plate_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "$SIF"   python3 scripts/render_phenotype_plate.py --labels data/v5_final     --cases '0004:five rib-free (typical);0231:four, with a lumbar rib;0151:four, no lumbar rib;0208:six rib-free'     --out paper/mpda/figures --name fig_phenotypes
