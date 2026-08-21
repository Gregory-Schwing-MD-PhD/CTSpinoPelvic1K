#!/usr/bin/env bash
#SBATCH --job-name=ctsp_gallery
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gallery_%j.out
#SBATCH --error=logs/gallery_%j.out

# Marching cubes + decimation over every structure in five cases is minutes per case, which
# is longer than a one-shot SSH tolerates -- so it is a batch job rather than an ssh call.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. gallery meshes ==="
"${RUN[@]}" python3 scripts/export_gallery_meshes.py \
    --labels data/v5_final --cases 0431,0033,0631,1035,0004 \
    --descriptor data/itksnap_v5_labels.txt --out gallery_meshes

echo; echo "=== 2. distribution panels from the FINAL morphometrics ==="
"${RUN[@]}" python3 scripts/build_gallery_data.py \
    --csv morphometrics/transition_morphometrics.csv \
    --out gallery_meshes/distributions.json

echo; echo "=== done ==="