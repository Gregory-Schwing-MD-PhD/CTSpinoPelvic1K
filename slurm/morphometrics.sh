#!/usr/bin/env bash
#SBATCH --job-name=ctsp_morph
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/morph_%j.out
#SBATCH --error=logs/morph_%j.out

# Transitional morphometrics over every densified case, then the figure.
# Chained behind a 12-case smoke test: if the extractor is broken, this never runs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. measure ==="
"${RUN[@]}" python3 scripts/extract_transition_morphometrics.py \
    --labels data/v5_final --manifest data/hf_export_v4/manifest.json \
    --workers 24 --out morphometrics

echo
echo "=== 2. plot ==="
"${RUN[@]}" python3 scripts/plot_transition_morphometrics.py \
    --csv morphometrics/transition_morphometrics.csv --out morphometrics

echo
echo "=== done ==="
