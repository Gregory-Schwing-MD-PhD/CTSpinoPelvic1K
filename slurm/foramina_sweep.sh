#!/usr/bin/env bash
#SBATCH --job-name=ctsp_foramsweep
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/foramsweep_%j.out
#SBATCH --error=logs/foramsweep_%j.out

# Map the foramina-detector parameter space against the known LSTV labels, then plot how
# unstable the count is. 24 workers: the work is embarrassingly parallel over cases, and
# each case reads a ~215MB CT once and evaluates all 81 settings in memory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. sweep ==="
"${RUN[@]}" python3 scripts/sweep_foramina_params.py \
    --labels data/v5_final --ct data/hf_export_v4/ct \
    --manifest data/hf_export_v4/manifest.json \
    --n-normal 60 --workers 24 --out morphometrics

echo; echo "=== 2. plot ==="
"${RUN[@]}" python3 scripts/plot_foramina_sweep.py \
    --csv morphometrics/foramina_param_sweep.csv --out morphometrics

echo; echo "=== done ==="
