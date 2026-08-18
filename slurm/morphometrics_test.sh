#!/usr/bin/env bash
#SBATCH --job-name=ctsp_morphtest
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/morphtest_%j.out
#SBATCH --error=logs/morphtest_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
echo "=== transitional morphometrics over every densified case ==="
"${RUN[@]}" python3 scripts/extract_transition_morphometrics.py \
    --labels data/v5_final --manifest data/hf_export_v4/manifest.json \
    --workers 8 --limit 12 --out morphometrics_test
echo "=== done ==="
