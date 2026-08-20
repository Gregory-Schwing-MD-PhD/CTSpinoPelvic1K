#!/usr/bin/env bash
#SBATCH --job-name=ctsp_hw
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hw_%j.out
#SBATCH --error=logs/hw_%j.out

# 12 workers not 24: each holds a CT plus a dilated boolean region. An 8-worker/64G run of
# a similar job died with BrokenProcessPool, so this trades parallelism for headroom.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 0. sanity: 0068 is a known positive (interbody cage) ==="
"${RUN[@]}" python3 scripts/scan_hardware.py --cases 0068 --workers 1 --out qc_hardware_test

echo; echo "=== 1. full scan ==="
"${RUN[@]}" python3 scripts/scan_hardware.py \
    --labels data/v5_final --ct data/hf_export_v4/ct \
    --manifest data/hf_export_v4/manifest.json --workers 12 --out qc_hardware
echo; echo "=== done ==="
