#!/usr/bin/env bash
#SBATCH --job-name=hipcmp
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/hipcmp_%j.out
#SBATCH --error=logs/hipcmp_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/compare_pieces_before_after.py \
      --before /data/hip_crossover_backup --after /data/hf_export_v6 \
      --cases 0012 0065 0135 0146 0172 0186 0376 0410 0471 0513 0746 0830 0917 0938 0957 1124 1145 1148
