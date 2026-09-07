#!/usr/bin/env bash
#SBATCH --job-name=repairman
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/repairman_%j.out
#SBATCH --error=logs/repairman_%j.err
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
    python3 -u /w/scripts/repair_manifest_counts.py \
      --src "${DATA_DIR}/hf_export_v6" \
      --also "${DATA_DIR}/zenodo_upload/manifest.json" "${DATA_DIR}/zenodo_v6/manifest.json"
