#!/usr/bin/env bash
#SBATCH --job-name=ctsp_verify_v6
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_v6_%j.out
#SBATCH --error=logs/verify_v6_%j.err
# =============================================================================
# verify_v6.sh — check the v6 tree before publishing it.
#
#   sbatch slurm/verify_v6.sh
#
# The baseline is data/v5_final, the tree build_v6 actually READ -- not the
# published hf_export_v5. Those two copies of v5 have drifted (1153 differs by
# 19,010 voxels), and comparing against the published one blames v6 for a change
# it inherited.
#
# Exits non-zero if anything fails, so it can gate the push.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/verify_v6.py \
        --v6 /data/hf_export_v6 --v5 /data/v5_final
