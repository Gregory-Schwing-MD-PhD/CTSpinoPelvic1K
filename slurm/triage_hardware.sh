#!/usr/bin/env bash
#SBATCH --job-name=ctsp_triage_hw
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/triage_hw_%j.out
#SBATCH --error=logs/triage_hw_%j.err
# =============================================================================
# triage_hardware.sh — separate "the 1800 HU flag over-called it" from "the
# implant is real and the batch missed it".
#
#   sbatch slurm/triage_hardware.sh
#
# 84 cases were flagged at 1800 HU; seed_hardware.py at the literature threshold
# of 2500 produced a proposal for 45. The other 39 are either not instrumented
# at all or carry an implant too small to clear the 40-voxel floor, and those
# need opposite treatment: one is a manifest correction, the other is a case to
# annotate.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs qc_hardware
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1,MPLBACKEND=Agg "${SIF_PATH}" \
    python3 -u /w/scripts/triage_hardware_batch.py \
        --scan qc_hardware/hardware_scan.csv \
        --proposals /data/hardware_fix \
        --ct /data/hf_export_v5/ct \
        --labels /data/v5_final \
        --out qc_hardware/triage.csv
