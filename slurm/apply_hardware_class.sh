#!/usr/bin/env bash
#SBATCH --job-name=ctsp_apply_hw
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/apply_hw_%j.out
#SBATCH --error=logs/apply_hw_%j.err
# =============================================================================
# apply_hardware_class.sh — write the reviewed hardware class into the label,
# for the eleven cases a reader confirmed are real instrumentation.
#
#   sbatch slurm/apply_hardware_class.sh
#
# Authored as a FILE and copied over, not written through ssh with a heredoc.
# Every attempt to inline a script through wsl -> ssh -> heredoc in this project
# has been mangled by one quoting layer or another: expanded ${USER} on the
# wrong machine, split a python command so bash tried to `export` its flags, and
# turned an escaped single quote into the literal text x27. One layer of quoting
# and no such failure mode.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/hardware_final}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs "${OUT_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/apply_hardware_class.py \
        --manifest qc_hardware/hardware_manifest.csv \
        --proposals /data/hardware_fix \
        --labels /data/v5_final \
        --out "/data/$(realpath --relative-to="${DATA_DIR}" "${OUT_DIR}")" \
        --override "0068=/w/thoracic_fix/0068/0068_label_v6.nii.gz"
