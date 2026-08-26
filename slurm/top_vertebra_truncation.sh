#!/usr/bin/env bash
#SBATCH --job-name=ctsp_top_trunc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/top_trunc_%j.out
#SBATCH --error=logs/top_trunc_%j.err
# =============================================================================
# top_vertebra_truncation.sh — over the whole release, is the TOP vertebra of
# each record cut by the edge of the reconstruction?
#
#   sbatch slurm/top_vertebra_truncation.sh
#
# Asked because 0068 images a truncated T11 above a whole T12, and whether to
# label a truncated level is a convention rather than a per-case judgement:
# whatever the other 801 records do is what 0068 should do.
#
# A SCRIPT RATHER THAN --wrap. The first attempt passed this through
# wsl -> ssh -> sbatch --wrap, and the layered quoting expanded ${USER} on the
# LOCAL machine and dropped ${SLURM_JOB_ID} entirely, so singularity was handed
# a scratch path that named the wrong user and did not exist. A file has one
# layer of quoting and no such failure mode.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

LABEL_DIR="${LABEL_DIR:-${DATA_DIR}/v5_final}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}"
mkdir -p "${LOGS_DIR}" qc_final

NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

echo "labels: ${LABEL_DIR}   ->  qc_final/top_vertebra_truncation.csv"
singularity exec --bind "${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data" --pwd /workspace \
    "${SIF_PATH}" python3 -u /workspace/scripts/qc_top_vertebra_truncation.py \
    --labels "/data/$(realpath --relative-to="${DATA_DIR}" "${LABEL_DIR}")" \
    --out /workspace/qc_final
