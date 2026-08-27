#!/usr/bin/env bash
#SBATCH --job-name=ctsp_hw_render
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/hw_render_%A_%a.out
#SBATCH --error=logs/hw_render_%A_%a.err
# =============================================================================
# hardware_renders.sh — the review pictures, for cases that already have a
# proposal.
#
#   sbatch slurm/hardware_renders.sh
#
# Separate from hardware_batch.sh because the seeding is the expensive half and
# it is already done: the first run produced every proposal but no 3-D view at
# all, because scripts/render3d.py had never been copied to the grid and the
# import failed 51 times in a row. Re-running the whole batch to recover one
# picture per case would repeat several hours of thresholding for nothing.
#
# Reads *_hardware_only.nii.gz and writes *_hardware_mip.png,
# *_hardware_3d.png and *_hardware_geometry.json beside it. Touches no label.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

CT_DIR="${CT_DIR:-${DATA_DIR}/hf_export_v5/ct}"
LABEL_DIR="${LABEL_DIR:-${DATA_DIR}/v5_final}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/hardware_fix}"
SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N="${SLURM_ARRAY_TASK_COUNT:-1}"

mapfile -t ALL < <(ls "${OUT_DIR}"/*_hardware_only.nii.gz 2>/dev/null \
                   | xargs -n1 basename | cut -c1-4 | sort)
CASES=()
for i in "${!ALL[@]}"; do
  (( i % N == SHARD )) && CASES+=("${ALL[$i]}")
done
echo "shard ${SHARD}/${N}: ${#CASES[@]} of ${#ALL[@]} cases with a proposal"

RUN=(singularity exec --bind "$(pwd)":/workspace,"${DATA_DIR}":/data --pwd /workspace
     --env PYTHONPATH=/workspace/scripts,PYTHONUNBUFFERED=1,MPLBACKEND=Agg "${SIF_PATH}")
REL() { realpath --relative-to="${DATA_DIR}" "$1"; }

ok=0; bad=0
for CASE in "${CASES[@]}"; do
  echo ""; echo "--- ${CASE} ---"
  if "${RUN[@]}" python3 -u /workspace/scripts/render_hardware.py \
        --case "${CASE}" \
        --ct    "/data/$(REL "${CT_DIR}")/${CASE}_ct.nii.gz" \
        --label "/data/$(REL "${LABEL_DIR}")/${CASE}_label.nii.gz" \
        --hardware "/data/$(REL "${OUT_DIR}")/${CASE}_hardware_only.nii.gz" \
        --out   "/data/$(REL "${OUT_DIR}")"; then
    ok=$((ok + 1))
  else
    echo "  ! render failed for ${CASE}"; bad=$((bad + 1))
  fi
done
echo ""; echo "shard ${SHARD}: ${ok} rendered, ${bad} failed"
