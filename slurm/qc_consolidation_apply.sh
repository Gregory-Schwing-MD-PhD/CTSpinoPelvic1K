#!/usr/bin/env bash
#SBATCH --job-name=ctsp_cons_apply
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cons_apply_%j.out
#SBATCH --error=logs/cons_apply_%j.err
# =============================================================================
# qc_consolidation_apply.sh — remove the dust the hardware subtraction left.
#
#   sbatch slurm/qc_consolidation_apply.sh
#
# Approved against qc_hardware/consolidation.csv: 26 structures, 3,988 mm3 of
# isolated specks, none more than 0.88% of the structure it sits on, every one
# beside an intact main piece holding 92-100%.
#
# ONLY DUST. Pieces under 30 mm3, and only in structures the implant took voxels
# from. 1035's hips are genuinely fragmented -- the largest piece holds 66% --
# and are left exactly as they are, because that is a finding rather than debris.
#
# The labels are BACKED UP first. data/ is gitignored, so there is no
# `git checkout` to undo this with.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
SRC="${SRC:-${DATA_DIR}/hardware_final}"
BAK="${BAK:-${DATA_DIR}/hardware_final_predust}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs qc_hardware
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

if [[ ! -d "${BAK}" ]]; then
  mkdir -p "${BAK}"
  cp -a "${SRC}"/*.nii.gz "${BAK}/" 2>/dev/null || true
  echo "backup: $(ls "${BAK}" | wc -l) label(s) copied to ${BAK}"
else
  echo "backup already exists at ${BAK} -- left alone"
fi

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/qc_hardware_consolidation.py \
        --labels "/data/$(realpath --relative-to="${DATA_DIR}" "${SRC}")" \
        --original /data/v5_final \
        --out qc_hardware/consolidation_after.csv \
        --apply --dust-mm3 30
