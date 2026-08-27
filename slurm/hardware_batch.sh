#!/usr/bin/env bash
#SBATCH --job-name=ctsp_hardware_batch
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%12
#SBATCH --output=logs/hardware_batch_%A_%a.out
#SBATCH --error=logs/hardware_batch_%A_%a.err
# =============================================================================
# hardware_batch.sh — propose a hardware label for every instrumented case, and
# render each one so a person can review it.
#
#   sbatch slurm/hardware_batch.sh
#
# 84 of the 802 records carry metal (qc_hardware/hardware_scan.csv). Exactly one
# of them has its hardware LABELLED. This produces, per case:
#
#   <case>_label_hardware.nii.gz             additions on background only
#   <case>_label_hardware_reassigned.nii.gz  metal outranks bone -- usually the
#                                            useful one, because a bone segmenter
#                                            absorbs an implant it can see
#   <case>_hardware_only.nii.gz              the mask by itself
#   <case>_hardware.json                     what was proposed, and why
#   <case>_hardware_mip.png                  sagittal/coronal/axial projections
#   <case>_hardware_3d.png                   the construct in 3-D
#   <case>_hardware_geometry.json            size, position, what it looks like
#
# NOTHING IS PROMOTED. Every output is a proposal in ${OUT_DIR}; v5 is opened
# read-only. The renders exist because the subtype call -- cage, screw, rod,
# plate -- is a judgement, and 0068 showed it can be right for the wrong reason:
# whole-component PCA called a pair of cages "compact", which a screw-rod-screw
# construct also is. A reviewer looking at a picture catches that; a threshold
# never will.
#
# CPU, not GPU: this is a threshold, connected components and marching cubes.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

CT_DIR="${CT_DIR:-${DATA_DIR}/hf_export_v5/ct}"
LABEL_DIR="${LABEL_DIR:-${DATA_DIR}/v5_final}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/hardware_fix}"
SCAN="${SCAN:-qc_hardware/hardware_scan.csv}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}"
HU="${HU:-2500}"

[[ -f "${SCAN}" ]] || { echo "ERROR: no scan at ${SCAN}"; exit 1; }
mkdir -p "${LOGS_DIR}" "${OUT_DIR}"

NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N="${SLURM_ARRAY_TASK_COUNT:-1}"

# the instrumented case list, straight from the scan; this shard takes every Nth
mapfile -t ALL < <(tail -n +2 "${SCAN}" | cut -d, -f1)
CASES=()
for i in "${!ALL[@]}"; do
  (( i % N == SHARD )) && CASES+=("${ALL[$i]}")
done

echo "======================================================================"
echo " hardware batch   shard ${SHARD}/${N}   ${#CASES[@]} of ${#ALL[@]} cases"
echo "   threshold ${HU} HU   out ${OUT_DIR}   (v5 read only)"
echo "======================================================================"

RUN=(singularity exec --bind "${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data" --pwd /workspace
     --env PYTHONPATH=/workspace/scripts,PYTHONUNBUFFERED=1,MPLBACKEND=Agg "${SIF_PATH}")
REL() { realpath --relative-to="${DATA_DIR}" "$1"; }

ok=0; fail=0
for CASE in "${CASES[@]}"; do
  CT="${CT_DIR}/${CASE}_ct.nii.gz"
  LB="${LABEL_DIR}/${CASE}_label.nii.gz"
  if [[ ! -f "${CT}" || ! -f "${LB}" ]]; then
    echo "  ${CASE}: missing CT or label, skipped"; continue
  fi
  echo ""; echo "--- ${CASE} ------------------------------------------------------"
  # a case that fails must not take the shard down with it
  if "${RUN[@]}" python3 -u /workspace/scripts/seed_hardware.py \
        --case "${CASE}" --ct "/data/$(REL "${CT}")" --label "/data/$(REL "${LB}")" \
        --out "/data/$(REL "${OUT_DIR}")" --hu "${HU}"; then
    "${RUN[@]}" python3 -u /workspace/scripts/render_hardware.py \
        --case "${CASE}" --ct "/data/$(REL "${CT}")" --label "/data/$(REL "${LB}")" \
        --hardware "/data/$(REL "${OUT_DIR}")/${CASE}_hardware_only.nii.gz" \
        --out "/data/$(REL "${OUT_DIR}")" || echo "  ! render failed for ${CASE}"
    ok=$((ok + 1))
  else
    echo "  ! seed failed for ${CASE}"; fail=$((fail + 1))
  fi
done

echo ""
echo "======================================================================"
echo " shard ${SHARD} done: ${ok} proposed, ${fail} failed   ->  ${OUT_DIR}"
echo "======================================================================"
