#!/usr/bin/env bash
#SBATCH --job-name=ctsp_build_v6
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/build_v6_%j.out
#SBATCH --error=logs/build_v6_%j.err
# =============================================================================
# build_v6.sh — assemble the v6 tree.
#
#   sbatch slurm/build_v6.sh
#
# v6 = v5 + surgical instrumentation, labelled, for the eleven cases a
# radiologist confirmed. The hardware block (76-82) has been declared in every
# release so far and populated in none of them.
#
# Scope is deliberately narrow: hardware, plus 0068's corrections. The
# release-wide body de-mixing is NOT in this build -- 0068 was a pseudolabel
# case and there is no evidence the other 801 share that defect.
#
# CTs are hardlinked, not copied: they are byte-identical to v5 and there is no
# reason to hold 1.8 GB twice.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
V5_DIR="${V5_DIR:-${DATA_DIR}/hf_export_v5}"
V6_DIR="${V6_DIR:-${DATA_DIR}/hf_export_v6}"
FINAL="${FINAL:-${DATA_DIR}/hardware_final}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs "${V6_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

REL() { realpath --relative-to="${DATA_DIR}" "$1"; }

echo "  v5 : ${V5_DIR}"
echo "  hw : ${FINAL}  ($(ls "${FINAL}"/*_label_hw.nii.gz 2>/dev/null | wc -l) reviewed)"
echo "  out: ${V6_DIR}"

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/scripts/build_v6.py \
        --v5     "/data/$(REL "${V5_DIR}")" \
        --labels /data/v5_final \
        --proposals "/data/$(REL "${FINAL}")" \
        --out    "/data/$(REL "${V6_DIR}")" \
        --project-root /w

echo ""
echo "  labels: $(ls "${V6_DIR}/labels" 2>/dev/null | wc -l)   cts: $(ls "${V6_DIR}/ct" 2>/dev/null | wc -l)"
