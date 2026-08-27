#!/usr/bin/env bash
#SBATCH --job-name=ctsp_zenodo_v6
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/zenodo_v6_%j.out
#SBATCH --error=logs/zenodo_v6_%j.err
# =============================================================================
# zenodo_v6.sh — assemble the Zenodo deposit from v6, not v5.
#
#   sbatch slurm/zenodo_v6.sh
#
# The deposit is the LABELS and the CROSSWALK: 1.8 GB against 193 GB of CT that
# is already in TCIA. What the source collections never published is the mapping
# from each annotation to the series it was drawn on, and manifest.json carries
# it as SeriesInstanceUIDs.
#
# Built from v6 so the archived copy is the one with surgical hardware labelled,
# 0068 renumbered, and 1035's hips lateralised -- rather than v5, which has none
# of those and whose published tree is additionally missing five hand-corrections
# that were never re-exported.
#
# --sidedness -1 checks all 802. It was previously left unset, which means SKIP --
# the comment here claimed the check was running and it was not. It exists because
# three records once shipped with left_hip and right_hip swapped and the release QC
# could not see it, and because 1035 shipped with half a hip wearing the other hip's
# label. The second of those needed a second test, which assemble_deposit.py now has.
#
# Resources are deliberately small. This job copies files, hashes them, and reads one label
# volume at a time -- about 210 MB each. The cluster is memory-allocated to within ~5 GB per
# node with CPUs idle, so a large --mem request is unschedulable for reasons unrelated to the
# work; 4 GB is what it needs and 4 GB is what gets placed.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
SRC="${SRC:-/data/hf_export_v6}"
OUT="${OUT:-/data/zenodo_v6}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/zenodo/assemble_deposit.py --src "${SRC}" --check --sidedness -1

echo ""
echo "=== assembling ==="
singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" \
    python3 -u /w/zenodo/assemble_deposit.py --src "${SRC}" --build "${OUT}"
