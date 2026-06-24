#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_verify_verse
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_verse_%j.out
#SBATCH --error=logs/verify_verse_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# verify_verse_native — assert a built export tree is VerSe-native + collision-free.
# Runs scripts/verify_label_scheme.py inside the project container. NON-zero exit
# (and a FAIL line in the log) means the tree still carries the old scheme — do NOT
# trust the push; fix and rebuild.
#
#   V3_DIR=<tree> sbatch slurm/verify_verse_native.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-verify$$}"
source configs/default.env

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"

echo "[verify_verse_native] checking ${V3_DIR} against scripts/label_scheme.py"
singularity exec --bind "${PROJECT_ROOT}:/workspace" --pwd /workspace "${SIF_PATH}" \
  python3 scripts/verify_label_scheme.py --tree "${V3_DIR}"
echo "[verify_verse_native] PASS — ${V3_DIR} is VerSe-native."
