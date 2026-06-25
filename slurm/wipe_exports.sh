#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_wipe_exports
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wipe_exports_%j.out
#SBATCH --error=logs/wipe_exports_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# wipe_exports — delete ALL hf_export* trees (incl. the _work resume markers /
# prediction caches) on a COMPUTE node, so the login node never blocks on a big
# NFS delete. The rebuild DAG (v1 base export) is chained afterok this job, so a
# clean VerSe-native rebuild can't pick up any old-scheme leftover.
#
# Triggered by NUKE=1 in ship_v2.sh / launch_all.sh — you don't run it directly.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env
DATA_DIR="${DATA_DIR:?DATA_DIR not set — refusing to rm}"   # guard against rm -rf /hf_export*

echo "[wipe_exports] DATA_DIR=${DATA_DIR}"
echo "[wipe_exports] removing:"
ls -d "${DATA_DIR}"/hf_export* 2>/dev/null | sed 's/^/   /' || echo "   (nothing present)"
rm -rf "${DATA_DIR}"/hf_export*
echo "[wipe_exports] done at $(date) — all hf_export* removed."
