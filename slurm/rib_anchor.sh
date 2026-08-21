#!/usr/bin/env bash
#SBATCH --job-name=ctsp_ribanchor
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ribanchor_%j.out
#SBATCH --error=logs/ribanchor_%j.out

# The login node killed this twice (exit 137). Point-distance work over full volumes is a
# batch job, not an ssh one-liner.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
CASES=0179,0412,1153,0378,0487,0787

echo "=== 1. anchor-and-increment, DRY RUN on the offset cases ==="
"${RUN[@]}" python3 scripts/anchor_and_increment_ribs.py \
    --labels data/v5_final --cases "${CASES}" --out qc_rib_anchor

echo; echo "=== 2. structure availability across the release ==="
"${RUN[@]}" python3 avail.py || true

echo; echo "=== done ==="