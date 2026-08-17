#!/usr/bin/env bash
#SBATCH --job-name=ctsp_rib_final
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rib_final_%j.out
#SBATCH --error=logs/rib_final_%j.out

# =============================================================================
# finalize_rib_review — act on the reviewed decisions, then re-QC to prove it.
#
# The reads are slow enough to matter: incidence() takes a point cloud per rib and
# per vertebra and a pairwise min-distance for every pair, so a case is minutes, not
# seconds. That is why this is a job and not a login-node one-liner.
#
#   1. apply_rib_decisions --apply    shift / lumbar / keep / flag / reannotate
#   2. qc_rib_vertebra_incidence      the same check that found the problem, re-run on
#                                     the rewritten labels, so the claim is measured
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w
     --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. applying reviewed decisions ==="
"${RUN[@]}" python3 scripts/apply_rib_decisions.py \
    --qc qc_rib_incidence_v5_fixed \
    --csv rib_review_sheets/decisions.csv \
    --labels data/v5_final --apply

echo
echo "=== 2. re-QC on the rewritten labels ==="
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py \
    --labels data/v5_final --workers 8 --out qc_rib_incidence_v5_final

echo
echo "=== done ==="
