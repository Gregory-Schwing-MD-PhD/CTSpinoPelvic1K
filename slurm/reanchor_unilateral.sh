#!/usr/bin/env bash
#SBATCH --job-name=ctsp_rib_uni
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rib_uni_%j.out
#SBATCH --error=logs/rib_uni_%j.out

# =============================================================================
# reanchor_unilateral — finish the unilateral cages, and look at the lumbar-rib cases
# the review sheets never showed us.
#
# THE GAP THIS CLOSES. The review set was built from cases with OFFSET ribs, so a case
# carrying a lumbar rib on an otherwise correctly-numbered cage never appeared in it.
# v5 has 14 lumbar-rib cases and only 9 reached review. Shipping the other 4 as plain
# "rib 12" while the 9 carry class 74/75 would put identical anatomy under two labels.
#
# 0344 is deliberately NOT here: l6->L3 and l7->L5 is not this phenotype. A 6th rib
# cannot reach L3, so that is a gross mislabel and belongs with the reannotation set.
#
# No final QC in this job. Four more cases change right after it, and an 802-case QC run
# that is stale on arrival is 30 minutes spent to measure a state nobody ships.
#
# Depends on the finalize job: both write data/v5_final.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w
     --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 0. rib, or mislabelled transverse process? the unilateral cases ==="
for C in 0315 0660 0720; do
  echo "--- $C"
  "${RUN[@]}" python3 scripts/render_stub_axial.py --labels data/v5_final \
      --case "$C" --struct 75 --vert 20 --n 8 || \
  "${RUN[@]}" python3 scripts/render_stub_axial.py --labels data/v5_final \
      --case "$C" --struct 57 --vert 20 --n 8
  echo
done

echo "=== 1. re-anchor the two unilateral cages already reclassed (APPLY) ==="
"${RUN[@]}" python3 scripts/lumbar_rib_class_v5.py \
    --labels data/v5_final --qc qc_rib_incidence_v5_fixed \
    --cases 0315,0660 --apply

echo
echo "=== 2. the four never reviewed (DRY RUN — read before applying) ==="
"${RUN[@]}" python3 scripts/lumbar_rib_class_v5.py \
    --labels data/v5_final --qc qc_rib_incidence_v5_fixed \
    --cases 0231,0389,0473,0720

echo
echo "=== done — no QC here on purpose; run it once, after step 2 is applied ==="
