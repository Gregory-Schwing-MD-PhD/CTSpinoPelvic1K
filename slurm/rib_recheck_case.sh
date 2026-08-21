#!/usr/bin/env bash
# Re-anchor and re-QC a single case after its vertebrae have been corrected by hand.
#
#   sbatch --export=CASE=0412 slurm/rib_recheck_case.sh
#
# Small on purpose: one label volume, seconds of work. It is a batch job rather than an
# ssh one-liner because nothing belongs on the login node, not because it is heavy.
#SBATCH --job-name=ctsp_ribcase
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=logs/ribcase_%j.out
#SBATCH --error=logs/ribcase_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
: "${CASE:?set CASE=NNNN}"

echo "=== 1. what the anchor pass would do to ${CASE} (dry run) ==="
"${RUN[@]}" python3 scripts/anchor_and_increment_ribs.py \
    --labels data/v5_final --cases "${CASE}" --out "qc_rib_anchor_${CASE}"

echo
echo "=== 2. apply ==="
"${RUN[@]}" python3 scripts/anchor_and_increment_ribs.py \
    --labels data/v5_final --cases "${CASE}" --out "qc_rib_anchor_${CASE}" --apply

echo
echo "=== 3. rib-vertebra incidence QC for ${CASE} ==="
# qc_rib_vertebra_incidence takes a LABELS DIRECTORY, not a case list, so give it a
# directory holding exactly this one case. Symlinks rather than copies: the volumes are
# hundreds of megabytes and this runs after every hand correction.
ONE="qc_one_${CASE}"
rm -rf "${ONE}"; mkdir -p "${ONE}"
ln -s "$(pwd)/data/v5_final/${CASE}_label.nii.gz" "${ONE}/${CASE}_label.nii.gz"
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py     --labels "${ONE}" --workers 4 --out "qc_rib_case_${CASE}"
echo
echo "--- offsets remaining for ${CASE} ---"
grep -h offset "qc_rib_case_${CASE}/rib_incidence.csv" || echo "  none"
rm -rf "${ONE}"

echo "=== done ==="
