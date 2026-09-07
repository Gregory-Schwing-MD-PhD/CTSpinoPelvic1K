#!/usr/bin/env bash
#SBATCH --job-name=hipcross
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hipcross_%j.out
#SBATCH --error=logs/hipcross_%j.err
# =============================================================================
# The eighteen crossover records carry the 1035 fault: a large contiguous block
# of one hip bone wearing the other hip's label. Measured against the SPINE
# midline -- which does not move when a hip label is wrong -- the worst are
# 0186 at 49.7% and 0376 at 48.3%, with wrong-side blocks of 464k and 764k
# voxels. That is not a rind at the symphysis.
#
# relabel_hips_by_midline.py re-derives every hip voxel's side from the lumbar
# and sacral centroid, through the affine, so it is orientation-robust and
# idempotent on records that were already right.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT
RUN() { singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
        --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" "$@"; }

CASES="0012,0065,0135,0146,0172,0186,0376,0410,0471,0513,0746,0830,0917,0938,0957,1124,1145,1148"

mkdir -p "${DATA_DIR}/hip_crossover_backup"
for c in $(echo "$CASES" | tr ',' ' '); do
  cp -n "${DATA_DIR}/hf_export_v6/labels/${c}_label.nii.gz" \
        "${DATA_DIR}/hip_crossover_backup/" 2>/dev/null || true
done
echo "  backed up $(ls "${DATA_DIR}/hip_crossover_backup" | wc -l) label(s)"

echo ""
echo "=== 1. re-derive hip laterality from the spine midline ==="
RUN python3 -u /w/scripts/relabel_hips_by_midline.py \
    --labels_dir /data/hf_export_v6/labels --tokens "$CASES"

echo ""
echo "=== 2. did it work? measure the same records again ==="
RUN python3 -u /w/scripts/measure_hip_crossover.py --src /data/hf_export_v6 \
    --out /data/hip_crossover_after.json \
    --cases $(echo "$CASES" | tr ',' ' ')

echo ""
echo "=== 3. full checks on the corrected release ==="
RUN python3 -u /w/zenodo/assemble_deposit.py --src /data/hf_export_v6 --check --sidedness -1

echo ""
echo "=== 4. detached pieces, recounted ==="
RUN python3 -u /w/scripts/recount_detached_pieces.py --src /data/hf_export_v6 \
    --out /data/detached_pieces_v6.json

echo ""
echo "=== 5. assemble ==="
rm -rf "${DATA_DIR}/zenodo_v6"
RUN python3 -u /w/zenodo/assemble_deposit.py --src /data/hf_export_v6 --build /data/zenodo_v6

echo ""
echo "=== 6. docs vs the assembled deposit ==="
RUN python3 -u /w/scripts/check_docs_against_deposit.py --deposit /data/zenodo_v6

echo ""
echo "=== 7. checksums against the files as written ==="
cd "${DATA_DIR}/zenodo_v6" && sha256sum -c SHA256SUMS.txt | grep -v ': OK$' || true
echo "  files: $(find . -type f | wc -l)"
echo "=== deposit ready ==="
