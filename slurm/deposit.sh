#!/usr/bin/env bash
#SBATCH --job-name=ctsp_deposit
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/deposit_%j.out
#SBATCH --error=logs/deposit_%j.err
# =============================================================================
# Everything that has to be true before the deposit is uploaded, in one job.
#
#   1. laterality across all 802 -- left_hip and right_hip once shipped swapped
#      and the release QC could not see it; 1035 was a partial version of the
#      same fault, which it also missed
#   2. the detached-piece counts in KNOWN_ISSUES section 4, which were measured
#      on v5 before thirteen records changed
#   3. assemble, with checksums
#   4. the docs, checked against the assembled directory: every backticked file
#      and manifest field has to resolve
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env

SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
SRC="/data/hf_export_v6"
OUT="/data/zenodo_v6"
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
mkdir -p "${SINGULARITY_TMPDIR}" logs
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT

RUN() {
  singularity exec --bind "$(pwd)":/w,"${DATA_DIR}":/data --pwd /w \
      --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF_PATH}" "$@"
}

echo "=== 1. checks, including laterality on all 802 ==="
RUN python3 -u /w/zenodo/assemble_deposit.py --src "${SRC}" --check --sidedness -1

echo ""
echo "=== 2. detached pieces, recounted on what is shipping ==="
RUN python3 -u /w/scripts/recount_detached_pieces.py \
    --src "${SRC}" --out /data/detached_pieces_v6.json

echo ""
echo "=== 3. assemble ==="
rm -rf "${DATA_DIR}/zenodo_v6"
RUN python3 -u /w/zenodo/assemble_deposit.py --src "${SRC}" --build "${OUT}"

echo ""
echo "=== 4. do the docs name anything the deposit does not contain? ==="
RUN python3 -u /w/scripts/check_docs_against_deposit.py --deposit "${OUT}"

echo ""
echo "=== 5. checksums verify against the files as written ==="
cd "${DATA_DIR}/zenodo_v6" && sha256sum -c SHA256SUMS.txt | grep -v ': OK$' || true
echo "  files: $(find . -type f | wc -l)"
echo "=== deposit ready ==="
