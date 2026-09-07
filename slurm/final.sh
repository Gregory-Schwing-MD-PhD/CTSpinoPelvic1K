#!/usr/bin/env bash
#SBATCH --job-name=deposit_final
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
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

echo "=== checks, laterality included, on the corrected release ==="
RUN python3 -u /w/zenodo/assemble_deposit.py --src /data/hf_export_v6 --check --sidedness -1

echo ""
echo "=== assemble ==="
rm -rf "${DATA_DIR}/zenodo_v6"
RUN python3 -u /w/zenodo/assemble_deposit.py --src /data/hf_export_v6 --build /data/zenodo_v6

echo ""
echo "=== docs vs the assembled deposit ==="
RUN python3 -u /w/scripts/check_docs_against_deposit.py --deposit /data/zenodo_v6

echo ""
echo "=== checksums, against the files as written ==="
cd "${DATA_DIR}/zenodo_v6"
BAD=$(sha256sum -c SHA256SUMS.txt 2>/dev/null | grep -vc ': OK$' || true)
echo "  files: $(find . -type f | wc -l)   checksum failures: ${BAD}"
echo "  size:  $(du -sh . | cut -f1)"
echo "=== deposit ready ==="
