#!/usr/bin/env bash
#SBATCH --job-name=hipfix
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hipfix_%j.out
#SBATCH --error=logs/hipfix_%j.err
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

echo "=== 1. swap the four transposed hip pairs ==="
RUN python3 -u /w/scripts/verify_hip_laterality.py --src /data/hf_export_v6 \
    --cases 0027 0107 0790 0935 --fix --backup /data/hip_transpose_backup

echo ""
echo "=== 2. the 18 crossover records, measured against the spine midline ==="
RUN python3 -u /w/scripts/measure_hip_crossover.py --src /data/hf_export_v6 \
    --out /data/hip_crossover_v6.json \
    --cases 0012 0065 0135 0146 0172 0186 0376 0410 0471 0513 0746 0830 0917 0938 0957 1124 1145 1148

echo ""
echo "=== 3. full checks again, on the corrected labels ==="
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
echo "=== 7. checksums, against the files as written ==="
cd "${DATA_DIR}/zenodo_v6" && sha256sum -c SHA256SUMS.txt | grep -v ': OK$' || true
echo "  files: $(find . -type f | wc -l)"
echo "=== deposit ready ==="
