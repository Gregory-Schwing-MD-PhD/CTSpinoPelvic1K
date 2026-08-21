#!/usr/bin/env bash
#SBATCH --job-name=ctsp_ribfinal
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=05:00:00
#SBATCH --output=logs/ribfinal_%j.out
#SBATCH --error=logs/ribfinal_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. apply the anchor-and-increment to 0179 ==="
"${RUN[@]}" python3 scripts/anchor_and_increment_ribs.py \
    --labels data/v5_final --cases 0179 --out qc_rib_anchor --apply

echo; echo "=== 2. rib QC: the measured result ==="
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py \
    --labels data/v5_final --workers 16 --out qc_rib_incidence_v5_final

echo; echo "=== 3. surgical morphometrics (PI, lateral corridor, pedicle, canal) ==="
"${RUN[@]}" python3 scripts/extract_surgical_morphometrics.py \
    --labels data/v5_final --manifest data/hf_export_v4/manifest.json \
    --workers 16 --out morphometrics

echo; echo "=== done ==="