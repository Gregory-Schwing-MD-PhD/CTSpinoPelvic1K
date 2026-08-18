#!/usr/bin/env bash
#SBATCH --job-name=ctsp_overnight
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/overnight_%j.out
#SBATCH --error=logs/overnight_%j.out

# Everything left, in the only order that is safe: the steps that WRITE data/v5_final run
# before the steps that read it, so no QC or measurement ever describes a half-written set.
#
#   1. 0231 re-anchor      the reverse lookup found rib 11 at 16.9/19.3mm from a labelled
#                          T12 with the runner-up at 56/62mm, and the sacrum reads 4
#                          foramina pairs by eye -- so it is a lumbar rib, and the +1 holds
#   2. final rib QC        the measured number for v5, run once, at the end
#   3. morphometrics       every densified case
#   4. figure              the transitional-anatomy landscape
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

echo "=== 1. re-anchor 0231 (vertebra-end evidence) ==="
"${RUN[@]}" python3 scripts/lumbar_rib_class_v5.py \
    --labels data/v5_final --qc qc_rib_incidence_v5_fixed --cases 0231 --apply

echo; echo "=== 2. final rib QC for v5 ==="
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py \
    --labels data/v5_final --workers 24 --out qc_rib_incidence_v5_final

echo; echo "=== 3. transitional morphometrics ==="
"${RUN[@]}" python3 scripts/extract_transition_morphometrics.py \
    --labels data/v5_final --manifest data/hf_export_v4/manifest.json \
    --workers 24 --out morphometrics

echo; echo "=== 4. figure ==="
"${RUN[@]}" python3 scripts/plot_transition_morphometrics.py \
    --csv morphometrics/transition_morphometrics.csv --out morphometrics

echo; echo "=== done ==="
