#!/usr/bin/env bash
#SBATCH --job-name=ctsp_final_v5
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --output=logs/final_v5_%j.out
#SBATCH --error=logs/final_v5_%j.out

# The whole finalisation, in the only order that is safe: everything that WRITES labels runs
# before anything that reads them, so no QC ever describes a half-written dataset.
#
#   1. speckle strip   stray vertebra fragments. Keeps any piece TOUCHING A FACE of the
#                      volume at any size -- a vertebra clipped by the scan legitimately
#                      splits, and largest-component-wins would delete hand-drawn bone.
#   2. re-anchor       13 cases just gained thoracic vertebrae, so ribs that were
#                      unnameable ("no thoracic body to name them against") become
#                      nameable under the rule: a rib is named for the vertebra it
#                      articulates with.
#   3. rib QC          the measured result, on the rewritten labels.
#   4. morphometrics   thoracic coverage feeds n_rib_bearing and lowest_rib_bearing.
#   5. figures         transitional landscape + the cross-sectional report with demographics.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
EDITED=0033,0167,0241,0344,0357,0383,0389,0409,0424,0428,0720,0730,1004

echo "=== 1a. speckle: dry run over the whole release ==="
"${RUN[@]}" python3 scripts/strip_vertebra_speckle.py --labels data/v5_final --out qc_speckle

echo; echo "=== 1b. speckle: apply ==="
"${RUN[@]}" python3 scripts/strip_vertebra_speckle.py --labels data/v5_final --out qc_speckle --apply

echo; echo "=== 2. re-anchor ribs on the newly-thoracic cases ==="
"${RUN[@]}" python3 scripts/lumbar_rib_class_v5.py --labels data/v5_final \
    --qc qc_rib_incidence_v5_fixed --cases "${EDITED}" --apply

echo; echo "=== 3. rib QC (the measured result) ==="
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py \
    --labels data/v5_final --workers 16 --out qc_rib_incidence_v5_final

echo; echo "=== 4. morphometrics ==="
"${RUN[@]}" python3 scripts/extract_transition_morphometrics.py \
    --labels data/v5_final --manifest data/hf_export_v4/manifest.json \
    --workers 16 --out morphometrics

echo; echo "=== 5a. transitional landscape ==="
"${RUN[@]}" python3 scripts/plot_transition_morphometrics.py \
    --csv morphometrics/transition_morphometrics.csv --out morphometrics

echo; echo "=== 5b. cross-sectional report (demographics) ==="
"${RUN[@]}" python3 scripts/plot_anatomical_report.py \
    --csv morphometrics/transition_morphometrics.csv --out morphometrics

echo; echo "=== done ==="
