#!/usr/bin/env bash
# Rerun the surgical morphometrics after the geometry rewrite.
#
# The first run (39996283) reported sacral slope 83 deg, lordosis 10 deg and a Torg
# ratio of 0.2 across all 802 cases -- every one of them impossible, and none of them
# caught by the run itself. The script now checks its own medians against published
# adult ranges and exits non-zero if any land outside, so a repeat of that failure
# stops here rather than reaching a figure.
#
# Smaller footprint than the job it replaces: this is one measurement pass, not the
# rib chain, so it does not need 110G or five hours -- and a smaller ask gets scheduled
# sooner.
#SBATCH --job-name=ctsp_surgrecheck
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/surgrecheck_%j.out
#SBATCH --error=logs/surgrecheck_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

echo "=== surgical morphometrics, rebuilt geometry ==="
singularity exec --bind "$(pwd)":/w --pwd /w \
  --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/extract_surgical_morphometrics.py \
    --labels data/v5_final --workers "${SLURM_CPUS_PER_TASK:-12}" \
    --out morphometrics
rc=$?

echo
if [ "$rc" -eq 0 ]; then
  echo "=== PASS: every measure inside its plausible range ==="
elif [ "$rc" -eq 2 ]; then
  echo "=== FAIL: at least one median is still implausible. Do not report these. ==="
else
  echo "=== ERROR: exit ${rc} ==="
fi
exit "$rc"
