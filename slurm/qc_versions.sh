#!/usr/bin/env bash
# Version-progression QC — ONE VERSION PER ARRAY TASK.
#
# WHY THIS IS AN ARRAY NOW. Job 40011919 ran all four versions in a single eight-hour task.
# It finished v2, v3 and v5pre, reached 600/802 of v5, and hit the wall clock — and because
# the script wrote only after all four completed, it produced nothing at all. Three
# versions of finished work discarded because the fourth ran long.
#
# Two changes fix that, and both are needed:
#   * the script writes each version's results the moment that version finishes, atomically,
#     and skips a version whose part file already exists (see qc_version_progression.py);
#   * this submits one task per version, so a version that runs long costs only itself, and
#     the four run concurrently rather than end to end.
#
# The expensive step is rib–vertebra incidence: every rib is compared against every
# candidate vertebra as a point cloud, so cost scales with ribs times vertebrae per case.
# That is what made a whole-corpus pass take hours, and it is why the per-version budget
# below is generous.
#
#   sbatch slurm/qc_versions.sh              # all four versions
#   sbatch --array=3 slurm/qc_versions.sh    # just v5
#
# After the array finishes, assemble the combined tables (cheap — reads the part files):
#   bash slurm/qc_versions.sh --assemble
#
#SBATCH --job-name=ctsp_qcver
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/qcver_%A_%a.out
#SBATCH --error=logs/qcver_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

VERSIONS=(
  "v2=data/hf_export_v2/labels"
  "v3=data/hf_export_v3/labels"
  "v5pre=data/hf_export_v5/labels"
  "v5=data/v5_final"
)

run() {
  singularity exec --bind "$(pwd)":/w --pwd /w \
    --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "$SIF" \
    python3 scripts/qc_version_progression.py --workers 24 "$@"
}

# `--assemble` re-reads the per-version part files and writes the combined tables. It does
# no segmentation work, so it is safe to run on the login node and safe to run repeatedly;
# a version still missing is reported and omitted rather than silently treated as zero.
if [ "${1:-}" = "--assemble" ]; then
  run --versions "${VERSIONS[@]}" --out qc_final/version_progression.csv
  exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
SPEC="${VERSIONS[$IDX]}"
echo "task ${IDX}: ${SPEC}"
run --versions "$SPEC" --out "qc_final/version_progression_${IDX}.csv" \
    --per-case-out "qc_final/version_progression_percase_${IDX}.csv"
