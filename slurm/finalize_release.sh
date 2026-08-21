#!/usr/bin/env bash
# Everything the release has to pass, in one place, in dependency order.
#
# The invariant check runs FIRST and stops the chain. A transposed label or a stray id
# poisons every measurement downstream, and there is no value in computing morphometrics
# on a corpus that has not established it is internally consistent -- that only produces
# numbers that look finished.
#SBATCH --job-name=ctsp_finalize
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
W="${SLURM_CPUS_PER_TASK:-12}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")

fail=0

echo "=== 1. release invariants (geometry, ids, sidedness) ==="
"${RUN[@]}" python3 scripts/check_release_invariants.py \
    --labels data/v5_final --ct data/hf_export/ct --workers "$W" --out qc_invariants.csv
rc=$?
if [ "$rc" -ne 0 ]; then
  echo
  echo "*** INVARIANTS FAILED. Stopping: measurements computed on an inconsistent"
  echo "*** corpus would look finished and be wrong. Fix the cases above and re-run."
  exit 2
fi

echo
echo "=== 2. rib-vertebra incidence, release-wide ==="
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py \
    --labels data/v5_final --workers "$W" --out qc_rib_incidence_final || fail=1

echo
echo "=== 3. transition morphometrics (count-free) ==="
"${RUN[@]}" python3 scripts/extract_transition_morphometrics.py \
    --labels data/v5_final --workers "$W" --out morphometrics || fail=1

echo
echo "=== 4. surgical morphometrics ==="
"${RUN[@]}" python3 scripts/extract_surgical_morphometrics.py \
    --labels data/v5_final --workers "$W" --out morphometrics
srg=$?
[ "$srg" -eq 2 ] && echo "*** surgical morphometrics flagged an implausible median"

echo
echo "=== 5. gallery distributions ==="
"${RUN[@]}" python3 scripts/build_gallery_data.py \
    --csv morphometrics/transition_morphometrics.csv \
    --out gallery_data/distributions.json || fail=1

echo
echo "================ SUMMARY ================"
echo "invariants          : PASS"
echo "rib QC              : see qc_rib_incidence_final/summary.json"
echo "surgical plausibility: $([ "$srg" -eq 0 ] && echo PASS || echo FLAGGED)"
[ "$fail" -ne 0 ] && echo "one or more steps errored -- read the log above"
echo "=== done ==="
exit $(( fail || (srg == 2) ))
