#!/usr/bin/env bash
#SBATCH --job-name=ctsp_build_v5
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_v5_%j.out
#SBATCH --error=logs/build_v5_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# build_v5 — merge the reviewed ribs and the reviewed spine onto the v4 base,
# then QC the result on the question the merge is supposed to fix.
#
# build_final_dataset writes labels FLAT into --out, not into out/labels.
#
#   1. build_final_dataset.py   v4 + ribs (34-57, reviews-ribs) + thoracic
#                               (8-19, reviews-spine). Anatomy-pure: each cohort
#                               only ever writes its own id range, so a reviewer
#                               who strayed outside their anatomy cannot leak.
#   2. qc_rib_vertebra_incidence.py on the MERGED tree -> does each rib now sit
#                               on the vertebra its number claims?
#   3. analyze_rib_incidence.py -> figure + table.
#
# The QC is run on the merge rather than only on v4 because the whole point of
# the rib cohort was to fix numbering; comparing the two runs is the evidence
# that it worked. Run the v4 baseline first (slurm/qc_rib_incidence.sh).
#
# Reads the org ledgers, so it needs the org token, kept OUTSIDE the repo at
# ~/.hf_org_token so it cannot be committed or scanned.
#
# Options (env):
#   OUT        merged tree            (default: data/v5_final)
#   V4_REV     base revision          (default: v4)
#   LIMIT      cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs
source configs/default.env          # singularity lives in a conda env here, not /usr/bin

OUT="${OUT:-data/v5_final}"
V4_REV="${V4_REV:-v4}"
LIMIT="${LIMIT:-0}"
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

HF_TOKEN="$(cat "$HOME/.hf_org_token" 2>/dev/null || true)"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "need an anonymous-mlhc token at ~/.hf_org_token (the ledgers are org-private)" >&2
  exit 1
fi
export HF_TOKEN

run() {
  singularity exec \
    --bind "${PROJECT_ROOT}":/w --bind "${HOME}/.cache/huggingface":/hf --pwd /w \
    --env HF_TOKEN="${HF_TOKEN}",HF_HOME=/hf,PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 \
    "${SIF}" "$@"
}

echo "=== 1. merge ribs + spine onto ${V4_REV} ==="
ARGS=(--out "/w/${OUT}" --v4-rev "${V4_REV}")
[[ "${LIMIT}" != "0" ]] && ARGS+=(--limit "${LIMIT}")
run python3 scripts/build_final_dataset.py "${ARGS[@]}"

echo
echo "=== 2. rib-vertebra incidence QC on the MERGED tree ==="
run python3 scripts/qc_rib_vertebra_incidence.py \
  --labels "/w/${OUT}" --workers "${WORKERS}" --out "/w/qc_rib_incidence_v5"

echo
echo "=== 3. figure + table ==="
run python3 scripts/analyze_rib_incidence.py --qc "/w/qc_rib_incidence_v5"

echo
echo "=== done ==="
echo "merged labels: $(ls "${OUT}"/*.nii.gz 2>/dev/null | wc -l)"
ls -l qc_rib_incidence_v5 2>/dev/null
