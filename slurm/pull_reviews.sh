#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pull_reviews
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pull_reviews_%j.out
#SBATCH --error=logs/pull_reviews_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# pull_reviews — download finalized reviews from the REVIEW_REPO (the private
# review ledger dataset) and build the reduce_to_v3 inputs:
#   <REVIEWS_PULL_DIR>/review_repo/   downloaded cases/ + reviews/
#   <REVIEWS_PULL_DIR>/finals.json    {case_id: case['final']} for finalized cases
# plus a status summary (how many finalized vs need adjudication vs in review).
# See scripts/pull_reviews.py. Network, light resources.
#
# Options (env):
#   HF_TOKEN          read token for REVIEW_REPO        (REQUIRED)
#   REVIEW_REPO       review ledger, e.g.
#                     gregoryschwingmdphd/CTSpinoPelvic1K-reviews-triaged (REQUIRED)
#   REVIEWS_PULL_DIR  output dir   (default: data/reviews_pull)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_TOKEN="${HF_TOKEN:-}"
REVIEW_REPO="${REVIEW_REPO:-}"
REVIEWS_PULL_DIR="${REVIEWS_PULL_DIR:-${DATA_DIR}/reviews_pull}"

mkdir -p "${LOGS_DIR}" "${REVIEWS_PULL_DIR}"
[[ -z "${REVIEW_REPO}" ]] && { echo "ERROR: REVIEW_REPO required (the review ledger dataset)"; exit 1; }
[[ -z "${HF_TOKEN}" ]]    && { echo "ERROR: HF_TOKEN required (read access to ${REVIEW_REPO})"; exit 1; }
[[ -f "${SIF_PATH}" ]]   || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }

echo "======================================================================"
echo " pull_reviews <- ${REVIEW_REPO}"
echo "   Job ID  : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   out     : ${REVIEWS_PULL_DIR}"
echo "   Started : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,HF_TOKEN=${HF_TOKEN},REVIEW_REPO=${REVIEW_REPO}"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/pull_reviews.py \
        --out "/data/$(basename "${REVIEWS_PULL_DIR}")"

echo ""
echo "======================================================================"
echo " pull_reviews done at $(date)"
echo "   finals.json + review_repo/ -> ${REVIEWS_PULL_DIR}"
echo "   read the STATUS summary above; if any NEED ADJUDICATION, resolve them"
echo "   with 'python -m reviewtool adjudicate', then re-run pull-reviews."
echo "   NEXT: make reduce-v3"
echo "======================================================================"
