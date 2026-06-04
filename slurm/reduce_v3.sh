#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_reduce_v3
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/reduce_v3_%j.out
#SBATCH --error=logs/reduce_v3_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# reduce_v3 — fold finalized reviews into the corrected v3 tree: manual labels
# untouched, reviewed regions get the corrected label swapped in, rejected cases
# dropped. See scripts/review/reduce_to_v3.py. Run AFTER make pull-reviews.
#
# V3_LABELS_ONLY=1 (default) skips copying the ~240 GB of CT volumes — labels +
# manifest only, which is fast and all the GT-free QC needs. Set V3_LABELS_ONLY=0
# for a full, pushable release tree (copies CTs too).
#
# Options (env):
#   PSEUDO_OUT_DIR    v2 tree            (default: data/hf_export_v2)
#   REVIEWS_PULL_DIR  pull-reviews out   (default: data/reviews_pull)
#   V3_OUT_DIR        v3 tree to create  (default: data/hf_export_v3)
#   V3_LABELS_ONLY    1=labels only (default), 0=full tree with CTs
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REVIEWS_PULL_DIR="${REVIEWS_PULL_DIR:-${DATA_DIR}/reviews_pull}"
V3_OUT_DIR="${V3_OUT_DIR:-${DATA_DIR}/hf_export_v3}"
V3_LABELS_ONLY="${V3_LABELS_ONLY:-1}"

mkdir -p "${LOGS_DIR}"
[[ -f "${SIF_PATH}" ]]                       || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -f "${PSEUDO_OUT_DIR}/manifest.json" ]]   || { echo "ERROR: no manifest.json in ${PSEUDO_OUT_DIR}"; exit 1; }
[[ -f "${REVIEWS_PULL_DIR}/finals.json" ]]   || { echo "ERROR: no finals.json in ${REVIEWS_PULL_DIR} — run 'make pull-reviews' first"; exit 1; }

EXTRA=""
[[ "${V3_LABELS_ONLY}" != "0" ]] && EXTRA="--labels_only"

echo "======================================================================"
echo " reduce_v3 — corrected v3 tree"
echo "   Job ID      : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   v2 tree     : ${PSEUDO_OUT_DIR}"
echo "   finals      : ${REVIEWS_PULL_DIR}/finals.json"
echo "   out (v3)    : ${V3_OUT_DIR}   (labels_only=${V3_LABELS_ONLY})"
echo "   Started     : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

# shellcheck disable=SC2086
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/review/reduce_to_v3.py \
        --v2          "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --finals      "/data/$(basename "${REVIEWS_PULL_DIR}")/finals.json" \
        --labels_root "/data/$(basename "${REVIEWS_PULL_DIR}")/review_repo" \
        --out         "/data/$(basename "${V3_OUT_DIR}")" ${EXTRA}

echo ""
echo "======================================================================"
echo " reduce_v3 done at $(date)"
echo "   corrected tree -> ${V3_OUT_DIR}"
echo "   QC it:  make vertebra-qc  PSEUDO_OUT_DIR=${V3_OUT_DIR} QC_PSEUDO_CSV=${DATA_DIR}/qc_v3.csv"
echo "           make structure-qc PSEUDO_OUT_DIR=${V3_OUT_DIR} STRUCT_PSEUDO_CSV=${DATA_DIR}/struct_v3.csv"
echo "           make merge-qc QC_PSEUDO_CSV=${DATA_DIR}/qc_v3.csv STRUCT_PSEUDO_CSV=${DATA_DIR}/struct_v3.csv QC_MASTER_CSV=${DATA_DIR}/qc_master_v3.csv"
echo "======================================================================"
