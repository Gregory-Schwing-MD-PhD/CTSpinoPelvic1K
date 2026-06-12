#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_qc_dashboard
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/qc_dashboard_%j.out
#SBATCH --error=logs/qc_dashboard_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# qc_dashboard — aggregate the propagation/completion QC CSVs into summary figures
# (placement drop, before/after overlap, GT-vs-model Dice, completeness, model
# completion = GT incompleteness). See scripts/qc_dashboard.py.
#
# Options (env):
#   PROP_OUT_DIR  propagate_pelvis dir   (default: data/placed/pelvic_propagated)
#   PSEUDO_OUT_DIR v2 tree w/ completion (default: data/hf_export_v2)
#   DASH_OUT_DIR  figure output dir      (default: data/qc_dashboard)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

PROP_OUT_DIR="${PROP_OUT_DIR:-${DATA_DIR}/placed/pelvic_propagated}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
DASH_OUT_DIR="${DASH_OUT_DIR:-${DATA_DIR}/qc_dashboard}"

mkdir -p "${LOGS_DIR}" "${DASH_OUT_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }

REL() { echo "/data/$(realpath --relative-to="${DATA_DIR}" "$1")"; }
ARGS=( --out_dir "$(REL "${DASH_OUT_DIR}")" )
[[ -f "${PROP_OUT_DIR}/propagate_qc.csv" ]] \
    && ARGS+=( --propagate_qc "$(REL "${PROP_OUT_DIR}/propagate_qc.csv")" )
[[ -f "${PSEUDO_OUT_DIR}/propagated_completion_qc.csv" ]] \
    && ARGS+=( --completion_qc "$(REL "${PSEUDO_OUT_DIR}/propagated_completion_qc.csv")" )

echo "======================================================================"
echo " qc_dashboard — dataset QC figures"
echo "   Job ID : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   out    : ${DASH_OUT_DIR}"
echo "   Started: $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,MPLBACKEND=Agg"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/qc_dashboard.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " qc_dashboard done at $(date)   figures: ${DASH_OUT_DIR}"
echo "======================================================================"
