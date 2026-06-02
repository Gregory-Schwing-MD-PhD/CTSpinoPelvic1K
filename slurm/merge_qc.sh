#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_merge_qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_qc_%j.out
#SBATCH --error=logs/merge_qc_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# merge_qc — join the per-case GT-free QC CSVs (neighbour-mixing / bone-leak /
# structure) into ONE ranked triage worklist (qc_master.csv), the list of cases
# students review. Two steps (CPU, fast):
#   1. build the radiologist BASELINE master from the manual QC CSVs (so the
#      continuous leak threshold = manual p-th percentile), if the manual CSVs
#      are present;
#   2. build the pseudo master, calibrated to that baseline, with QC_EXCLUDE
#      checks kept-but-not-triggering.
#
# QC_EXCLUDE defaults to `leak`: off-bone bleed is too hard to fix by hand, so
# we still RECORD it (column stays) but a leak-only case no longer lands on the
# worklist. See scripts/merge_qc.py (--exclude). Run AFTER vertebra-qc /
# bone-leak-qc / structure-qc, BEFORE export-crops.
#
# Options (env):
#   QC_PSEUDO_CSV     mixing (pseudo)    (default: data/qc_pseudo.csv)
#   LEAK_PSEUDO_CSV   leak   (pseudo)    (default: data/leak_pseudo.csv)
#   STRUCT_PSEUDO_CSV struct (pseudo)    (default: data/struct_pseudo.csv)
#   QC_MANUAL_CSV     mixing (manual)    (default: data/qc_manual.csv)
#   LEAK_MANUAL_CSV   leak   (manual)    (default: data/leak_manual.csv)
#   STRUCT_MANUAL_CSV struct (manual)    (default: data/struct_manual.csv)
#   QC_BASELINE       baseline master    (default: data/qc_manual_master.csv)
#   QC_MASTER_CSV     out worklist       (default: data/qc_master.csv)
#   QC_PCT            baseline percentile (default: 95)
#   QC_EXCLUDE        checks to drop from the trigger (default: leak; "" = none)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

QC_PSEUDO_CSV="${QC_PSEUDO_CSV:-${DATA_DIR}/qc_pseudo.csv}"
LEAK_PSEUDO_CSV="${LEAK_PSEUDO_CSV:-${DATA_DIR}/leak_pseudo.csv}"
STRUCT_PSEUDO_CSV="${STRUCT_PSEUDO_CSV:-${DATA_DIR}/struct_pseudo.csv}"
QC_MANUAL_CSV="${QC_MANUAL_CSV:-${DATA_DIR}/qc_manual.csv}"
LEAK_MANUAL_CSV="${LEAK_MANUAL_CSV:-${DATA_DIR}/leak_manual.csv}"
STRUCT_MANUAL_CSV="${STRUCT_MANUAL_CSV:-${DATA_DIR}/struct_manual.csv}"
QC_BASELINE="${QC_BASELINE:-${DATA_DIR}/qc_manual_master.csv}"
QC_MASTER_CSV="${QC_MASTER_CSV:-${DATA_DIR}/qc_master.csv}"
QC_PCT="${QC_PCT:-95}"
QC_EXCLUDE="${QC_EXCLUDE:-leak}"

mkdir -p "${LOGS_DIR}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
# the pseudo mixing CSV is the minimum input — without it there is nothing to triage
[[ -f "${QC_PSEUDO_CSV}" ]] || { echo "ERROR: no pseudo mixing CSV: ${QC_PSEUDO_CSV} (run make vertebra-qc)"; exit 1; }

# Build --flag args only for the CSVs that actually exist (any source is optional).
mk_args() {  # $1=mixing $2=leak $3=structure  -> echoes the present --flags
    local a=""
    [[ -f "$1" ]] && a+=" --mixing    /data/$(basename "$1")"
    [[ -f "$2" ]] && a+=" --leak      /data/$(basename "$2")"
    [[ -f "$3" ]] && a+=" --structure /data/$(basename "$3")"
    echo "${a}"
}

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"
RUN=(singularity exec --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}")

echo "======================================================================"
echo " merge_qc — build triage worklist"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   exclude   : '${QC_EXCLUDE}'   baseline pct: ${QC_PCT}"
echo "   master out: ${QC_MASTER_CSV}"
echo "   Started   : $(date)"
echo "======================================================================"

# ── 1. baseline master from the manual QC CSVs (for the leak percentile) ─────
BASELINE_ARG=""
MANUAL_ARGS="$(mk_args "${QC_MANUAL_CSV}" "${LEAK_MANUAL_CSV}" "${STRUCT_MANUAL_CSV}")"
if [[ -n "${MANUAL_ARGS// /}" ]]; then
    echo ""; echo "=== [1/2] radiologist baseline master -> ${QC_BASELINE} ==="
    # shellcheck disable=SC2086
    stdbuf -oL -eL "${RUN[@]}" python3 -u /workspace/scripts/merge_qc.py \
        ${MANUAL_ARGS} --out "/data/$(basename "${QC_BASELINE}")"
    BASELINE_ARG="--baseline /data/$(basename "${QC_BASELINE}") --pct ${QC_PCT}"
else
    echo ""; echo "=== [1/2] no manual QC CSVs found — skipping baseline calibration ==="
    echo "    (mixing/structure don't use it; leak is excluded by default anyway)"
fi

# ── 2. pseudo master, calibrated + QC_EXCLUDE dropped from the trigger ───────
echo ""; echo "=== [2/2] pseudo triage worklist -> ${QC_MASTER_CSV} ==="
EXCLUDE_ARG=""
[[ -n "${QC_EXCLUDE// /}" ]] && EXCLUDE_ARG="--exclude ${QC_EXCLUDE}"
PSEUDO_ARGS="$(mk_args "${QC_PSEUDO_CSV}" "${LEAK_PSEUDO_CSV}" "${STRUCT_PSEUDO_CSV}")"
# shellcheck disable=SC2086
stdbuf -oL -eL "${RUN[@]}" python3 -u /workspace/scripts/merge_qc.py \
    ${PSEUDO_ARGS} ${BASELINE_ARG} ${EXCLUDE_ARG} \
    --out "/data/$(basename "${QC_MASTER_CSV}")"

echo ""
echo "======================================================================"
echo " merge_qc done at $(date)"
echo "   worklist : ${QC_MASTER_CSV}   (sorted worst-first; needs_review=1 rows)"
echo "   excluded from trigger (still recorded): '${QC_EXCLUDE}'"
echo "   NEXT: make export-crops  (re-cut crops from the new worklist)"
echo "======================================================================"
