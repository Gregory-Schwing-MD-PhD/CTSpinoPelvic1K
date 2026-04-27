#!/usr/bin/env bash
#SBATCH --job-name=audit_lr_full
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/audit_lr_full_%j.out
#SBATCH --error=logs/audit_lr_full_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Full-dataset L/R hip alignment audit
#
# For every hip-bearing record in data/hf_export/manifest.json (configs:
# fused, pelvic_native), compares cached TotalSegmentator predictions
# against the GT label via no-swap vs swap Dice. Verdicts: OK, FLIPPED,
# MIXED, SKIP. Outputs a per-record CSV plus stdout summary.
#
# Cross-references FLIPPED tokens against configs/flip_list.json so the 9
# manually AP-flipped tokens (which have stale TS predictions cached
# against the pre-flip CT) are flagged as "expected" rather than real
# bugs.
#
# Usage:
#   sbatch slurm/audit_lr_full.sh
#
# Options (env):
#   AUDIT_WORKERS=16     parallel workers (default = SLURM cpus-per-task)
#   OUT_CSV=...          override output CSV path
#   TS_GLOB="..."        override TS prediction glob (defaults to all runs)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Sourced env (DATA_DIR, LOGS_DIR, etc.) — match other slurm scripts
if [[ -f configs/default.env ]]; then
    source configs/default.env
fi

LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
TS_CONTAINER="${TS_CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"

AUDIT_WORKERS="${AUDIT_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
OUT_CSV="${OUT_CSV:-${DATA_DIR}/audit_lr_full.csv}"
TS_GLOB="${TS_GLOB:-/results/totalseg_bench_*/ts_preds}"

mkdir -p "${LOGS_DIR}"

echo "======================================================================"
echo " L/R Hip Alignment Audit — Full Dataset"
echo "   Job ID       : ${SLURM_JOB_ID:-local}"
echo "   Node         : $(hostname)"
echo "   CPUs         : ${AUDIT_WORKERS}"
echo "   HF export    : ${HF_EXPORT_DIR}"
echo "   TS results   : ${RESULTS_DIR}"
echo "   TS glob      : ${TS_GLOB}"
echo "   Output CSV   : ${OUT_CSV}"
echo "   Container    : ${TS_CONTAINER}"
echo "   Started      : $(date)"
echo "======================================================================"

# Pre-flight
if [[ ! -f "${TS_CONTAINER}" ]]; then
    echo "ERROR: TS container not found at ${TS_CONTAINER}"
    echo "       Run: make build-container  (or: make hpc-pull)"
    exit 1
fi

if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: ${HF_EXPORT_DIR}/manifest.json not found."
    echo "       Run Stage 3 first:  make export-dataset"
    exit 1
fi

# Locate the audit script — prefer scripts/ over root
AUDIT_PY="${PROJECT_ROOT}/scripts/audit_lr_full.py"
if [[ ! -f "${AUDIT_PY}" ]]; then
    AUDIT_PY="${PROJECT_ROOT}/audit_lr_full.py"
fi
if [[ ! -f "${AUDIT_PY}" ]]; then
    echo "ERROR: audit_lr_full.py not found in scripts/ or project root."
    exit 1
fi
echo "  Using script : ${AUDIT_PY}"

# Singularity binds: project root, data, results
BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${RESULTS_DIR}:/results"

# Container-side paths
C_HF_EXPORT="/data/hf_export"
C_OUT_CSV="/data/$(basename ${OUT_CSV})"
C_AUDIT_PY="/workspace/$(realpath --relative-to=${PROJECT_ROOT} ${AUDIT_PY})"

echo ""
echo "Running audit inside container..."
echo ""

singularity exec --nv \
    --env "AUDIT_WORKERS=${AUDIT_WORKERS},PYTHONPATH=/workspace/scripts:/workspace" \
    --bind "${BINDS}" \
    --pwd /workspace \
    "${TS_CONTAINER}" \
    python3 "${C_AUDIT_PY}" \
        --hf_export "${C_HF_EXPORT}" \
        --ts_glob   "${TS_GLOB}" \
        --workers   "${AUDIT_WORKERS}" \
        --out_csv   "${C_OUT_CSV}"

echo ""
echo "======================================================================"
echo " Audit complete at $(date)"
echo "   CSV: ${OUT_CSV}"
echo ""
echo " Quick scan:"
if [[ -f "${OUT_CSV}" ]]; then
    echo ""
    echo "   Verdict counts:"
    awk -F, 'NR>1 {print $3}' "${OUT_CSV}" | sort | uniq -c | awk '{printf "     %-12s %d\n", $2, $1}'
    echo ""
    echo "   Top 10 lines of any FLIPPED rows:"
    head -1 "${OUT_CSV}" | awk -F, '{printf "     %-8s %-14s %-8s  %-7s/%-7s  %-7s/%-7s  flip?\n", $1, $2, $3, $5, $6, $7, $8}'
    awk -F, 'NR>1 && $3=="FLIPPED" {printf "     %-8s %-14s %-8s  %-7s/%-7s  %-7s/%-7s  %s\n", $1, $2, $3, $5, $6, $7, $8, $11}' "${OUT_CSV}" | head -10
fi
echo "======================================================================"
