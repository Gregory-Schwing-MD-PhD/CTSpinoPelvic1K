#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pelvis_opposing_qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pelvis_opposing_qc_%j.out
#SBATCH --error=logs/pelvis_opposing_qc_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# pelvis_opposing_qc — validate a PSEUDOLABELLED pelvis against the SAME
# patient's REAL pelvis GT from the OPPOSING acquisition, using pose-invariant
# shape descriptors (no registration). Catches class mixing / irregular borders
# (vol+extent %% off) and L/R hip swaps (laterality disagreeing with GT).
# CPU-only. See scripts/pelvis_opposing_qc.py.
#
# Applies to "separate" patients: a token with a spine_only record (pseudo
# pelvis, in the dense v2 tree) AND a pelvic_native record (real GT pelvis, in
# the base v1 tree). Fused patients have a single shared scan -> nothing to check.
#
# Options (env):
#   PSEUDO_TREE   dense tree w/ pseudo pelves   (default: data/hf_export_v2)
#   GT_TREE       base tree w/ GT pelves        (default: data/hf_export)
#   OPP_CSV       output CSV                    (default: data/pelvis_opposing_qc.csv)
#   OPP_TOL_PCT   flag threshold (%%)           (default: 15)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

PSEUDO_TREE="${PSEUDO_TREE:-${DATA_DIR}/hf_export_v2}"
GT_TREE="${GT_TREE:-${DATA_DIR}/hf_export}"
OPP_CSV="${OPP_CSV:-${DATA_DIR}/pelvis_opposing_qc.csv}"
OPP_TOL_PCT="${OPP_TOL_PCT:-15}"

mkdir -p "${LOGS_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -f "${PSEUDO_TREE}/manifest.json" ]] || { echo "ERROR: no manifest.json in ${PSEUDO_TREE}"; exit 1; }
[[ -f "${GT_TREE}/manifest.json" ]]     || { echo "ERROR: no manifest.json in ${GT_TREE}"; exit 1; }

echo "======================================================================"
echo " pelvis_opposing_qc (pseudo pelvis vs same-patient opposing-position GT)"
echo "   Job ID      : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   pseudo tree : ${PSEUDO_TREE}"
echo "   gt tree     : ${GT_TREE}"
echo "   out csv     : ${OPP_CSV}"
echo "   tol %%       : ${OPP_TOL_PCT}"
echo "   Started     : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

stdbuf -oL -eL singularity exec \
    --env "PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1" \
    --bind "${BINDS}" --pwd /workspace \
    "${SIF_PATH}" \
    python3 -u /workspace/scripts/pelvis_opposing_qc.py \
        --pseudo_tree "/data/$(realpath --relative-to="${DATA_DIR}" "${PSEUDO_TREE}")" \
        --gt_tree     "/data/$(realpath --relative-to="${DATA_DIR}" "${GT_TREE}")" \
        --out_csv     "/data/$(basename "${OPP_CSV}")" \
        --tol_pct     "${OPP_TOL_PCT}"

echo ""
echo "======================================================================"
echo " pelvis_opposing_qc done at $(date)"
echo "   csv: ${OPP_CSV}"
echo "======================================================================"
