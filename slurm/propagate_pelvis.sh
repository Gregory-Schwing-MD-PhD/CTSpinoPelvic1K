#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_propagate_pelvis
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/propagate_pelvis_%j.out
#SBATCH --error=logs/propagate_pelvis_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# propagate_pelvis — carry each "separate" patient's REAL radiologist pelvis GT
# across acquisitions onto their spine-side scan by deterministic, bone-masked
# deformable registration (SimpleITK — already in the container, no rebuild).
# Replaces the MODEL pelvis with real GT on the dominant cohort; the gate proves
# bone-HU overlap does not degrade by more than --max_bone_drop pp vs the native
# placement. CONSTRUCTION stage: run AFTER create_dataset, BEFORE export.
# See scripts/propagate_pelvis.py.
#
# Options (env):
#   MODE         test | production                 (default: production)
#   MANIFEST     placed_manifest.json              (default: data/placed/placed_manifest.json)
#   NIFTI_DIR    TCIA NIfTIs                        (default: data/tcia_nifti)
#   PELVIC_DIR   placed pelvic masks               (default: data/placed/pelvic)
#   PROP_OUT_DIR propagated pelves + qc            (default: data/placed/pelvic_propagated)
#   PROP_WORKERS parallel registrations            (default: $SLURM_CPUS_PER_TASK - 2)
#   MAX_BONE_DROP bone-HU overlap drop gate (pp)   (default: 1.0)
#   PROP_LIMIT   cap cases (debug)                 (default: 0 = all in production)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

MODE="${MODE:-production}"
MANIFEST="${MANIFEST:-${DATA_DIR}/placed/placed_manifest.json}"
NIFTI_DIR="${NIFTI_DIR:-${DATA_DIR}/tcia_nifti}"
PELVIC_DIR="${PELVIC_DIR:-${DATA_DIR}/placed/pelvic}"
PROP_OUT_DIR="${PROP_OUT_DIR:-${DATA_DIR}/placed/pelvic_propagated}"
PROP_WORKERS="${PROP_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} > 2 ? ${SLURM_CPUS_PER_TASK:-8} - 2 : 1 ))}"
MAX_BONE_DROP="${MAX_BONE_DROP:-1.0}"
PROP_LIMIT="${PROP_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${PROP_OUT_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: no placed_manifest at ${MANIFEST} (run create_dataset first)"; exit 1; }

ARGS=( --manifest "/data/$(realpath --relative-to="${DATA_DIR}" "${MANIFEST}")"
       --nifti_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${NIFTI_DIR}")"
       --pelvic_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${PELVIC_DIR}")"
       --out_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${PROP_OUT_DIR}")"
       --mode "${MODE}" --workers "${PROP_WORKERS}"
       --max_bone_drop "${MAX_BONE_DROP}" )
[[ "${PROP_LIMIT}" != "0" ]] && ARGS+=( --limit "${PROP_LIMIT}" )

echo "======================================================================"
echo " propagate_pelvis — real-GT pelvis carried across acquisitions"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   mode      : ${MODE}    workers: ${PROP_WORKERS}"
echo "   manifest  : ${MANIFEST}"
echo "   out_dir   : ${PROP_OUT_DIR}"
echo "   bone-drop gate (pp) : ${MAX_BONE_DROP}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/propagate_pelvis.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " propagate_pelvis done at $(date)"
echo "   out: ${PROP_OUT_DIR}  (propagate_qc.csv + propagate_manifest.json)"
echo "======================================================================"
