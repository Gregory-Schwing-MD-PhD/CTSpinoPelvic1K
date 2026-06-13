#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_propagate_pelvis
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# ~7 GB/worker: a 512^3 rigid registration peaks at ~5-6 GB (the float32 CT pair +
# SimpleITK pyramid buffers), and we run cpus-2 (=22) of them concurrently. 96G
# (=4.4 GB/worker) OOM-kills; 160G keeps all workers with headroom.
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/propagate_pelvis_%j.out
#SBATCH --error=logs/propagate_pelvis_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# propagate_pelvis — carry each "separate" patient's REAL radiologist pelvis GT
# across acquisitions onto their spine-side scan by deterministic, bone-masked
# deformable registration (SimpleITK — already in the container, no rebuild).
# Replaces the MODEL pelvis with real GT on the dominant cohort; the gate proves
# bone-HU overlap stays within --drop_target pp of the native placement (reported,
# not enforced — a real pelvis beats a model guess). CONSTRUCTION stage: run AFTER
# create_dataset, BEFORE export.
# See scripts/propagate_pelvis.py.
#
# Options (env):
#   MODE         test | production                 (default: production)
#   MANIFEST     placed_manifest.json              (default: data/placed/placed_manifest.json)
#   NIFTI_DIR    TCIA NIfTIs                        (default: data/tcia_nifti)
#   PELVIC_DIR   placed pelvic masks               (default: data/placed/pelvic)
#   SPINE_DIR    placed spine GT (L5/S1 landmark)  (default: data/placed/spine)
#   PROP_OUT_DIR propagated pelves + qc            (default: data/placed/pelvic_propagated)
#   PROP_WORKERS parallel registrations            (default: $SLURM_CPUS_PER_TASK - 2)
#   DROP_TARGET  bone-HU overlap report ref (pp)   (default: 1.0, report-only)
#   FAIL_DROP    fall back to model if bone-HU drop > this many pp (default: 8.0)
#   PER_BONE     1 = ALSO per-bone rigid refinement; 0 = whole-pelvis rigid (default)
#   RESUME       1 = skip already-accepted cases from a prior propagate_qc.csv
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
SPINE_DIR="${SPINE_DIR:-${DATA_DIR}/placed/spine}"
PROP_OUT_DIR="${PROP_OUT_DIR:-${DATA_DIR}/placed/pelvic_propagated}"
PROP_WORKERS="${PROP_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} > 2 ? ${SLURM_CPUS_PER_TASK:-8} - 2 : 1 ))}"
DROP_TARGET="${DROP_TARGET:-1.0}"
FAIL_DROP="${FAIL_DROP:-8.0}"
PER_BONE="${PER_BONE:-0}"
REG_LOG_EVERY="${REG_LOG_EVERY:-10}"
RESUME="${RESUME:-0}"
PROP_LIMIT="${PROP_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${PROP_OUT_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: no placed_manifest at ${MANIFEST} (run create_dataset first)"; exit 1; }

ARGS=( --manifest "/data/$(realpath --relative-to="${DATA_DIR}" "${MANIFEST}")"
       --nifti_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${NIFTI_DIR}")"
       --pelvic_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${PELVIC_DIR}")"
       --out_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${PROP_OUT_DIR}")"
       --mode "${MODE}" --workers "${PROP_WORKERS}"
       --reg_log_every "${REG_LOG_EVERY}"
       --drop_target "${DROP_TARGET}" )
ARGS+=( --fail_drop "${FAIL_DROP}" )
[[ -d "${SPINE_DIR}" ]] && ARGS+=( --spine_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${SPINE_DIR}")" ) || echo "NOTE: no spine dir at ${SPINE_DIR} — L5/S1 landmark disabled"
[[ "${PER_BONE}" == "1" ]] && ARGS+=( --per_bone )
[[ "${AFFINE:-0}" == "1" ]] && ARGS+=( --affine )
[[ "${RESUME}" == "1" ]] && ARGS+=( --resume )
[[ -n "${TOKENS:-}" ]] && ARGS+=( --tokens "${TOKENS}" )
[[ "${PROP_LIMIT}" != "0" ]] && ARGS+=( --limit "${PROP_LIMIT}" )

echo "======================================================================"
echo " propagate_pelvis — real-GT pelvis carried across acquisitions"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   mode      : ${MODE}    workers: ${PROP_WORKERS}"
echo "   manifest  : ${MANIFEST}"
echo "   out_dir   : ${PROP_OUT_DIR}"
echo "   bone-drop report ref (pp) : ${DROP_TARGET}   fail_drop (pp): ${FAIL_DROP}"
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
echo "   out: ${PROP_OUT_DIR}  (propagate_qc.csv + placed_manifest_propagated.json)"
echo "======================================================================"
