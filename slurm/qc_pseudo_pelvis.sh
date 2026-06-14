#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_qc_pseudo_pelvis
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qc_pseudo_pelvis_%j.out
#SBATCH --error=logs/qc_pseudo_pelvis_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# qc_pseudo_pelvis — GT-free anatomical triage of the model-pseudolabelled pelves.
# Writes a triage-sorted CSV INTO the v2 tree (qc/pseudo_pelvis_triage.csv) so it
# ships with the push and you review the suspect cases first. See
# scripts/qc_pseudo_pelvis.py.
#
# NON-FATAL BY DESIGN: this is a review aid, not a gate. It always exits 0 so the
# downstream v2 push (afterok) is never blocked by a QC hiccup.
#
# Options (env):
#   PSEUDO_OUT_DIR  v2 tree (ct/, labels/, manifest.json)  (default: data/hf_export_v2)
#   BONE_HU         soft bone-fit HU threshold             (default: 200)
# =============================================================================
set -uo pipefail            # NOT -e: a QC failure must not fail the job

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
BONE_HU="${BONE_HU:-200}"
OUT_CSV="${PSEUDO_OUT_DIR}/qc/pseudo_pelvis_triage.csv"

mkdir -p "${LOGS_DIR}" "${PSEUDO_OUT_DIR}/qc"
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "WARN: container missing at ${SIF_PATH} — skipping QC (non-fatal)"; exit 0
fi
if [[ ! -f "${PSEUDO_OUT_DIR}/manifest.json" ]]; then
    echo "WARN: no v2 manifest at ${PSEUDO_OUT_DIR} — skipping QC (non-fatal)"; exit 0
fi

REL() { echo "/data/$(realpath --relative-to="${DATA_DIR}" "$1")"; }

echo "======================================================================"
echo " qc_pseudo_pelvis — anatomical triage of model pelves"
echo "   Job ID : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   v2 tree: ${PSEUDO_OUT_DIR}"
echo "   out csv: ${OUT_CSV}"
echo "   Started: $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/qc_pseudo_pelvis.py \
        --v2_dir  "$(REL "${PSEUDO_OUT_DIR}")" \
        --out_csv "$(REL "${OUT_CSV}")" \
        --bone_hu "${BONE_HU}" \
    || echo "WARN: qc_pseudo_pelvis.py exited non-zero — continuing (non-fatal)"

echo ""
echo "======================================================================"
echo " qc_pseudo_pelvis done at $(date)   csv: ${OUT_CSV}"
echo "======================================================================"
exit 0