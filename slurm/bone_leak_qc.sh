#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_bone_leak_qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bone_leak_qc_%j.out
#SBATCH --error=logs/bone_leak_qc_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# bone_leak_qc — GT-free off-bone label-leak metrics on BOTH the radiologist
# tree and the pseudolabel tree, then a side-by-side summary (CPU). Loads each
# CT (big), so this is heavier than vertebra-qc. See scripts/bone_leak_qc.py.
#
# Options (env):
#   HF_EXPORT_DIR   radiologist/manual tree (default: data/hf_export)
#   PSEUDO_OUT_DIR  pseudolabel tree        (default: data/hf_export_v2)
#   LEAK_MANUAL_CSV out CSV for manual      (default: data/leak_manual.csv)
#   LEAK_PSEUDO_CSV out CSV for pseudo      (default: data/leak_pseudo.csv)
#   LEAK_BONE_HU    bone HU threshold       (default: 150)
#   QC_LIMIT        cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
LEAK_MANUAL_CSV="${LEAK_MANUAL_CSV:-${DATA_DIR}/leak_manual.csv}"
LEAK_PSEUDO_CSV="${LEAK_PSEUDO_CSV:-${DATA_DIR}/leak_pseudo.csv}"
LEAK_BONE_HU="${LEAK_BONE_HU:-150}"
QC_WORKERS="${QC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
QC_LIMIT="${QC_LIMIT:-0}"

mkdir -p "${LOGS_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
for d in "${HF_EXPORT_DIR}" "${PSEUDO_OUT_DIR}"; do
    if [[ ! -f "${d}/manifest.json" ]]; then
        echo "ERROR: no manifest.json in ${d}"; exit 1
    fi
done

echo "======================================================================"
echo " bone_leak_qc — off-bone label leak (manual vs pseudo)"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   manual    : ${HF_EXPORT_DIR}"
echo "   pseudo    : ${PSEUDO_OUT_DIR}"
echo "   bone_hu   : ${LEAK_BONE_HU}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

EXTRA=""
[[ "${QC_LIMIT}" != "0" ]] && EXTRA="--limit ${QC_LIMIT}"

echo ""; echo "============= manual tree ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/bone_leak_qc.py \
        --tree    "/data/$(basename "${HF_EXPORT_DIR}")" \
        --out     "/data/$(basename "${LEAK_MANUAL_CSV}")" \
        --bone_hu "${LEAK_BONE_HU}" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""; echo "============= pseudo tree (with compare) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/bone_leak_qc.py \
        --tree    "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out     "/data/$(basename "${LEAK_PSEUDO_CSV}")" \
        --compare "/data/$(basename "${LEAK_MANUAL_CSV}")" \
        --bone_hu "${LEAK_BONE_HU}" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""
echo "======================================================================"
echo " bone_leak_qc done at $(date)"
echo "   manual CSV : ${LEAK_MANUAL_CSV}"
echo "   pseudo CSV : ${LEAK_PSEUDO_CSV}   (sorted leak_flag-first)"
echo "   compare summary is printed above (manual vs pseudo)"
echo "======================================================================"
