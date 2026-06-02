#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_vertebra_qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vertebra_qc_%j.out
#SBATCH --error=logs/vertebra_qc_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# vertebra_qc — GT-free neighbour-mixing metrics on BOTH the radiologist tree
# and the pseudolabel tree, then a side-by-side summary (CPU). See
# scripts/vertebra_topology_qc.py.
#
# Options (env):
#   HF_EXPORT_DIR   radiologist/manual tree (default: data/hf_export)
#   PSEUDO_OUT_DIR  pseudolabel tree        (default: data/hf_export_v2)
#   QC_MANUAL_CSV   out CSV for manual      (default: data/qc_manual.csv)
#   QC_PSEUDO_CSV   out CSV for pseudo      (default: data/qc_pseudo.csv)
#   QC_LIMIT        cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
QC_MANUAL_CSV="${QC_MANUAL_CSV:-${DATA_DIR}/qc_manual.csv}"
QC_PSEUDO_CSV="${QC_PSEUDO_CSV:-${DATA_DIR}/qc_pseudo.csv}"
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
echo " vertebra_qc — neighbour-mixing metrics (manual vs pseudo)"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   manual    : ${HF_EXPORT_DIR}"
echo "   pseudo    : ${PSEUDO_OUT_DIR}"
echo "   out manual: ${QC_MANUAL_CSV}"
echo "   out pseudo: ${QC_PSEUDO_CSV}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

EXTRA=""
[[ "${QC_LIMIT}" != "0" ]] && EXTRA="--limit ${QC_LIMIT}"

echo ""; echo "============= manual tree ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/vertebra_topology_qc.py \
        --tree    "/data/$(basename "${HF_EXPORT_DIR}")" \
        --out     "/data/$(basename "${QC_MANUAL_CSV}")" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""; echo "============= pseudo tree (with compare) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/vertebra_topology_qc.py \
        --tree    "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out     "/data/$(basename "${QC_PSEUDO_CSV}")" \
        --compare "/data/$(basename "${QC_MANUAL_CSV}")" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""
echo "======================================================================"
echo " vertebra_qc done at $(date)"
echo "   manual CSV : ${QC_MANUAL_CSV}"
echo "   pseudo CSV : ${QC_PSEUDO_CSV}   (sorted mixing_flag-first)"
echo "   compare summary is printed above (manual vs pseudo)"
echo "======================================================================"
