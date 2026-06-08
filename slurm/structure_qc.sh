#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_structure_qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/structure_qc_%j.out
#SBATCH --error=logs/structure_qc_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# structure_qc — presence / duplication / vertebra-gap / LEFT-RIGHT hip swap on
# BOTH the radiologist tree and the pseudolabel tree, then a side-by-side
# summary (CPU, label-only). See scripts/structure_qc.py.
#
# Options (env):
#   HF_EXPORT_DIR    radiologist/manual tree (default: data/hf_export)
#   PSEUDO_OUT_DIR   pseudolabel tree        (default: data/hf_export_v2)
#   STRUCT_MANUAL_CSV / STRUCT_PSEUDO_CSV  (defaults: data/struct_{manual,pseudo}.csv)
#   STRUCT_FLIP_LR   1 = invert L-R convention (if manual lr_swap ~100%)
#   QC_LIMIT         cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
STRUCT_MANUAL_CSV="${STRUCT_MANUAL_CSV:-${DATA_DIR}/struct_manual.csv}"
STRUCT_PSEUDO_CSV="${STRUCT_PSEUDO_CSV:-${DATA_DIR}/struct_pseudo.csv}"
STRUCT_FLIP_LR="${STRUCT_FLIP_LR:-0}"
STRUCT_DUP_RATIO="${STRUCT_DUP_RATIO:-0.2}"   # 2nd-component size ratio for dup flag
QC_WORKERS="${QC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
QC_LIMIT="${QC_LIMIT:-0}"
# The manual/radiologist tree never changes between QC runs, so on a re-run
# (e.g. QC-ing a corrected v3 tree) recomputing it is pure waste. Set
# QC_SKIP_MANUAL=1 to reuse an existing STRUCT_MANUAL_CSV and only score the
# pseudo tree against it — roughly halves the job.
QC_SKIP_MANUAL="${QC_SKIP_MANUAL:-0}"

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
echo " structure_qc — presence/duplication/gap/L-R swap (manual vs pseudo)"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   manual    : ${HF_EXPORT_DIR}"
echo "   pseudo    : ${PSEUDO_OUT_DIR}"
echo "   flip_lr   : ${STRUCT_FLIP_LR}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"

EXTRA=""
[[ "${QC_LIMIT}" != "0" ]]      && EXTRA="${EXTRA} --limit ${QC_LIMIT}"
[[ "${STRUCT_FLIP_LR}" == "1" ]] && EXTRA="${EXTRA} --flip_lr"

if [[ "${QC_SKIP_MANUAL}" == "1" && -f "${STRUCT_MANUAL_CSV}" ]]; then
    echo ""; echo "============= manual tree (SKIPPED — reusing ${STRUCT_MANUAL_CSV}) ============="; echo ""
else
    [[ "${QC_SKIP_MANUAL}" == "1" ]] && echo "QC_SKIP_MANUAL=1 but ${STRUCT_MANUAL_CSV} missing — computing it."
    echo ""; echo "============= manual tree ============="; echo ""
    stdbuf -oL -eL singularity exec \
        --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
        python3 -u /workspace/scripts/structure_qc.py \
            --tree    "/data/$(basename "${HF_EXPORT_DIR}")" \
            --out     "/data/$(basename "${STRUCT_MANUAL_CSV}")" \
            --dup_ratio "${STRUCT_DUP_RATIO}" \
            --workers "${QC_WORKERS}" ${EXTRA}
fi

echo ""; echo "============= pseudo tree (with compare) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/structure_qc.py \
        --tree    "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out     "/data/$(basename "${STRUCT_PSEUDO_CSV}")" \
        --compare "/data/$(basename "${STRUCT_MANUAL_CSV}")" \
        --dup_ratio "${STRUCT_DUP_RATIO}" \
        --workers "${QC_WORKERS}" ${EXTRA}

echo ""
echo "======================================================================"
echo " structure_qc done at $(date)"
echo "   manual CSV : ${STRUCT_MANUAL_CSV}"
echo "   pseudo CSV : ${STRUCT_PSEUDO_CSV}   (sorted struct_flag-first)"
echo "   compare summary is printed above (manual vs pseudo)"
echo "======================================================================"
