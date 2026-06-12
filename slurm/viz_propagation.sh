#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_viz_propagation
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_propagation_%j.out
#SBATCH --error=logs/viz_propagation_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# viz_propagation — overlay each propagated pelvis on its SPINE CT for eyeballing
# (sacrum + ilia on target bone?), titled with accept / bone-HU drop / overlap.
# One PNG per case. See scripts/viz_propagation.py.
#
# Options (env):
#   PROP_OUT_DIR  propagate_pelvis output dir  (default: data/placed/pelvic_propagated)
#   NIFTI_DIR     TCIA NIfTIs                  (default: data/tcia_nifti)
#   VIZ_OUT_DIR   PNG output dir               (default: $PROP_OUT_DIR/qc)
#   VIZ_TOKENS    comma-separated subset       (default: all)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

PROP_OUT_DIR="${PROP_OUT_DIR:-${DATA_DIR}/placed/pelvic_propagated}"
NIFTI_DIR="${NIFTI_DIR:-${DATA_DIR}/tcia_nifti}"
VIZ_OUT_DIR="${VIZ_OUT_DIR:-${PROP_OUT_DIR}/qc}"
VIZ_TOKENS="${VIZ_TOKENS:-}"

mkdir -p "${LOGS_DIR}" "${VIZ_OUT_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: container missing.  Run: make build-container"; exit 1; }
[[ -f "${PROP_OUT_DIR}/propagate_qc.csv" ]] || { echo "ERROR: no propagate_qc.csv in ${PROP_OUT_DIR} (run propagate-pelvis first)"; exit 1; }

ARGS=( --qc_csv "/data/$(realpath --relative-to="${DATA_DIR}" "${PROP_OUT_DIR}")/propagate_qc.csv"
       --nifti_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${NIFTI_DIR}")"
       --out_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${VIZ_OUT_DIR}")"
       --propagated_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${PROP_OUT_DIR}")" )
[[ -n "${VIZ_TOKENS}" ]] && ARGS+=( --tokens "${VIZ_TOKENS}" )

echo "======================================================================"
echo " viz_propagation — propagated pelvis overlaid on the spine CT"
echo "   Job ID  : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   prop dir: ${PROP_OUT_DIR}"
echo "   out dir : ${VIZ_OUT_DIR}"
echo "   tokens  : ${VIZ_TOKENS:-all}"
echo "   Started : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,MPLBACKEND=Agg"

stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/viz_propagation.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " viz_propagation done at $(date)   PNGs: ${VIZ_OUT_DIR}"
echo "======================================================================"
