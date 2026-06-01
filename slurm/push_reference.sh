#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_push_reference
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/push_reference_%j.out
#SBATCH --error=logs/push_reference_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# push_reference — crop ONE clean, COMPLETE example case and upload it to
# crops/reference/ in the v2 dataset, so the review tool opens it as the
# side-by-side good example for every case. Defaults to a clean SCOPED case
# (config != fused, needs_review=0) from the v2 tree — small crop, like the
# worklist — rather than a fused gold case (whose full diagnostic CT crops big).
# See scripts/export_review_crops.py
#
# Options (env):
#   HF_TOKEN       write token (REQUIRED)
#   HF_REPO_ID     dataset repo, e.g. gregoryschwingmdphd/CTSpinoPelvic1K (REQUIRED)
#   HF_REVISION    branch                              (default: v2)
#   REF_TOKEN      example as 'token:config' (default: auto-pick a clean scoped)
#   PSEUDO_OUT_DIR tree to crop from                   (default: data/hf_export_v2)
#   REF_QC_CSV     calibrated master for auto-pick     (default: data/qc_master.csv)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REVISION="${HF_REVISION:-v2}"
REF_TOKEN="${REF_TOKEN:-}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REF_QC_CSV="${REF_QC_CSV:-${DATA_DIR}/qc_master.csv}"
REF_DIR="${DATA_DIR}/ref_tmp"

mkdir -p "${LOGS_DIR}" "${REF_DIR}"

[[ -z "${HF_TOKEN}" ]]   && { echo "ERROR: HF_TOKEN required (write token)"; exit 1; }
[[ -z "${HF_REPO_ID}" ]] && { echo "ERROR: HF_REPO_ID required"; exit 1; }
[[ -f "${SIF_PATH}" ]]   || { echo "ERROR: container missing. make build-container"; exit 1; }
[[ -f "${PSEUDO_OUT_DIR}/manifest.json" ]] || { echo "ERROR: no manifest in ${PSEUDO_OUT_DIR}"; exit 1; }

# REF_SPEC = 'token:config' fed to export_review_crops --tokens. Auto-pick a
# clean, NON-fused (small) case if not given.
REF_SPEC="${REF_TOKEN:-}"
if [[ -z "${REF_SPEC}" ]]; then
    if [[ -f "${REF_QC_CSV}" ]]; then
        REF_SPEC=$(awk -F, 'NR>1 && $3==0 && $2!="fused" {print $1":"$2; exit}' "${REF_QC_CSV}")
    fi
    if [[ -z "${REF_SPEC}" ]]; then
        echo "ERROR: could not auto-pick a clean scoped case from ${REF_QC_CSV}."
        echo "       Pass one: REF_TOKEN=<token>:<config> make push-reference ..."; exit 1
    fi
fi
echo "======================================================================"
echo " push_reference — example -> ${HF_REPO_ID}@${HF_REVISION}:crops/reference/"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   example   : ${REF_SPEC}   (clean, complete; small crop)"
echo "   tree      : ${PSEUDO_OUT_DIR}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,HF_TOKEN=${HF_TOKEN}"

# 1) crop the example case
rm -rf "${REF_DIR:?}"/* 2>/dev/null || true
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/export_review_crops.py \
        --tokens "${REF_SPEC}" \
        --tree   "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --out    "/data/ref_tmp"

# 2) upload the single crop dir to crops/reference/
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u -c "
import glob, os
from huggingface_hub import upload_folder
dirs = sorted(glob.glob('/data/ref_tmp/*/'))
if not dirs:
    raise SystemExit('no crop produced for ${REF_SPEC} (not in ${PSEUDO_OUT_DIR}?)')
d = dirs[0]
upload_folder(folder_path=d, path_in_repo='crops/reference',
              repo_id='${HF_REPO_ID}', repo_type='dataset', revision='${HF_REVISION}',
              token=os.environ['HF_TOKEN'], commit_message='gold reference example')
print('uploaded', d, '-> crops/reference/')
"

echo ""
echo "======================================================================"
echo " push_reference done at $(date)"
echo "   reviewtool next now opens this gold example in a second window."
echo "======================================================================"
