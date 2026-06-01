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
# push_reference — crop ONE clean gold (fused) case and upload it to
# crops/reference/ in the v2 dataset, so the review tool opens it as the
# side-by-side GOLD example for every case. Auto-picks a clean fused token if
# REF_TOKEN is not given. See scripts/export_review_crops.py
#
# Options (env):
#   HF_TOKEN       write token (REQUIRED)
#   HF_REPO_ID     dataset repo, e.g. gregoryschwingmdphd/CTSpinoPelvic1K (REQUIRED)
#   HF_REVISION    branch                              (default: v2)
#   REF_TOKEN      the gold case token (default: auto-pick a clean fused)
#   HF_EXPORT_DIR  manual/gold tree                    (default: data/hf_export)
#   REF_QC_CSV     radiologist master for auto-pick    (default: data/qc_master_manual.csv)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REVISION="${HF_REVISION:-v2}"
REF_TOKEN="${REF_TOKEN:-}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
REF_QC_CSV="${REF_QC_CSV:-${DATA_DIR}/qc_master_manual.csv}"
REF_DIR="${DATA_DIR}/ref_tmp"

mkdir -p "${LOGS_DIR}" "${REF_DIR}"

[[ -z "${HF_TOKEN}" ]]   && { echo "ERROR: HF_TOKEN required (write token)"; exit 1; }
[[ -z "${HF_REPO_ID}" ]] && { echo "ERROR: HF_REPO_ID required"; exit 1; }
[[ -f "${SIF_PATH}" ]]   || { echo "ERROR: container missing. make build-container"; exit 1; }
[[ -f "${HF_EXPORT_DIR}/manifest.json" ]] || { echo "ERROR: no manifest in ${HF_EXPORT_DIR}"; exit 1; }

# Auto-pick the cleanest fused case (config=fused, needs_review=0) if not named.
if [[ -z "${REF_TOKEN}" ]]; then
    if [[ -f "${REF_QC_CSV}" ]]; then
        REF_TOKEN=$(awk -F, 'NR>1 && $2=="fused" && $3==0 {print $1; exit}' "${REF_QC_CSV}")
    fi
    if [[ -z "${REF_TOKEN}" ]]; then
        echo "ERROR: could not auto-pick a clean fused token from ${REF_QC_CSV}."
        echo "       Pass one explicitly: REF_TOKEN=<token> make push-reference ..."; exit 1
    fi
fi
echo "======================================================================"
echo " push_reference — gold example -> ${HF_REPO_ID}@${HF_REVISION}:crops/reference/"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   gold token: ${REF_TOKEN}  (fused)"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
ENV_VARS="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1,HF_TOKEN=${HF_TOKEN}"

# 1) crop the gold case
rm -rf "${REF_DIR:?}"/* 2>/dev/null || true
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/export_review_crops.py \
        --tokens "${REF_TOKEN}:fused" \
        --tree   "/data/$(basename "${HF_EXPORT_DIR}")" \
        --out    "/data/ref_tmp"

# 2) upload the single crop dir to crops/reference/
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u -c "
import glob, os
from huggingface_hub import upload_folder
dirs = sorted(glob.glob('/data/ref_tmp/*/'))
if not dirs:
    raise SystemExit('no crop produced for ${REF_TOKEN} (token not in ${HF_EXPORT_DIR}?)')
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
