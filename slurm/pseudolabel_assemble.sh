#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pseudolabel_assemble
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pseudolabel_assemble_%j.out
#SBATCH --error=logs/pseudolabel_assemble_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# pseudolabel_assemble — the REDUCE pass after the sharded pseudolabel array.
#
# Runs pseudolabel.py with n_shards=1 so it iterates ALL records, RESUMES every
# shard's per-case markers (<v2>_work/done/*.json) + on-disk labels — reusing every
# shard's predictions with NO re-inference, NO GPU — and writes the final v2
# manifest.json + splits. Chained afterok the array, so every case is already done.
#
# Runs in the LEAN project container (no nnU-Net): assembly never predicts — nnU-Net
# is only imported inside run_nnunet_folder, which the all-resume pass never calls.
# It MUST get the SAME INCLUDE_CONFIGS as the array workers so the kept-record set
# (and therefore the manifest) matches exactly. ship_v2.sh wires this for you.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${nnUNet_results:-${PROJECT_ROOT}/nnunet/results}}"
mkdir -p "${LOGS_DIR}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ -f "${HF_EXPORT_DIR}/manifest.json" ]] || { echo "ERROR: no v1 base at ${HF_EXPORT_DIR}"; exit 1; }

EXTRA_ARGS=""
# Same record filter as the workers, or the assembled manifest won't match.
[[ -n "${INCLUDE_CONFIGS:-}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --include_configs ${INCLUDE_CONFIGS}"

echo "======================================================================"
echo " pseudolabel ASSEMBLE (reduce) — reuse shard predictions, write v2 manifest"
echo "   v1 source : ${HF_EXPORT_DIR}"
echo "   v2 out    : ${PSEUDO_OUT_DIR}"
echo "   include   : ${INCLUDE_CONFIGS:-<all scoped>}"
echo "   container : ${SIF_PATH}  (lean — no GPU, no inference)"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${NNUNET_RESULTS}:${NNUNET_RESULTS}"
CENV="PYTHONPATH=/workspace/scripts:/workspace/src:/workspace,PYTHONUNBUFFERED=1"

stdbuf -oL -eL singularity exec --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${SIF_PATH}" python3 -u /workspace/scripts/pseudolabel.py \
    --hf_export      "/data/$(basename "${HF_EXPORT_DIR}")" \
    --out            "/data/$(basename "${PSEUDO_OUT_DIR}")" \
    --models_config  "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
    --splits         "/data/$(basename "${HF_EXPORT_DIR}")/splits_5fold.json" \
    --nnunet_results "${NNUNET_RESULTS}" \
    --device         cpu \
    --skip_download \
    --n_shards 1 --shard_id 0 \
    ${EXTRA_ARGS}

echo ""
echo "======================================================================"
echo " pseudolabel assemble done at $(date)  ->  ${PSEUDO_OUT_DIR}/manifest.json"
echo "======================================================================"
