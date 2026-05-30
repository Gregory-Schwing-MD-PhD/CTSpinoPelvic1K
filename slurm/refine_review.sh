#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_refine_review
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/refine_review_%j.out
#SBATCH --error=logs/refine_review_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Compete-refine + change-review — ONE CPU job, two stages.
#
#   Stage 1  intensity_refine.py --mode compete  -> refined tree +
#            review_flags.json (touching/fused components left for human review)
#   Stage 2  refine_review.py    -> per changed case: a 3D change-map NIfTI
#            (1=removed 2=added 3=relabeled) to overlay on the CT in ITK-SNAP,
#            2D PNG overlays of the changed slices, summary.csv + index.html.
#
# Set RUN_REFINE=0 to skip Stage 1 and only (re)build the review from an
# existing refined tree.
#
# Options (env):
#   HF_EXPORT_DIR   v1 manual tree     (default: data/hf_export)
#   PSEUDO_OUT_DIR  v2 pseudo tree     (default: data/hf_export_v2)
#   REFINE_OUT_DIR  refined out tree   (default: data/hf_export_v2_compete)
#   REVIEW_OUT_DIR  review output dir  (default: data/refine_review)
#   REFINE_MODE     refine mode        (default: compete)
#   REFINE_PCTL/REFINE_ERODE/REFINE_GROW/REFINE_FILL  (see intensity_refine.py)
#   PURITY_TOL      compete purity tolerance     (default: 0.15)
#   MIN_BLEED_VOX   compete small-bleed cutoff   (default: 50)
#   REFINE_LIMIT    cap cases (debug, both stages)
#   REFINE_OVERWRITE 1 = re-refine existing cases
#   RUN_REFINE      1 = run Stage 1 (default); 0 = review an existing tree only
#   REVIEW_AXIS     slice axis for PNGs          (default: 2)
#   REVIEW_MAX_SLICES  max changed slices/case   (default: 12)
#   REVIEW_NO_PNGS  1 = change-maps + csv only (no PNGs)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
REFINE_OUT_DIR="${REFINE_OUT_DIR:-${DATA_DIR}/hf_export_v2_compete}"
REVIEW_OUT_DIR="${REVIEW_OUT_DIR:-${DATA_DIR}/refine_review}"
REFINE_MODE="${REFINE_MODE:-compete}"
REFINE_PCTL="${REFINE_PCTL:-10}"
REFINE_ERODE="${REFINE_ERODE:-1}"
REFINE_GROW="${REFINE_GROW:-0}"
REFINE_FILL="${REFINE_FILL:-1}"
PURITY_TOL="${PURITY_TOL:-0.15}"
MIN_BLEED_VOX="${MIN_BLEED_VOX:-50}"
REFINE_WORKERS="${REFINE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
REFINE_LIMIT="${REFINE_LIMIT:-0}"
REFINE_OVERWRITE="${REFINE_OVERWRITE:-0}"
RUN_REFINE="${RUN_REFINE:-1}"
REVIEW_AXIS="${REVIEW_AXIS:-2}"
REVIEW_MAX_SLICES="${REVIEW_MAX_SLICES:-12}"
REVIEW_NO_PNGS="${REVIEW_NO_PNGS:-0}"

mkdir -p "${LOGS_DIR}" "${REFINE_OUT_DIR}" "${REVIEW_OUT_DIR}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"; exit 1
fi
for d in "${HF_EXPORT_DIR}" "${PSEUDO_OUT_DIR}"; do
    if [[ ! -f "${d}/manifest.json" ]]; then
        echo "ERROR: no manifest.json in ${d}"; exit 1
    fi
done

echo "======================================================================"
echo " compete-refine + change-review (one job)"
echo "   Job ID      : ${SLURM_JOB_ID:-local}"
echo "   Node        : $(hostname)"
echo "   v1 manual   : ${HF_EXPORT_DIR}"
echo "   v2 pseudo   : ${PSEUDO_OUT_DIR}"
echo "   refined out : ${REFINE_OUT_DIR}"
echo "   review out  : ${REVIEW_OUT_DIR}"
echo "   mode        : ${REFINE_MODE}  pctl=${REFINE_PCTL} erode=${REFINE_ERODE} grow=${REFINE_GROW}"
echo "   compete     : purity_tol=${PURITY_TOL}  min_bleed_vox=${MIN_BLEED_VOX}"
echo "   RUN_REFINE  : ${RUN_REFINE}"
echo "   Started     : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"
ENV_VARS="PYTHONPATH=${PPATH},PYTHONUNBUFFERED=1,MPLBACKEND=Agg"

REVIEW_FLAGS="/data/$(basename "${REFINE_OUT_DIR}")/review_flags.json"

if [[ "${RUN_REFINE}" == "1" ]]; then
    EXTRA=""
    [[ "${REFINE_FILL}" == "0" ]]      && EXTRA="${EXTRA} --no_fill_holes"
    [[ "${REFINE_LIMIT}" != "0" ]]     && EXTRA="${EXTRA} --limit ${REFINE_LIMIT}"
    [[ "${REFINE_OVERWRITE}" == "1" ]] && EXTRA="${EXTRA} --overwrite"

    echo ""; echo "============= STAGE 1: intensity_refine (${REFINE_MODE}) ============="; echo ""
    stdbuf -oL -eL singularity exec \
        --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
        python3 -u /workspace/scripts/intensity_refine.py \
            --manual_from  "/data/$(basename "${HF_EXPORT_DIR}")" \
            --in           "/data/$(basename "${PSEUDO_OUT_DIR}")" \
            --out          "/data/$(basename "${REFINE_OUT_DIR}")" \
            --mode         "${REFINE_MODE}" \
            --percentile   "${REFINE_PCTL}" \
            --erode_iter   "${REFINE_ERODE}" \
            --grow_iters   "${REFINE_GROW}" \
            --purity_tol   "${PURITY_TOL}" \
            --min_bleed_vox "${MIN_BLEED_VOX}" \
            --workers      "${REFINE_WORKERS}" \
            ${EXTRA}
else
    echo "RUN_REFINE=0 — skipping Stage 1; reviewing existing ${REFINE_OUT_DIR}"
fi

REVIEW_EXTRA=""
[[ -f "${REFINE_OUT_DIR}/review_flags.json" ]] && REVIEW_EXTRA="${REVIEW_EXTRA} --flags ${REVIEW_FLAGS}"
[[ "${REFINE_LIMIT}" != "0" ]]   && REVIEW_EXTRA="${REVIEW_EXTRA} --limit ${REFINE_LIMIT}"
[[ "${REVIEW_NO_PNGS}" == "1" ]] && REVIEW_EXTRA="${REVIEW_EXTRA} --no_pngs"

echo ""; echo "============= STAGE 2: refine_review (change overlays) ============="; echo ""
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd /workspace "${SIF_PATH}" \
    python3 -u /workspace/scripts/refine_review.py \
        --before       "/data/$(basename "${PSEUDO_OUT_DIR}")" \
        --after        "/data/$(basename "${REFINE_OUT_DIR}")" \
        --manual_from  "/data/$(basename "${HF_EXPORT_DIR}")" \
        --out          "/data/$(basename "${REVIEW_OUT_DIR}")" \
        --axis         "${REVIEW_AXIS}" \
        --max_slices   "${REVIEW_MAX_SLICES}" \
        ${REVIEW_EXTRA}

echo ""
echo "======================================================================"
echo " refine_review done at $(date)"
echo "   refined tree : ${REFINE_OUT_DIR}"
echo "   review       : ${REVIEW_OUT_DIR}/index.html   (sorted relabeled-first)"
echo "   per-case     : ${REVIEW_OUT_DIR}/<token>/change_map.nii.gz  (overlay on CT)"
echo "   summary      : ${REVIEW_OUT_DIR}/summary.csv"
echo "======================================================================"
