#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_ts_viz
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_ts_%j.out
#SBATCH --error=logs/viz_ts_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Visualize TotalSegmentator vs ground truth for one or more cases.
# =============================================================================
# Why this exists
# ---------------
# benchmark_totalseg.py gives you per-class Dice numbers. When a class
# scores unexpectedly low (e.g. sacrum at 0.80 on a normal case while
# adjacent vertebrae score 0.95), you need eyes on the actual masks to
# tell whether:
#
#   - The model is missing the structure (real failure), OR
#   - The model finds the structure but disagrees with the GT on
#     boundary placement (annotation convention drift, not a model
#     defect)
#
# This wraps scripts/viz_ts_case.py, which produces per-case PNGs:
#   - axial_mosaic.png    : 12 axial slices through the case, three
#                           rows: CT alone, CT+GT overlay, CT+TS overlay
#   - orthogonal.png      : axial + coronal + sagittal through L5, three
#                           rows: CT, CT+GT, CT+TS
#   - ct.nii.gz, gt_unified.nii.gz, pred_unified.nii.gz, pred_diff.nii.gz
#                         : NIfTIs for opening in Slicer/ITK-SNAP if you
#                           want to scrub through interactively
#
# CPU only — no GPU needed. The TS prediction was already produced by
# benchmark_totalseg.sh and cached in pred_dir; this only resamples,
# remaps, and renders.
#
# Prereqs
# -------
#   1. benchmark_totalseg.sh has been run. Predictions sit in
#      <bench_out>/ts_preds/<token>_<config>/segmentation.nii.gz
#   2. Dataset (CT + labels) is at DATASET_DIR.
#
# USAGE
# -----
#   # Single case (most common)
#   TOKEN=77 CONFIG=fused BENCH_DIR=results/totalseg_bench_35761234 \
#       sbatch slurm/viz_ts_case.sh
#
#   # Multiple cases (parallel arrays of tokens + configs)
#   TOKENS="77,189,640" CONFIGS="fused,fused,fused" \
#       BENCH_DIR=results/totalseg_bench_35761234 \
#       sbatch slurm/viz_ts_case.sh
#
#   # Mixed configs
#   TOKENS="77,189,640" CONFIGS="fused,spine_only,pelvic_native" \
#       BENCH_DIR=results/totalseg_bench_35761234 \
#       sbatch slurm/viz_ts_case.sh
#
# Output goes to <BENCH_DIR>/viz/<token>_<config>/ by default.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"

# BENCH_DIR is the output dir from benchmark_totalseg.sh; it must contain a
# ts_preds/ subdirectory with cached predictions. If you can't remember which
# job ID it was, ls results/totalseg_bench_* | sort | tail -n 1.
BENCH_DIR="${BENCH_DIR:-}"
if [[ -z "${BENCH_DIR}" ]]; then
    echo "ERROR: BENCH_DIR not set. Find your benchmark dir with:" >&2
    echo "       ls -d ${PROJECT_ROOT}/results/totalseg_bench_*" >&2
    exit 1
fi
PRED_DIR="${PRED_DIR:-${BENCH_DIR}/ts_preds}"
OUT_DIR="${OUT_DIR:-${BENCH_DIR}/viz}"

# Case selection: either single (TOKEN+CONFIG) or batch (TOKENS+CONFIGS)
TOKEN="${TOKEN:-}"
CONFIG="${CONFIG:-}"
TOKENS="${TOKENS:-}"
CONFIGS="${CONFIGS:-}"

# ── Singularity runtime dirs ─────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

mkdir -p logs "${OUT_DIR}"

# ── Preflight ───────────────────────────────────────────────────────────────
[[ ! -f "${SIF_PATH}"        ]] && { echo "ERROR: SIF not found: ${SIF_PATH}" >&2; exit 1; }
[[ ! -d "${DATASET_DIR}/ct"  ]] && { echo "ERROR: ${DATASET_DIR}/ct missing" >&2; exit 1; }
[[ ! -d "${PRED_DIR}"        ]] && {
    echo "ERROR: TS predictions dir missing: ${PRED_DIR}" >&2
    echo "       Did benchmark_totalseg.sh finish? Check BENCH_DIR." >&2
    exit 1
}

# Build viz_ts_case.py args. Single-case mode requires TOKEN+CONFIG;
# batch mode requires TOKENS+CONFIGS (comma-separated, equal length).
VIZ_ARGS=(
    --dataset_dir /dataset
    --pred_dir    /pred
    --out_dir     /viz_out
)
if [[ -n "${TOKENS}" && -n "${CONFIGS}" ]]; then
    VIZ_ARGS+=( --tokens "${TOKENS}" --configs "${CONFIGS}" )
    SUMMARY="${TOKENS} :: ${CONFIGS}"
elif [[ -n "${TOKEN}" && -n "${CONFIG}" ]]; then
    VIZ_ARGS+=( --token "${TOKEN}" --config "${CONFIG}" )
    SUMMARY="${TOKEN}/${CONFIG}"
else
    echo "ERROR: provide either (TOKEN + CONFIG) or (TOKENS + CONFIGS)." >&2
    echo "  TOKEN=77 CONFIG=fused sbatch slurm/viz_ts_case.sh" >&2
    echo "  TOKENS=77,189 CONFIGS=fused,fused sbatch slurm/viz_ts_case.sh" >&2
    exit 1
fi

# Bind paths under stable mount points so the container args don't leak
# host paths. /pred and /viz_out are otherwise unused inside the container.
BINDS="${PROJECT_ROOT}:/workspace"
BINDS+=",${DATASET_DIR}:/dataset"
BINDS+=",${PRED_DIR}:/pred"
BINDS+=",${OUT_DIR}:/viz_out"

PPATH="/workspace/scripts:/workspace"
CONTAINER_ENV="PYTHONPATH=${PPATH}"

echo "======================================================================"
echo " viz_ts_case  (CPU)"
echo " Job        : ${SLURM_JOB_ID:-local}"
echo " Node       : $(hostname)"
echo " Dataset    : ${DATASET_DIR}"
echo " Predictions: ${PRED_DIR}"
echo " Output     : ${OUT_DIR}"
echo " Cases      : ${SUMMARY}"
echo " Started    : $(date)"
echo "======================================================================"

# No --nv (no GPU); no --writable-tmpfs (no TS license-file write).
singularity exec \
    --env "${CONTAINER_ENV}" \
    --bind "${BINDS}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    python scripts/viz_ts_case.py "${VIZ_ARGS[@]}"

echo ""
echo "======================================================================"
echo " viz COMPLETE  $(date)"
echo ""
echo " Output PNGs (open with scp + image viewer):"
find "${OUT_DIR}" -maxdepth 2 -name '*.png' | sort
echo ""
echo " To pull a single case to your laptop:"
echo "   scp -r ${USER}@warrior:${OUT_DIR}/<token>_<config>/ ."
echo "======================================================================"
