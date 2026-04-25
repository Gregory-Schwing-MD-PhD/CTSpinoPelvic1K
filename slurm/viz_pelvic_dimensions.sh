#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pelvic_viz
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pelvic_viz_%j.out
#SBATCH --error=logs/pelvic_viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Pelvic principal-axis figure (sacrum + hips) for the methods section.
#
# Produces two figures, both at 300 DPI:
#
#   1. Single-case figure: three orthogonal CT views through the
#      sacrum centroid, with sacrum + L hip + R hip masks shaded
#      and PCA axes drawn through each centroid. Optional dashed
#      patch boxes overlay candidate patch sizes.
#
#   2. Cohort figure: 4-panel histogram across all cases:
#        - sacrum SI extent
#        - sacrum AP extent
#        - sacrum ML extent
#        - inter-hip ML extent (pelvic-ring width — the binding
#                                in-plane patch constraint)
#      Each panel shows median/p90/p99 + per-patch coverage %.
#
# Cohort scan caches results to JSON next to the output PNG. Subsequent
# runs (e.g. with different --plans to overlay) reuse the cache and just
# redraw the figure.
#
# CPU only.
#
# USAGE
# -----
#   # Single representative case + cohort with patch overlays
#   TOKEN=77 \
#     PLANS_GLOB="nnunet/preprocessed/Dataset802_*/nnUNetResEncUNetPlans_{60,100}G.json" \
#     sbatch slurm/viz_pelvic_dimensions.sh
#
#   # Cohort only
#   COHORT_ONLY=1 \
#     PLANS_GLOB="nnunet/preprocessed/Dataset802_*/nnUNetResEncUNetPlans_{60,100}G.json" \
#     sbatch slurm/viz_pelvic_dimensions.sh
#
#   # Custom output stem
#   TOKEN=77 OUT_STEM=figures/methods_pelvic \
#     sbatch slurm/viz_pelvic_dimensions.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/hf_export}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"

TOKEN="${TOKEN:-}"
CONFIG="${CONFIG:-fused}"
PLANS_GLOB="${PLANS_GLOB:-}"
COHORT_ONLY="${COHORT_ONLY:-0}"
COHORT_LIMIT="${COHORT_LIMIT:-}"

OUT_STEM="${OUT_STEM:-${PROJECT_ROOT}/figures/pelvic_dimensions}"

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

mkdir -p logs "$(dirname "${OUT_STEM}")"

# ── Preflight ───────────────────────────────────────────────────────────────
[[ ! -f "${SIF_PATH}"            ]] && { echo "ERROR: SIF not found: ${SIF_PATH}" >&2; exit 1; }
[[ ! -d "${DATASET_DIR}/labels"  ]] && { echo "ERROR: ${DATASET_DIR}/labels missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/scripts/viz_pelvic_dimensions.py" ]] && {
    echo "ERROR: scripts/viz_pelvic_dimensions.py missing" >&2; exit 1; }

if [[ "${COHORT_ONLY}" != "1" && -z "${TOKEN}" ]]; then
    echo "ERROR: provide TOKEN=<n> for the single-case figure, or set COHORT_ONLY=1." >&2
    exit 1
fi

# ── Resolve PLANS_GLOB to absolute paths (host-side) ────────────────────────
PLAN_PATHS=()
if [[ -n "${PLANS_GLOB}" ]]; then
    eval "for f in ${PLANS_GLOB}; do
        if [[ -f \"\$f\" ]]; then
            PLAN_PATHS+=(\"\$f\")
        fi
    done"
    if [[ ${#PLAN_PATHS[@]} -eq 0 ]]; then
        echo "WARN: PLANS_GLOB matched no files: ${PLANS_GLOB}" >&2
        echo "      Continuing without patch-box overlays." >&2
    else
        echo "  resolved ${#PLAN_PATHS[@]} plans file(s):"
        for p in "${PLAN_PATHS[@]}"; do echo "    ${p}"; done
    fi
fi

# ── Build viz args ──────────────────────────────────────────────────────────
VIZ_ARGS=( --dataset_dir /dataset --out /out_stem )
if [[ "${COHORT_ONLY}" != "1" ]]; then
    VIZ_ARGS+=( --token "${TOKEN}" --config "${CONFIG}" )
fi
VIZ_ARGS+=( --cohort )
[[ -n "${COHORT_LIMIT}" ]] && VIZ_ARGS+=( --cohort_limit "${COHORT_LIMIT}" )

# Per-plans-file binds so container args don't leak host paths
PLANS_BINDS=""
PLAN_INDEX=0
for p in "${PLAN_PATHS[@]:-}"; do
    [[ -z "${p}" ]] && continue
    bn="$(basename "${p}")"
    PLANS_BINDS+=",${p}:/plans/${bn}:ro"
    VIZ_ARGS+=( --plans "/plans/${bn}" )
    PLAN_INDEX=$((PLAN_INDEX + 1))
done

OUT_DIR_HOST="$(dirname "${OUT_STEM}")"
OUT_BASE="$(basename "${OUT_STEM}")"
mkdir -p "${OUT_DIR_HOST}"

BINDS="${PROJECT_ROOT}:/workspace"
BINDS+=",${DATASET_DIR}:/dataset"
BINDS+=",${OUT_DIR_HOST}:/out_dir"
[[ -n "${PLANS_BINDS}" ]] && BINDS+="${PLANS_BINDS}"

# Replace --out /out_stem placeholder with the actual in-container stem
VIZ_ARGS_FIXED=()
for arg in "${VIZ_ARGS[@]}"; do
    if [[ "${arg}" == "/out_stem" ]]; then
        VIZ_ARGS_FIXED+=( "/out_dir/${OUT_BASE}" )
    else
        VIZ_ARGS_FIXED+=( "${arg}" )
    fi
done

PPATH="/workspace/scripts:/workspace"
CONTAINER_ENV="PYTHONPATH=${PPATH}"

echo "======================================================================"
echo " viz_pelvic_dimensions  (CPU)"
echo " Job        : ${SLURM_JOB_ID:-local}"
echo " Node       : $(hostname)"
echo " Dataset    : ${DATASET_DIR}"
echo " Out stem   : ${OUT_STEM}"
echo " Single case: $([[ "${COHORT_ONLY}" == "1" ]] && echo 'skipped' || echo "${TOKEN}/${CONFIG}")"
echo " Cohort     : enabled"
[[ -n "${COHORT_LIMIT}" ]] && echo " Cohort lim : ${COHORT_LIMIT}"
echo " Plans      : ${PLAN_INDEX} file(s)"
echo " Started    : $(date)"
echo "======================================================================"

singularity exec \
    --env "${CONTAINER_ENV}" \
    --bind "${BINDS}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    python scripts/viz_pelvic_dimensions.py "${VIZ_ARGS_FIXED[@]}"

echo ""
echo "======================================================================"
echo " viz COMPLETE  $(date)"
echo ""
echo " Output PNGs:"
ls -la "${OUT_DIR_HOST}/${OUT_BASE}"*.png 2>/dev/null || true
echo ""
echo " Pull to laptop:"
echo "   scp ${USER}@warrior:${OUT_DIR_HOST}/${OUT_BASE}*.png ."
echo "======================================================================"
