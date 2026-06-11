#!/usr/bin/env bash
#SBATCH --job-name=refresh_lstv_v3
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/refresh_lstv_v3_%j.out
#SBATCH --error=logs/refresh_lstv_v3_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# refresh_lstv_v3.sh — recompute has_l6 / n_lumbar_labels from the ACTUAL v3
# label voxels and (optionally) rewrite the manifest + re-split. Runs the scan
# inside the container (nibabel lives there, not in the nextflow env).
#
# REPORT-FIRST: a DRY RUN by default — it only prints which tokens flip
# has_l6 False->True (the L6 cases that first appeared in corrected/pseudo
# labels). Nothing is modified unless WRITE=1.
#
# Why this is needed: reduce_to_v3.py swaps in corrected labels and
# refresh_hf_manifests.py updates lstv_class, but NEITHER recomputes has_l6 /
# n_lumbar_labels from the new label voxels. So an L6 that first appears in a
# corrected label leaves the manifest stale (has_l6=False), and a later
# generate_5fold_splits run mislabels those cases as `normal`. See
# scripts/refresh_lstv_from_labels.py.
#
# Usage:
#   sbatch slurm/refresh_lstv_v3.sh                          # dry-run audit only
#   FINALS=reviews/finalized_index.json sbatch slurm/refresh_lstv_v3.sh   # + attribution
#   WRITE=1 sbatch slurm/refresh_lstv_v3.sh                  # apply the fix
#   WRITE=1 RESPLIT=1 sbatch slurm/refresh_lstv_v3.sh        # apply + re-split
#
# Env:
#   HF_DIR   tree to scan (default ${DATA_DIR}/hf_export_v3)
#   FINALS   finalized-reviews index, path RELATIVE TO THE REPO ROOT (optional;
#            adds "corrected vs pseudo" attribution to each flip)
#   WRITE    1 = rewrite manifest + splits (default 0 = dry run)
#   RESPLIT  1 = also re-run generate_5fold_splits (only with WRITE=1)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_DIR="${HF_DIR:-${DATA_DIR}/hf_export_v3}"
FINALS="${FINALS:-}"
WRITE="${WRITE:-0}"
RESPLIT="${RESPLIT:-0}"
# Spine-authoritative on by default for v3: trust the spine-bearing record's v3
# label for has_l6 (reviewer corrections included), neutralise pelvic_native
# pseudolabels. KEEP_PELVIC = confirmed pelvic-only L6 tokens to include;
# EXCLUDE_TOKENS = tokens to leave untouched (e.g. a known bad label).
SPINE_AUTH="${SPINE_AUTH:-1}"
KEEP_PELVIC="${KEEP_PELVIC:-}"
EXCLUDE_TOKENS="${EXCLUDE_TOKENS:-}"

mkdir -p "${LOGS_DIR:-logs}"

# ── Preflight ────────────────────────────────────────────────────────────────
[[ ! -f "${SIF_PATH}" ]] && { echo "ERROR: container missing: ${SIF_PATH}" >&2; exit 1; }
if [[ ! -f "${HF_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_DIR}" >&2
    echo "       Is the v3 tree built? (run scripts/review/reduce_to_v3.py first,)" >&2
    echo "       or override the tree with HF_DIR=data/hf_export_vX." >&2
    exit 1
fi
if [[ -n "${FINALS}" && ! -f "${PROJECT_ROOT}/${FINALS}" ]]; then
    echo "ERROR: FINALS=${FINALS} not found under repo root ${PROJECT_ROOT}" >&2
    exit 1
fi

# ── Singularity runtime ──────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_refresh_${SLURM_JOB_ID:-$$}"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

# Tight bind so the ONLY writable data path is the v3 tree itself: the repo is
# mounted read-only (just for the scripts) and only ${HF_DIR} is mounted rw at
# /hf. The v2 tree and the canonical hf_export are not mounted at all, so this
# job physically cannot modify anything outside ${HF_DIR}. --no-home drops the
# home auto-mount for the same reason.
BINDS="${PROJECT_ROOT}:/workspace:ro,${HF_DIR}:/hf"
PPATH="/workspace/scripts:/workspace"
C_HF="/hf"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

_run() {
    singularity exec --no-home --env "PYTHONPATH=${PPATH}" --bind "${BINDS}" \
        --pwd /workspace "${SIF_PATH}" "$@"
}

echo "======================================================================"
echo " refresh_lstv_v3"
echo "   tree     : ${HF_DIR}  ->  ${C_HF}"
echo "   finals   : ${FINALS:-<none>}"
echo "   WRITE    : ${WRITE}     RESPLIT : ${RESPLIT}"
echo "   container: ${SIF_PATH}"
echo "   started  : $(date)"
echo "======================================================================"

ARGS=( --hf_dir "${C_HF}" --workers "${WORKERS}" )
[[ -n "${FINALS}" ]] && ARGS+=( --finals "/workspace/${FINALS}" )
[[ "${WRITE}" == "1" ]] && ARGS+=( --write )
[[ "${SPINE_AUTH}" == "1" ]] && ARGS+=( --spine_authoritative )
[[ -n "${KEEP_PELVIC}" ]] && ARGS+=( --keep_pelvic "${KEEP_PELVIC}" )
[[ -n "${EXCLUDE_TOKENS}" ]] && ARGS+=( --exclude_tokens "${EXCLUDE_TOKENS}" )

echo "   workers  : ${WORKERS}   spine_auth: ${SPINE_AUTH}   keep_pelvic: ${KEEP_PELVIC:-<none>}   exclude: ${EXCLUDE_TOKENS:-<none>}"
_run python -u scripts/refresh_lstv_from_labels.py "${ARGS[@]}"

if [[ "${WRITE}" == "1" && "${RESPLIT}" == "1" ]]; then
    echo ""
    echo "----- re-splitting ${C_HF} (generate_5fold_splits) -----"
    _run python -u scripts/generate_5fold_splits.py \
        --hf_dir "${C_HF}" --out "${C_HF}/splits_5fold.json" \
        --n_folds 5 --seed 42
elif [[ "${WRITE}" == "1" ]]; then
    echo ""
    echo "NOTE: manifest updated. Re-split when ready:"
    echo "  WRITE=1 RESPLIT=1 sbatch slurm/refresh_lstv_v3.sh"
fi

echo ""
echo " Completed at $(date)"
echo "======================================================================"
