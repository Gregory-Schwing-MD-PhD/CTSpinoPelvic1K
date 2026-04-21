#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_export_dataset
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/export_dataset_%j.out
#SBATCH --error=logs/export_dataset_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 3 — export dataset + push to HuggingFace
#
# Reads placed_manifest_orientation_fixed.json from Stage 2 by default (since
# that manifest carries AP-inverted flips for tokens like 480 where the
# anatomy was wrong in the original scan). To export against the un-fixed
# manifest instead, set MANIFEST_FILE=placed_manifest.json.
#
# Produces:
#   data/hf_export/ct/          CT NIfTIs (PIR, PHI-stripped)
#   data/hf_export/labels/      10-class label NIfTIs (voxel-aligned with CT)
#   data/hf_export/qc/          QC overlays (optional)
#   data/hf_export/manifest.json
#   data/hf_export/manifest.csv
#   data/hf_export/manifest_train.json       LSTV-stratified 70/15/15 splits
#   data/hf_export/manifest_validation.json
#   data/hf_export/manifest_test.json
#   data/hf_export/data_splits.json
#   data/hf_export/dataset_interface.py      runtime Python API
#   data/hf_export/README.md                 dataset card
#
# Then optionally pushes to HuggingFace Hub via upload_large_folder.
#
# Usage:
#   sbatch slurm/export_dataset.sh                              # export only
#   HF_TOKEN=hf_xxx PUSH=1 sbatch slurm/export_dataset.sh       # export + push
#   HF_TOKEN=hf_xxx PUSH=1 SKIP_EXPORT=1 sbatch slurm/export_dataset.sh
#                                                               # push existing
#
# Options:
#   MANIFEST_FILE=placed_manifest.json   use un-fixed manifest (default: fixed)
#   SKIP_QC=1        skip QC figure generation
#   NO_PIR=1         skip PIR reorientation (native voxel space)
#   HF_PRIVATE=1     create HF repo as private
#   HF_WORKERS=8     HF upload workers (default 8)
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Manifest selection ───────────────────────────────────────────────────────
# Default to the orientation-fixed manifest. Its schema is a strict superset of
# the original placed_manifest.json (same top-level keys + additional
# orientation_check/orientation_fixed fields per case + patched series_uid
# for flipped cases). export_hf.py reads series_uid to locate CT files, so
# using this manifest automatically picks up the flipped CT NIfTIs.
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"

# Fallback: if the orientation-fixed manifest doesn't exist (e.g. Stage 2 ran
# without Step C), fall back to the original with a loud warning.
if [[ ! -f "${HOST_MANIFEST}" && "${MANIFEST_FILE}" == "placed_manifest_orientation_fixed.json" ]]; then
    echo "WARNING: ${HOST_MANIFEST} not found."
    echo "         Falling back to placed_manifest.json (no AP-inversion fix applied)."
    echo "         Run Stage 2 with Step C enabled to produce the orientation-fixed manifest."
    MANIFEST_FILE="placed_manifest.json"
    HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"
fi

mkdir -p "${LOGS_DIR}" "${HF_EXPORT_DIR}"

echo "======================================================================"
echo " Stage 3: Export dataset"
echo "   Job ID       : ${SLURM_JOB_ID:-local}"
echo "   Node         : $(hostname)"
echo "   Manifest     : ${HOST_MANIFEST}"
echo "   Export dir   : ${HF_EXPORT_DIR}"
echo "   HF repo      : ${HF_REPO_ID}"
echo "   PUSH         : ${PUSH}"
echo "   SKIP_EXPORT  : ${SKIP_EXPORT}"
echo "   SKIP_QC      : ${SKIP_QC}"
echo "   NO_PIR       : ${NO_PIR}"
echo "   HF_PRIVATE   : ${HF_PRIVATE}"
echo "   Started      : $(date)"
echo "======================================================================"

# ── Token handling (with redaction for logs) ─────────────────────────────────
if [[ "${PUSH}" == "1" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: PUSH=1 requires HF_TOKEN."
        echo "       Submit via:  HF_TOKEN=hf_xxx make export-dataset PUSH=1"
        exit 1
    fi
    echo "  HF_TOKEN : ${HF_TOKEN:0:8}***  (full token passed via env, redacted in logs)"
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"
    exit 1
fi

# ── Pre-flight ───────────────────────────────────────────────────────────────
if [[ "${SKIP_EXPORT}" != "1" ]]; then
    if [[ ! -f "${HOST_MANIFEST}" ]]; then
        echo "ERROR: ${MANIFEST_FILE} not found at ${HOST_MANIFEST}"
        echo "       Run Stage 2 first:  make create-dataset"
        exit 1
    fi

    echo ""
    python3 - "${HOST_MANIFEST}" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  Input manifest: {m.get('n_cases','?')} cases "
      f"(fused={m.get('n_fused','?')}  separate={m.get('n_separate','?')}  "
      f"spine_only={m.get('n_spine_only','?')}  pelvic_only={m.get('n_pelvic_only','?')})")
# Surface orientation-fix fields when present
if "n_ap_inverted" in m:
    print(f"  Orientation   : ok={m.get('n_ap_ok','?')}  "
          f"inverted={m.get('n_ap_inverted','?')}  "
          f"indeterminate={m.get('n_ap_indeterminate','?')}  "
          f"skipped={m.get('n_ap_skipped','?')}")
    if m.get('schema_version'):
        print(f"  Schema        : {m['schema_version']}")
PYEOF
fi

# ── Singularity runtime ──────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_stage3_${SLURM_JOB_ID:-$$}"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

_run() {
    local env_args="PYTHONPATH=${PPATH}"
    if [[ "${PUSH}" == "1" ]]; then
        # Pass token via env so it does not appear in `ps aux` or CLI args
        env_args="${env_args},HF_TOKEN=${HF_TOKEN}"
    fi
    singularity exec \
        --env "${env_args}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

# ── Container-side paths ─────────────────────────────────────────────────────
C_MANIFEST="/data/placed/${MANIFEST_FILE}"
C_NIFTI="/data/tcia_nifti"
C_PLACED_SPINE="/data/placed/spine"
C_PLACED_PELVIC="/data/placed/pelvic"
C_HF_EXPORT="/data/hf_export"

# ── Stage the dataset card and interface script ─────────────────────────────
if [[ "${SKIP_EXPORT}" != "1" ]]; then
    # Copy dataset card to export dir so export_hf.py's push picks it up
    if [[ -f "${PROJECT_ROOT}/docs/dataset_card.md" ]]; then
        cp "${PROJECT_ROOT}/docs/dataset_card.md" "${HF_EXPORT_DIR}/README.md"
        echo "  Staged dataset card → ${HF_EXPORT_DIR}/README.md"
    fi
    if [[ -f "${PROJECT_ROOT}/scripts/dataset_interface.py" ]]; then
        cp "${PROJECT_ROOT}/scripts/dataset_interface.py" "${HF_EXPORT_DIR}/dataset_interface.py"
        echo "  Staged dataset_interface.py"
    fi
fi

# ── Flag construction ────────────────────────────────────────────────────────
EXPORT_FLAGS=""
[[ "${SKIP_EXPORT}" == "1" ]] && EXPORT_FLAGS="${EXPORT_FLAGS} --skip_export"
[[ "${SKIP_QC}"     == "1" ]] && EXPORT_FLAGS="${EXPORT_FLAGS} --skip_qc"
[[ "${NO_PIR}"      == "1" ]] && EXPORT_FLAGS="${EXPORT_FLAGS} --no_pir"

if [[ "${PUSH}" == "1" ]]; then
    EXPORT_FLAGS="${EXPORT_FLAGS} --push_to_hub"
    EXPORT_FLAGS="${EXPORT_FLAGS} --hf_repo_id ${HF_REPO_ID}"
    EXPORT_FLAGS="${EXPORT_FLAGS} --hf_workers ${HF_WORKERS}"
    [[ "${HF_PRIVATE}" == "1" ]] && EXPORT_FLAGS="${EXPORT_FLAGS} --hf_private"
    # Token is NOT passed as CLI arg — it comes in via HF_TOKEN env var
fi

# =============================================================================
# Run export_hf.py
# =============================================================================
echo ""
echo "======================================================================"
echo " Running export_hf.py ..."
echo "======================================================================"

_run python3 /workspace/scripts/export_hf.py \
    --manifest   "${C_MANIFEST}" \
    --nifti_dir  "${C_NIFTI}" \
    --spine_dir  "${C_PLACED_SPINE}" \
    --pelvic_dir "${C_PLACED_PELVIC}" \
    --out_dir    "${C_HF_EXPORT}" \
    --workers    "${WORKERS}" \
    ${EXPORT_FLAGS}

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================================"
echo " Stage 3 summary"
echo "======================================================================"

N_CT=$(find "${HF_EXPORT_DIR}/ct"     -name "*.nii.gz" 2>/dev/null | wc -l)
N_LB=$(find "${HF_EXPORT_DIR}/labels" -name "*.nii.gz" 2>/dev/null | wc -l)
N_QC=$(find "${HF_EXPORT_DIR}/qc"     -name "*.png"    2>/dev/null | wc -l)

printf "  CT volumes   : %d\n" "${N_CT}"
printf "  Label maps   : %d\n" "${N_LB}"
printf "  QC figures   : %d\n" "${N_QC}"
printf "  Export size  : %s\n" "$(du -sh ${HF_EXPORT_DIR} 2>/dev/null | cut -f1)"

if [[ -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    python3 - "${HF_EXPORT_DIR}/manifest.json" << 'PYEOF'
import json, sys
from collections import Counter
m = json.load(open(sys.argv[1]))
cfg = Counter(r["config"] for r in m)
lbl = Counter(r["lstv_label"] for r in m)
bad = sum(1 for r in m if not r.get("alignment_ok", True))
print(f"  Configs      : {dict(cfg)}")
print(f"  LSTV         : {dict(lbl)}")
print(f"  Align fails  : {bad}")
PYEOF
fi

if [[ "${PUSH}" == "1" ]]; then
    echo ""
    echo "  Pushed to: https://huggingface.co/datasets/${HF_REPO_ID}"
else
    echo ""
    echo "  To push when ready:"
    echo "    HF_TOKEN=hf_xxx make export-dataset PUSH=1 SKIP_EXPORT=1"
fi

echo ""
echo " Completed at $(date)"
echo "======================================================================"
