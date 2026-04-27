#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_export_dataset
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/export_dataset_%j.out
#SBATCH --error=logs/export_dataset_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 3 — export dataset + push to HuggingFace
#
# Reads placed_manifest_orientation_fixed.json from Stage 2 by default (since
# that manifest carries manually-reviewed AP-inversion flips for tokens where
# the anatomy was wrong in the original scan). To export against the un-fixed
# manifest instead, set MANIFEST_FILE=placed_manifest.json.
#
# Produces:
#   data/hf_export/ct/          CT NIfTIs (PIR, PHI-stripped)
#   data/hf_export/labels/      10-class label NIfTIs (voxel-aligned with CT)
#   data/hf_export/qc/          QC overlays (optional)
#   data/hf_export/manifest.json
#   data/hf_export/manifest.csv
#   data/hf_export/splits_5fold.json         LSTV-stratified 5-fold + test
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
#   SKIP_SPLITS=1    skip 5-fold splits generation
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Manifest selection ───────────────────────────────────────────────────────
# Default to the orientation-fixed manifest. Its schema (v6_manual_flips_with_exclusions)
# is a strict superset of the original placed_manifest.json — same top-level
# keys plus additional orientation_check / orientation_fixed fields per case
# plus patched series_uid for flipped cases. export_hf.py reads series_uid to
# locate CT files, so using this manifest automatically picks up the flipped
# CT NIfTIs.
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"

# Fallback: if the orientation-fixed manifest doesn't exist (e.g. Stage 2 ran
# without Step C), fall back to the original with a loud warning.
if [[ ! -f "${HOST_MANIFEST}" && "${MANIFEST_FILE}" == "placed_manifest_orientation_fixed.json" ]]; then
    echo "WARNING: ${HOST_MANIFEST} not found."
    echo "         Falling back to placed_manifest.json (no manual-flip review applied)."
    echo "         Run Stage 2 with Step C enabled to produce the orientation-fixed manifest."
    MANIFEST_FILE="placed_manifest.json"
    HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"
fi

SKIP_SPLITS="${SKIP_SPLITS:-0}"

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
echo "   SKIP_SPLITS  : ${SKIP_SPLITS}"
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

# Surface v6 manual-flip review fields when present (orientation-fixed manifest)
if "n_manually_flipped" in m:
    print(f"  Manual flips  : flipped={m.get('n_manually_flipped','?')}  "
          f"requested={m.get('n_flip_requested','?')}  "
          f"missing={m.get('n_flip_missing','?')}  "
          f"failed={m.get('n_flip_failed','?')}")
    if "n_excluded" in m:
        print(f"  Exclusions    : applied={m.get('n_excluded','?')}  "
              f"requested={m.get('n_exclude_requested','?')}  "
              f"missing={m.get('n_exclude_missing','?')}")
    if m.get('schema_version'):
        print(f"  Schema        : {m['schema_version']}")
    if m.get('excluded_tokens'):
        print(f"  Excluded      : {m['excluded_tokens']}")
else:
    print(f"  Schema        : {m.get('schema_version','unknown (pre-v6, no manual flips)')}")
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
# Generate 5-fold CV splits (from the HF manifest we just produced)
# =============================================================================
if [[ "${SKIP_EXPORT}" != "1" && "${SKIP_SPLITS}" != "1" ]]; then
    if [[ -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
        echo ""
        echo "======================================================================"
        echo " Generating 5-fold CV splits"
        echo "======================================================================"

        _run python3 /workspace/scripts/generate_5fold_splits.py \
            --placed_manifest "${C_MANIFEST}" \
            --out             "${C_HF_EXPORT}/splits_5fold.json" \
            --n_folds         5 \
            --test_fraction   0.15 \
            --seed            42

        if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
            echo "  Splits file: ${HF_EXPORT_DIR}/splits_5fold.json"
        else
            echo "  WARNING: splits file was not produced."
        fi
    else
        echo "  Skipping splits: ${HF_EXPORT_DIR}/manifest.json not found."
    fi
elif [[ "${SKIP_SPLITS}" == "1" ]]; then
    echo "  Splits generation skipped (SKIP_SPLITS=1)"
fi

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
# manifest.json may be a flat list OR {"records": [...]}
recs = m if isinstance(m, list) else m.get("records", [])
cfg = Counter(r["config"] for r in recs)
lbl = Counter(r.get("lstv_label", "") for r in recs)
bad = sum(1 for r in recs if not r.get("alignment_ok", True))
print(f"  Configs      : {dict(cfg)}")
print(f"  LSTV         : {dict(lbl)}")
print(f"  Align fails  : {bad}")
PYEOF
fi

if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
    python3 - "${HF_EXPORT_DIR}/splits_5fold.json" << 'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  Splits       : schema v{s.get('schema_version','?')}  "
      f"strata={s.get('strata_scheme','?')}")
print(f"    total     = {s.get('n_tokens_total','?')}")
print(f"    test      = {s.get('n_tokens_test','?')}")
print(f"    trainval  = {s.get('n_tokens_trainval','?')}")
print(f"    n_folds   = {s.get('n_folds','?')}")
print(f"    validated = {s.get('invariants_validated', False)}")
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
