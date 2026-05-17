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
# Usage (HF_REPO_ID is REQUIRED for any push — there is no default repo):
#   sbatch slurm/export_dataset.sh                              # export only
#   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name PUSH=1 \
#       sbatch slurm/export_dataset.sh                          # export + push
#   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name PUSH=1 SKIP_EXPORT=1 \
#       sbatch slurm/export_dataset.sh                          # push existing
#
# Multi-repo push (e.g. one repo per conference submission). HF_REPO_ID and
# HF_TOKEN are independent per-invocation env vars — export ONCE, then push
# the same data/hf_export/ to each repo with its own repo id + token:
#   # 1. export once (no push)
#   sbatch slurm/export_dataset.sh
#   # 2. push to repo A
#   HF_TOKEN=hf_tokenA HF_REPO_ID=anonymous-neurips-ED/CTSpinoPelvic1K \
#     PUSH=1 SKIP_EXPORT=1 sbatch slurm/export_dataset.sh
#   # 3. push to repo B (different repo, different token)
#   HF_TOKEN=hf_tokenB HF_REPO_ID=anonymous-other-venue/CTSpinoPelvic1K \
#     PUSH=1 SKIP_EXPORT=1 sbatch slurm/export_dataset.sh
# Nothing is hardcoded to a single repo; the resolved target is echoed
# prominently right before the push so a mis-set HF_REPO_ID is caught.
#
# Wipe orphan files on HF (e.g., after a filename schema change):
#   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name PUSH=1 WIPE_REMOTE=1 \
#       sbatch slurm/export_dataset.sh                          # wipe + re-push
# WIPE_REMOTE clears ALL files in the HF repo before push (so files from
# prior schemas don't linger) but PRESERVES the repo itself — its URL, git
# history, stars and discussions stay intact (the URL is under anonymous
# paper review and must stay continuously live). Always paired with PUSH=1
# (the script enforces this). The non-interactive FORCE_WIPE_REMOTE=1 flag
# is auto-set when WIPE_REMOTE=1 in a SLURM context — there's no terminal
# for the confirmation prompt.
#
# Options:
#   MANIFEST_FILE=placed_manifest.json   use un-fixed manifest (default: fixed)
#   SKIP_QC=1        skip QC figure generation
#   NO_PIR=1         skip PIR reorientation (native voxel space)
#   HF_PRIVATE=1     create HF repo as private
#   HF_WORKERS=8     HF upload workers (default 8)
#   SKIP_SPLITS=1    skip 5-fold splits generation
#   WIPE_REMOTE=1    clear all files in the HF repo before push; repo,
#                    URL, git history & discussions preserved
#
# CHANGE LOG (relevant)
# ---------------------
# May 2026: generate_5fold_splits.py CLI was changed to take --hf_dir
#   instead of --placed_manifest, and --test_fraction was dropped.
#   Updated this script's Step 2 invocation to match. Earlier runs of
#   this script crashed at the splits step with "the following arguments
#   are required: --hf_dir" while the export+push had already succeeded.
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Manifest selection ───────────────────────────────────────────────────────
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"

if [[ ! -f "${HOST_MANIFEST}" && "${MANIFEST_FILE}" == "placed_manifest_orientation_fixed.json" ]]; then
    echo "WARNING: ${HOST_MANIFEST} not found."
    echo "         Falling back to placed_manifest.json (no manual-flip review applied)."
    echo "         Run Stage 2 with Step C enabled to produce the orientation-fixed manifest."
    MANIFEST_FILE="placed_manifest.json"
    HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"
fi

SKIP_SPLITS="${SKIP_SPLITS:-0}"
WIPE_REMOTE="${WIPE_REMOTE:-0}"

if [[ "${WIPE_REMOTE}" == "1" ]]; then
    export FORCE_WIPE_REMOTE=1
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
echo "   WIPE_REMOTE  : ${WIPE_REMOTE}"
echo "   SKIP_EXPORT  : ${SKIP_EXPORT}"
echo "   SKIP_QC      : ${SKIP_QC}"
echo "   SKIP_SPLITS  : ${SKIP_SPLITS}"
echo "   NO_PIR       : ${NO_PIR}"
echo "   HF_PRIVATE   : ${HF_PRIVATE}"
echo "   Started      : $(date)"
echo "======================================================================"

# ── Token handling (with redaction for logs) ─────────────────────────────────
if [[ "${PUSH}" == "1" ]]; then
    if [[ -z "${HF_REPO_ID:-}" ]]; then
        echo "ERROR: PUSH=1 requires HF_REPO_ID — there is no default repo."
        echo "       The same export is pushed to multiple venue repos, so"
        echo "       the target must be explicit. Submit via:"
        echo "         HF_TOKEN=hf_xxx HF_REPO_ID=org/Name make export-dataset PUSH=1"
        exit 1
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: PUSH=1 requires HF_TOKEN."
        echo "       Submit via:  HF_TOKEN=hf_xxx HF_REPO_ID=org/Name make export-dataset PUSH=1"
        exit 1
    fi
    echo "  HF_TOKEN : ${HF_TOKEN:0:8}***  (full token passed via env, redacted in logs)"
fi

if [[ "${WIPE_REMOTE}" == "1" && "${PUSH}" != "1" ]]; then
    echo "ERROR: WIPE_REMOTE=1 requires PUSH=1."
    echo "       Wiping the HF repo without repushing would leave it offline."
    exit 1
fi

if [[ "${WIPE_REMOTE}" == "1" ]]; then
    echo ""
    echo "  ⚠  WIPE_REMOTE=1: ALL files in HF repo ${HF_REPO_ID} will be"
    echo "     cleared before push. The repo, its URL, git history, stars"
    echo "     and discussions are PRESERVED (URL stays continuously live)."
    echo "     Local files at ${HF_EXPORT_DIR} are unaffected."
    echo ""
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
        env_args="${env_args},HF_TOKEN=${HF_TOKEN}"
        if [[ "${WIPE_REMOTE}" == "1" ]]; then
            env_args="${env_args},FORCE_WIPE_REMOTE=1"
        fi
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
    if [[ "${WIPE_REMOTE}" == "1" ]]; then
        EXPORT_FLAGS="${EXPORT_FLAGS} --wipe_remote --force_wipe_remote"
    fi
fi

# =============================================================================
# Run export_hf.py
# =============================================================================
echo ""
echo "======================================================================"
echo " Running export_hf.py ..."
echo "======================================================================"

if [[ "${PUSH}" == "1" ]]; then
    echo ""
    echo "######################################################################"
    echo "#  PUSH TARGET (verify before this proceeds):"
    echo "#    HF_REPO_ID  = ${HF_REPO_ID}"
    echo "#    URL         = https://huggingface.co/datasets/${HF_REPO_ID}"
    echo "#    WIPE_REMOTE = ${WIPE_REMOTE}  (1 = clear all files first; repo/URL/history kept)"
    echo "#    HF_TOKEN    = ${HF_TOKEN:0:8}***  (redacted)"
    echo "######################################################################"
    echo ""
fi

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
#
# CLI (May 2026): --hf_dir replaces --placed_manifest, and --test_fraction
# was removed (the 15% test split is now derived inside the script from
# the manifest's existing train/validation/test partition). Earlier
# versions of this script passed the old args and crashed here while the
# export and HF push had already succeeded. If you see a missing-arg
# crash again, diff this invocation against
#   python3 /workspace/scripts/generate_5fold_splits.py --help
# and update accordingly.
# =============================================================================
if [[ "${SKIP_EXPORT}" != "1" && "${SKIP_SPLITS}" != "1" ]]; then
    if [[ -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
        echo ""
        echo "======================================================================"
        echo " Generating 5-fold CV splits"
        echo "======================================================================"

        _run python3 /workspace/scripts/generate_5fold_splits.py \
            --hf_dir  "${C_HF_EXPORT}" \
            --out     "${C_HF_EXPORT}/splits_5fold.json" \
            --n_folds 5 \
            --seed    42

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
recs = m if isinstance(m, list) else m.get("records", [])
cfg = Counter(r["config"] for r in recs)
lbl = Counter(r.get("lstv_label", "") for r in recs)
bad = sum(1 for r in recs if not r.get("alignment_ok", True))
n_partial = sum(1 for r in recs if r.get("partial_annotation"))
print(f"  Configs      : {dict(cfg)}")
print(f"  LSTV         : {dict(lbl)}")
print(f"  Align fails  : {bad}")
print(f"  Partial annots (ignore=10): {n_partial}/{len(recs)}")
PYEOF
fi

if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
    python3 - "${HF_EXPORT_DIR}/splits_5fold.json" << 'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  Splits       : schema v{s.get('schema_version','?')}")
print(f"  n_patients   : {s.get('n_patients','?')}")
print(f"  n_folds      : {s.get('n_folds','?')}")
counts = s.get('subtype_counts')
if counts:
    print(f"  Subtype counts:")
    for st, n in counts.items():
        print(f"    {st:<22} {n}")
PYEOF
fi

if [[ "${PUSH}" == "1" ]]; then
    echo ""
    if [[ "${WIPE_REMOTE}" == "1" ]]; then
        echo "  Pushed (cleared files + repush, repo preserved) to: https://huggingface.co/datasets/${HF_REPO_ID}"
    else
        echo "  Pushed (additive) to: https://huggingface.co/datasets/${HF_REPO_ID}"
    fi
else
    echo ""
    echo "  To push when ready:"
    echo "    HF_TOKEN=hf_xxx make export-dataset PUSH=1 SKIP_EXPORT=1"
    echo ""
    echo "  To wipe orphan files on HF and re-push from scratch:"
    echo "    HF_TOKEN=hf_xxx PUSH=1 WIPE_REMOTE=1 SKIP_EXPORT=1 \\"
    echo "      sbatch slurm/export_dataset.sh"
fi

echo ""
echo " Completed at $(date)"
echo "======================================================================"
