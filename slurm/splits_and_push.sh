#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_finalize
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 3 RESUMPTION — generate splits + push METADATA ONLY
#
# Use case: the export phase finished and the CT/label NIfTIs are already
# on HuggingFace from a previous push. generate_5fold_splits.py crashed
# before splits_5fold.json was produced, so the splits + interface +
# README never got uploaded.
#
# This script does ONLY:
#   1. Run generate_5fold_splits.py (against the placed_manifest)
#   2. Upload splits_5fold.json + dataset_interface.py + README.md to HF
#      using upload_file (one file at a time, by exact path).
#
# It does NOT:
#   - Touch the CT or label NIfTIs on HF (they stay exactly as they are)
#   - Wipe or delete anything on HF
#   - Re-run the export pipeline
#   - Use upload_large_folder (which would re-scan/upload everything)
#
# Three small files, individually uploaded to specific paths in the repo.
# Surgical, low-risk, fast.
#
# Usage
# -----
#   HF_TOKEN=hf_xxx sbatch slurm/finalize_and_push.sh
#
# Options (env)
# -------------
#   SKIP_SPLITS=1    skip splits regeneration (push existing splits_5fold.json
#                    that's already on disk locally)
#   SKIP_README=1    don't push README.md
#   SKIP_INTERFACE=1 don't push dataset_interface.py
#   SKIP_PUSH=1      regenerate splits but don't push to HF (dry run)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Manifest selection (mirrors export_dataset.sh) ──────────────────────────
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"

if [[ ! -f "${HOST_MANIFEST}" && "${MANIFEST_FILE}" == "placed_manifest_orientation_fixed.json" ]]; then
    echo "WARNING: ${HOST_MANIFEST} not found, falling back to placed_manifest.json"
    MANIFEST_FILE="placed_manifest.json"
    HOST_MANIFEST="${PLACED_DIR}/${MANIFEST_FILE}"
fi

SKIP_SPLITS="${SKIP_SPLITS:-0}"
SKIP_README="${SKIP_README:-0}"
SKIP_INTERFACE="${SKIP_INTERFACE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

mkdir -p "${LOGS_DIR}" "${HF_EXPORT_DIR}"

echo "======================================================================"
echo " Stage 3 RESUMPTION — splits + metadata-only push"
echo "   Job ID         : ${SLURM_JOB_ID:-local}"
echo "   Node           : $(hostname)"
echo "   Manifest       : ${HOST_MANIFEST}"
echo "   Export dir     : ${HF_EXPORT_DIR}"
echo "   HF repo        : ${HF_REPO_ID}"
echo "   SKIP_SPLITS    : ${SKIP_SPLITS}"
echo "   SKIP_README    : ${SKIP_README}"
echo "   SKIP_INTERFACE : ${SKIP_INTERFACE}"
echo "   SKIP_PUSH      : ${SKIP_PUSH}"
echo ""
echo "   *** This script does NOT wipe HF or touch any CT/label files. ***"
echo "   *** Only splits_5fold.json + dataset_interface.py + README.md ***"
echo "   *** are uploaded individually via upload_file.                ***"
echo ""
echo "   Started        : $(date)"
echo "======================================================================"

# ── Pre-flight ───────────────────────────────────────────────────────────────
if [[ "${SKIP_PUSH}" != "1" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: HF_TOKEN required (or set SKIP_PUSH=1 to dry-run)"
        echo "       HF_TOKEN=hf_xxx sbatch $0"
        exit 1
    fi
    echo "  HF_TOKEN : ${HF_TOKEN:0:8}***"
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing: ${SIF_PATH}"
    exit 1
fi

if [[ ! -f "${HOST_MANIFEST}" ]]; then
    echo "ERROR: manifest not found at ${HOST_MANIFEST}"
    exit 1
fi

# ── Singularity runtime ──────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_finalize_${SLURM_JOB_ID:-$$}"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

_run() {
    local env_args="PYTHONPATH=${PPATH}"
    if [[ "${SKIP_PUSH}" != "1" ]]; then
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
C_HF_EXPORT="/data/hf_export"

# =============================================================================
# Step 1: Generate 5-fold CV splits
# =============================================================================
if [[ "${SKIP_SPLITS}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step 1: Generate 5-fold CV splits"
    echo "======================================================================"

    _run python3 /workspace/scripts/generate_5fold_splits.py \
        --placed_manifest "${C_MANIFEST}" \
        --out             "${C_HF_EXPORT}/splits_5fold.json" \
        --n_folds         5 \
        --test_fraction   0.15 \
        --seed            42

    if [[ ! -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        echo "ERROR: splits file was not produced at ${HF_EXPORT_DIR}/splits_5fold.json"
        exit 1
    fi
    echo "  ✓ Splits file produced: ${HF_EXPORT_DIR}/splits_5fold.json"

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
else
    echo "  Step 1: skipped (SKIP_SPLITS=1)"
    if [[ ! -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        echo "  WARNING: SKIP_SPLITS=1 but ${HF_EXPORT_DIR}/splits_5fold.json missing."
    fi
fi

# =============================================================================
# Step 2: Stage README + interface locally
# =============================================================================
echo ""
echo "======================================================================"
echo " Step 2: Stage README + dataset_interface.py"
echo "======================================================================"

if [[ "${SKIP_README}" != "1" ]]; then
    if [[ -f "${PROJECT_ROOT}/docs/dataset_card.md" ]]; then
        cp "${PROJECT_ROOT}/docs/dataset_card.md" "${HF_EXPORT_DIR}/README.md"
        echo "  ✓ Staged docs/dataset_card.md → ${HF_EXPORT_DIR}/README.md"
    elif [[ -f "${PROJECT_ROOT}/README.md" ]]; then
        cp "${PROJECT_ROOT}/README.md" "${HF_EXPORT_DIR}/README.md"
        echo "  ✓ Staged README.md → ${HF_EXPORT_DIR}/README.md"
    else
        echo "  WARNING: no README.md found in docs/ or project root."
    fi
else
    echo "  README staging skipped (SKIP_README=1)"
fi

if [[ "${SKIP_INTERFACE}" != "1" ]]; then
    if [[ -f "${PROJECT_ROOT}/scripts/dataset_interface.py" ]]; then
        cp "${PROJECT_ROOT}/scripts/dataset_interface.py" "${HF_EXPORT_DIR}/dataset_interface.py"
        echo "  ✓ Staged scripts/dataset_interface.py"
    else
        echo "  WARNING: scripts/dataset_interface.py not found."
    fi
else
    echo "  Interface staging skipped (SKIP_INTERFACE=1)"
fi

# =============================================================================
# Step 3: Push ONLY the named files to HF (additive, surgical)
# =============================================================================
if [[ "${SKIP_PUSH}" == "1" ]]; then
    echo ""
    echo "  Step 3: skipped (SKIP_PUSH=1, dry run)"
else
    echo ""
    echo "======================================================================"
    echo " Step 3: Upload metadata files individually to HF"
    echo "======================================================================"
    echo "  Uploading to repo: ${HF_REPO_ID}"
    echo "  Mode: per-file upload via huggingface_hub.upload_file"
    echo "  No CTs, labels, or other repo files are touched."
    echo ""

    # Build the file list dynamically based on what's staged + skip flags.
    # Each entry is "local_path:repo_path"
    FILES_TO_UPLOAD=()
    if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/splits_5fold.json:splits_5fold.json")
    fi
    if [[ "${SKIP_README}" != "1" && -f "${HF_EXPORT_DIR}/README.md" ]]; then
        FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/README.md:README.md")
    fi
    if [[ "${SKIP_INTERFACE}" != "1" && -f "${HF_EXPORT_DIR}/dataset_interface.py" ]]; then
        FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/dataset_interface.py:dataset_interface.py")
    fi

    if [[ ${#FILES_TO_UPLOAD[@]} -eq 0 ]]; then
        echo "  Nothing to upload. Exiting."
        exit 0
    fi

    # Echo what we're about to push so it's auditable in the log
    echo "  Files queued for upload:"
    for entry in "${FILES_TO_UPLOAD[@]}"; do
        local_path="${entry%%:*}"
        repo_path="${entry##*:}"
        size=$(du -h "${local_path}" 2>/dev/null | awk '{print $1}')
        printf "    %-12s  →  %s\n" "${size}" "${repo_path}"
    done
    echo ""

    # Convert to container-side paths and a comma-separated list for the
    # python step, since arrays don't survive heredoc subprocess boundary.
    UPLOAD_ARGS=""
    for entry in "${FILES_TO_UPLOAD[@]}"; do
        local_path="${entry%%:*}"
        repo_path="${entry##*:}"
        # Translate ${HF_EXPORT_DIR} (host) → /data/hf_export (container)
        c_local_path="${local_path/${HF_EXPORT_DIR}/${C_HF_EXPORT}}"
        UPLOAD_ARGS="${UPLOAD_ARGS}${c_local_path}:${repo_path};"
    done

    _run python3 -u - "${HF_REPO_ID}" "${UPLOAD_ARGS}" << 'PYEOF'
"""
Surgical metadata push. ONE huggingface_hub.upload_file call per file.

Does NOT call upload_large_folder, does NOT delete anything, does NOT
touch any file other than the ones explicitly named.
"""
import os
import sys

repo_id     = sys.argv[1]
upload_spec = sys.argv[2].rstrip(";")

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN env var missing in container", file=sys.stderr)
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=token)

# Sanity: confirm the repo exists. We do NOT create it (would mask a typo).
try:
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"  Target repo confirmed: {repo_id} (last commit: {info.sha[:8]})")
except Exception as e:
    print(f"ERROR: could not access repo {repo_id}: {e}", file=sys.stderr)
    sys.exit(2)

uploads = [s for s in upload_spec.split(";") if s]
print(f"  Will upload {len(uploads)} file(s) individually.")
print()

failures = []
for i, spec in enumerate(uploads, 1):
    local_path, repo_path = spec.split(":", 1)
    if not os.path.isfile(local_path):
        print(f"  [{i}/{len(uploads)}] SKIP (missing): {local_path}", file=sys.stderr)
        failures.append((local_path, "missing"))
        continue
    size = os.path.getsize(local_path)
    print(f"  [{i}/{len(uploads)}] uploading {repo_path} ({size} bytes) ...")
    try:
        url = api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = repo_path,
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = f"Update {repo_path} (metadata-only push)",
        )
        print(f"      ✓ {url}")
    except Exception as e:
        print(f"      ✗ FAILED: {e}", file=sys.stderr)
        failures.append((local_path, str(e)))

print()
if failures:
    print(f"  {len(failures)} upload(s) failed:")
    for path, reason in failures:
        print(f"    - {path}: {reason}")
    sys.exit(3)
else:
    print(f"  ✓ All {len(uploads)} uploads succeeded.")
    print(f"  Browse: https://huggingface.co/datasets/{repo_id}/tree/main")
PYEOF

fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================================"
echo " Finalize summary"
echo "======================================================================"

# Local state on disk
N_CT=$(find "${HF_EXPORT_DIR}/ct"     -name "*.nii.gz" 2>/dev/null | wc -l)
N_LB=$(find "${HF_EXPORT_DIR}/labels" -name "*.nii.gz" 2>/dev/null | wc -l)
printf "  Local CT volumes  : %d (untouched)\n" "${N_CT}"
printf "  Local label maps  : %d (untouched)\n" "${N_LB}"
printf "  Local export size : %s (untouched)\n" "$(du -sh ${HF_EXPORT_DIR} 2>/dev/null | cut -f1)"

if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
    SPLITS_SIZE=$(du -h "${HF_EXPORT_DIR}/splits_5fold.json" | cut -f1)
    printf "  splits_5fold.json : present (%s)\n" "${SPLITS_SIZE}"
fi

if [[ "${SKIP_PUSH}" != "1" ]]; then
    echo ""
    echo "  HF repo (additive, no wipe): https://huggingface.co/datasets/${HF_REPO_ID}"
fi

echo ""
echo " Completed at $(date)"
echo "======================================================================"
