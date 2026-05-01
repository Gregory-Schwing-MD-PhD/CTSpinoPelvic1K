#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_refresh
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/refresh_%j.out
#SBATCH --error=logs/refresh_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 3 RESUMPTION — refresh manifests + v6 splits + metadata-only push
#
# Use case: the parser bug fix in mask_index.py changed lstv_class for
# tokens 22 and 120 (class 3 -> class 2). The CT and label NIfTI files
# on disk are byte-identical to the previous push (placement was unchanged),
# so re-running export_hf.py end-to-end would be ~800 wasted file copies.
#
# This script does ONLY:
#   1. Refresh manifest.json + manifest.csv + manifest_train/val/test.json
#      + data_splits.json + splits/test.json + splits_summary.json
#      from the updated placed_manifest.json (refresh_hf_manifests.py)
#   2. Generate v6 splits (generate_5fold_splits.py — new CLI: --hf_dir)
#   3. Upload the refreshed metadata files to HF using upload_file
#      (one file at a time, by exact path).
#
# It does NOT:
#   - Touch the CT or label NIfTIs on HF (they stay byte-identical)
#   - Wipe or delete anything on HF
#   - Re-run export_hf.py
#   - Use upload_large_folder
#
# Usage
# -----
#   HF_TOKEN=hf_xxx sbatch slurm/refresh_and_push.sh
#
# Options (env)
# -------------
#   SKIP_REFRESH=1   skip manifest refresh (use existing manifest.json)
#   SKIP_SPLITS=1    skip splits regeneration (push existing splits_5fold.json)
#   SKIP_README=1    don't push README.md
#   SKIP_INTERFACE=1 don't push dataset_interface.py
#   SKIP_PUSH=1      regenerate everything but don't push to HF (dry run)
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

SKIP_REFRESH="${SKIP_REFRESH:-0}"
SKIP_SPLITS="${SKIP_SPLITS:-0}"
SKIP_README="${SKIP_README:-0}"
SKIP_INTERFACE="${SKIP_INTERFACE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

mkdir -p "${LOGS_DIR}" "${HF_EXPORT_DIR}"

echo "======================================================================"
echo " Stage 3 RESUMPTION — refresh manifests + v6 splits + metadata push"
echo "   Job ID         : ${SLURM_JOB_ID:-local}"
echo "   Node           : $(hostname)"
echo "   Manifest       : ${HOST_MANIFEST}"
echo "   Export dir     : ${HF_EXPORT_DIR}"
echo "   HF repo        : ${HF_REPO_ID}"
echo "   SKIP_REFRESH   : ${SKIP_REFRESH}"
echo "   SKIP_SPLITS    : ${SKIP_SPLITS}"
echo "   SKIP_README    : ${SKIP_README}"
echo "   SKIP_INTERFACE : ${SKIP_INTERFACE}"
echo "   SKIP_PUSH      : ${SKIP_PUSH}"
echo ""
echo "   *** NO NIfTIs are re-exported. CT/label files on disk and on HF ***"
echo "   *** stay byte-identical. Only manifest JSON/CSV files change.    ***"
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

if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: ${HF_EXPORT_DIR}/manifest.json not found."
    echo "       This script requires a previous export_hf.py run to have"
    echo "       populated hf_dir. If hf_dir is empty, run export_dataset.sh"
    echo "       end-to-end first."
    exit 1
fi

# ── Singularity runtime ──────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_refresh_${SLURM_JOB_ID:-$$}"
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
# Step 1: Refresh HF manifests from updated placed_manifest.json
# =============================================================================
if [[ "${SKIP_REFRESH}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step 1: Refresh HF manifests from updated placed_manifest"
    echo "======================================================================"

    if [[ ! -f "${PROJECT_ROOT}/scripts/refresh_hf_manifests.py" ]]; then
        echo "ERROR: scripts/refresh_hf_manifests.py not found."
        exit 1
    fi

    _run python3 /workspace/scripts/refresh_hf_manifests.py \
        --placed_manifest "${C_MANIFEST}" \
        --hf_dir          "${C_HF_EXPORT}"

    # Sanity check the refreshed manifest
    python3 - "${HF_EXPORT_DIR}/manifest.json" << 'PYEOF'
import json, sys
from collections import Counter
m = json.load(open(sys.argv[1]))
classes = Counter(r.get('lstv_class', 0) for r in m)
labels = Counter(r.get('lstv_label', '') for r in m)
print(f"  Refreshed manifest.json: {len(m)} records")
print(f"  lstv_class distribution: {dict(sorted(classes.items()))}")
print(f"  lstv_label distribution: {dict(labels)}")
PYEOF
else
    echo "  Step 1: skipped (SKIP_REFRESH=1)"
fi

# =============================================================================
# Step 2: Generate v6 5-fold CV splits
# =============================================================================
if [[ "${SKIP_SPLITS}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step 2: Generate v6 5-fold CV splits"
    echo "======================================================================"

    _run python3 /workspace/scripts/generate_5fold_splits.py \
        --hf_dir  "${C_HF_EXPORT}" \
        --out     "${C_HF_EXPORT}/splits_5fold.json" \
        --n_folds 5 \
        --seed    42

    if [[ ! -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        echo "ERROR: splits file not produced at ${HF_EXPORT_DIR}/splits_5fold.json"
        exit 1
    fi

    python3 - "${HF_EXPORT_DIR}/splits_5fold.json" << 'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  Splits        : schema v{s.get('schema_version','?')}")
print(f"  n_patients    : {s.get('n_patients','?')}")
print(f"  n_folds       : {s.get('n_folds','?')}")
print(f"  Subtype counts:")
for st, n in s.get('subtype_counts', {}).items():
    print(f"    {st:<22} {n}")
PYEOF
else
    echo "  Step 2: skipped (SKIP_SPLITS=1)"
    if [[ ! -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        echo "  WARNING: SKIP_SPLITS=1 but splits_5fold.json missing locally."
    fi
fi

# =============================================================================
# Step 3: Stage README + interface locally
# =============================================================================
echo ""
echo "======================================================================"
echo " Step 3: Stage README + dataset_interface.py"
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
# Step 4: Push refreshed metadata files to HF
# =============================================================================
if [[ "${SKIP_PUSH}" == "1" ]]; then
    echo ""
    echo "  Step 4: skipped (SKIP_PUSH=1, dry run)"
else
    echo ""
    echo "======================================================================"
    echo " Step 4: Upload metadata files individually to HF"
    echo "======================================================================"
    echo "  Uploading to repo: ${HF_REPO_ID}"
    echo "  Mode: per-file upload via huggingface_hub.upload_file"
    echo "  No CTs, labels, or other repo files are touched."
    echo ""

    # Build upload list. Each entry is "local_path:repo_path".
    # We push EVERYTHING that the parser-bug fix touched, not just splits.
    FILES_TO_UPLOAD=()

    # Manifests refreshed by refresh_hf_manifests.py
    for fn in manifest.json manifest.csv \
              manifest_train.json manifest_validation.json manifest_test.json \
              data_splits.json splits_summary.json; do
        if [[ -f "${HF_EXPORT_DIR}/${fn}" ]]; then
            FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/${fn}:${fn}")
        fi
    done

    # splits/test.json (subdirectory)
    if [[ -f "${HF_EXPORT_DIR}/splits/test.json" ]]; then
        FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/splits/test.json:splits/test.json")
    fi

    # New 5-fold CV splits (v6)
    if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
        FILES_TO_UPLOAD+=("${HF_EXPORT_DIR}/splits_5fold.json:splits_5fold.json")
    fi

    # README and interface (optional)
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

    echo "  Files queued for upload:"
    for entry in "${FILES_TO_UPLOAD[@]}"; do
        local_path="${entry%%:*}"
        repo_path="${entry##*:}"
        size=$(du -h "${local_path}" 2>/dev/null | awk '{print $1}')
        printf "    %-12s  →  %s\n" "${size}" "${repo_path}"
    done
    echo ""

    UPLOAD_ARGS=""
    for entry in "${FILES_TO_UPLOAD[@]}"; do
        local_path="${entry%%:*}"
        repo_path="${entry##*:}"
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

# Confirm the repo exists. We do NOT create it (would mask a typo).
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
            commit_message  = f"Refresh {repo_path} (parser bug fix Apr 2026)",
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
echo " Refresh + push summary"
echo "======================================================================"

N_CT=$(find "${HF_EXPORT_DIR}/ct"     -name "*.nii.gz" 2>/dev/null | wc -l)
N_LB=$(find "${HF_EXPORT_DIR}/labels" -name "*.nii.gz" 2>/dev/null | wc -l)
printf "  Local CT volumes  : %d (untouched)\n" "${N_CT}"
printf "  Local label maps  : %d (untouched)\n" "${N_LB}"
printf "  Local export size : %s (untouched)\n" "$(du -sh ${HF_EXPORT_DIR} 2>/dev/null | cut -f1)"

if [[ -f "${HF_EXPORT_DIR}/splits_5fold.json" ]]; then
    printf "  splits_5fold.json : present (%s)\n" "$(du -h ${HF_EXPORT_DIR}/splits_5fold.json | cut -f1)"
fi

if [[ "${SKIP_PUSH}" != "1" ]]; then
    echo ""
    echo "  HF repo (additive, no wipe): https://huggingface.co/datasets/${HF_REPO_ID}"
fi

echo ""
echo " Completed at $(date)"
echo "======================================================================"
