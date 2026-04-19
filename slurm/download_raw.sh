#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_download_raw
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/download_raw_%j.out
#SBATCH --error=logs/download_raw_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 1 — download raw data
#
# Downloads all three upstream datasets into data/:
#   • TCIA COLONOGRAPHY  → data/tcia/         (~250 GB, ~3451 series)
#   • CTSpine1K          → data/ctspine1k/    (~4 GB,   784 NIfTI segs)
#   • CTPelvic1K         → data/ctpelvic1k/   (~15 GB,  Zenodo archives)
#
# All three sub-downloads are idempotent: re-submitting the job resumes.
#
# Usage:
#   sbatch slurm/download_raw.sh                                # all 3
#   TCIA_ONLY=1    sbatch slurm/download_raw.sh                 # TCIA only
#   SPINE_ONLY=1   sbatch slurm/download_raw.sh                 # CTSpine1K only
#   PELVIC_ONLY=1  sbatch slurm/download_raw.sh                 # CTPelvic1K only
#
# Env:
#   HF_TOKEN    required for CTSpine1K (HuggingFace gated dataset)
#
# Next stage:
#   make create-dataset
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Selective run flags ──────────────────────────────────────────────────────
RUN_TCIA=1
RUN_SPINE=1
RUN_PELVIC=1

if [[ "${TCIA_ONLY:-0}"   == "1" ]]; then RUN_SPINE=0; RUN_PELVIC=0; fi
if [[ "${SPINE_ONLY:-0}"  == "1" ]]; then RUN_TCIA=0;  RUN_PELVIC=0; fi
if [[ "${PELVIC_ONLY:-0}" == "1" ]]; then RUN_TCIA=0;  RUN_SPINE=0;  fi

mkdir -p "${LOGS_DIR}" "${DATA_DIR}" \
         "${TCIA_DIR}" "${CTSPINE1K_DIR}" "${CTPELVIC1K_DIR}" \
         "${DATA_DIR}/hf_cache"

echo "======================================================================"
echo " Stage 1: Download raw data"
echo "   Job ID      : ${SLURM_JOB_ID:-local}"
echo "   Node        : $(hostname)"
echo "   Project     : ${PROJECT_ROOT}"
echo "   Data root   : ${DATA_DIR}"
echo "   Container   : ${SIF_PATH}"
echo "   Run flags   : TCIA=${RUN_TCIA}  SPINE=${RUN_SPINE}  PELVIC=${RUN_PELVIC}"
echo "   Started     : $(date)"
echo "======================================================================"

# ── Container runtime ────────────────────────────────────────────────────────
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container not found: ${SIF_PATH}"
    echo "       Run: sbatch slurm/hpc_pull.sh    (or: make build-container)"
    exit 1
fi

if ! command -v singularity &>/dev/null; then
    echo "ERROR: singularity not in PATH.  module load singularity?"
    exit 1
fi

export SINGULARITY_TMPDIR="/tmp/${USER}_stage1_${SLURM_JOB_ID:-$$}"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace"

_run() {
    singularity exec \
        --env PYTHONPATH="${PPATH}",HF_HOME="/data/hf_cache",HF_TOKEN="${HF_TOKEN:-}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

# =============================================================================
# 1/3: TCIA COLONOGRAPHY
# =============================================================================
if [[ "${RUN_TCIA}" == "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " 1/3  TCIA COLONOGRAPHY  →  ${TCIA_DIR}"
    echo "======================================================================"

    _run python3 /workspace/scripts/download_tcia_colonog.py \
        --out_dir  /data/tcia \
        --workers  "${WORKERS}"

    echo "  TCIA done.  Series on disk: $(find ${TCIA_DIR} -maxdepth 1 -type d 2>/dev/null | wc -l)"
fi

# =============================================================================
# 2/3: CTSpine1K (HuggingFace)
# =============================================================================
if [[ "${RUN_SPINE}" == "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " 2/3  CTSpine1K  →  ${CTSPINE1K_DIR}"
    echo "======================================================================"

    if [[ -z "${HF_TOKEN}" ]]; then
        echo "WARNING: HF_TOKEN not set. CTSpine1K is gated; download may fail."
    fi

    _run python3 - << 'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

dest  = Path("/data/ctspine1k")
cache = Path("/data/hf_cache")
token = os.environ.get("HF_TOKEN") or None

print(f"Downloading alexanderdann/CTSpine1K -> {dest}", flush=True)

# Retry up to 20 times with backoff for HF rate limits
import time
for attempt in range(1, 21):
    try:
        snapshot_download(
            repo_id    = "alexanderdann/CTSpine1K",
            repo_type  = "dataset",
            local_dir  = str(dest),
            cache_dir  = str(cache),
            token      = token,
            ignore_patterns = ["*.arrow", "*.parquet", "data/*.arrow"],
        )
        break
    except Exception as e:
        print(f"[attempt {attempt}] failed: {e}", flush=True)
        if attempt == 20:
            print("  giving up after 20 attempts", flush=True)
            sys.exit(1)
        wait = min(600, 30 * attempt)
        print(f"  sleeping {wait}s and retrying ...", flush=True)
        time.sleep(wait)

n = len(list(dest.rglob("*.nii.gz")))
print(f"Done.  .nii.gz files: {n}", flush=True)
if n < 100:
    print(f"WARNING: only {n} NIfTI files — expected ~1568 (784 img + 784 seg)", flush=True)
PYEOF

    echo "  CTSpine1K done.  NIfTIs: $(find ${CTSPINE1K_DIR} -name '*.nii.gz' 2>/dev/null | wc -l)"
fi

# =============================================================================
# 3/3: CTPelvic1K (Zenodo + HuggingFace metadata)
# =============================================================================
if [[ "${RUN_PELVIC}" == "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " 3/3  CTPelvic1K  →  ${CTPELVIC1K_DIR}"
    echo "======================================================================"

    ZENODO_RECORD=4588403
    ZENODO_BASE="https://zenodo.org/record/${ZENODO_RECORD}/files"

    mkdir -p "${CTPELVIC1K_DIR}/masks" "${CTPELVIC1K_DIR}/downloads" "${CTPELVIC1K_DIR}/metadata"

    _download() {
        local url="$1" dest="$2" label="$3"
        if [[ -f "${dest}" ]]; then
            echo "  already have: $(basename ${dest})"
            return
        fi
        echo "  fetching ${label} ..."
        wget --continue --progress=bar:force --timeout=120 --tries=5 --waitretry=30 \
             --output-document="${dest}.part" "${url}" \
          && mv "${dest}.part" "${dest}"
    }

    # Mask archives (datasets 1-5)
    for DS in 1 2 3 4 5; do
        _download \
            "${ZENODO_BASE}/CTPelvic1K_dataset${DS}_mask_mappingback.tar.gz?download=1" \
            "${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset${DS}_mask_mappingback.tar.gz" \
            "masks dataset${DS}"
    done

    # CLINIC imaging data (datasets 6-7)
    _download \
        "${ZENODO_BASE}/CTPelvic1K_dataset6_data.tar.gz?download=1" \
        "${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset6_data.tar.gz" \
        "CLINIC (dataset6)"
    _download \
        "${ZENODO_BASE}/CTPelvic1K_dataset7_data.tar.gz?download=1" \
        "${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset7_data.tar.gz" \
        "CLINIC-metal (dataset7)"

    # Extract masks
    for DS in 1 2 3 4 5; do
        MARKER="${CTPELVIC1K_DIR}/downloads/.dataset${DS}_extracted"
        if [[ ! -f "${MARKER}" && -f "${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset${DS}_mask_mappingback.tar.gz" ]]; then
            echo "  extracting masks dataset${DS} ..."
            tar -xzf "${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset${DS}_mask_mappingback.tar.gz" \
                -C "${CTPELVIC1K_DIR}/masks/"
            touch "${MARKER}"
        fi
    done

    # Extract imaging data
    for DS in 6 7; do
        MARKER="${CTPELVIC1K_DIR}/downloads/.dataset${DS}_extracted"
        ARCHIVE="${CTPELVIC1K_DIR}/downloads/CTPelvic1K_dataset${DS}_data.tar.gz"
        if [[ ! -f "${MARKER}" && -f "${ARCHIVE}" ]]; then
            echo "  extracting data dataset${DS} ..."
            tar -xzf "${ARCHIVE}" -C "${CTPELVIC1K_DIR}/"
            touch "${MARKER}"
        fi
    done

    echo "  CTPelvic1K done."
    for DS in 1 2 3 4 5; do
        DIR="${CTPELVIC1K_DIR}/masks/CTPelvic1K_dataset${DS}_mask_mappingback"
        printf "    dataset%-2s masks : %4d\n" "${DS}" \
            "$(find ${DIR} -name '*.nii.gz' 2>/dev/null | wc -l || echo 0)"
    done
fi

echo ""
echo "======================================================================"
echo " Stage 1 complete  at $(date)"
echo ""
echo " Disk usage:"
du -sh "${TCIA_DIR}" "${CTSPINE1K_DIR}" "${CTPELVIC1K_DIR}" 2>/dev/null || true
echo ""
echo " Next stage:"
echo "   make create-dataset"
echo "======================================================================"
