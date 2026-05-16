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
#   # Paste a HuggingFace token on the submit line to avoid throttling.
#   # sbatch exports the submitter's env by default, so this just works:
#   HF_TOKEN=hf_xxx sbatch slurm/download_raw.sh
#
# Resumability:
#   Each stage writes a .download_complete marker on success and is skipped
#   on resubmission (so a finished TCIA download is not re-scanned, and only
#   the missing stages — e.g. the CTPelvic1K masks — actually run).  In
#   addition, every individual download is itself idempotent.
#   Force a stage to re-run with:  FORCE=1 sbatch slurm/download_raw.sh
#
# Env:
#   HF_TOKEN    required for CTSpine1K (HuggingFace gated dataset); pass it
#               on the submit line (see above) to lift HF rate limits
#   FORCE=1     ignore .download_complete markers and re-run every stage
#
# Next stage:
#   make create-dataset
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Capture any HF_TOKEN passed on the submit command line *before* sourcing
# default.env, so a command-line value always wins over a file default.
_CLI_HF_TOKEN="${HF_TOKEN:-}"
source configs/default.env
HF_TOKEN="${_CLI_HF_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN

# Honour markers from previous successful runs unless FORCE=1.
FORCE="${FORCE:-0}"

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
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "   HF_TOKEN    : provided (${#HF_TOKEN} chars) — authenticated HF downloads"
else
    echo "   HF_TOKEN    : NOT set — CTSpine1K may be rate-limited or blocked"
fi
echo "   FORCE       : ${FORCE}  (1 = ignore .download_complete markers)"
echo "   Started     : $(date)"
echo "======================================================================"

# ── Completion-marker helpers ────────────────────────────────────────────────
# A stage that finished cleanly drops a .download_complete marker in its data
# dir.  _stage_done short-circuits a stage on resubmission (unless FORCE=1).
_stage_done() {  # $1 = stage data dir
    [[ "${FORCE}" != "1" && -f "${1}/.download_complete" ]]
}
_mark_done() {   # $1 = stage data dir
    touch "${1}/.download_complete"
}

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
if [[ "${RUN_TCIA}" == "1" ]] && _stage_done "${TCIA_DIR}"; then
    echo ""
    echo " 1/3  TCIA COLONOGRAPHY  —  already complete, skipping"
    echo "      (FORCE=1 to re-download)"
    RUN_TCIA=0
fi
if [[ "${RUN_TCIA}" == "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " 1/3  TCIA COLONOGRAPHY  →  ${TCIA_DIR}"
    echo "======================================================================"

    _run python3 /workspace/scripts/download_tcia_colonog.py \
        --out_dir  /data/tcia \
        --workers  "${WORKERS}"

    _mark_done "${TCIA_DIR}"
    echo "  TCIA done.  Series on disk: $(find ${TCIA_DIR} -maxdepth 1 -type d 2>/dev/null | wc -l)"
fi

# =============================================================================
# 2/3: CTSpine1K (HuggingFace)
# =============================================================================
if [[ "${RUN_SPINE}" == "1" ]] && _stage_done "${CTSPINE1K_DIR}"; then
    echo ""
    echo " 2/3  CTSpine1K  —  already complete, skipping"
    echo "      (FORCE=1 to re-download)"
    RUN_SPINE=0
fi
if [[ "${RUN_SPINE}" == "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " 2/3  CTSpine1K  →  ${CTSPINE1K_DIR}"
    echo "======================================================================"

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "WARNING: HF_TOKEN not set. CTSpine1K is gated; download may fail."
        echo "         Re-submit with:  HF_TOKEN=hf_xxx sbatch slurm/download_raw.sh"
    fi

    _run python3 - << 'PYEOF'
import os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

dest  = Path("/data/ctspine1k")
cache = Path("/data/hf_cache")
token = os.environ.get("HF_TOKEN") or None

# Minimum .nii.gz expected (~1568 = 784 img + 784 seg). Anything materially
# below this means the snapshot is incomplete (almost always HF rate-limiting).
EXPECTED_MIN = int(os.environ.get("CTSPINE1K_MIN_NII", "1500"))
# Lower concurrency keeps us under HF's 1000-requests/5min API quota even
# unauthenticated; authenticated tokens get a much higher limit.
MAX_WORKERS  = int(os.environ.get("HF_MAX_WORKERS", "4"))

print(f"Downloading alexanderdann/CTSpine1K -> {dest}", flush=True)
if not token:
    print("WARNING: no HF_TOKEN — unauthenticated requests are throttled at "
          "1000/5min and may not complete in one pass.", flush=True)

last_err = None
for attempt in range(1, 21):
    try:
        snapshot_download(
            repo_id    = "alexanderdann/CTSpine1K",
            repo_type  = "dataset",
            local_dir  = str(dest),
            cache_dir  = str(cache),
            token      = token,
            ignore_patterns = ["*.arrow", "*.parquet", "data/*.arrow"],
            max_workers = MAX_WORKERS,
        )
    except Exception as e:
        last_err = e
        print(f"[attempt {attempt}] download error: {e}", flush=True)
    else:
        # snapshot_download silently returns the (possibly stale/partial)
        # local dir when the repo can't be reached — e.g. HTTP 429 — instead
        # of raising. The only trustworthy completion signal is file count.
        n = len(list(dest.rglob("*.nii.gz")))
        if n >= EXPECTED_MIN:
            print(f"Done.  .nii.gz files: {n}", flush=True)
            sys.exit(0)
        print(f"[attempt {attempt}] incomplete: {n} .nii.gz "
              f"(< {EXPECTED_MIN}) — likely rate-limited; will retry", flush=True)

    if attempt == 20:
        print(f"  giving up after 20 attempts (last error: {last_err})", flush=True)
        sys.exit(1)
    # 429 quota is per-5-min, so back off long enough to clear the window.
    wait = min(600, 60 * attempt)
    print(f"  sleeping {wait}s and retrying (resumes from cache) ...", flush=True)
    time.sleep(wait)

sys.exit(1)
PYEOF

    _mark_done "${CTSPINE1K_DIR}"
    echo "  CTSpine1K done.  NIfTIs: $(find ${CTSPINE1K_DIR} -name '*.nii.gz' 2>/dev/null | wc -l)"
fi

# =============================================================================
# 3/3: CTPelvic1K (Zenodo + HuggingFace metadata)
# =============================================================================
if [[ "${RUN_PELVIC}" == "1" ]] && _stage_done "${CTPELVIC1K_DIR}"; then
    echo ""
    echo " 3/3  CTPelvic1K  —  already complete, skipping"
    echo "      (FORCE=1 to re-download)"
    RUN_PELVIC=0
fi
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

    _mark_done "${CTPELVIC1K_DIR}"
    echo "  CTPelvic1K done."
    for DS in 1 2 3 4 5; do
        DIR="${CTPELVIC1K_DIR}/masks/CTPelvic1K_dataset${DS}_mask_mappingback"
        count=0
        if [[ -d "${DIR}" ]]; then
            count=$(find "${DIR}" -name '*.nii.gz' 2>/dev/null | wc -l)
        fi
        printf "    dataset%-2s masks : %4d\n" "${DS}" "${count}"
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
