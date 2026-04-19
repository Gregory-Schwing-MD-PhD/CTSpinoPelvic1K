#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_create_dataset
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/create_dataset_%j.out
#SBATCH --error=logs/create_dataset_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 2 — create dataset
#
# Reads raw data from Stage 1 and produces:
#   Step A  build_db.py           → data/patient_db.json   (canonical DB)
#   Step B  place_fused_masks.py  → data/placed/*.nii.gz   (placed masks)
#                                 → data/placed/placed_manifest.json
#   Step C  visualize_qc.py       → data/qc_figures/       (optional, per-case)
#
# patient_db.json REPLACES colonog_matched_pairs.json as the source of truth.
# Mask-to-series resolution is patient-anchored (DICOM PatientID equality),
# not affine-based.  place_fused_masks.py has been adapted to consume
# patient_db.json directly.
#
# Usage:
#   make create-dataset                                           # full run
#   DEBUG_N=5        make create-dataset                          # first 5 patients
#   DEBUG_TOKENS="145,184,205"  make create-dataset               # specific tokens
#   SKIP_PLACE=1     make create-dataset                          # only build_db
#   SKIP_QC=1        make create-dataset                          # skip QC figures
#
# Next stage:
#   HF_TOKEN=hf_xxx make export-dataset PUSH=1
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Step toggles ─────────────────────────────────────────────────────────────
SKIP_BUILD_DB="${SKIP_BUILD_DB:-0}"
SKIP_PLACE="${SKIP_PLACE:-0}"
SKIP_QC="${SKIP_QC:-0}"
REBUILD_TCIA_INDEX="${REBUILD_TCIA_INDEX:-0}"
REBUILD_MASK_CACHE="${REBUILD_MASK_CACHE:-0}"

mkdir -p "${LOGS_DIR}" "${NIFTI_DIR}" \
         "${PLACED_DIR}/spine" "${PLACED_DIR}/pelvic" "${PLACED_DIR}/fused" \
         "${DATA_DIR}/qc_figures"

echo "======================================================================"
echo " Stage 2: Create dataset"
echo "   Job ID         : ${SLURM_JOB_ID:-local}"
echo "   Node           : $(hostname)"
echo "   CPUs           : ${WORKERS}"
echo "   TCIA dir       : ${TCIA_DIR}"
echo "   Spine root     : ${CTSPINE1K_DIR}"
echo "   Pelvis root    : ${CTPELVIC1K_DIR}"
echo "   PatientDB      : ${PATIENT_DB}"
echo "   Placed dir     : ${PLACED_DIR}"
echo "   DEBUG_N        : ${DEBUG_N}"
echo "   DEBUG_TOKENS   : ${DEBUG_TOKENS:-<none>}"
echo "   Started        : $(date)"
echo "======================================================================"

# ── Pre-flight ───────────────────────────────────────────────────────────────
PREFLIGHT_FAIL=0
if [[ ! -d "${TCIA_DIR}" ]]; then
    echo "ERROR: TCIA dir missing: ${TCIA_DIR} — run  make download-raw"
    PREFLIGHT_FAIL=1
fi
if [[ ! -d "${CTSPINE1K_DIR}" ]]; then
    echo "ERROR: CTSpine1K missing: ${CTSPINE1K_DIR}"
    PREFLIGHT_FAIL=1
fi
if [[ ! -d "${CTPELVIC1K_DIR}" ]]; then
    echo "ERROR: CTPelvic1K missing: ${CTPELVIC1K_DIR}"
    PREFLIGHT_FAIL=1
fi
[[ ${PREFLIGHT_FAIL} -eq 1 ]] && exit 1

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: container missing.  Run: make build-container"
    exit 1
fi

# ── Singularity runtime ──────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_stage2_${SLURM_JOB_ID:-$$}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export NUMEXPR_MAX_THREADS=2
export OMP_NUM_THREADS=2
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
PPATH="/workspace/scripts:/workspace/src:/workspace"

_run() {
    singularity exec \
        --env PYTHONPATH="${PPATH}" \
        --bind "${BINDS}" \
        --pwd /workspace \
        "${SIF_PATH}" "$@"
}

# ── Translate host paths to container paths ──────────────────────────────────
C_TCIA="/data/tcia"
C_SPINE="/data/ctspine1k"
C_PELVIC="/data/ctpelvic1k"
C_OUTDIR="/data"
C_NIFTI="/data/tcia_nifti"
C_PLACED="/data/placed"
C_PATIENT_DB="/data/patient_db.json"
C_MANIFEST="/data/placed/placed_manifest.json"
C_QC="/data/qc_figures"

# ── Flag construction ────────────────────────────────────────────────────────
TOKENS_ARG=""
DEBUG_ARG=""
if [[ -n "${DEBUG_TOKENS}" ]]; then
    TOKENS_ARG="--tokens ${DEBUG_TOKENS// /,}"
elif [[ "${DEBUG_N}" -gt 0 ]]; then
    DEBUG_ARG="--debug_n ${DEBUG_N}"
fi

BUILD_DB_FLAGS=""
[[ "${REBUILD_TCIA_INDEX}" == "1" ]] && BUILD_DB_FLAGS="${BUILD_DB_FLAGS} --rebuild_tcia_index"
[[ "${REBUILD_MASK_CACHE}" == "1" ]] && BUILD_DB_FLAGS="${BUILD_DB_FLAGS} --rebuild_mask_cache"
[[ "${DEBUG_N}"            -gt 0  ]] && BUILD_DB_FLAGS="${BUILD_DB_FLAGS} --debug_n ${DEBUG_N}"

# =============================================================================
# Step A: build_db.py  →  patient_db.json
# =============================================================================
if [[ "${SKIP_BUILD_DB}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step A: build_db.py → patient_db.json"
    echo "   Patient-anchored mask→series resolution (DICOM PatientID equality)"
    echo "======================================================================"

    _run python3 /workspace/scripts/build_db.py \
        --spine_root  "${C_SPINE}" \
        --pelvis_root "${C_PELVIC}" \
        --tcia_dir    "${C_TCIA}" \
        --out_dir     "${C_OUTDIR}" \
        --workers     "${WORKERS}" \
        ${BUILD_DB_FLAGS}

    if [[ -f "${PATIENT_DB}" ]]; then
        echo ""
        python3 - "${PATIENT_DB}" << 'PYEOF'
import json, sys
db = json.load(open(sys.argv[1]))
m  = db.get("metadata", {})
print(f"  Patients         : {m.get('n_patients','?')}")
print(f"  TCIA series      : {m.get('n_tcia_series_total','?')}")
print(f"  Spine masks      : {m.get('n_spine_masks','?')}")
print(f"  Pelvic masks     : {m.get('n_pelvic_masks','?')}")
print(f"  Complete (both)  : {m.get('n_complete_patients','?')}")
print(f"    fusion         : {m.get('n_fusion','?')}")
print(f"    separate       : {m.get('n_separate','?')}")
print(f"  Ambiguous        : {m.get('n_ambiguous_assignment','?')}")
print(f"  Unresolved       : {m.get('n_unresolved','?')}")
PYEOF
    else
        echo "ERROR: patient_db.json not produced."
        exit 1
    fi
else
    echo "  Step A skipped (SKIP_BUILD_DB=1)"
fi

# =============================================================================
# Step B: place_fused_masks.py  →  data/placed/*.nii.gz + placed_manifest.json
# =============================================================================
if [[ "${SKIP_PLACE}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step B: place_fused_masks.py → placed masks + manifest"
    echo "   Consumes patient_db.json (NOT colonog_matched_pairs.json)"
    echo "   Per-mask independent best-series selection by bone coverage"
    echo "======================================================================"

    _run python3 /workspace/scripts/place_fused_masks.py \
        --patient_db  "${C_PATIENT_DB}" \
        --spine_root  "${C_SPINE}" \
        --pelvis_root "${C_PELVIC}" \
        --tcia_dir    "${C_TCIA}" \
        --nifti_dir   "${C_NIFTI}" \
        --out_dir     "${C_PLACED}" \
        --workers     "${WORKERS}" \
        --dcm2niix_workers "${DCM2NIIX_WORKERS}" \
        ${TOKENS_ARG} \
        ${DEBUG_ARG}

    if [[ -f "${PLACED_MANIFEST}" ]]; then
        echo ""
        python3 - "${PLACED_MANIFEST}" << 'PYEOF'
import json, sys, statistics
m = json.load(open(sys.argv[1]))
print(f"  Manifest cases   : {m.get('n_cases','?')}")
print(f"    fused          : {m.get('n_fused','?')}")
print(f"    separate       : {m.get('n_separate','?')}")
print(f"    spine_only     : {m.get('n_spine_only','?')}")
print(f"    pelvic_only    : {m.get('n_pelvic_only','?')}")
cases = m.get("cases", [])
sbp = [c["spine"]["bone_pct"]  for c in cases if c.get("spine")  and c["spine"] .get("bone_pct") is not None]
pbp = [c["pelvic"]["bone_pct"] for c in cases if c.get("pelvic") and c["pelvic"].get("bone_pct") is not None]
if sbp:
    print(f"  Spine bone_pct   : mean={statistics.mean(sbp):.1f}%  min={min(sbp):.1f}%  low(<20%): {sum(1 for v in sbp if v<20)}")
if pbp:
    print(f"  Pelvic bone_pct  : mean={statistics.mean(pbp):.1f}%  min={min(pbp):.1f}%  low(<15%): {sum(1 for v in pbp if v<15)}")
PYEOF
    else
        echo "ERROR: placed_manifest.json not produced."
        exit 1
    fi
else
    echo "  Step B skipped (SKIP_PLACE=1)"
fi

# =============================================================================
# Step C: QC figures  (optional, but recommended)
# =============================================================================
if [[ "${SKIP_QC}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step C: visualize_qc.py → per-case QC figures"
    echo "======================================================================"

    QC_EXTRA=""
    [[ -n "${DEBUG_TOKENS}" ]] && QC_EXTRA="--tokens ${DEBUG_TOKENS// /,}"

    _run python3 /workspace/scripts/visualize_qc.py \
        --manifest   "${C_MANIFEST}" \
        --nifti_dir  "${C_NIFTI}" \
        --placed_dir "${C_PLACED}" \
        --out_dir    "${C_QC}" \
        --per_case \
        --workers    "${WORKERS}" \
        ${QC_EXTRA}

    N_QC=$(find "${DATA_DIR}/qc_figures" -name "*.png" 2>/dev/null | wc -l)
    echo "  QC figures produced: ${N_QC}"
else
    echo "  Step C skipped (SKIP_QC=1)"
fi

echo ""
echo "======================================================================"
echo " Stage 2 complete at $(date)"
echo ""
echo " Artifacts:"
for f in "${PATIENT_DB}" "${PLACED_MANIFEST}"; do
    if [[ -f "${f}" ]]; then
        printf "   %-40s  %s\n" "$(basename ${f})" "$(du -sh ${f} | cut -f1)"
    fi
done
echo "   Placed spine masks  : $(find ${PLACED_DIR}/spine  -name '*.nii.gz' 2>/dev/null | wc -l)"
echo "   Placed pelvic masks : $(find ${PLACED_DIR}/pelvic -name '*.nii.gz' 2>/dev/null | wc -l)"
echo "   QC figures          : $(find ${DATA_DIR}/qc_figures -name '*.png'    2>/dev/null | wc -l)"
echo ""
echo " Next stage:"
echo "   HF_TOKEN=hf_xxx make export-dataset PUSH=1"
echo "======================================================================"
