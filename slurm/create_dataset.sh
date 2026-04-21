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
#   Step A  build_db.py          → data/patient_db.json   (canonical DB)
#   Step B  place_fused_masks.py → data/placed/*.nii.gz   (placed masks)
#                                → data/placed/placed_manifest.json
#   Step C  orientation_fix.py   → data/placed/*_orientation_fixed.nii.gz
#                                → data/placed/placed_manifest_orientation_fixed.json
#   Step D  visualize_qc.py      → data/qc_figures/       (optional, per-case)
#
# patient_db.json REPLACES colonog_matched_pairs.json as the source of truth.
# Mask-to-series resolution is patient-anchored (DICOM PatientID equality),
# not affine-based.  place_fused_masks.py has been adapted to consume
# patient_db.json directly.
#
# orientation_fix.py is a second-pass AP orientation consistency check run
# AFTER placement.  place_fused_masks.py always places masks on the original
# images; any AP-inverted scans detected here emit parallel files with the
# `_orientation_fixed` suffix alongside the originals.  Originals are never
# modified.  For fused cases, CT + spine mask + pelvic mask all flip together.
#
# Usage:
#   make create-dataset                                           # full run
#   DEBUG_N=5        make create-dataset                          # first 5 patients
#   DEBUG_TOKENS="145,184,205"  make create-dataset               # specific tokens
#   SKIP_PLACE=1     make create-dataset                          # only build_db
#   SKIP_ORIENTATION_FIX=1  make create-dataset                   # skip orientation pass
#   SKIP_QC=1        make create-dataset                          # skip QC figures
#   ORIENTATION_FIX_DRY_RUN=1 make create-dataset                 # detect only, no flips
#   ORIENTATION_FIX_THRESHOLD_MM=20 make create-dataset           # stricter threshold
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
SKIP_ORIENTATION_FIX="${SKIP_ORIENTATION_FIX:-0}"
SKIP_QC="${SKIP_QC:-0}"
REBUILD_TCIA_INDEX="${REBUILD_TCIA_INDEX:-0}"
REBUILD_MASK_CACHE="${REBUILD_MASK_CACHE:-0}"

# Orientation-fix knobs (v3: body-center detector + explicit ct_nifti paths
# written into the output manifest. No staging dir; viz reads paths directly.)
ORIENTATION_FIX_THRESHOLD_MM="${ORIENTATION_FIX_THRESHOLD_MM:-10.0}"
ORIENTATION_FIX_DRY_RUN="${ORIENTATION_FIX_DRY_RUN:-0}"
# Also run a second QC pass against the orientation-fixed manifest.
SKIP_QC_ORIFIX="${SKIP_QC_ORIFIX:-0}"

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
echo "   OrientFix thr  : ${ORIENTATION_FIX_THRESHOLD_MM} mm  (dry_run=${ORIENTATION_FIX_DRY_RUN})"
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
C_MANIFEST_ORIFIX="/data/placed/placed_manifest_orientation_fixed.json"
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
# Step C: orientation_fix.py  →  AP orientation consistency pass
# =============================================================================
#
# Runs AFTER placement.  Detects AP-inverted scans via sacrum vs combined hip
# centroids projected onto PIR axis 0.  Flagged cases get parallel files with
# the `_orientation_fixed` suffix next to the originals.  Originals are never
# modified.  For fused cases, CT + spine mask + pelvic mask flip together.
#
# Skip with SKIP_ORIENTATION_FIX=1.
# Run detection-only with ORIENTATION_FIX_DRY_RUN=1 (no files written).
# Tune sensitivity with ORIENTATION_FIX_THRESHOLD_MM (default 15.0).
# =============================================================================
if [[ "${SKIP_ORIENTATION_FIX}" != "1" ]]; then
    if [[ ! -f "${PLACED_MANIFEST}" ]]; then
        echo ""
        echo "  Step C skipped: placed_manifest.json missing (did Step B run?)"
    else
        echo ""
        echo "======================================================================"
        echo " Step C: orientation_fix.py → AP orientation consistency pass (v3)"
        echo "   Detects AP-inverted scans via pelvic mass vs body-bbox center"
        echo "   along world Y (body-center detector)."
        echo "   Threshold: |delta_posterior_mm| > ${ORIENTATION_FIX_THRESHOLD_MM}"
        echo "              → pelvic mass anterior of body middle = INVERTED"
        echo "   Flagged cases → parallel _orientation_fixed files next to originals."
        echo "   For fused cases, CT + spine mask + pelvic mask flip together."
        echo "   Output manifest carries explicit spine.ct_nifti and pelvic.ct_nifti"
        echo "   for every case (originals for OK cases, flipped paths for inverted)."
        echo "======================================================================"

        ORIFIX_FLAGS=""
        if [[ "${ORIENTATION_FIX_DRY_RUN}" == "1" ]]; then
            ORIFIX_FLAGS="--dry_run"
            echo "  DRY RUN — no _orientation_fixed files will be written"
        fi

        _run python3 /workspace/scripts/orientation_fix.py \
            --manifest     "${C_MANIFEST}" \
            --nifti_dir    "${C_NIFTI}" \
            --placed_dir   "${C_PLACED}" \
            --workers      "${WORKERS}" \
            --threshold_mm "${ORIENTATION_FIX_THRESHOLD_MM}" \
            ${ORIFIX_FLAGS}

        ORIFIX_MANIFEST="${PLACED_DIR}/placed_manifest_orientation_fixed.json"
        if [[ -f "${ORIFIX_MANIFEST}" ]]; then
            echo ""
            python3 - "${ORIFIX_MANIFEST}" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  Total cases          : {m.get('n_cases','?')}")
print(f"  AP ok                : {m.get('n_ap_ok','?')}")
print(f"  AP inverted          : {m.get('n_ap_inverted','?')}")
print(f"  AP indeterminate     : {m.get('n_ap_indeterminate','?')}")
print(f"  AP skipped (no pelv) : {m.get('n_ap_skipped','?')}")
print(f"  Detector             : {m.get('detector','?')}")
print(f"  Threshold (mm)       : {m.get('threshold_mm','?')}")
print(f"  Dry run              : {m.get('dry_run', False)}")
inv = [c for c in m.get("cases", [])
       if (c.get("orientation_check") or {}).get("status") == "inverted"]
if inv:
    def _key(c):
        oc = c.get("orientation_check") or {}
        return oc.get("delta_posterior_mm") or 0
    inv.sort(key=_key)   # most negative first = strongest inversion signal
    print("")
    print("  Inverted tokens (most negative delta_posterior_mm first):")
    for c in inv[:25]:
        oc = c.get("orientation_check") or {}
        n_files = len((c.get("orientation_fixed") or {}) or {})
        print(f"    token={str(c.get('patient_token','?')):<12}  "
              f"delta_post={oc.get('delta_posterior_mm',0):+7.1f} mm  "
              f"delta_sh={oc.get('delta_sacrum_hip_mm',0):+7.1f} mm  "
              f"match={c.get('match_type','?'):<10}  files_flipped={n_files}")
    if len(inv) > 25:
        print(f"    ... and {len(inv) - 25} more")
PYEOF
        else
            echo "  WARNING: orientation_fix did not produce ${ORIFIX_MANIFEST}"
        fi
    fi
else
    echo ""
    echo "  Step C skipped (SKIP_ORIENTATION_FIX=1)"
fi

# =============================================================================
# Step D: QC figures  (optional, but recommended)
#
# Uses placed_manifest.json (the original) by default.  QC figures therefore
# show the ORIGINAL placed masks — which is what you want for identifying
# inversions by eye.  To re-run QC on the orientation-fixed files:
#
#   _run python3 /workspace/scripts/visualize_qc.py \
#       --manifest /data/placed/placed_manifest_orientation_fixed.json \
#       --nifti_dir /data/tcia_nifti --placed_dir /data/placed \
#       --out_dir /data/qc_figures_orifix --per_case --workers 32
# =============================================================================
if [[ "${SKIP_QC}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step D: visualize_qc.py → per-case QC figures"
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
    echo "  Step D skipped (SKIP_QC=1)"
fi

# =============================================================================
# Step E: QC figures on the orientation-FIXED manifest  (optional)
#
# The v3 orientation-fixed manifest is self-contained: every case has
# explicit spine.ct_nifti and pelvic.ct_nifti pointing at the file to load
# (flipped for inverted cases, original otherwise). visualize_qc.py should
# read these paths directly — no UID-based fallback, no staging dir.
#
# REQUIRED viz change (one-line, if not already done):
#   ct_path = Path(case["spine"]["ct_nifti"])   # was: nifti_dir / f"{uid}.nii.gz"
#
# Skip with SKIP_QC_ORIFIX=1.
# =============================================================================
ORIFIX_MANIFEST="${PLACED_DIR}/placed_manifest_orientation_fixed.json"

if [[ "${SKIP_QC}" != "1" && "${SKIP_QC_ORIFIX}" != "1" \
      && -f "${ORIFIX_MANIFEST}" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step E: visualize_qc.py → QC on orientation-fixed manifest"
    echo "   Manifest   : placed_manifest_orientation_fixed.json"
    echo "   CT paths   : read from case.{spine,pelvic}.ct_nifti in manifest"
    echo "   Output dir : ${DATA_DIR}/qc_figures_orifix"
    echo "======================================================================"

    QC_EXTRA=""
    [[ -n "${DEBUG_TOKENS}" ]] && QC_EXTRA="--tokens ${DEBUG_TOKENS// /,}"

    _run python3 /workspace/scripts/visualize_qc.py \
        --manifest   /data/placed/placed_manifest_orientation_fixed.json \
        --nifti_dir  "${C_NIFTI}" \
        --placed_dir "${C_PLACED}" \
        --out_dir    /data/qc_figures_orifix \
        --per_case \
        --workers    "${WORKERS}" \
        ${QC_EXTRA}

    N_QC_ORIFIX=$(find "${DATA_DIR}/qc_figures_orifix" -name "*.png" 2>/dev/null | wc -l)
    echo "  Orientation-fixed QC figures produced: ${N_QC_ORIFIX}"
elif [[ "${SKIP_QC_ORIFIX}" == "1" ]]; then
    echo "  Step E skipped (SKIP_QC_ORIFIX=1)"
elif [[ ! -f "${ORIFIX_MANIFEST}" ]]; then
    echo "  Step E skipped (orientation-fixed manifest not found)"
fi

echo ""
echo "======================================================================"
echo " Stage 2 complete at $(date)"
echo ""
echo " Artifacts:"
for f in "${PATIENT_DB}" "${PLACED_MANIFEST}" \
         "${PLACED_DIR}/placed_manifest_orientation_fixed.json"; do
    if [[ -f "${f}" ]]; then
        printf "   %-45s  %s\n" "$(basename ${f})" "$(du -sh ${f} | cut -f1)"
    fi
done
echo "   Placed spine masks           : $(find ${PLACED_DIR}/spine  -name '*_seg_placed.nii.gz'    ! -name '*_orientation_fixed*' 2>/dev/null | wc -l)"
echo "   Placed pelvic masks          : $(find ${PLACED_DIR}/pelvic -name '*_pelvic_placed.nii.gz' ! -name '*_orientation_fixed*' 2>/dev/null | wc -l)"
echo "   Orientation-fixed spine      : $(find ${PLACED_DIR}/spine  -name '*_orientation_fixed.nii.gz' 2>/dev/null | wc -l)"
echo "   Orientation-fixed pelvic     : $(find ${PLACED_DIR}/pelvic -name '*_orientation_fixed.nii.gz' 2>/dev/null | wc -l)"
echo "   Orientation-fixed CT         : $(find ${NIFTI_DIR}        -name '*_orientation_fixed.nii.gz' 2>/dev/null | wc -l)"
echo "   QC figures                   : $(find ${DATA_DIR}/qc_figures -name '*.png' 2>/dev/null | wc -l)"
echo "   QC figures (orientation-fix) : $(find ${DATA_DIR}/qc_figures_orifix -name '*.png' 2>/dev/null | wc -l)"
echo ""
echo " Next stage:"
echo "   HF_TOKEN=hf_xxx make export-dataset PUSH=1"
echo "   (point export_hf.py at placed_manifest_orientation_fixed.json to use fixed files)"
echo "======================================================================"
