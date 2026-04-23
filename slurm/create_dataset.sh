#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_create_dataset
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/create_dataset_%j.out
#SBATCH --error=logs/create_dataset_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Stage 2 — create dataset  (v4: manual flip list, no heuristic detector)
#
# Reads raw data from Stage 1 and produces:
#   Step A  build_db.py           → data/patient_db.json
#   Step B  place_fused_masks.py  → data/placed/*.nii.gz + placed_manifest.json
#   Step C  apply_manual_flips.py → data/placed/*_orientation_fixed.nii.gz
#                                 → data/placed/placed_manifest_orientation_fixed.json
#                                   (only tokens listed in configs/flip_list.json)
#   Step D  visualize_qc.py       → data/qc_figures/        (QC on originals)
#   Step E  visualize_qc.py       → data/qc_figures_orifix/ (QC on flipped manifest)
#
# The AP-orientation heuristic (orientation_fix.py) is NOT run. Instead,
# reviewers populate configs/flip_list.json with tokens whose CT + masks need
# flipping, and Step C flips exactly those listed tokens via world-space
# resampling. Tokens not listed are left untouched.
#
# Workflow:
#   1. Run this script once with an empty configs/flip_list.json "flips" list
#   2. Inspect data/qc_figures/ to identify tokens needing flips
#   3. Add entries to configs/flip_list.json for each
#   4. Re-run this script — Step C picks up the new entries and re-generates
#      placed_manifest_orientation_fixed.json accordingly
#   5. Inspect data/qc_figures_orifix/ to confirm
#
# Usage:
#   make create-dataset                                           # full run
#   DEBUG_N=5        make create-dataset                          # first 5 patients
#   DEBUG_TOKENS="145,184,205"  make create-dataset               # specific tokens
#   SKIP_PLACE=1     make create-dataset                          # only build_db
#   SKIP_MANUAL_FLIPS=1  make create-dataset                      # skip Step C
#   SKIP_QC=1        make create-dataset                          # skip QC figures
#   SKIP_QC_ORIFIX=1 make create-dataset                          # skip Step E only
# =============================================================================

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

# ── Step toggles ─────────────────────────────────────────────────────────────
SKIP_BUILD_DB="${SKIP_BUILD_DB:-0}"
SKIP_PLACE="${SKIP_PLACE:-0}"
SKIP_MANUAL_FLIPS="${SKIP_MANUAL_FLIPS:-0}"
SKIP_QC="${SKIP_QC:-0}"
SKIP_QC_ORIFIX="${SKIP_QC_ORIFIX:-0}"
REBUILD_TCIA_INDEX="${REBUILD_TCIA_INDEX:-0}"
REBUILD_MASK_CACHE="${REBUILD_MASK_CACHE:-0}"

# Flip list location (on the host). Default lives under configs/.
FLIP_LIST="${FLIP_LIST:-${PROJECT_ROOT}/configs/flip_list.json}"

mkdir -p "${LOGS_DIR}" "${NIFTI_DIR}" \
         "${PLACED_DIR}/spine" "${PLACED_DIR}/pelvic" "${PLACED_DIR}/fused" \
         "${DATA_DIR}/qc_figures" "${DATA_DIR}/qc_figures_orifix"

echo "======================================================================"
echo " Stage 2: Create dataset  (manual-flip workflow)"
echo "   Job ID         : ${SLURM_JOB_ID:-local}"
echo "   Node           : $(hostname)"
echo "   CPUs           : ${WORKERS}"
echo "   TCIA dir       : ${TCIA_DIR}"
echo "   Spine root     : ${CTSPINE1K_DIR}"
echo "   Pelvis root    : ${CTPELVIC1K_DIR}"
echo "   PatientDB      : ${PATIENT_DB}"
echo "   Placed dir     : ${PLACED_DIR}"
echo "   Flip list      : ${FLIP_LIST}"
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

# ── Container-side paths ─────────────────────────────────────────────────────
C_TCIA="/data/tcia"
C_SPINE="/data/ctspine1k"
C_PELVIC="/data/ctpelvic1k"
C_OUTDIR="/data"
C_NIFTI="/data/tcia_nifti"
C_PLACED="/data/placed"
C_PATIENT_DB="/data/patient_db.json"
C_MANIFEST="/data/placed/placed_manifest.json"
C_MANIFEST_ORIFIX="/data/placed/placed_manifest_orientation_fixed.json"
C_FLIP_LIST="/workspace/configs/flip_list.json"
C_QC="/data/qc_figures"
C_QC_ORIFIX="/data/qc_figures_orifix"

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
# Step B: place_fused_masks.py  →  placed_manifest.json
# =============================================================================
if [[ "${SKIP_PLACE}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step B: place_fused_masks.py → placed masks + manifest"
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
    print(f"  Spine bone_pct   : mean={statistics.mean(sbp):.1f}%  min={min(sbp):.1f}%")
if pbp:
    print(f"  Pelvic bone_pct  : mean={statistics.mean(pbp):.1f}%  min={min(pbp):.1f}%")
PYEOF
    else
        echo "ERROR: placed_manifest.json not produced."
        exit 1
    fi
else
    echo "  Step B skipped (SKIP_PLACE=1)"
fi

# =============================================================================
# Step C: apply_manual_flips.py  →  placed_manifest_orientation_fixed.json
# =============================================================================
# Reads configs/flip_list.json — ONLY tokens listed there get flipped.
# Non-listed tokens pass through with explicit ct_nifti and series_uid
# paths so downstream (export_hf.py, visualize_qc.py) can consume this
# single manifest without special-casing.
# =============================================================================
if [[ "${SKIP_MANUAL_FLIPS}" != "1" ]]; then
    if [[ ! -f "${PLACED_MANIFEST}" ]]; then
        echo ""
        echo "  Step C skipped: placed_manifest.json missing (did Step B run?)"
    elif [[ ! -f "${FLIP_LIST}" ]]; then
        echo ""
        echo "  Step C skipped: flip list not found at ${FLIP_LIST}"
        echo "  (create configs/flip_list.json to enable manual flips)"
    else
        echo ""
        echo "======================================================================"
        echo " Step C: apply_manual_flips.py → AP flips from curated review list"
        echo "   Flip list : ${FLIP_LIST}"
        python3 - "${FLIP_LIST}" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
flips = d.get("flips", [])
print(f"   {len(flips)} token(s) listed")
for f in flips[:25]:
    tok    = f.get("token", "?")
    rev    = f.get("reviewer", "?")
    dt     = f.get("date", "?")
    notes  = (f.get("notes", "") or "").replace("\n", " ")
    if len(notes) > 80:
        notes = notes[:77] + "..."
    print(f"     token={tok:<14}  reviewer={rev:<24}  date={dt:<12}  {notes}")
if len(flips) > 25:
    print(f"     ... and {len(flips) - 25} more")
PYEOF
        echo "======================================================================"

        _run python3 /workspace/scripts/apply_manual_flips.py \
            --manifest   "${C_MANIFEST}" \
            --flip_list  "${C_FLIP_LIST}" \
            --nifti_dir  "${C_NIFTI}" \
            --placed_dir "${C_PLACED}" \
            --workers    "${WORKERS}"

        ORIFIX_MANIFEST="${PLACED_DIR}/placed_manifest_orientation_fixed.json"
        if [[ -f "${ORIFIX_MANIFEST}" ]]; then
            echo ""
            python3 - "${ORIFIX_MANIFEST}" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  Total cases (output)  : {m.get('n_cases','?')}")
print(f"  Total cases (input)   : {m.get('n_cases_input','?')}")
print(f"  Flip requested        : {m.get('n_flip_requested','?')}")
print(f"  Flip missing          : {m.get('n_flip_missing','?')}   (listed but not in manifest)")
print(f"  Flipped OK            : {m.get('n_manually_flipped','?')}")
print(f"  Flip failed           : {m.get('n_flip_failed','?')}")
print(f"  Exclude requested     : {m.get('n_exclude_requested','?')}")
print(f"  Excluded (applied)    : {m.get('n_excluded','?')}")
print(f"  Exclude missing       : {m.get('n_exclude_missing','?')}   (listed but not in manifest)")
print(f"  Schema                : {m.get('schema_version','?')}")
excluded = m.get('excluded_tokens') or []
if excluded:
    print(f"  Excluded tokens       : {excluded}")
flipped_cases = [c for c in m.get("cases", [])
                 if (c.get("orientation_check") or {}).get("status") == "flipped"]
if flipped_cases:
    print("")
    print("  Flipped tokens — post-flip alignment:")
    for c in flipped_cases:
        oc = c.get("orientation_check") or {}
        tok = c.get("patient_token", "?")
        bp  = oc.get("post_flip_bone_pct")
        hu  = oc.get("post_flip_mean_hu")
        rev = oc.get("reviewer", "")
        sides = oc.get("sides") or []
        print(f"    token={str(tok):<12}  sides={str(sides):<22}  bone%={bp if bp is not None else '?':>5}  "
              f"mean_HU={hu if hu is not None else '?':>7}  reviewer={rev}")
PYEOF
        else
            echo "  WARNING: apply_manual_flips did not produce ${ORIFIX_MANIFEST}"
        fi
    fi
else
    echo ""
    echo "  Step C skipped (SKIP_MANUAL_FLIPS=1)"
fi

# =============================================================================
# Step D: QC figures on the ORIGINAL manifest
# =============================================================================
if [[ "${SKIP_QC}" != "1" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step D: visualize_qc.py → QC on ORIGINAL manifest (pre-flip)"
    echo "   Output dir : ${DATA_DIR}/qc_figures"
    echo "======================================================================"

    if [[ -d "${DATA_DIR}/qc_figures" ]]; then
        echo "  Clearing stale figures in ${DATA_DIR}/qc_figures/"
        find "${DATA_DIR}/qc_figures" -type f -name '*.png' -delete 2>/dev/null || true
    fi

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
    echo "  QC figures (original) produced: ${N_QC}"
else
    echo "  Step D skipped (SKIP_QC=1)"
fi

# =============================================================================
# Step E: QC figures on the ORIENTATION-FIXED manifest
# =============================================================================
ORIFIX_MANIFEST="${PLACED_DIR}/placed_manifest_orientation_fixed.json"

if [[ "${SKIP_QC}" != "1" && "${SKIP_QC_ORIFIX}" != "1" \
      && -f "${ORIFIX_MANIFEST}" ]]; then
    echo ""
    echo "======================================================================"
    echo " Step E: visualize_qc.py → QC on ORIENTATION-FIXED manifest (post-flip)"
    echo "   Manifest   : placed_manifest_orientation_fixed.json"
    echo "   Output dir : ${DATA_DIR}/qc_figures_orifix"
    echo "======================================================================"

    if [[ -d "${DATA_DIR}/qc_figures_orifix" ]]; then
        echo "  Clearing stale figures in ${DATA_DIR}/qc_figures_orifix/"
        find "${DATA_DIR}/qc_figures_orifix" -type f -name '*.png' -delete 2>/dev/null || true
    fi

    QC_EXTRA=""
    [[ -n "${DEBUG_TOKENS}" ]] && QC_EXTRA="--tokens ${DEBUG_TOKENS// /,}"

    _run python3 /workspace/scripts/visualize_qc.py \
        --manifest   "${C_MANIFEST_ORIFIX}" \
        --nifti_dir  "${C_NIFTI}" \
        --placed_dir "${C_PLACED}" \
        --out_dir    "${C_QC_ORIFIX}" \
        --per_case \
        --workers    "${WORKERS}" \
        ${QC_EXTRA}

    N_QC_ORIFIX=$(find "${DATA_DIR}/qc_figures_orifix" -name "*.png" 2>/dev/null | wc -l)
    echo "  QC figures (orientation-fixed) produced: ${N_QC_ORIFIX}"
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
echo "   QC figures (original)        : $(find ${DATA_DIR}/qc_figures -name '*.png' 2>/dev/null | wc -l)"
echo "   QC figures (orientation-fix) : $(find ${DATA_DIR}/qc_figures_orifix -name '*.png' 2>/dev/null | wc -l)"
echo ""
echo " Manual-flip review workflow:"
echo "   1. eyeball ${DATA_DIR}/qc_figures/ for AP-inverted cases"
echo "   2. add those tokens to ${FLIP_LIST}"
echo "   3. re-run (Steps C + E will pick up the new entries)"
echo ""
echo " Next stage:"
echo "   HF_TOKEN=hf_xxx make export-dataset PUSH=1"
echo "   (MANIFEST_FILE defaults to placed_manifest_orientation_fixed.json)"
echo "======================================================================"
