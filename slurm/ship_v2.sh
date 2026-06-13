#!/usr/bin/env bash
# =============================================================================
# ship_v2.sh — build + push BOTH releases in one shot (v1 on the way, then v2).
#
# v1 = the partial-annotation base (ALL configs + T12 anchor): the input that
#      trained the pseudolabeller. Built and pushed @v1 in step 1. UNCHANGED.
#
# v2 = the LSTV-segmenter training artifact, now built GT-FIRST:
#        (2) propagate_pelvis — carry each separate-cohort patient's OWN radiologist
#            pelvis GT across acquisitions onto their spine scan (deterministic
#            rigid registration), producing placed_manifest_propagated.json.   [CPU]
#        (3) pseudolabel — lay that propagated REAL pelvis GT onto the base, then
#            let the model COMPLETE only the bone the (sometimes partial) GT missed
#            (never overwriting GT); pelvic_native dropped. Writes the v2 tree +
#            propagated_completion_qc.csv (GT-vs-model Dice, completeness).   [GPU]
#        (4) QC — overlay each propagated pelvis on its spine CT, and aggregate the
#            placement/completion CSVs into dataset summary figures.          [CPU]
#        (5) push the v2 tree -> <repo>@v2.                                   [CPU]
#
# So a single run ships v1 AND v2. SKIP_BASE=1 reuses an existing base (skip step 1);
# SKIP_PROP=1 reuses an existing propagation (skip step 2).
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/ship_v2.sh
#
# DRY_RUN=1 plans the pseudolabel step (no inference); pair with SKIP_PROP=1 to also
# skip the (heavy) registration. Optional env: HF_WORKERS, HF_PRIVATE, PROP_MODE,
# COMPLETE_PROPAGATED, NNUNET_RESULTS, MODELS_CONFIG, MANIFEST_FILE.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
# This launcher runs on the login node (no SLURM_JOB_ID); default.env references
# it for the singularity tmpdir. Give it a placeholder so `set -u` + source is
# happy — the actual sbatch jobs get real SLURM_JOB_IDs.
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

: "${HF_TOKEN:?paste your token -> HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/ship_v2.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${PROJECT_ROOT}/nnunet/results}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"          # filtered base (scratch)
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"     # the v2 tree we push
NIFTI_DIR="${NIFTI_DIR:-${DATA_DIR}/tcia_nifti}"
PELVIC_DIR="${PELVIC_DIR:-${DATA_DIR}/placed/pelvic}"
SPINE_DIR="${SPINE_DIR:-${DATA_DIR}/placed/spine}"
PROP_OUT_DIR="${PROP_OUT_DIR:-${DATA_DIR}/placed/pelvic_propagated}"  # propagation output
DASH_OUT_DIR="${DASH_OUT_DIR:-${DATA_DIR}/qc_dashboard}"
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SKIP_QC="${SKIP_QC:-0}"
NO_PIR="${NO_PIR:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROP_MODE="${PROP_MODE:-production}"             # propagate_pelvis mode
COMPLETE_PROPAGATED="${COMPLETE_PROPAGATED:-1}"  # 1 = model completes GT-missed bone
# WIPE=1 (default): clear each target branch's files on HF before pushing it
# (v1 in step 1, v2 in step 5), so no stale files survive. Set WIPE=0 to skip.
WIPE="${WIPE:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
PLACED_MANIFEST="${DATA_DIR}/placed/${MANIFEST_FILE}"
# Inject a QOS / extra sbatch flags into EVERY job in the chain (e.g. so the whole
# pipeline runs on a queue you can actually get nodes on): SBATCH_QOS=secondary.
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"
SB="${SB} ${SBATCH_EXTRA:-}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ "${DRY_RUN}" == "1" || -f "${NNUNET_SIF}" ]] || {
    echo "ERROR: nnUNet container missing at ${NNUNET_SIF} (needed unless DRY_RUN=1)."
    echo "       set NNUNET_SIF=/path/to/ctspinopelvic1k-ts.sif"; exit 1; }

# Guards: force fresh v2 labels (so the pelvis fill is re-applied onto the
# anchored base) and, unless SKIP_BASE=1, fresh base labels too. NEVER touch
# *_work — those are the cached preds the pseudolabel step reuses (no GPU run).
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_PROP="${SKIP_PROP:-0}"
echo "[ship_v2] clearing stale v2 labels    ${PSEUDO_OUT_DIR}/{labels,qc,manifest.json}  (KEEPING ${PSEUDO_OUT_DIR}_work)"
rm -rf "${PSEUDO_OUT_DIR}"/labels "${PSEUDO_OUT_DIR}"/qc "${PSEUDO_OUT_DIR}"/manifest.json \
       "${PSEUDO_OUT_DIR}"/propagated_completion_qc.csv

# ---------------------------------------------------------------------------
# (1) Build the v1 BASE = ALL configs + anchor (NOT filtered). v2 is derived
# from this base; the fused+spine_only filter is applied at the pseudolabel step,
# not by re-exporting a filtered tree. Skip with SKIP_BASE=1.
BASE_DEP=""
if [[ "${SKIP_BASE}" == "1" ]]; then
    echo "[ship_v2] (1) SKIP_BASE=1 — reusing existing all-configs base at ${HF_EXPORT_DIR}"
    [[ -f "${HF_EXPORT_DIR}/manifest.json" ]] || { echo "ERROR: no base at ${HF_EXPORT_DIR}; run ship_v1 first or unset SKIP_BASE"; exit 1; }
else
    echo "[ship_v2] clearing stale base labels  ${HF_EXPORT_DIR}/{labels,qc,manifest.json}"
    rm -rf "${HF_EXPORT_DIR}"/labels "${HF_EXPORT_DIR}"/qc "${HF_EXPORT_DIR}"/manifest.json
    echo "[ship_v2] (1) export the v1 base (ALL configs + anchor) + PUSH @v1 [CPU]"
    J1=$(sbatch --parsable ${SB} \
      --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=0,SKIP_QC=${SKIP_QC},NO_PIR=${NO_PIR},WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v1,HF_EXPORT_DIR=${HF_EXPORT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
      slurm/export_dataset.sh)
    BASE_DEP=":${J1}"
fi

# ---------------------------------------------------------------------------
# (2) propagate_pelvis — carry the REAL pelvis GT across acquisitions onto the
# spine scans [CPU]. Independent of the export (reads placed_manifest), so it runs
# in PARALLEL with step 1. Output: ${PROP_OUT_DIR}/placed_manifest_propagated.json.
PROP_DEP=""
if [[ "${SKIP_PROP}" == "1" ]]; then
    echo "[ship_v2] (2) SKIP_PROP=1 — reusing existing propagation at ${PROP_OUT_DIR}"
    [[ -f "${PROP_OUT_DIR}/placed_manifest_propagated.json" ]] || { echo "ERROR: no propagation at ${PROP_OUT_DIR}; unset SKIP_PROP"; exit 1; }
else
    echo "[ship_v2] (2) propagate_pelvis (MODE=${PROP_MODE}) -> ${PROP_OUT_DIR} [CPU]"
    JPROP=$(sbatch --parsable ${SB} \
      --export=ALL,SIF_PATH=${SIF_PATH},MODE=${PROP_MODE},MANIFEST=${PLACED_MANIFEST},NIFTI_DIR=${NIFTI_DIR},PELVIC_DIR=${PELVIC_DIR},SPINE_DIR=${SPINE_DIR},PROP_OUT_DIR=${PROP_OUT_DIR} \
      slurm/propagate_pelvis.sh)
    PROP_DEP=":${JPROP}"
fi

# ---------------------------------------------------------------------------
# (3) pseudolabel — GT-first union: lay the propagated pelvis GT, then model
# COMPLETES the missed bone; DROP pelvic_native [GPU]. Depends on base + propagation.
export INCLUDE_CONFIGS="fused,spine_only"
PSEUDO_DEP="afterok${BASE_DEP}${PROP_DEP}"
[[ "${PSEUDO_DEP}" == "afterok" ]] && PSEUDO_DEP=""    # no deps (both skipped)
DEP_ARG=""; [[ -n "${PSEUDO_DEP}" ]] && DEP_ARG="--dependency=${PSEUDO_DEP}"
echo "[ship_v2] (3) pseudolabel: propagated GT + model-complete, keep ${INCLUDE_CONFIGS} (DRY_RUN=${DRY_RUN}) [GPU]  ${DEP_ARG:-no dep}"
J2=$(sbatch --parsable ${SB} ${DEP_ARG} \
  --export=ALL,SIF_PATH=${SIF_PATH},NNUNET_SIF=${NNUNET_SIF},NNUNET_RESULTS=${NNUNET_RESULTS},HF_EXPORT_DIR=${HF_EXPORT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},MODELS_CONFIG=${MODELS_CONFIG},DRY_RUN=${DRY_RUN},HF_TOKEN=${HF_TOKEN},PROPAGATED_DIR=${PROP_OUT_DIR},COMPLETE_PROPAGATED=${COMPLETE_PROPAGATED} \
  slurm/pseudolabel.sh)

# ---------------------------------------------------------------------------
# (4) QC — overlays of each propagated pelvis (after propagation) + dataset summary
# figures (after pseudolabel, which writes propagated_completion_qc.csv) [CPU].
if [[ "${SKIP_QC}" != "1" ]]; then
    VIZ_DEP=""; [[ -n "${PROP_DEP}" ]] && VIZ_DEP="--dependency=afterok${PROP_DEP}"
    echo "[ship_v2] (4a) viz_propagation (overlays) [CPU]  ${VIZ_DEP:-no dep}"
    JVIZ=$(sbatch --parsable ${SB} ${VIZ_DEP} \
      --export=ALL,SIF_PATH=${SIF_PATH},PROP_OUT_DIR=${PROP_OUT_DIR},NIFTI_DIR=${NIFTI_DIR} \
      slurm/viz_propagation.sh)
    echo "[ship_v2] (4b) qc_dashboard (figures) [CPU]  after ${J2}"
    JDASH=$(sbatch --parsable ${SB} --dependency=afterok:${J2} \
      --export=ALL,SIF_PATH=${SIF_PATH},PROP_OUT_DIR=${PROP_OUT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},DASH_OUT_DIR=${DASH_OUT_DIR} \
      slurm/qc_dashboard.sh)
fi

# ---------------------------------------------------------------------------
# (5) push the v2 tree (export step skipped — it already exists from step 3) [CPU].
echo "[ship_v2] (5) push ${PSEUDO_OUT_DIR} -> ${HF_REPO_ID}@v2 [CPU]  after ${J2}"
J3=$(sbatch --parsable ${SB} --dependency=afterok:${J2} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v2,HF_EXPORT_DIR=${PSEUDO_OUT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

echo "[ship_v2] submitted:"
echo "[ship_v2]   v1 build+push : ${J1:-<skipped>}"
echo "[ship_v2]   propagate     : ${JPROP:-<skipped>}"
echo "[ship_v2]   pseudolabel   : ${J2}   (GT-first union)"
echo "[ship_v2]   qc viz/dash   : ${JVIZ:-<skipped>} / ${JDASH:-<skipped>}"
echo "[ship_v2]   v2 push       : ${J3}"
echo "[ship_v2]   monitor       : tail -f logs/*${J2}* logs/*${J3}*"
echo "[ship_v2]   v2 pelvis = propagated REAL GT, model completes only the missed bone."
