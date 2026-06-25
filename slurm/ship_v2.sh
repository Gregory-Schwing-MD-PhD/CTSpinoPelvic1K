#!/usr/bin/env bash
# =============================================================================
# ship_v2.sh — build + push BOTH releases in one shot (v1 on the way, then v2).
#
# v1 = the partial-annotation base (ALL configs + T12 anchor): the input that
#      trained the pseudolabeller. Built and pushed @v1 in step 1. UNCHANGED.
#
# v2 = GROUND-TRUTH spines + MODEL-pseudolabelled pelves:
#        (2) pseudolabel — keep the radiologist spine GT, and on spine_only
#            (separate-mode) scans let the 5-fold nnU-Net ENSEMBLE fill the pelvis
#            the spine annotator never traced. Manual voxels are never overwritten;
#            fused cases pass through unchanged; the ~3 pelvic_native cases are
#            DROPPED. NO cross-acquisition registration (propagation removed).  [GPU]
#        (3) QC — aggregate the completion CSV into dataset summary figures.    [CPU]
#        (4) push the v2 tree -> <repo>@v2 (manifest carries Castellvi grades).  [CPU]
#
# So a single run ships v1 AND v2. SKIP_BASE=1 reuses an existing base (skip step 1).
# Ribs are a v3 concern (see ship_v3.sh) — not built here.
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/ship_v2.sh
#
# DRY_RUN=1 plans the pseudolabel step (no inference). Optional env: HF_WORKERS,
# HF_PRIVATE, NNUNET_RESULTS, MODELS_CONFIG, MANIFEST_FILE.
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
DASH_OUT_DIR="${DASH_OUT_DIR:-${DATA_DIR}/qc_dashboard}"
HF_WORKERS="${HF_WORKERS:-8}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SKIP_QC="${SKIP_QC:-0}"
NO_PIR="${NO_PIR:-0}"
DRY_RUN="${DRY_RUN:-0}"
# WIPE=1 (default): clear each target branch's files on HF before pushing it
# (v1 in step 1, v2 in step 4), so no stale files survive. Set WIPE=0 to skip.
WIPE="${WIPE:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}"
# Inject a QOS / extra sbatch flags into EVERY job in the chain (e.g. so the whole
# pipeline runs on a queue you can actually get nodes on): SBATCH_QOS=secondary.
SB=""; [[ -n "${SBATCH_QOS:-}" ]] && SB="-q ${SBATCH_QOS}"
SB="${SB} ${SBATCH_EXTRA:-}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: project container missing at ${SIF_PATH}"; exit 1; }
[[ "${DRY_RUN}" == "1" || -f "${NNUNET_SIF}" ]] || {
    echo "ERROR: nnUNet container missing at ${NNUNET_SIF} (needed unless DRY_RUN=1)."
    echo "       set NNUNET_SIF=/path/to/ctspinopelvic1k-ts.sif"; exit 1; }

# v1 base: AUTO-REUSE if it already exists (already built + pushed @v1) — don't wipe
# or redo it, just proceed to v2. Only (re)build it when it's missing. Force either
# way with SKIP_BASE=1 (reuse) / SKIP_BASE=0 (rebuild + re-push v1). v2/v3 always
# rebuild + wipe-push regardless. NEVER touch *_work (cached preds).
if [[ -z "${SKIP_BASE:-}" ]]; then
    if [[ -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
        SKIP_BASE=1; echo "[ship_v2] v1 base present -> reuse (auto; SKIP_BASE=0 to rebuild)"
    else
        SKIP_BASE=0; echo "[ship_v2] no v1 base -> will build + push @v1"
    fi
fi
# NUKE=1 wipes EVERYTHING, so the base must be rebuilt regardless of SKIP_BASE.
[[ "${NUKE:-0}" == "1" ]] && { SKIP_BASE=0; echo "[ship_v2] NUKE=1 -> forcing SKIP_BASE=0 (base rebuilt)"; }

# (0) NUKE=1: wipe ALL hf_export* (trees + _work resume markers/pred caches) as a
# SLURM job, so the login node does NOT block on a big NFS delete. The whole DAG is
# chained afterok the wipe, so nothing old can leak into the clean VerSe-native rebuild.
WIPE_DEP_ARG=""
if [[ "${NUKE:-0}" == "1" ]]; then
    echo "[ship_v2] (0) NUKE=1 — wiping ${DATA_DIR}/hf_export* via SLURM (no login-node wait) [CPU]"
    J0=$(sbatch --parsable ${SB} --export=ALL,DATA_DIR=${DATA_DIR} slurm/wipe_exports.sh)
    WIPE_DEP_ARG="--dependency=afterok:${J0}"
    echo "[ship_v2] wipe job: ${J0}"
else
    echo "[ship_v2] clearing stale v2 labels    ${PSEUDO_OUT_DIR}/{labels,qc,manifest.json}  (KEEPING ${PSEUDO_OUT_DIR}_work)"
    rm -rf "${PSEUDO_OUT_DIR}"/labels "${PSEUDO_OUT_DIR}"/qc "${PSEUDO_OUT_DIR}"/manifest.json \
           "${PSEUDO_OUT_DIR}"/propagated_completion_qc.csv
fi

# ---------------------------------------------------------------------------
# (1) Build the v1 BASE = ALL configs + anchor (NOT filtered). v2 is derived
# from this base; the fused+spine_only filter is applied at the pseudolabel step,
# not by re-exporting a filtered tree. Skip with SKIP_BASE=1.
BASE_DEP=""
if [[ "${SKIP_BASE}" == "1" ]]; then
    echo "[ship_v2] (1) SKIP_BASE=1 — reusing existing all-configs base at ${HF_EXPORT_DIR}"
    [[ -f "${HF_EXPORT_DIR}/manifest.json" ]] || { echo "ERROR: no base at ${HF_EXPORT_DIR}; run ship_v1 first or unset SKIP_BASE"; exit 1; }
else
    if [[ "${NUKE:-0}" != "1" ]]; then        # under NUKE the wipe job owns deletion
        echo "[ship_v2] clearing stale base labels  ${HF_EXPORT_DIR}/{labels,qc,manifest.json}"
        rm -rf "${HF_EXPORT_DIR}"/labels "${HF_EXPORT_DIR}"/qc "${HF_EXPORT_DIR}"/manifest.json
    fi
    echo "[ship_v2] (1) export the v1 base (ALL configs + anchor) + PUSH @v1 [CPU]  ${WIPE_DEP_ARG:-no wipe dep}"
    J1=$(sbatch --parsable ${SB} ${WIPE_DEP_ARG} \
      --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=0,SKIP_QC=${SKIP_QC},NO_PIR=${NO_PIR},WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v1,HF_EXPORT_DIR=${HF_EXPORT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
      slurm/export_dataset.sh)
    BASE_DEP=":${J1}"
fi

# ---------------------------------------------------------------------------
# (2) pseudolabel — keep GT spines; MODEL-complete the pelvis on spine_only;
# DROP pelvic_native; fused passes through. USE_PROPAGATED=0 => no registration,
# pure model pelves [GPU]. Depends on the base export.
export INCLUDE_CONFIGS="fused,spine_only"
PSEUDO_DEP="afterok${BASE_DEP}"
[[ "${PSEUDO_DEP}" == "afterok" ]] && PSEUDO_DEP=""    # no dep (base skipped)
DEP_ARG=""; [[ -n "${PSEUDO_DEP}" ]] && DEP_ARG="--dependency=${PSEUDO_DEP}"

PSEUDO_EXPORT="SIF_PATH=${SIF_PATH},NNUNET_SIF=${NNUNET_SIF},NNUNET_RESULTS=${NNUNET_RESULTS},HF_EXPORT_DIR=${HF_EXPORT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},MODELS_CONFIG=${MODELS_CONFIG},DRY_RUN=${DRY_RUN},HF_TOKEN=${HF_TOKEN},USE_PROPAGATED=0"

if [[ "${DRY_RUN}" == "1" ]]; then
    # plan-only: single job, no sharding / assembly.
    echo "[ship_v2] (2) pseudolabel PLAN (DRY_RUN) keep ${INCLUDE_CONFIGS} [CPU]  ${DEP_ARG:-no dep}"
    J2=$(sbatch --parsable ${SB} ${DEP_ARG} --export=ALL,${PSEUDO_EXPORT} slurm/pseudolabel.sh)
    PSEUDO_FINAL="${J2}"
else
    # SHARDED like ship_v3's TS pass: an --array of GPU workers each predicting a
    # disjoint 1/N of the cases (split by index %% N in pseudolabel.py); predictions +
    # labels + resume markers land in the shared v2 tree on NFS. The assembly pass (2b)
    # then REUSES every shard's markers — no re-inference — and writes the manifest.
    PSEUDO_SHARDS="${PSEUDO_SHARDS:-8}"
    PSEUDO_CONCURRENT="${PSEUDO_CONCURRENT:-8}"
    echo "[ship_v2] (2) pseudolabel: GT spines + MODEL pelves — ${PSEUDO_SHARDS}-way array, %${PSEUDO_CONCURRENT} concurrent [GPU]  ${DEP_ARG:-no dep}"
    J2=$(sbatch --parsable ${SB} ${DEP_ARG} \
      --array=0-$((PSEUDO_SHARDS - 1))%${PSEUDO_CONCURRENT} \
      --export=ALL,${PSEUDO_EXPORT},N_SHARDS_OVERRIDE=${PSEUDO_SHARDS} \
      slurm/pseudolabel.sh)
    echo "[ship_v2] (2b) pseudolabel assemble (reduce) -> ${PSEUDO_OUT_DIR}/manifest.json [CPU]  after all ${PSEUDO_SHARDS} shards of ${J2}"
    J2B=$(sbatch --parsable ${SB} --dependency=afterok:${J2} \
      --export=ALL,SIF_PATH=${SIF_PATH},HF_EXPORT_DIR=${HF_EXPORT_DIR},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR},MODELS_CONFIG=${MODELS_CONFIG},NNUNET_RESULTS=${NNUNET_RESULTS} \
      slurm/pseudolabel_assemble.sh)
    PSEUDO_FINAL="${J2B}"
fi

# ---------------------------------------------------------------------------
# (3) QC triage — score every model-pseudolabelled pelvis on anatomical plausibility
# and write qc/pseudo_pelvis_triage.csv INTO the v2 tree so it ships with the push
# (review worst-first). Non-fatal: the job always exits 0, so it never blocks the
# push. Runs after pseudolabel (needs the v2 labels). [CPU]
QC_DEP=":${PSEUDO_FINAL}"
if [[ "${SKIP_QC}" != "1" ]]; then
    echo "[ship_v2] (3) qc_pseudo_pelvis triage -> ${PSEUDO_OUT_DIR}/qc [CPU]  after ${PSEUDO_FINAL}"
    JQC=$(sbatch --parsable ${SB} --dependency=afterok:${PSEUDO_FINAL} \
      --export=ALL,SIF_PATH=${SIF_PATH},PSEUDO_OUT_DIR=${PSEUDO_OUT_DIR} \
      slurm/qc_pseudo_pelvis.sh)
    QC_DEP=":${JQC}"      # push waits for the CSV so it ships inside v2
fi

# ---------------------------------------------------------------------------
# (4) push the v2 tree (export step skipped — it already exists from step 2) [CPU].
# (The old propagation QC dashboard is gone with propagation; the pseudolabel step
# writes its own per-case completion QC into the v2 tree.)
echo "[ship_v2] (4) push ${PSEUDO_OUT_DIR} -> ${HF_REPO_ID}@v2 [CPU]  after${QC_DEP}"
J3=$(sbatch --parsable ${SB} --dependency=afterok${QC_DEP} \
  --export=ALL,SIF_PATH=${SIF_PATH},PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=${WIPE},HF_TOKEN=${HF_TOKEN},HF_REPO_ID=${HF_REPO_ID},HF_REVISION=v2,HF_EXPORT_DIR=${PSEUDO_OUT_DIR},HF_WORKERS=${HF_WORKERS},HF_PRIVATE=${HF_PRIVATE},MANIFEST_FILE=${MANIFEST_FILE} \
  slurm/export_dataset.sh)

# Emit the terminal push job id so a parent launcher can chain v3 onto it.
echo "V2_PUSH_JOB=${J3}"
echo "[ship_v2] submitted:"
echo "[ship_v2]   wipe exports  : ${J0:-<skipped>}   (NUKE=1 -> rm hf_export* on a compute node)"
echo "[ship_v2]   v1 build+push : ${J1:-<skipped>}"
echo "[ship_v2]   pseudolabel   : ${J2}   (GT spines + MODEL pelves${J2B:+, ${PSEUDO_SHARDS:-?}-way array})"
echo "[ship_v2]   assemble      : ${J2B:-<single run>}   (reduce: reuse markers -> manifest)"
echo "[ship_v2]   qc triage     : ${JQC:-<skipped>}"
echo "[ship_v2]   v2 push       : ${J3}   (completeness-gated: aborts if labels/CTs missing)"
echo "[ship_v2]   monitor       : tail -f logs/*${J2}* logs/*${J3}*"
echo "[ship_v2]   v2 = radiologist spine GT + model-pseudolabelled pelves; pelvic_native dropped."
