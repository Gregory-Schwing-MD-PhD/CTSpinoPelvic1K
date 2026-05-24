#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_pseudolabel
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/pseudolabel_%j.out
#SBATCH --error=logs/pseudolabel_%j.err
#SBATCH --mail-type=END,FAIL
# msa1's tmpdir hangs the SIF->sandbox conversion (job 36124058 sat 12h with
# no output before Python ever started). Keep off it; the conversion below is
# also redirected to node-local /tmp, but exclude the node that has stalled.
#SBATCH --exclude=msa1

# =============================================================================
# Pseudo-label completion — builds a FULL v2 tree from a staged v1 export.
#
# Completes spine_only / pelvic_native (separate-mode) records by filling the
# MISSING region with a 5-fold nnU-Net ensemble. Manual voxels are never
# overwritten. fused cases pass through unchanged. See scripts/pseudolabel.py.
#
# This NEVER touches HuggingFace. Publishing the result is a separate,
# explicit step (the v2 goes to a BRANCH so the reviewed main URL is safe):
#   make hf-push HF_REPO_ID=org/Name HF_REVISION=v2 \
#       HF_EXPORT_DIR=$(pwd)/data/hf_export_v2     # + HF_TOKEN=hf_xxx
#
# Prereqs:
#   * `make hf-stage` already produced data/hf_export/ (the v1 tree).
#   * configs/pseudolabel_models.json has the relevant model(s) enabled
#     with final checkpoint identity + label_remap (model is in flux —
#     disabled models are skipped, their records left partial, not faked).
#   * An nnU-Net inference runtime. nnU-Net is NOT in the project container,
#     so point NNUNET_SIF at a container that has nnU-Net v2 + CUDA torch,
#     and NNUNET_RESULTS at the trained nnUNet_results root.
#
# Usage:
#   NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif sbatch slurm/pseudolabel.sh
#   DRY_RUN=1 sbatch slurm/pseudolabel.sh          # plan only, no inference
#
# The 5-fold Dataset803 checkpoints are DOWNLOADED automatically from
# HuggingFace (configs/pseudolabel_models.json) into NNUNET_RESULTS. The
# nnU-Net+CUDA container is containers/ctspinopelvic1k-ts.sif (TotalSegmentator,
# which ships nnunetv2 + huggingface_hub); the lean containers/ctspinopelvic1k.sif
# has NO nnU-Net. Inference is OUT-OF-FOLD: each
# training case is predicted only by the fold that held it out.
#
# Options (env overrides):
#   HF_EXPORT_DIR   v1 source tree   (default: data/hf_export)
#   PSEUDO_OUT_DIR  v2 output tree   (default: data/hf_export_v2)
#   MODELS_CONFIG   (default: configs/pseudolabel_models.json)
#   NNUNET_RESULTS  checkpoint download dir (default: <root>/nnunet/results)
#   NNUNET_SIF      nnU-Net+CUDA container (required for a real run)
#   SKIP_DOWNLOAD=1 reuse already-downloaded checkpoints
#   PSEUDO_LIMIT=N  cap pseudo-filled records (debug)
#   DRY_RUN=1       copy v1->v2 verbatim, log per-case held-out fold,
#                   run no download/inference
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${DATA_DIR}/hf_export}"
PSEUDO_OUT_DIR="${PSEUDO_OUT_DIR:-${DATA_DIR}/hf_export_v2}"
MODELS_CONFIG="${MODELS_CONFIG:-${PROJECT_ROOT}/configs/pseudolabel_models.json}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${nnUNet_results:-${PROJECT_ROOT}/nnunet/results}}"
# The checkpoints were trained under a CUSTOM trainer class name
# (nnUNetTrainerWandB_500ep_LSTVOversample). nnUNetv2_predict locates the
# trainer BY NAME and calls its build_network_architecture() (a @staticmethod
# inherited from nnUNetTrainer that builds the net from the plans) before
# loading weights — the trainer is never instantiated for inference. So all a
# real run needs is a class of that name subclassing nnUNetTrainer, which we
# vendor as a self-contained ~10-line shim in THIS repo and bind into the
# container. No dependency on the training repo being checked out.
# (Full training trainer: spinesurg-ct-nnunet/tools/nnunet_wandb_variant.py.)
TRAINER_SRC="${TRAINER_SRC:-${PROJECT_ROOT}/containers/nnunet_wandb_variant.py}"
TRAINER_DST="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PSEUDO_LIMIT="${PSEUDO_LIMIT:-0}"

mkdir -p "${LOGS_DIR}" "${PSEUDO_OUT_DIR}" "${NNUNET_RESULTS}"

# ── Split scratch (mirrors slurm/benchmark_totalseg.sh) ──────────────────────
# Job 36127105 died because BOTH the multi-GB SIF->sandbox unpack AND the
# container's runtime /tmp (nnU-Net's multiprocessing listener sockets
# /tmp/pymp-*, preprocessing scratch) shared the node's /tmp. When it filled /
# was reaped mid-run the listener socket vanished (RemoteError /
# FileNotFoundError), then libtorch_cpu.so became unreadable (Intel MKL FATAL
# ERROR) and every later fold silently fell through to passthrough.
#
# Fix = the proven TotalSeg-bench policy:
#   * sandbox unpack (read-mostly, ~10 GB)  -> node-local /tmp
#       (SINGULARITY_TMPDIR on NFS hangs the conversion for hours, job 36124058)
#   * container /tmp + XDG (churny runtime)  -> roomy project NFS
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
NFS_SCRATCH="${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"   # ~10 GB sandbox
HOST_CONTAINER_TMP="${NFS_SCRATCH}/container_tmp"                # bound at /tmp
export XDG_RUNTIME_DIR="${NFS_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" "${NFS_SCRATCH}" 2>/dev/null || true' EXIT TERM INT

# Preflight: fail loud & early if scratch is too tight — instead of dying
# three folds in. (Same thresholds as benchmark_totalseg.sh.)
_free_gib() {
    local kb
    kb=$(df -k --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ')
    echo $(( ${kb:-0} / 1024 / 1024 ))
}
NODE_FREE_GIB=$(_free_gib "${NODE_SCRATCH}")
NFS_FREE_GIB=$(_free_gib "${NFS_SCRATCH}")
if [[ "${NODE_FREE_GIB}" -lt 15 ]]; then
    echo "ERROR: node /tmp (${NODE_SCRATCH}) has only ${NODE_FREE_GIB} GiB free;" >&2
    echo "       need 15 for the singularity sandbox unpack. Likely too many" >&2
    echo "       concurrent jobs on $(hostname); re-submit, optionally with" >&2
    echo "       --exclude=$(hostname)." >&2
    exit 1
fi
if [[ "${NFS_FREE_GIB}" -lt 30 ]]; then
    echo "ERROR: project NFS (${NFS_SCRATCH}) has only ${NFS_FREE_GIB} GiB free;" >&2
    echo "       need 30. Free up space under ${PROJECT_ROOT}." >&2
    exit 1
fi

echo "======================================================================"
echo " Pseudo-label completion (v2 tree)"
echo "   Job ID        : ${SLURM_JOB_ID:-local}"
echo "   Node          : $(hostname)"
echo "   GPU           : $(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo N/A)"
echo "   v1 source     : ${HF_EXPORT_DIR}"
echo "   v2 out        : ${PSEUDO_OUT_DIR}"
echo "   Models config : ${MODELS_CONFIG}"
echo "   nnUNet_results: ${NNUNET_RESULTS:-<unset>}"
echo "   Trainer src   : ${TRAINER_SRC:-<unset>}"
echo "   Sandbox       : ${SINGULARITY_TMPDIR}  (node /tmp, ${NODE_FREE_GIB} GiB free)"
echo "   Ctr /tmp      : ${HOST_CONTAINER_TMP}  (NFS, ${NFS_FREE_GIB} GiB free)"
echo "   DRY_RUN       : ${DRY_RUN}"
echo "   Started       : $(date)"
echo "======================================================================"

# Snapshot the GPU BEFORE any inference. If memory.used is already large or
# foreign PIDs appear here, the GPU was handed to us already occupied — i.e.
# an OOM later is contention, not this job's footprint (single-fold predict
# on this model needs ~10 GB of a 140 GB H200).
echo " GPU state at job start (should be ~empty if we own it exclusively):"
nvidia-smi 2>/dev/null | sed 's/^/   /' || echo "   nvidia-smi unavailable"
echo "======================================================================"

# GPU-occupancy preflight (job 36155337): SLURM can hand us a GPU that a
# NON-SLURM process is already squatting — a foreign VLLM worker held 128/140
# GiB on msa3, so we OOM'd mid-fold AFTER a full sandbox unpack + model load.
# Single-fold predict needs ~10-15 GiB; bail in seconds if the assigned GPU
# lacks headroom rather than crashing 20 min in. These nodes have one H200
# each, so the first nvidia-smi row is our GPU. Override with MIN_GPU_FREE_MIB.
if [[ "${DRY_RUN}" != "1" ]]; then
    MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-20000}"
    GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | head -1 | tr -d ' ')
    if [[ "${GPU_FREE_MIB}" =~ ^[0-9]+$ ]] && (( GPU_FREE_MIB < MIN_GPU_FREE_MIB )); then
        echo "ERROR: assigned GPU has only ${GPU_FREE_MIB} MiB free (need >= ${MIN_GPU_FREE_MIB})."
        echo "       It is already occupied — almost certainly a process SLURM did"
        echo "       not schedule. Offending compute apps on $(hostname):"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null | sed 's/^/         /'
        echo "       Resubmit (lands on a clean node) or exclude this one:"
        echo "         sbatch --exclude=$(hostname) slurm/pseudolabel.sh"
        echo "       (override the floor with MIN_GPU_FREE_MIB=<MiB> if intended)."
        exit 1
    fi
    echo " GPU preflight OK: ${GPU_FREE_MIB:-?} MiB free (>= ${MIN_GPU_FREE_MIB} MiB)."
    echo "======================================================================"
fi

if [[ ! -f "${HF_EXPORT_DIR}/manifest.json" ]]; then
    echo "ERROR: no manifest.json in ${HF_EXPORT_DIR}."
    echo "       Run 'make hf-stage' first — pseudolabel never re-exports."
    exit 1
fi

EXTRA_ARGS=""
PSEUDO_DEVICE="cuda"
if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --dry_run"
    PSEUDO_DEVICE="cpu"
fi
[[ "${SKIP_DOWNLOAD}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --skip_download"
[[ "${PSEUDO_LIMIT}" != "0" ]] && EXTRA_ARGS="${EXTRA_ARGS} --limit ${PSEUDO_LIMIT}"

PPATH="/workspace/scripts:/workspace/src:/workspace"
# nnUNet_results is bound at the SAME host path so the container writes
# downloaded checkpoints back to NFS (persists across re-submits).
BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${NNUNET_RESULTS}:${NNUNET_RESULTS}"
# Container /tmp -> NFS so nnU-Net's /tmp/pymp-* sockets + preprocessing scratch
# never touch the node /tmp that holds the sandbox (the 36127105 failure mode).
BINDS="${BINDS},${HOST_CONTAINER_TMP}:/tmp"
CENV="PYTHONPATH=${PPATH},nnUNet_results=${NNUNET_RESULTS}"
[[ -n "${HF_TOKEN:-}" ]] && CENV="${CENV},HF_TOKEN=${HF_TOKEN}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${NNUNET_SIF:-}" || ! -f "${NNUNET_SIF:-}" ]]; then
        echo "ERROR: NNUNET_SIF not set / not found. A real run needs an"
        echo "       nnU-Net+CUDA container (the lean ctspinopelvic1k.sif lacks"
        echo "       nnunetv2; ctspinopelvic1k-ts.sif ships it). Re-submit with"
        echo "       NNUNET_SIF=\$(pwd)/containers/ctspinopelvic1k-ts.sif, or DRY_RUN=1."
        exit 1
    fi
    if [[ -z "${TRAINER_SRC:-}" || ! -f "${TRAINER_SRC:-}" ]]; then
        echo "ERROR: TRAINER_SRC not found: ${TRAINER_SRC:-<unset>}"
        echo "       The checkpoints use a custom trainer class"
        echo "       (nnUNetTrainerWandB_500ep_LSTVOversample); nnUNetv2_predict"
        echo "       needs a .py defining it bound into the container, or it"
        echo "       fails with 'Unable to locate trainer class …'. This repo"
        echo "       ships a self-contained shim at"
        echo "       containers/nnunet_wandb_variant.py (the default) — if you"
        echo "       see this, it was deleted or TRAINER_SRC points elsewhere."
        exit 1
    fi
fi

CENV="${CENV},PYTHONUNBUFFERED=1"
# Reduce CUDA fragmentation (nnU-Net's own OOM hint). NOTE: this only helps
# on an otherwise-free GPU — it cannot rescue a GPU already occupied by
# another process. If you OOM with most of the 140 GB held by a foreign
# PID, the GPU is shared/leaked; get a clean one (see job 36115436 logs).
CENV="${CENV},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

_run() {
    # DRY_RUN needs no GPU/nnU-Net (only huggingface_hub/nibabel, which the
    # project .sif has): use the project container. A real run uses the
    # caller-supplied nnU-Net+CUDA container with --nv.
    # stdbuf -oL -eL keeps logs streaming live through Singularity+SLURM.
    if [[ "${DRY_RUN}" == "1" ]]; then
        stdbuf -oL -eL singularity exec \
            --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
            "${SIF_PATH}" "$@"
    else
        stdbuf -oL -eL singularity exec --nv \
            --env "${CENV}" \
            --bind "${BINDS}" \
            --bind "${TRAINER_SRC}:${TRAINER_DST}" \
            --pwd /workspace \
            "${NNUNET_SIF}" "$@"
    fi
}

echo ""
echo "Entering container (first use converts the SIF to a sandbox in"
echo "${SINGULARITY_TMPDIR} — node-local, a few minutes with NO output;"
echo "it is NOT hung) ..."
echo ""

_run stdbuf -oL -eL python3 -u /workspace/scripts/pseudolabel.py \
    --hf_export      "/data/$(basename "${HF_EXPORT_DIR}")" \
    --out            "/data/$(basename "${PSEUDO_OUT_DIR}")" \
    --models_config  "/workspace/configs/$(basename "${MODELS_CONFIG}")" \
    --splits         "/data/$(basename "${HF_EXPORT_DIR}")/splits_5fold.json" \
    --nnunet_results "${NNUNET_RESULTS}" \
    --device         "${PSEUDO_DEVICE}" \
    ${EXTRA_ARGS}

echo ""
echo "======================================================================"
echo " Pseudo-label done at $(date)"
echo "   v2 tree: ${PSEUDO_OUT_DIR}"
echo ""
echo " Publish to a v2 BRANCH (main / review URL untouched):"
echo "   HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \\"
echo "     HF_EXPORT_DIR=${PSEUDO_OUT_DIR} make hf-push"
echo "======================================================================"
