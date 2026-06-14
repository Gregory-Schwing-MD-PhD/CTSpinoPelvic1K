#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_v3_ribs
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# ANY gpu — TotalSegmentator's rib task does NOT need an H200, and pinning that
# (the most contested card) is what left this job PENDING for ~24h. Pin a faster
# card only if you want, via:  SBATCH_EXTRA="--gres=gpu:nvidia_h200:1"  (ship_v3
# injects SBATCH_EXTRA into the sbatch call, overriding this default).
#SBATCH --gres=gpu:1
# 12h is more backfill-friendly than 24h; the job is RESUMABLE (per-case markers),
# so if it is preempted or hits the wall, just resubmit and it continues.
#SBATCH --time=12:00:00
#SBATCH --output=logs/v3_ribs_%j.out
#SBATCH --error=logs/v3_ribs_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# v3 ribs — derive the v3 tree from v2 by adding anatomically-numbered ribs.
#
# Runs TotalSegmentator (ribs ROI only) per case in the TS container, re-numbers
# each rib from the GT thoracic vertebrae (placed VerSe spine masks), and merges
# the result into the v2 labels ONLY on background (GT never overwritten). See
# scripts/build_v3_ribs.py and scripts/relabel_ribs.py.
#
# Options (env):
#   V2_DIR     v2 source tree     (default: data/hf_export_v2)
#   V3_DIR     v3 output tree     (default: data/hf_export_v3)
#   SPINE_DIR  placed VerSe masks (default: data/placed/spine)  [thoracic anchors]
#   NNUNET_SIF TS+CUDA container  (default: containers/ctspinopelvic1k-ts.sif)
#   V3_LIMIT   cap cases (debug)  (default: 0 = all)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

V2_DIR="${V2_DIR:-${DATA_DIR}/hf_export_v2}"
V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"
SPINE_DIR="${SPINE_DIR:-${DATA_DIR}/placed/spine}"
NNUNET_SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-${HOME}/totalseg_weights}"
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-${HOME}/.totalseg}"
V3_LIMIT="${V3_LIMIT:-0}"
RESUME="${RESUME:-1}"          # 1 = continue from .rib_done markers (default)

[[ -f "${NNUNET_SIF}" ]] || { echo "ERROR: TS container missing at ${NNUNET_SIF}"; exit 1; }
[[ -f "${V2_DIR}/manifest.json" ]] || { echo "ERROR: no v2 tree at ${V2_DIR} (run ship_v2 first)"; exit 1; }
mkdir -p "${LOGS_DIR}" "${V3_DIR}" "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

# Scratch policy mirrors slurm/benchmark_totalseg.sh: sandbox on node /tmp, runtime on NFS.
NODE_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
NFS_SCRATCH="${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NFS_SCRATCH}/container_tmp"
export XDG_RUNTIME_DIR="${NFS_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" "${NFS_SCRATCH}" 2>/dev/null || true' EXIT TERM INT

echo "======================================================================"
echo " v3 ribs  (resume=${RESUME})"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
echo "   v2 source : ${V2_DIR}"
echo "   v3 out    : ${V3_DIR}"
echo "   spine GT  : ${SPINE_DIR}  (thoracic anchors)"
echo "   TS SIF    : ${NNUNET_SIF}"
echo "   Started   : $(date)"
echo "======================================================================"

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data,${HOST_CONTAINER_TMP}:/tmp"
BINDS+=",${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
CENV="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"
CENV+=",TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS},TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CENV+=",HOME=${TOTALSEG_CONFIG_DIR},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

ARGS=( --v2_dir   "/data/$(realpath --relative-to="${DATA_DIR}" "${V2_DIR}")"
       --v3_dir   "/data/$(realpath --relative-to="${DATA_DIR}" "${V3_DIR}")"
       --spine_dir "/data/$(realpath --relative-to="${DATA_DIR}" "${SPINE_DIR}")"
       --device gpu )
[[ "${V3_LIMIT}" != "0" ]] && ARGS+=( --limit "${V3_LIMIT}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )

stdbuf -oL -eL singularity exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${NNUNET_SIF}" python3 -u /workspace/scripts/build_v3_ribs.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " v3 ribs done at $(date)  ->  ${V3_DIR}  (+ rib_qc.csv)"
echo "======================================================================"
