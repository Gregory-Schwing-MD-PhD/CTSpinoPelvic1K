#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_v3_totalseg
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/v3_totalseg_%j.out
#SBATCH --error=logs/v3_totalseg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=msa1
# =============================================================================
# Same GPU directives as the known-good pseudolabel.sh. 48h wall, but the job is
# RESUMABLE (per-case markers) so a wall-hit/preemption just continues on resubmit.
#
# v3 TotalSegmentator — derive the v3 tree from v2 with one TS pass per case:
# GT-vertebra-matched ribs + femurs + an S1 carve out of the GT sacrum (bone only).
#
# Ribs are emitted only where a GT thoracic vertebra backs them; femurs are added
# directly; S1 = (GT sacrum) ∩ (TS vertebrae_S1). All land on background / relabel
# the sacrum in place — GT boundaries are never overwritten.
# See scripts/build_v3_totalseg.py.
#
# Options (env):
#   V2_DIR     v2 source tree     (default: data/hf_export_v2)
#   V3_DIR     v3 output tree     (default: data/hf_export_v3)
#   SPINE_DIR  placed VerSe spine masks (rib/S1 anchors)  (default: data/placed/spine)
#   NNUNET_SIF TS+CUDA container  (default: containers/ctspinopelvic1k-ts.sif)
#   V3_LIMIT   cap cases (debug)  (default: 0 = all)
#   RESUME     1 = continue from .totalseg_done markers; 0 = full rebuild
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
RESUME="${RESUME:-1}"          # 1 = continue from .totalseg_done markers (default)

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
echo " v3 TotalSegmentator  (resume=${RESUME})"
echo "   Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "   GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
echo "   v2 source : ${V2_DIR}"
echo "   v3 out    : ${V3_DIR}"
echo "   numbering : TotalSegmentator native (rib_left/right_1..12)"
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
    "${NNUNET_SIF}" python3 -u /workspace/scripts/build_v3_totalseg.py "${ARGS[@]}"

echo ""
echo "======================================================================"
echo " v3 TotalSegmentator done at $(date)  ->  ${V3_DIR}  (+ totalseg_qc.csv)"
echo "======================================================================"
