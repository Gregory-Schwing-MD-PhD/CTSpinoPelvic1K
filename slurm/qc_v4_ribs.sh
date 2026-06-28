#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_qc_v4_ribs
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/qc_v4_ribs_%j.out
#SBATCH --error=logs/qc_v4_ribs_%j.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# qc_v4_ribs — aggregate the per-case rib-connection QC that build_v4_ribs wrote to
# <V4_DIR>/_v4ribs_done/*.json into (1) a connection-quality summary and (2) the
# student rib-correction worklist (whole ribs present-but-disconnected AND missed by
# Möller) + the ready `reviewtool review-cases ... --check ribs` command.
#
# Stdlib-only + reads local files only (no GPU, no network, no HF token). Runs in the
# TS container just for a consistent python3. Everything prints to the .out log; the
# worklist is also written to CSV.
#
#   sbatch slurm/qc_v4_ribs.sh
#   CSV=/path/out.csv HF_REVISION=v4 sbatch slurm/qc_v4_ribs.sh
# =============================================================================
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source configs/default.env

V4_DIR="${V4_DIR:-${DATA_DIR}/hf_export_v4}"
SIF="${NNUNET_SIF:-${PROJECT_ROOT}/containers/ctspinopelvic1k-ts.sif}"
CSV="${CSV:-${PROJECT_ROOT}/rib_review.csv}"
REPO="${HF_REPO_ID:-anonymous-mlhc/CTSpinoPelvic1K}"
REV="${HF_REVISION:-v4}"
mkdir -p "${LOGS_DIR}"

[[ -d "${V4_DIR}/_v4ribs_done" ]] || { echo "ERROR: no ${V4_DIR}/_v4ribs_done — run build_v4_ribs (ship_v4) first"; exit 1; }

BINDS="${PROJECT_ROOT}:/workspace,${DATA_DIR}:/data"
CENV="PYTHONPATH=/workspace/scripts:/workspace,PYTHONUNBUFFERED=1"
REL="$(realpath --relative-to="${DATA_DIR}" "${V4_DIR}")"

echo "[qc_v4_ribs] V4_DIR=${V4_DIR}  repo=${REPO}@${REV}  csv=${CSV}  $(date)"
stdbuf -oL -eL singularity exec --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
  "${SIF}" python3 -u /workspace/scripts/qc_v4_ribs.py \
    --v4_dir "/data/${REL}" --repo "${REPO}" --revision "${REV}" \
    --csv "/workspace/$(basename "${CSV}")"
echo "[qc_v4_ribs] done  $(date)"
