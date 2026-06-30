#!/usr/bin/env bash
#SBATCH --job-name=rib_space
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rib_space_%j.out
#SBATCH --error=logs/rib_space_%j.err
# =============================================================================
# deploy_rib_space.sh — get the v4 rib-fix DUAL-REVIEW Space ready (CPU + network only).
#   (1) (re)generate rib_worklist.json into V4_DIR + upload it to <dataset>@v4
#   (2) deploy + seed the rib review Space (TASK=rib_fix@v4, server-side CHECK=ribs gate)
# Runs inside the project SIF (which has huggingface_hub). One-time: future ship_v4 runs
# already push rib_worklist.json with the dataset, so the worklist step becomes a no-op refresh.
#
#   HF_TOKEN=hf_xxx sbatch slurm/deploy_rib_space.sh
#   # optional: ORG=anonymous-mlhc ADJUDICATORS=gregoryschwingmdphd V4_DIR=/path SIF_PATH=/path.sif
# =============================================================================
set -euo pipefail
# sbatch copies the script to a spool dir, so $0 is NOT the repo — use SLURM_SUBMIT_DIR (the dir
# you ran sbatch from). Run this from the repo root: cd ~/CTSpinoPelvic1K && sbatch slurm/...
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "${PROJECT_ROOT}"
[[ -f configs/default.env ]] || { echo "ERROR: configs/default.env not found in ${PROJECT_ROOT} — "\
  "run from the repo root: cd ~/CTSpinoPelvic1K && sbatch slurm/deploy_rib_space.sh"; exit 1; }
source configs/default.env

: "${HF_TOKEN:?HF_TOKEN=hf_xxx sbatch slurm/deploy_rib_space.sh}"
ORG="${ORG:-anonymous-mlhc}"
DATASET="${DATASET:-${ORG}/CTSpinoPelvic1K}"
V4_DIR="${V4_DIR:-${DATA_DIR}/hf_export_v4}"
SIF_PATH="${SIF_PATH:-${CONTAINER:-${PROJECT_ROOT}/containers/ctspinopelvic1k.sif}}"
ADJUDICATORS="${ADJUDICATORS:-gregoryschwingmdphd}"
mkdir -p logs

[[ -d "${V4_DIR}/_v4ribs_done" ]] || { echo "ERROR: ${V4_DIR}/_v4ribs_done missing — run ship_v4 first"; exit 1; }
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: no SIF at ${SIF_PATH} (set SIF_PATH=/path/to/image.sif)"; exit 1; }

ENVV="HF_TOKEN=${HF_TOKEN},ORG=${ORG},DATASET=${DATASET},ONLY=ribs,ADJUDICATORS=${ADJUDICATORS},PYTHONPATH=${PROJECT_ROOT}/scripts"
SX=( singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT},${DATA_DIR}:${DATA_DIR}"
     --pwd "${PROJECT_ROOT}" --env "${ENVV}" "${SIF_PATH}" )

echo "[rib_space] (1) regenerate rib_worklist.json + upload -> ${DATASET}@v4   $(date)"
"${SX[@]}" python3 scripts/qc_v4_ribs.py --v4_dir "${V4_DIR}"
"${SX[@]}" python3 - "${V4_DIR}" "${DATASET}" <<'PY'
import os, sys
from huggingface_hub import HfApi
v4_dir, dataset = sys.argv[1], sys.argv[2]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=os.path.join(v4_dir, "rib_worklist.json"),
    path_in_repo="rib_worklist.json", repo_id=dataset, repo_type="dataset", revision="v4")
print(f"[rib_space] uploaded rib_worklist.json -> {dataset}@v4")
PY

echo "[rib_space] (2) deploy + seed the rib review Space (ONLY=ribs adj=${ADJUDICATORS})   $(date)"
"${SX[@]}" python3 review_service/deploy_v4_spaces.py

URL="https://$(echo "${ORG}" | tr 'A-Z' 'a-z')-ctspinopelvic1k-review-ribs.hf.space"
echo "[rib_space] done. The Space now builds + seeds on HF — watch its Logs for 'seeded 165'."
echo "[rib_space]   ${URL}"
echo "[rib_space]   students: reviewtool login --service ${URL} ; python -m reviewtool next"
