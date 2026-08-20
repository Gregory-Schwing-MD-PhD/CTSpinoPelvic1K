#!/usr/bin/env bash
#SBATCH --job-name=ctsp_qc_rib_incidence
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qc_rib_incidence_%j.out
#SBATCH --error=logs/qc_rib_incidence_%j.out
#SBATCH --mail-type=END,FAIL

# =============================================================================
# qc_rib_incidence — does each rib sit on the vertebra its number claims?
#
# Runs scripts/qc_rib_vertebra_incidence.py over a whole dataset revision. Each
# case loads a full label volume and compares up to 24 ribs against up to 18
# vertebrae, so it is memory- and CPU-bound and belongs on a compute node.
#
# It was first run on the LOGIN NODE with --workers 12. The login node has 2
# cores: twelve workers took 13% CPU each and 40 cases had not finished in 34
# minutes. Hence this file.
#
# Options (env):
#   REV       dataset revision to read      (default: v4)
#   DS_REPO   dataset repo                  (default: anonymous-mlhc/CTSpinoPelvic1K)
#   LABELS    local label dir instead of HF (overrides REV)
#   OUTDIR    where the CSVs go             (default: qc_rib_incidence_$REV)
#   LIMIT     cap cases (debug)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs

# singularity lives in a conda env on this cluster, not /usr/bin. Without this the job
# dies inside the container launch with "fork/exec /usr/bin/singularity: no such file or
# directory" -- which reads like a missing image rather than a missing PATH.
source configs/default.env

REV="${REV:-v4}"
DS_REPO="${DS_REPO:-anonymous-mlhc/CTSpinoPelvic1K}"
OUTDIR="${OUTDIR:-qc_rib_incidence_${REV}}"
LIMIT="${LIMIT:-0}"
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

# One worker per allocated core, not per core on the box. Oversubscribing was the
# original failure and --cpus-per-task is the only number that reflects the grant.
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "no HF token: put one at ~/.cache/huggingface/token or export HF_TOKEN" >&2
  exit 1
fi

ARGS=(--workers "${WORKERS}" --out "/w/${OUTDIR}" --hf-repo "${DS_REPO}")
if [[ -n "${LABELS:-}" ]]; then ARGS+=(--labels "${LABELS}"); else ARGS+=(--hf-rev "${REV}"); fi
[[ "${LIMIT}" != "0" ]] && ARGS+=(--limit "${LIMIT}")

echo "=== rib-vertebra incidence QC ==="
echo "  source   : ${LABELS:-${DS_REPO}@${REV}}"
echo "  workers  : ${WORKERS} (cpus-per-task=${SLURM_CPUS_PER_TASK:-?})"
echo "  out      : ${OUTDIR}"
echo

singularity exec \
  --bind "${PROJECT_ROOT}":/w \
  --bind "${HOME}/.cache/huggingface":/hf \
  --pwd /w \
  --env HF_TOKEN="${HF_TOKEN}",HF_HOME=/hf,PYTHONPATH=/w/scripts \
  "${SIF}" \
  python3 scripts/qc_rib_vertebra_incidence.py "${ARGS[@]}"

echo
echo "=== done -> ${OUTDIR}/ ==="
ls -l "${OUTDIR}" 2>/dev/null || true
