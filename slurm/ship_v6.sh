#!/usr/bin/env bash
# =============================================================================
# ship_v6.sh — publish the v6 tree to HuggingFace as its own revision.
#
#   bash slurm/ship_v6.sh
#
# v6 = v5 + surgical instrumentation, labelled, on the eleven cases a radiologist
# confirmed, plus 0068's corrections (six lumbar bodies, T10-T12, the cages).
#
# WIPE_REMOTE=0, DELIBERATELY. v6 is a NEW branch: there is nothing on it to
# orphan, and the earlier revisions are what the paper under review points at.
# A wipe here would clear files the reviewers are reading.
#
# The token is the ORG token (~/.hf_org_token, mode 0600), because the target
# lives in the anonymous-mlhc org rather than a personal namespace. It is read
# here and passed through --export so it never appears in a command line or a
# log.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# default.env expects to run inside a job; from a login shell SLURM_JOB_ID is unset
# and `set -u` makes that fatal. ship_v3.sh sets the same placeholder.
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

V6_DIR="${V6_DIR:-${DATA_DIR}/hf_export_v6}"
HF_REPO_ID="${HF_REPO_ID:-anonymous-mlhc/CTSpinoPelvic1K}"
HF_REVISION="${HF_REVISION:-v6}"
TOKEN_FILE="${TOKEN_FILE:-${HOME}/.hf_org_token}"

[[ -f "${TOKEN_FILE}" ]] || { echo "ERROR: no token at ${TOKEN_FILE}"; exit 1; }
[[ -d "${V6_DIR}/labels" ]] || { echo "ERROR: no v6 tree at ${V6_DIR}"; exit 1; }
HF_TOKEN="$(tr -d ' \n\r' < "${TOKEN_FILE}")"
[[ -n "${HF_TOKEN}" ]] || { echo "ERROR: token file is empty"; exit 1; }

N_LAB=$(ls "${V6_DIR}/labels"/*_label.nii.gz 2>/dev/null | wc -l)
N_CT=$(ls "${V6_DIR}/ct"/*_ct.nii.gz 2>/dev/null | wc -l)
echo "  tree     : ${V6_DIR}  (${N_LAB} labels, ${N_CT} cts)"
echo "  target   : ${HF_REPO_ID} @ ${HF_REVISION}"
echo "  wipe     : no (new branch; earlier revisions are under review)"
[[ "${N_LAB}" == "802" && "${N_CT}" == "802" ]] || {
  echo "ERROR: expected 802 labels and 802 cts"; exit 1; }

JOB=$(sbatch --parsable \
  --export=ALL,SIF_PATH="${SIF_PATH:-containers/ctspinopelvic1k.sif}",PUSH=1,SKIP_EXPORT=1,WIPE_REMOTE=0,HF_TOKEN="${HF_TOKEN}",HF_REPO_ID="${HF_REPO_ID}",HF_REVISION="${HF_REVISION}",HF_EXPORT_DIR="${V6_DIR}",HF_WORKERS="${HF_WORKERS:-8}",HF_PRIVATE="${HF_PRIVATE:-0}",MANIFEST_FILE="${MANIFEST_FILE:-placed_manifest_orientation_fixed.json}" \
  slurm/export_dataset.sh)
echo "  submitted: ${JOB}"
echo "  watch    : tail -f logs/export_dataset_${JOB}.out"
