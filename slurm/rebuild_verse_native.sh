#!/usr/bin/env bash
# =============================================================================
# rebuild_verse_native.sh — ONE command to rebuild + push v1 + v2 + v3 in the
# VerSe-native label scheme, with every SLURM dependency wired for you.
#
#   v1 base (export_hf, VerSe-native merge_labels)  ->  @v1
#   v2 (GT spines + model pelves at 26/30/31)        ->  @v2   [afterok v1]   (pseudolabel SHARDED)
#   v3 (+ TS femurs/ribs/S1 carve)                   ->  @v3 + @main   [afterok v2 push]
#
# It FORCES SKIP_BASE=0 because the base on disk is the OLD scheme and MUST be
# re-exported — that is the whole point of this rebuild. Submits the full DAG and
# returns immediately; the cluster runs it in order. Correctness is guaranteed by the
# unit tests (label_scheme.verify + the rib-range regression guard), not a post-build gate.
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/rebuild_verse_native.sh
#
# Optional: SBATCH_QOS=secondary (run the DAG on another queue), HF_PRIVATE=1,
#           SYNC_MAIN=0 (don't move @main), HF_WORKERS=N.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-launcher$$}"
source configs/default.env

: "${HF_TOKEN:?paste your token -> HF_TOKEN=hf_xxx HF_REPO_ID=<org>/Name bash slurm/rebuild_verse_native.sh}"
: "${HF_REPO_ID:?set HF_REPO_ID=<org>/CTSpinoPelvic1K}"

# The decisive flags: NUKE everything first (so no old-scheme leftover can survive)
# and re-export the v1 base. NUKE=1 forces SKIP_BASE=0 and submits the wipe as a SLURM
# job (no login-node wait). Override with NUKE=0 to keep existing trees + reuse caches.
export NUKE="${NUKE:-1}"
export SKIP_BASE=0
export SHIP_V3=1
V3_DIR="${V3_DIR:-${DATA_DIR}/hf_export_v3}"

echo "=================================================================="
echo " rebuild_verse_native — v1 + v2 + v3, VerSe-native, SKIP_BASE=0"
echo "   repo: ${HF_REPO_ID}   tree: ${V3_DIR}"
echo "=================================================================="

# --- v2 chain (also v1). Capture the terminal v2 push job id. ------------------
V2_OUT="$(bash slurm/ship_v2.sh)"
echo "${V2_OUT}"
V2_PUSH_JOB="$(printf '%s\n' "${V2_OUT}" | sed -n 's/^V2_PUSH_JOB=//p' | tail -1)"
[[ -n "${V2_PUSH_JOB}" ]] || { echo "ERROR: could not parse V2_PUSH_JOB from ship_v2"; exit 1; }

# --- v3 chain, gated on the v2 push. Capture the v3 push job id. ---------------
echo "[rebuild_verse_native] chaining v3 after v2 push ${V2_PUSH_JOB}"
EXTRA_DEP="${V2_PUSH_JOB}" bash slurm/ship_v3.sh

echo "[rebuild_verse_native] full DAG submitted. Monitor: squeue -u ${USER}"
