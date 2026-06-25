#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — one command to ship every release in order: v1 + v2, then v3.
#
#   ship_v2.sh  ->  v1 (base) + v2 (GT spines + model pelves)        [submits its chain]
#   ship_v3.sh  ->  v3 (v2 + ribs), chained AFTER the v2 push        [SHIP_V3=1, default]
#
# Everything is submitted to SLURM with dependencies, so this returns immediately
# and the cluster runs the DAG. Watch it with `squeue -u $USER`.
#
#   HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K \
#     NNUNET_SIF=$(pwd)/containers/ctspinopelvic1k-ts.sif bash slurm/launch_all.sh
#
# Toggles:  NUKE=1 (wipe ALL hf_export* first, as a SLURM job — clean rebuild, no
#               login-node wait; forces SKIP_BASE=0),
#           SHIP_V3=0 (stop after v2),  SKIP_BASE=1 (reuse v1 base),
#           SBATCH_QOS=secondary (run the whole DAG on another queue),
#           DRY_RUN=1 (plan the pseudolabel step only).
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
SHIP_V3="${SHIP_V3:-1}"

echo "=============================================================="
echo " launch_all — shipping v1 + v2$( [[ "${SHIP_V3}" == "1" ]] && echo ' + v3' )"
echo "=============================================================="

# --- v2 chain (also builds/pushes v1). Capture its terminal push job id. -------
V2_OUT="$(bash slurm/ship_v2.sh)"
echo "${V2_OUT}"
V2_PUSH_JOB="$(printf '%s\n' "${V2_OUT}" | sed -n 's/^V2_PUSH_JOB=//p' | tail -1)"

if [[ "${SHIP_V3}" != "1" ]]; then
    echo "[launch_all] SHIP_V3=0 — stopping after v2 (v3 push job: ${V2_PUSH_JOB})"
    exit 0
fi
[[ -n "${V2_PUSH_JOB}" ]] || { echo "ERROR: could not parse V2_PUSH_JOB from ship_v2 output"; exit 1; }

# --- v3 chain, gated on the v2 push finishing so it reads a complete v2 tree. --
echo "[launch_all] chaining v3 after v2 push job ${V2_PUSH_JOB}"
EXTRA_DEP="${V2_PUSH_JOB}" bash slurm/ship_v3.sh

echo "[launch_all] all releases submitted. Monitor: squeue -u ${USER}"
