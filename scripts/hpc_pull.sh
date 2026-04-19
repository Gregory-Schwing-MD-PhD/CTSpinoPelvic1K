#!/usr/bin/env bash
# =============================================================================
# CTSpinoPelvic1K — HPC Singularity Pull
# scripts/hpc_pull.sh
#
# Run ON THE HPC GRID (warrior or equivalent).  Converts Docker Hub images
# to local .sif containers.
#
# Prereqs on HPC:
#   - Singularity >= 3.9   (module load singularity or in PATH)
#   - Internet access from login/compute node
#   - Disk: ~3 GB (lean) + ~5 GB (totalsegmentator)
#
# Usage:
#   DOCKERHUB_USER=myusername bash scripts/hpc_pull.sh
#
#   # Pull totalsegmentator only:
#   DOCKERHUB_USER=myusername TOTALSEG_ONLY=1 bash scripts/hpc_pull.sh
#
#   # Pull lean only:
#   DOCKERHUB_USER=myusername LEAN_ONLY=1 bash scripts/hpc_pull.sh
# =============================================================================
set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
TAG="${TAG:-latest}"
LEAN_ONLY="${LEAN_ONLY:-0}"
TOTALSEG_ONLY="${TOTALSEG_ONLY:-0}"
SIF_DIR="${SIF_DIR:-$(pwd)/containers}"

LEAN_SIF="${SIF_DIR}/ctspinopelvic1k.sif"
TS_SIF="${SIF_DIR}/ctspinopelvic1k-ts.sif"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v singularity >/dev/null 2>&1 || \
    { module load singularity 2>/dev/null || \
        die "Singularity not found. Try: module load singularity"; }

mkdir -p "${SIF_DIR}"

log "=== CTSpinoPelvic1K HPC Singularity Pull ==="
log "User        : ${DOCKERHUB_USER}"
log "SIF out dir : ${SIF_DIR}"

if [[ "${TOTALSEG_ONLY}" != "1" ]]; then
    log "Pulling lean image ..."
    singularity pull --force \
        "${LEAN_SIF}" \
        "docker://${DOCKERHUB_USER}/ctspinopelvic1k:${TAG}"
    log "  ✓ ${LEAN_SIF}  ($(du -sh "${LEAN_SIF}" | cut -f1))"
else
    log "(skipped lean image)"
fi

if [[ "${LEAN_ONLY}" != "1" ]]; then
    log "Pulling TotalSegmentator image ..."
    singularity pull --force \
        "${TS_SIF}" \
        "docker://${DOCKERHUB_USER}/ctspinopelvic1k-ts:${TAG}"
    log "  ✓ ${TS_SIF}  ($(du -sh "${TS_SIF}" | cut -f1))"
else
    log "(skipped TS image)"
fi

cat <<EOF

  Next steps:

    # Stage 1-3 (dataset construction + export):
    sbatch slurm/download_raw.sh
    sbatch slurm/create_dataset.sh
    sbatch slurm/export_dataset.sh

    # Stage 4 (TotalSegmentator benchmark):
    sbatch slurm/benchmark_totalseg.sh

EOF
log "Pull complete."
