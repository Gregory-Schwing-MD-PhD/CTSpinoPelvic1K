#!/usr/bin/env bash
# =============================================================================
# CTSpinoPelvic1K — Docker Build & Push
# scripts/docker_push.sh
#
# Run on your LOCAL WORKSTATION.  Builds and pushes:
#   1. ctspinopelvic1k            (lean: download / visualize / export)
#   2. ctspinopelvic1k-ts         (CUDA: TotalSegmentator benchmark)
#
# Prereqs:
#   - Docker Desktop running
#   - docker login
#
# Usage:
#   chmod +x scripts/docker_push.sh
#   DOCKERHUB_USER=myusername ./scripts/docker_push.sh
#
#   # TotalSegmentator image only:
#   DOCKERHUB_USER=myusername TOTALSEG_ONLY=1 ./scripts/docker_push.sh
#
#   # Lean image only:
#   DOCKERHUB_USER=myusername LEAN_ONLY=1 ./scripts/docker_push.sh
# =============================================================================
set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
TAG="${TAG:-latest}"
LEAN_ONLY="${LEAN_ONLY:-0}"
TOTALSEG_ONLY="${TOTALSEG_ONLY:-0}"

LEAN_IMAGE="${DOCKERHUB_USER}/ctspinopelvic1k:${TAG}"
TS_IMAGE="${DOCKERHUB_USER}/ctspinopelvic1k-ts:${TAG}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "Docker not found."
[[ -f docker/Dockerfile ]]                  || die "Run from the repo root."
[[ -f docker/Dockerfile.totalsegmentator ]] || die "docker/Dockerfile.totalsegmentator missing."

log "=== CTSpinoPelvic1K Docker Build & Push ==="
log "User : ${DOCKERHUB_USER}"
log "Tag  : ${TAG}"

# Lean image ------------------------------------------------------------------
if [[ "${TOTALSEG_ONLY}" != "1" ]]; then
    log "Building lean image: ${LEAN_IMAGE}"
    docker build \
        --file docker/Dockerfile \
        --tag  "${LEAN_IMAGE}" \
        --progress=plain .
    log "Pushing ${LEAN_IMAGE}"
    docker push "${LEAN_IMAGE}"
    log "  ✓ ${LEAN_IMAGE}"
else
    log "(skipped lean image: TOTALSEG_ONLY=1)"
fi

# TotalSegmentator image ------------------------------------------------------
if [[ "${LEAN_ONLY}" != "1" ]]; then
    log "Building TotalSegmentator image: ${TS_IMAGE}"
    docker build \
        --file docker/Dockerfile.totalsegmentator \
        --tag  "${TS_IMAGE}" \
        --progress=plain .
    log "Pushing ${TS_IMAGE}"
    docker push "${TS_IMAGE}"
    log "  ✓ ${TS_IMAGE}"
else
    log "(skipped TS image: LEAN_ONLY=1)"
fi

cat <<EOF

  ┌──────────────────────────────────────────────────────────────────┐
  │  On HPC:                                                          │
  │    DOCKERHUB_USER=${DOCKERHUB_USER} bash scripts/hpc_pull.sh     │
  │                                                                    │
  │  Or directly:                                                      │
  │    singularity pull ctspinopelvic1k.sif \\                         │
  │        docker://${LEAN_IMAGE}                                      │
  │    singularity pull ctspinopelvic1k-ts.sif \\                      │
  │        docker://${TS_IMAGE}                                        │
  └──────────────────────────────────────────────────────────────────┘
EOF
log "Done."
