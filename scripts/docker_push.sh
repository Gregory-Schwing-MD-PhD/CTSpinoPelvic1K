#!/usr/bin/env bash
# =============================================================================
# CTSpinoPelvic1K — Docker Build & Push
# scripts/docker_push.sh
#
# Run on your LOCAL WORKSTATION.  Builds and pushes:
#   1. ctspinopelvic1k            (lean: download / visualize / export)
#   2. ctspinopelvic1k-ts         (CUDA: TotalSegmentator benchmark)
#   3. ctspinopelvic1k-spineps    (CUDA: SPINEPS CT + Möller ribs + rib measurements)
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
#
#   # SPINEPS image only (pin upstream with SPINEPS_REF / RIBSEG_REF):
#   DOCKERHUB_USER=myusername SPINEPS_ONLY=1 ./scripts/docker_push.sh
# =============================================================================
set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
TAG="${TAG:-latest}"
LEAN_ONLY="${LEAN_ONLY:-0}"
TOTALSEG_ONLY="${TOTALSEG_ONLY:-0}"
SPINEPS_ONLY="${SPINEPS_ONLY:-0}"
SPINEPS_REF="${SPINEPS_REF:-main}"
RIBSEG_REF="${RIBSEG_REF:-main}"

LEAN_IMAGE="${DOCKERHUB_USER}/ctspinopelvic1k:${TAG}"
TS_IMAGE="${DOCKERHUB_USER}/ctspinopelvic1k-ts:${TAG}"
SPINEPS_IMAGE="${DOCKERHUB_USER}/ctspinopelvic1k-spineps:${TAG}"

# Any *_ONLY flag suppresses the others.
if [[ "${SPINEPS_ONLY}" == "1" ]]; then
    BUILD_SPINEPS=1; LEAN_ONLY=1; TOTALSEG_ONLY=1
elif [[ "${LEAN_ONLY}" != "1" && "${TOTALSEG_ONLY}" != "1" ]]; then
    BUILD_SPINEPS=1
else
    BUILD_SPINEPS=0
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "Docker not found."
[[ -f docker/Dockerfile ]]                  || die "Run from the repo root."
[[ -f docker/Dockerfile.totalsegmentator ]] || die "docker/Dockerfile.totalsegmentator missing."
[[ -f docker/Dockerfile.spineps ]]          || die "docker/Dockerfile.spineps missing."

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

# SPINEPS image ---------------------------------------------------------------
# Big (~10 GB): the CT weights are baked in so compute nodes never need the internet.
if [[ "${BUILD_SPINEPS}" == "1" ]]; then
    log "Building SPINEPS image: ${SPINEPS_IMAGE}  (spineps@${SPINEPS_REF}, rib-segmentation@${RIBSEG_REF})"
    docker build \
        --file docker/Dockerfile.spineps \
        --build-arg "SPINEPS_REF=${SPINEPS_REF}" \
        --build-arg "RIBSEG_REF=${RIBSEG_REF}" \
        --tag  "${SPINEPS_IMAGE}" \
        --progress=plain .
    log "Pushing ${SPINEPS_IMAGE}"
    docker push "${SPINEPS_IMAGE}"
    log "  ✓ ${SPINEPS_IMAGE}"
else
    log "(skipped SPINEPS image)"
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
