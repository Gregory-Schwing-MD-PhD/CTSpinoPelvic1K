#!/usr/bin/env bash
#SBATCH --job-name=ctspinopelvic1k_hpc_pull
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc_pull_%j.out
#SBATCH --error=logs/hpc_pull_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Pulls both Docker Hub images and converts them to local .sif containers:
#
#   containers/ctspinopelvic1k.sif        lean (Stages 1-3, all utilities)
#   containers/ctspinopelvic1k-ts.sif     CUDA + TotalSegmentator (Stage 4)
#
# Usage (from the project root):
#     sbatch slurm/hpc_pull.sh
#     DOCKERHUB_USER=myuser sbatch slurm/hpc_pull.sh
#
# Or via make:
#     make hpc-pull
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs containers

# ── Scrub host env that can leak into the pull ───────────────────────────────
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# ── Nextflow/conda env (same layout used elsewhere in the project) ───────────
export CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/mambaforge/envs/nextflow}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ── Singularity runtime dirs ─────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

echo "======================================================================"
echo " ctspinopelvic1k hpc_pull"
echo "   Job ID        : ${SLURM_JOB_ID:-local}"
echo "   Node          : $(hostname)"
echo "   DOCKERHUB_USER: ${DOCKERHUB_USER:-gregoryschwingmdphd}"
echo "   SIF out dir   : ${PROJECT_ROOT}/containers"
echo "   Cache dir     : ${NXF_SINGULARITY_CACHEDIR}"
echo "   Started       : $(date)"
echo "======================================================================"

# Call the pull script directly (NOT `make hpc-pull` — that would re-submit).
SIF_DIR="${PROJECT_ROOT}/containers" \
    bash "${PROJECT_ROOT}/scripts/hpc_pull.sh"

echo ""
echo " Completed at $(date)"
echo "======================================================================"
