#!/usr/bin/env bash
#SBATCH --job-name=ctsp_rib_v5fin
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rib_v5fin_%j.out
#SBATCH --error=logs/rib_v5fin_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
echo '=== apply the four never-reviewed lumbar-rib cases ==='
"${RUN[@]}" python3 scripts/lumbar_rib_class_v5.py --labels data/v5_final --qc qc_rib_incidence_v5_fixed --cases 0231,0389,0473,0720 --apply
echo; echo '=== the one final QC for v5 ==='
"${RUN[@]}" python3 scripts/qc_rib_vertebra_incidence.py --labels data/v5_final --workers 8 --out qc_rib_incidence_v5_final
echo; echo '=== done ==='
