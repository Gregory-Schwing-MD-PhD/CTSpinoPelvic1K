#!/usr/bin/env bash
# Re-export the Gallery meshes at higher resolution, in the ITK-SNAP colour scheme.
#
# Two reasons to rebuild. The colours came from the old descriptor, which ran every
# label along one hue ramp -- adjacent vertebrae were the same yellow-green on the site
# just as they were in ITK-SNAP. And the meshes were built at a third of resolution with
# a tight triangle budget, which faceted the vertebrae and eroded thin ribs to nothing.
#SBATCH --job-name=ctsp_meshes
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/meshes_%j.out
#SBATCH --error=logs/meshes_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

singularity exec --bind "$(pwd)":/w --pwd /w \
  --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" \
  python3 scripts/export_gallery_meshes.py --labels data/v5_final \
    --cases 0431,0033,0631,1035,0004 \
    --descriptor data/itksnap_v5_labels.txt --out gallery_meshes

echo
echo "=== payload ==="
du -ch gallery_meshes/*.bin | tail -1
echo "=== done ==="
