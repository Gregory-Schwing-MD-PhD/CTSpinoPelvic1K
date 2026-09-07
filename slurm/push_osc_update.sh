#!/usr/bin/env bash
#SBATCH --job-name=osc_update
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/osc_update_%j.out
#SBATCH --error=logs/osc_update_%j.err
set -euo pipefail
cd /wsu/home/go/go24/go2432/CTSpinoPelvic1K
module load singularity/3.5.2
singularity exec   --bind /wsu/home/go/go24/go2432/CTSpinoPelvic1K:/workspace   --bind $HOME/.hf_osc_token:/tmp/hf_token:ro   --pwd /workspace containers/ctspinopelvic1k.sif   python3 scripts/push_osc_release.py     --folder data/hf_export_osc     --repo   OpenSpineConsortium/CTSpinoPelvic1K     --token-file /tmp/hf_token     --workers 2
