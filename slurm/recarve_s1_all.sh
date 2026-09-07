#!/usr/bin/env bash
#SBATCH --job-name=s1_recarve
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s1_recarve_%j.out
#SBATCH --error=logs/s1_recarve_%j.err
set -euo pipefail
cd /wsu/home/go/go24/go2432/CTSpinoPelvic1K
module load singularity/3.5.2
singularity exec --bind /wsu/home/go/go24/go2432/CTSpinoPelvic1K:/workspace --pwd /workspace   containers/ctspinopelvic1k.sif   python3 scripts/recarve_s1_all.py     --in_dir  data/hf_export_osc     --out_dir data/s1_recarve     --report  data/s1_recarve/s1_recarve_qc.csv     --workers 16
