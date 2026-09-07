#!/usr/bin/env bash
#SBATCH --job-name=rescreen_metal
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rescreen_metal_%j.out
#SBATCH --error=logs/rescreen_metal_%j.err
set -euo pipefail
cd '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'
module load singularity/3.5.2
singularity exec --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K':/workspace --pwd /workspace   containers/ctspinopelvic1k.sif   python3 scripts/rescreen_metal_aggregate.py     --tree data/hf_export_osc     --out  rescreen_metal.csv     --workers 12
