#!/usr/bin/env bash
#SBATCH --job-name=detect_metal_v6
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/detect_metal_v6_%j.out
#SBATCH --error=logs/detect_metal_v6_%j.err
set -euo pipefail
cd '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'
module load singularity/3.5.2
singularity exec --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K':/workspace --pwd /workspace   containers/ctspinopelvic1k.sif   python3 scripts/detect_metal.py     --labels data/hf_export_osc/labels     --ct     data/hf_export_osc/ct     --workers 16     --out    morphometrics_v6_regrouped
