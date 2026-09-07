#!/usr/bin/env bash
#SBATCH --job-name=v6_label_occupancy
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/label_occupancy_%j.out
#SBATCH --error=logs/label_occupancy_%j.err
set -euo pipefail
cd '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'
module load singularity/3.5.2
singularity exec   --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K':/workspace   --pwd /workspace   containers/ctspinopelvic1k.sif   python3 scripts/label_occupancy_v6.py     --tree data/hf_export_osc     --out  data/hf_export_osc/label_occupancy.json     --workers 12
