#!/usr/bin/env bash
#SBATCH --job-name=fix0816_complete
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fix0816_complete_%j.out
#SBATCH --error=logs/fix0816_complete_%j.err
#SBATCH --exclude=msa1
set -euo pipefail
cd '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'
module load singularity/3.5.2
# TS writes its weights and config under $HOME; the container's /opt/totalseg is
# read-only, so point both at the cached copies and bind them in (mirrors v3_totalseg.sh).
export SINGULARITYENV_PYTHONPATH=/workspace/scripts
export SINGULARITYENV_TOTALSEG_WEIGHTS_PATH='/wsu/home/go/go24/go2432/totalseg_weights'
export SINGULARITYENV_TOTALSEG_HOME_DIR='/wsu/home/go/go24/go2432/.totalseg'
export SINGULARITYENV_HOME='/wsu/home/go/go24/go2432/.totalseg'
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
singularity exec --nv   --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K':/workspace   --bind '/wsu/home/go/go24/go2432/totalseg_weights':'/wsu/home/go/go24/go2432/totalseg_weights'   --bind '/wsu/home/go/go24/go2432/.totalseg':'/wsu/home/go/go24/go2432/.totalseg'   --pwd /workspace   containers/ctspinopelvic1k-ts.sif   python3 scripts/complete_0816.py     --ct    data/fix_0816/ct/0816_ct.nii.gz     --label data/fix_0816_pseudo/labels/0816_label.nii.gz     --out   data/fix_0816_pseudo/labels/0816_label_completed.nii.gz     --report data/fix_0816_pseudo/0816_completion_report.json     --device gpu
