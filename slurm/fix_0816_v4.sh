#!/usr/bin/env bash
#SBATCH --job-name=fix0816_v4
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fix0816_v4_%j.out
#SBATCH --error=logs/fix0816_v4_%j.err
#SBATCH --exclude=msa1
set -euo pipefail
cd /wsu/home/go/go24/go2432/CTSpinoPelvic1K
module load singularity/3.5.2
export SINGULARITYENV_PYTHONPATH=/workspace/scripts
export SINGULARITYENV_TOTALSEG_WEIGHTS_PATH=/wsu/home/go/go24/go2432/totalseg_weights
export SINGULARITYENV_TOTALSEG_HOME_DIR=/wsu/home/go/go24/go2432/.totalseg
export SINGULARITYENV_HOME=/wsu/home/go/go24/go2432/.totalseg
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- step 1: completion WITHOUT the hardware label --------------------------
rm -rf data/fix_0816_v3 data/fix_0816_v4
mkdir -p data/fix_0816_v3/ct data/fix_0816_v3/labels
cp data/fix_0816/ct/0816_ct.nii.gz data/fix_0816_v3/ct/
singularity exec --nv --bind /wsu/home/go/go24/go2432/CTSpinoPelvic1K:/workspace --bind /wsu/home/go/go24/go2432/totalseg_weights:/wsu/home/go/go24/go2432/totalseg_weights --bind /wsu/home/go/go24/go2432/.totalseg:/wsu/home/go/go24/go2432/.totalseg   --pwd /workspace containers/ctspinopelvic1k-ts.sif   python3 scripts/complete_0816.py     --ct    data/fix_0816/ct/0816_ct.nii.gz     --label data/fix_0816_pseudo/labels/0816_label.nii.gz     --out   data/fix_0816_v3/labels/0816_label.nii.gz     --report data/fix_0816_v3/0816_completion_report.json     --no_hardware --device gpu

# --- step 2: Moller binary rib net, grafted onto the TS rib numbering -------
singularity exec --nv --bind /wsu/home/go/go24/go2432/CTSpinoPelvic1K:/workspace --bind /wsu/home/go/go24/go2432/totalseg_weights:/wsu/home/go/go24/go2432/totalseg_weights --bind /wsu/home/go/go24/go2432/.totalseg:/wsu/home/go/go24/go2432/.totalseg   --pwd /workspace containers/ctspinopelvic1k-ts.sif   python3 scripts/build_v4_ribs.py     --v3_dir data/fix_0816_v3     --out_dir data/fix_0816_v4     --model_folder /wsu/home/go/go24/go2432/CTSpinoPelvic1K/models/moller_ribseg/ribseg_model_weights     --folds 0 --device cuda
echo "ALL DONE"
ls -la data/fix_0816_v4/labels/
