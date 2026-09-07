#!/usr/bin/env bash
#SBATCH --job-name=fix0816_pelvis
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fix0816_pelvis_%j.out
#SBATCH --error=logs/fix0816_pelvis_%j.err
#SBATCH --exclude=msa1
set -euo pipefail
cd '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'
module load singularity/3.5.2
export SINGULARITYENV_HF_TOKEN="$(cat $HOME/.hf_org_token)"
export SINGULARITYENV_PYTHONPATH=/workspace/scripts
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the Dataset803 checkpoints were trained under a custom trainer class; nnU-Net
# looks it up BY NAME at predict time, so the repo's inference shim is bound in.
singularity exec --nv   --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K':/workspace   --bind '/wsu/home/go/go24/go2432/CTSpinoPelvic1K'/containers/nnunet_wandb_variant.py:'/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py':ro   --pwd /workspace   containers/ctspinopelvic1k-ts.sif   python3 scripts/pseudolabel.py     --hf_export data/fix_0816     --out       data/fix_0816_pseudo     --nnunet_results ckpt_mirror     --splits    data/fix_0816/splits_5fold.json     --skip_download     --device cuda
