#!/usr/bin/env bash
#SBATCH --job-name=zenodo_up_v7
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/zenodo_up_v7_%j.out
#SBATCH --error=logs/zenodo_up_v7_%j.err
set -euo pipefail
cd /wsu/home/go/go24/go2432/CTSpinoPelvic1K
# Authorised explicitly. The deposit passed every assemble_deposit check
# and each file's md5 is verified against the server's after upload.
python3 zenodo/zenodo_newversion.py \
  --dir data/zenodo_v7 \
  --record 22139643 \
  --version v7 \
  --metadata zenodo/zenodo.json \
  --token-file $HOME/.zenodo_token \
  --publish
