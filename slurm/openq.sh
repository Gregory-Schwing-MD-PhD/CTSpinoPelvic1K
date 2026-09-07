#!/usr/bin/env bash
#SBATCH -q primary
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/openq_%j.out
#SBATCH -e logs/openq_%j.err
set -euo pipefail
cd source configs/default.env
export SINGULARITY_TMPDIR=/tmp/oq_mkdir -p singularity exec --bind \/mnt/c/Users/grego/OneDrive/Desktop/CTSpinoPelvic1K-1:/w,\:/data --pwd /w containers/ctspinopelvic1k.sif python3 -u /w/scripts/openq.py
