#!/usr/bin/env bash
#SBATCH -q primary
#SBATCH -J upload_archive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Upload the build archive (slurm/archive_build_inputs.sh output) into the Zenodo draft
# that already carries the reserved DOI. Resumable: files already up at the right size
# are skipped. Does NOT publish; publishing is a separate, deliberate step:
#     ZENODO_TOKEN=$(cat ~/.zenodo_token) python zenodo/upload.py --publish 22647933
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DEP="${DEPOSITION:-22647933}"
A="${ARCHIVE_DIR:-$HOME/build_archive}"
export ZENODO_TOKEN
ZENODO_TOKEN="$(tr -d '\r\n' < "$HOME/.zenodo_token")"
ls -la "$A"
"$HOME/mambaforge/bin/python" zenodo/upload.py --deposition "$DEP" --dir "$A" \
    --metadata zenodo/build_archive.json
echo "done $(date)"
