#!/usr/bin/env bash
#SBATCH --job-name=ctsp_ship_v5
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ship_v5_%j.out
#SBATCH --error=logs/ship_v5_%j.out

# Build the v5 release tree and push it to <repo>@v5.
#
# v5 = v4 with the FINAL labels: reviewed ribs + reviewed spine, the lumbar-rib class
# applied, the hand-added thoracic vertebrae, and the speckle stripped. The CT volumes are
# byte-identical to v4, so they are HARDLINKED rather than copied -- 225GB of images does
# not need duplicating on disk to ship a label revision.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}")
V4=data/hf_export_v4
V5="${V5_DIR:-data/hf_export_v5}"

echo "=== 1. assemble the v5 tree ==="
mkdir -p "${V5}"
# images: hardlink (same bytes as v4, no duplication)
if [ ! -d "${V5}/ct" ]; then
  cp -al "${V4}/ct" "${V5}/ct"
fi
# labels: the finalised ones
rm -rf "${V5}/labels"; mkdir -p "${V5}/labels"
cp -a data/v5_final/*_label.nii.gz "${V5}/labels/"
for f in manifest.json splits_5fold.json dataset_interface.py README.md; do
  [ -f "${V4}/${f}" ] && cp -a "${V4}/${f}" "${V5}/${f}"
done
echo "  ct:     $(ls "${V5}/ct" | wc -l) files"
echo "  labels: $(ls "${V5}/labels" | wc -l) files"

echo; echo "=== 2. document the classes the release actually ships ==="
"${RUN[@]}" python3 scripts/update_label_docs.py --export-dir "${V5}"

echo; echo "=== 3. carry the QC and morphometrics with the release ==="
mkdir -p "${V5}/qc"
for d in qc_rib_incidence_v5_final qc_speckle morphometrics qc_hardware; do
  [ -d "$d" ] && cp -a "$d" "${V5}/qc/" || true
done
[ -f docs/DEFERRED_CASES.md ] && cp -a docs/DEFERRED_CASES.md "${V5}/qc/"
[ -f docs/LAB_JOURNAL.md ]    && cp -a docs/LAB_JOURNAL.md    "${V5}/qc/"
du -sh "${V5}/qc" 2>/dev/null || true

echo; echo "=== 4. push -> ${HF_REPO_ID:-anonymous-mlhc/CTSpinoPelvic1K}@v5 ==="
: "${HF_TOKEN:?set HF_TOKEN}"
export HF_REPO_ID="${HF_REPO_ID:-anonymous-mlhc/CTSpinoPelvic1K}"
PUSH=1 SKIP_EXPORT=1 WIPE_REMOTE=0 \
  HF_REVISION=v5 HF_EXPORT_DIR="${V5}" HF_WORKERS=8 HF_PRIVATE=0 \
  bash slurm/export_dataset.sh

echo; echo "=== done ==="
