#!/usr/bin/env bash
#SBATCH -q primary
#SBATCH -J reclaim_space
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# reclaim_grid_space.sh — remove what the build archive (docs/BUILD_CHAIN.md, Zenodo
# 10.5281/zenodo.22647933) and the public sources make redundant, and keep the v10
# working copy at ~/data/CTSpinoPelvic1K.
#
#   DRY_RUN=1 (default) prints the size of every target and deletes nothing.
#   DRY_RUN=0 deletes. Run it only after the archive record is PUBLISHED and its
#   checksums verified; this script refuses to run otherwise unless FORCE=1.
#
# Kept on purpose: ~/data/CTSpinoPelvic1K (v10 + CTs), ~/VerSeFusion (next project),
# ~/XRSpinoPelvic1K, the RNA-seq study, mambaforge, the .sif images in
# ~/singularity_cache, spinesurg-ct-nnunet/nnunet/results/Dataset802 (checkpoints that
# are not on Hugging Face), spinopelvic-seg/.git.
# =============================================================================
set -uo pipefail
cd "$HOME"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
ARCHIVE_DOI_RECORD="${ARCHIVE_DOI_RECORD:-22647933}"

if [ "$DRY_RUN" = "0" ] && [ "$FORCE" != "1" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://zenodo.org/api/records/$ARCHIVE_DOI_RECORD")
  if [ "$code" != "200" ]; then
    echo "archive record $ARCHIVE_DOI_RECORD is not published (HTTP $code); refusing to delete. FORCE=1 overrides."
    exit 2
  fi
fi

C="CTSpinoPelvic1K/data"
TARGETS=(
  # public sources, cited in the archive README; their labels/masks are archived
  "$C/ctspine1k"
  "$C/ctpelvic1k"
  # superseded export trees (CTs are hard links; the store survives in ~/data/CTSpinoPelvic1K/ct)
  "$C/hf_export_v2_work" "$C/hf_export_v3_work" "$C/hf_export_v4" "$C/hf_export_v5"
  "$C/hf_export_v6" "$C/hf_export_osc"
  # published Zenodo staging folders
  "$C/zenodo_v6" "$C/zenodo_v7" "$C/zenodo_upload"
  # archived intermediates
  "$C/placed" "$C/v5_final" "$C/v5_smoke" "$C/s1_recarve"
  "$C/hardware_final" "$C/hardware_final_predust" "$C/hardware_fix"
  "$C/fix_0816" "$C/fix_0816_v3" "$C/fix_0816_v4" "$C/fix_0816_pseudo" "$C/fix_0816_pseudo_work"
  "$C/hip_crossover_backup" "$C/hip_transpose_backup" "$C/thoracic_fix"
  # checkpoints that are on Hugging Face (OpenSpineConsortium/spinopelvic-seg-checkpoints)
  "CTSpinoPelvic1K/nnunet/results" "CTSpinoPelvic1K/ckpt_mirror" "spinopelvic-seg/nnunet/results"
  # public weights (Zenodo 10.5281/zenodo.14850928) and regenerable QC
  "CTSpinoPelvic1K/models/moller_ribseg" "CTSpinoPelvic1K/qc_speckle/pre_speckle" "CTSpinoPelvic1K/.totalsegmentator"
  # nnU-Net raw tree (regenerable from the labels by the converter; preprocessed already gone)
  "spinesurg-ct-nnunet/nnunet/raw"
  # caches and installers
  ".singularity/cache" "singularity_cache/cache" ".cache/huggingface/hub" "conda_cache"
  "tmp/Anaconda3-2019.10-Linux-x86_64.sh" "Anaconda3-2021.11-Linux-x86_64.sh" "Anaconda2-5.3.0-Linux-x86_64.sh"
  "Miniforge3-Linux-x86_64.sh" "Mambaforge-Linux-x86_64.sh" "gcc-14.2.0.tar.xz" "gcc-9.1.0.tar.gz"
  "grid_untracked_aside" "build_archive_stage/verify"
)

total=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then
    sz=$(du -s --apparent-size "$t" 2>/dev/null | cut -f1)
    total=$((total + sz))
    printf '%10s  %s\n' "$(du -sh --apparent-size "$t" | cut -f1)" "$t"
    if [ "$DRY_RUN" = "0" ]; then rm -rf "$t"; fi
  else
    printf '%10s  %s (absent)\n' "-" "$t"
  fi
done
echo "total: $((total / 1024 / 1024)) GB  (DRY_RUN=$DRY_RUN)"
[ "$DRY_RUN" = "0" ] && { echo "deleted at $(date)"; wsuquota 2>/dev/null | head -3; }
exit 0
