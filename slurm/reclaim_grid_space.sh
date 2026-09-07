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
#   PHASE=A   public sources, checkpoint copies that are on Hugging Face, superseded
#             export trees, published Zenodo staging, caches. Nothing a running archive
#             job reads.
#   PHASE=B   the archived intermediates the archive job reads (v5_final, s1_recarve,
#             placed, hardware and 0816 folders, hip backups, v6 export, v7 staging).
#             Run only after slurm/archive_build_inputs.sh has finished.
#   PHASE=all both.
#   DRY_RUN=1 (default) prints sizes and deletes nothing; DRY_RUN=0 deletes.
#
# Kept on purpose (Greg, 2026-09-07): ~/data/CTSpinoPelvic1K (v10 + CTs), ~/VerSeFusion,
# ~/XRSpinoPelvic1K, the RNA-seq study, mambaforge, every .sif image, the nnU-Net raw
# tree, installers, the Möller weights, the speckle QC backup, the TotalSegmentator
# weights, spinesurg-ct-nnunet/nnunet/results/Dataset802, spinopelvic-seg/.git.
# =============================================================================
set -uo pipefail
cd "$HOME"
DRY_RUN="${DRY_RUN:-1}"
PHASE="${PHASE:-A}"
C="CTSpinoPelvic1K/data"

PHASE_A=(
  "$C/ctspine1k" "$C/ctpelvic1k"
  "CTSpinoPelvic1K/nnunet/results" "CTSpinoPelvic1K/ckpt_mirror" "spinopelvic-seg/nnunet/results"
  "$C/hf_export_v2_work" "$C/hf_export_v3_work" "$C/hf_export_v4" "$C/hf_export_v5" "$C/hf_export_osc"
  "$C/zenodo_v6" "$C/zenodo_upload"
  ".singularity/cache" "singularity_cache/cache" ".cache/huggingface/hub"
)
PHASE_B=(
  "$C/placed" "$C/v5_final" "$C/v5_smoke" "$C/s1_recarve"
  "$C/hardware_final" "$C/hardware_final_predust" "$C/hardware_fix"
  "$C/fix_0816" "$C/fix_0816_v3" "$C/fix_0816_v4" "$C/fix_0816_pseudo" "$C/fix_0816_pseudo_work"
  "$C/hip_crossover_backup" "$C/hip_transpose_backup" "$C/thoracic_fix"
  "$C/hf_export_v6" "$C/zenodo_v7"
  "grid_untracked_aside" "build_archive_stage/verify"
)
case "$PHASE" in
  A) TARGETS=("${PHASE_A[@]}") ;;
  B) TARGETS=("${PHASE_B[@]}") ;;
  all) TARGETS=("${PHASE_A[@]}" "${PHASE_B[@]}") ;;
  *) echo "PHASE must be A, B or all"; exit 1 ;;
esac

# the CT store must survive as hard links in the working copy before any export tree goes
W="$HOME/data/CTSpinoPelvic1K"
nct=$(ls "$W/ct" 2>/dev/null | wc -l)
if [ "$nct" -ne 802 ]; then echo "working copy has $nct CT volumes, not 802; refusing"; exit 2; fi
if [ "$DRY_RUN" = "0" ] && [ "$PHASE" != "B" ] && squeue -u "$USER" -h -o %j | grep -q archive_build; then
  echo "archive_build is running; phase A is still safe, but re-check the list. Continuing."
fi
if [ "$DRY_RUN" = "0" ] && [ "$PHASE" != "A" ] && squeue -u "$USER" -h -o %j | grep -q archive_build; then
  echo "archive_build is still running and reads phase-B folders; refusing"; exit 3
fi

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
echo "listed: $((total / 1024 / 1024)) GB apparent (export trees over-count the hard-linked CTs)  PHASE=$PHASE DRY_RUN=$DRY_RUN"
if [ "$DRY_RUN" = "0" ]; then echo "deleted at $(date)"; wsuquota 2>/dev/null | head -3; fi
exit 0
