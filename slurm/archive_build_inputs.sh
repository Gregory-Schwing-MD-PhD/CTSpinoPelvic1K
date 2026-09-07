#!/usr/bin/env bash
#SBATCH -q primary
#SBATCH -J archive_build
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =============================================================================
# archive_build_inputs.sh — assemble the build archive described in docs/BUILD_CHAIN.md,
# set up the v10 working copy, and prove the v7 -> v9 -> v10 hops regenerate the
# published volumes.
#
#   cd ~/CTSpinoPelvic1K && sbatch slurm/archive_build_inputs.sh
#
# Before this runs, grid_worktree_snapshot.sh has put the checkout's uncommitted state in
# $STAGE_DIR/08_grid_worktree, and delete_nnunet_preprocessed.sh has left the nnU-Net
# plans in $STAGE_DIR/09_nnunet_meta.
#
# Outputs:
#   $ARCHIVE_DIR   flat folder of tar.zst components + README + checksums, ready for
#                  zenodo/upload.py --dir $ARCHIVE_DIR --metadata zenodo/build_archive.json
#   $WORKCOPY      ~/data/CTSpinoPelvic1K: v10 labels and metadata from Zenodo, CTs
#                  hard-linked from the existing export (no second copy)
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load singularity 2>/dev/null || true
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"
# singularity 3.5 has no --env; SINGULARITYENV_* is the portable spelling
export SINGULARITYENV_PYTHONPATH=/w/scripts SINGULARITYENV_PYTHONUNBUFFERED=1
RUN=(singularity exec --bind "$(pwd)":/w --pwd /w "$SIF")
SKIP_TO="${SKIP_TO:-1}"     # rerun from a given step (1..5) after a partial failure
HFPY="$HOME/mambaforge/envs/hfup/bin/python"
A="${ARCHIVE_DIR:-$HOME/build_archive}"
S="${STAGE_DIR:-$HOME/build_archive_stage}"
W="${WORKCOPY:-$HOME/data/CTSpinoPelvic1K}"
W8="${SLURM_CPUS_PER_TASK:-8}"
ZEN="https://zenodo.org/api/records/22642578/files"
mkdir -p "$A" "$S" "$W" logs
D="$PWD/data"

# pack NAME [-C ABSDIR] path...   missing paths are reported and skipped, never fatal
pack() {
  local name=$1; shift
  local args=() dir="$PWD"
  while [ $# -gt 0 ]; do
    if [ "$1" = "-C" ]; then dir=$2; args+=(-C "$2"); shift 2; continue; fi
    if [ -e "$dir/$1" ]; then args+=("$1"); else echo "  ! missing: $dir/$1"; fi
    shift
  done
  echo "== $name"
  tar --owner=0 --group=0 --numeric-owner -cf - "${args[@]}" \
    | zstd -T"$W8" -3 -q --force -o "$A/$name.tar.zst"
  ls -la "$A/$name.tar.zst"
}

if [ "$SKIP_TO" -le 3 ]; then
echo "### 1. v10 working copy at $W  ($(date))"
if [ ! -s "$S/labels_v10.zip" ]; then
  curl -sSL --retry 5 -o "$S/labels_v10.zip" "$ZEN/labels.zip/content"
fi
echo "1ad7afa871daee5d545a40e561a7c09f  $S/labels_v10.zip" | md5sum -c -
for f in README.md LICENSE fetch_from_tcia.py reconstruct_ct.py splits_5fold.json \
         dataset_labels.json KNOWN_ISSUES.md manifest.json SHA256SUMS.txt; do
  curl -sSL --retry 5 -o "$W/$f" "$ZEN/$f/content"
done
rm -rf "$W/labels"
unzip -q -o "$S/labels_v10.zip" -d "$W"
( cd "$W" && sha256sum --quiet -c SHA256SUMS.txt ) && echo "  v10 checksums verified"
mkdir -p "$W/ct"
n=0
for f in "$D"/hf_export_osc/ct/*.nii.gz; do ln -f "$f" "$W/ct/$(basename "$f")"; n=$((n+1)); done
echo "  $n CT volumes hard-linked; $(ls "$W/labels" | wc -l) labels"

echo "### 2. sources and placement  ($(date))"
pack 01_sources -C "$D" ctspine1k/raw_data/labels ctspine1k/metadata ctspine1k/README.md \
     ctspine1k/README.txt ctspine1k/CTSpine1K.py ctpelvic1k/masks ctpelvic1k/metadata
pack 02_placement -C "$D" patient_db.json patient_db_summary.txt placed/placed_manifest.json \
     placed/placed_manifest_orientation_fixed.json placed/spine placed/pelvic placed/pelvic_propagated \
     -C "$PWD" configs/flip_list.json configs/default.env
pack 03_v4_base -C "$D/hf_export_v4" labels manifest.json dataset_labels.json rib_worklist.json \
     splits_5fold.json README.md dataset_interface.py _v4ribs_done -C "$PWD" manifest_v4.json

echo "### 3. student reviews  ($(date))"
"$HFPY" scripts/archive_reviews.py --out "$S/04_reviews" \
    --private-map "$HOME/build_archive_private/annotator_map.json"
pack 04_reviews -C "$S" 04_reviews
fi

if [ "$SKIP_TO" -le 4 ]; then
echo "### 4. v5, v6 inputs, v7  ($(date))"
pack 05_v5 -C "$D" v5_final thoracic_fix fix_0816/labels fix_0816/manifest.json \
     fix_0816/splits_5fold.json fix_0816_v3/0816_completion_report.json fix_0816_v3/labels \
     fix_0816_v4/labels fix_0816_pseudo fix_0816_pseudo/manifest.csv \
     0816_label_v4.nii.gz detached_pieces_final.json detached_pieces_v6.json l6_audit.json l6_truth.json \
     itksnap_v4_labels.txt itksnap_v5_labels.txt itksnap_v6_labels.txt \
     -C "$PWD" qc_speckle/speckle_report.json final_build_report.csv qc_final qc_hardware
pack 06_v6_inputs -C "$D" hardware_final hardware_final_predust hardware_fix hip_crossover_backup \
     hip_transpose_backup hip_crossover_after.json hip_crossover_v6.json \
     -C "$D/hf_export_v6" manifest.json dataset_labels.json
pack 07_v7 -C "$D" s1_recarve -C "$D/zenodo_v7" manifest.json dataset_labels.json SHA256SUMS.txt \
     README.md KNOWN_ISSUES.md splits_5fold.json
pack 08_grid_worktree -C "$S" 08_grid_worktree
if [ -d "$S/09_nnunet_meta" ]; then pack 09_nnunet_meta -C "$S" 09_nnunet_meta; fi
fi

echo "### 5. chain check: v7 -> v9 -> v10 against the published v10  ($(date))"
rm -rf "$S/verify"; mkdir -p "$S/verify"
"${RUN[@]}" python3 scripts/renumber_labels.py --map v9 --labels data/s1_recarve/labels \
    --out "$S/verify/v9" --workers "$W8"
"${RUN[@]}" python3 scripts/renumber_labels.py --map v10 --labels "$S/verify/v9" \
    --out "$S/verify/v10" --workers "$W8"
"${RUN[@]}" python3 scripts/compare_label_trees.py --a "$S/verify/v10" --b "$W/labels" \
    --workers "$W8" --out "$A/CHAIN_CHECK_v7_to_v10.txt" || echo "  chain check reported differences; read $A/CHAIN_CHECK_v7_to_v10.txt"

echo "### 6. manifest and checksums  ($(date))"
cp docs/BUILD_CHAIN.md "$A/README.md"
( cd "$A" && for t in *.tar.zst; do printf '%s\t%s files\t%s\n' "$t" "$(zstd -dc "$t" | tar -tf - | wc -l)" "$(du -h --apparent-size "$t" | cut -f1)"; done ) > "$A/CONTENTS.txt"
( cd "$A" && sha256sum *.tar.zst README.md CONTENTS.txt CHAIN_CHECK_v7_to_v10.txt ) > "$A/SHA256SUMS.txt"
cat "$A/CONTENTS.txt"
du -sh --apparent-size "$A"
echo "done $(date)"
