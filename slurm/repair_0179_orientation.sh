#!/usr/bin/env bash
# Rebuild 0179 in its own orientation.
#
# The anchor pass loaded the label through as_closest_canonical and then saved the
# reoriented array under the CANONICAL affine. Renumbering ribs is pure id arithmetic
# and never needed the reorientation; the effect was to transpose the label away from
# its CT -- 512x512x645 against a 512x645x512 scan -- so ITK-SNAP refuses the pair.
#
# This restores the pre-anchor backup and REPLAYS THE RECORDED REMAP, rather than
# recomputing it: the plan in anchor_plan.json is what was reviewed and accepted, and a
# recomputation today would run under a different vertebra-selection rule and could
# legitimately differ. Repairing a file is not the moment to also change its answer.
#SBATCH --job-name=ctsp_fix0179
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fix0179_%j.out
#SBATCH --error=logs/fix0179_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/default.env
SIF="${SIF_PATH:-containers/ctspinopelvic1k.sif}"

singularity exec --bind "$(pwd)":/w --pwd /w \
  --env PYTHONPATH=/w/scripts,PYTHONUNBUFFERED=1 "${SIF}" python3 - <<'PY'
import json
import numpy as np
import nibabel as nib

BACKUP = "qc_rib_anchor/pre_anchor/0179_label.nii.gz"
TARGET = "data/v5_final/0179_label.nii.gz"
CT = "data/hf_export/ct/0179_ct.nii.gz"

plan = json.load(open("qc_rib_anchor/anchor_plan.json"))
remap = {}
for c in plan["cases"]:
    if c["case"] == "0179":
        remap = {int(k): int(v) for k, v in c["remap"].items()}
assert remap, "no recorded remap for 0179"

img = nib.load(BACKUP)                      # NO as_closest_canonical
lab = np.asanyarray(img.dataobj)
ct_shape = nib.load(CT).shape
print("backup shape", lab.shape, " ct shape", ct_shape)
assert lab.shape == ct_shape, "the backup itself does not match the CT"

lut = np.arange(int(lab.max()) + 1, dtype=np.int64)
for o, n in remap.items():
    if o < len(lut):
        lut[o] = n
out = lut[lab.astype(np.int64)].astype(img.get_data_dtype())

assert out.shape == lab.shape
assert int((out > 0).sum()) == int((lab > 0).sum()), "voxel count changed"
rib_ids = set(range(34, 58))
changed = {int(v) for v in np.unique(lab[lab != out])}
assert changed <= rib_ids, f"non-rib ids changed: {sorted(changed)}"
before = sum(1 for i in rib_ids if (lab == i).any())
after = sum(1 for i in rib_ids if (out == i).any())
assert after == before, f"rib count {before} -> {after}: the remap collided"

nib.save(nib.Nifti1Image(out, img.affine, img.header), TARGET)
chk = nib.load(TARGET)
print("rewritten shape", chk.shape, " matches CT:", chk.shape == ct_shape)
print("remap replayed:", ", ".join(f"{o}->{n}" for o, n in sorted(remap.items())))
PY

echo "=== done ==="
