"""Laterality of the sixteen lumbar ribs, read from those sixteen volumes only.

The full repair is rescanning all 802 volumes and will take the better part of an hour. The
manuscript sentence that needs correcting concerns sixteen named records, and reading sixteen
files answers it in seconds. The repair still has to run -- it rewrites the manifest for every
record -- but the paper does not have to wait on it.

The sentence being fixed reports twelve bilateral and three unilateral out of fifteen, all
three on the right. The count is now known to be sixteen, so the split behind it cannot be
right either.
"""
import os

import numpy as np
import nibabel as nib

CASES = ["0008", "0172", "0231", "0315", "0389", "0428", "0431", "0473",
         "0516", "0568", "0660", "0680", "0720", "0787", "1004", "1090"]
SRC = os.path.expanduser("~/CTSpinoPelvic1K/data/hf_export_v6/labels")
RIB_L, RIB_R = 74, 75

bil, left, right = [], [], []
for cid in CASES:
    arr = np.asanyarray(nib.load(f"{SRC}/{cid}_label.nii.gz").dataobj)
    u = {int(v) for v in np.unique(arr) if v}
    l, r = RIB_L in u, RIB_R in u
    (bil if (l and r) else (left if l else right)).append(cid)
    nl = int((arr == RIB_L).sum())
    nr = int((arr == RIB_R).sum())
    side = "bilateral" if (l and r) else ("left only" if l else "right only")
    print(f"  {cid}  {side:11}  left {nl:>7,} vox   right {nr:>7,} vox")

print(f"\n  bilateral  : {len(bil)}  {bil}")
print(f"  left only  : {len(left)}  {left}")
print(f"  right only : {len(right)}  {right}")
print(f"  total      : {len(bil) + len(left) + len(right)}")
