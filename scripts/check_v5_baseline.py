"""Which v5 is v5? The verify compared against one copy and the build read another.

build_v6.py takes its labels from data/v5_final, the working directory. verify_v6.py
compared the result against data/hf_export_v5/labels, the published tree. If those two have
drifted, then "1153 changed and should not have" says nothing about v6 -- it says the two
copies of v5 disagree, and v6 inherited whichever one it read.

That is worth knowing either way, and it has to be settled before publishing: if the
published v5 and the working v5 differ, then v6 built from the working copy contains changes
that were never in v5, and nobody has looked at them.

Also separates the dust the verify flagged: dust in a structure the implant NEVER TOUCHED is
pre-existing, and the consolidation pass deliberately only cleaned structures the metal took
voxels from.
"""
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

V5F = Path("data/v5_final")
V5E = Path("data/hf_export_v5/labels")
V6 = Path("data/hf_export_v6/labels")
HW = (76, 77, 78, 79, 80, 81, 82)
HWCASES = {"0068", "0188", "0247", "0443", "0485", "0515", "0671", "0974", "1003",
           "1035", "1128"}

cases = sorted(p.name[:4] for p in V5F.glob("*_label.nii.gz"))
print(f"  {len(cases)} cases in v5_final")

# ---- 1. do the two v5 copies agree? --------------------------------------------------
import random
random.seed(0)
sample = [c for c in cases if c not in HWCASES]
random.shuffle(sample)
sample = sample[:60] + ["1153"]
differ, same, missing = [], 0, 0
for cid in sample:
    a = V5F / f"{cid}_label.nii.gz"
    b = V5E / f"{cid}_label.nii.gz"
    if not b.exists():
        missing += 1
        continue
    x = np.asanyarray(nib.load(str(a)).dataobj)
    y = np.asanyarray(nib.load(str(b)).dataobj)
    if x.shape != y.shape or not np.array_equal(x, y):
        n = int((x != y).sum()) if x.shape == y.shape else -1
        differ.append((cid, n))
    else:
        same += 1
print(f"\n  v5_final vs hf_export_v5/labels, {len(sample)} sampled: "
      f"{same} identical, {len(differ)} differ, {missing} missing")
for cid, n in differ[:10]:
    print(f"    {cid}: {n:,} voxels differ" if n >= 0 else f"    {cid}: different shape")

# ---- 2. is the flagged dust pre-existing? --------------------------------------------
print("\n  dust the verify flagged -- was it already in v5?")
for cid, sid in (("0068", 31), ("0188", 30), ("0188", 32), ("0443", 31)):
    out = []
    for tag, d in (("v5_final", V5F), ("v6", V6)):
        f = d / f"{cid}_label.nii.gz"
        if not f.exists():
            out.append(f"{tag}: missing")
            continue
        img = nib.load(str(f))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        m = arr == sid
        if not m.any():
            out.append(f"{tag}: absent")
            continue
        cc, n = ndimage.label(m)
        sizes = ndimage.sum(m, cc, range(1, n + 1))
        out.append(f"{tag}: {int(sum(1 for s in sizes if s * vox < 30))} specks")
    print(f"    {cid} label {sid}:  " + "   ".join(out))
