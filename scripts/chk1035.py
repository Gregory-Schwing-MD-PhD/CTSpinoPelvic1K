import numpy as np, nibabel as nib
from scipy import ndimage
for tag, f in (("v5_final (before any hardware work)", "/data/v5_final/1035_label.nii.gz"),
               ("v6 (after)", "/data/hf_export_v6/labels/1035_label.nii.gz")):
    img = nib.load(f); lab = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
    print(f"\n  {tag}")
    for sid, nm in ((30, "left_hip"), (31, "right_hip")):
        m = lab == sid
        if not m.any():
            print(f"    {nm}: absent"); continue
        cc, n = ndimage.label(m)
        sizes = ndimage.sum(m, cc, range(1, n+1))
        order = np.argsort(sizes)[::-1]
        parts = []
        for k in order[:3]:
            idx = np.argwhere(cc == int(k)+1)
            x = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, 0].mean()
            parts.append(f"{int(sizes[k]):,}@x={x:.0f}")
        print(f"    {nm}: {n} piece(s)  " + "  ".join(parts))
