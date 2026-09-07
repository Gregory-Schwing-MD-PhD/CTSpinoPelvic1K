import json
import numpy as np, nibabel as nib
from scipy import ndimage

# --- 1035 as it now stands IN THE DEPOSIT ------------------------------------
img = nib.load("/data/zenodo_v6/labels/1035_label.nii.gz")
a = np.asanyarray(img.dataobj).astype(np.int16); aff = img.affine
print("  1035 hips, as deposited:")
for sid, nm in ((30, "left_hip"), (31, "right_hip")):
    m = a == sid
    cc, n = ndimage.label(m)
    s = np.sort(ndimage.sum(m, cc, range(1, n+1)))[::-1]
    vox = float(np.prod(img.header.get_zooms()[:3]))
    tail = f", rest {int(s[1:].sum()):,} vox / {s[1:].sum()*vox:,.0f} mm3" if n > 1 else ""
    print(f"    {nm:<10} {n} piece(s), largest holds {100*s[0]/s.sum():.2f}%{tail}")

# --- 0068: what the manifest says about its transitional status --------------
recs = json.load(open("/data/zenodo_v6/manifest.json"))
r = [x for x in recs if x["volume_id"] == "0068"][0]
print("\n  0068 manifest fields bearing on the question:")
for k in ("lstv_label", "lstv_vertebral", "lstv_pelvic", "castellvi_type",
          "has_l6", "n_lumbar_labels", "prov_spine", "fusion", "fusion_level",
          "hardware_type"):
    print(f"    {k:<20} {r.get(k)}")
