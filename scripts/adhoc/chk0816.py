import json, numpy as np, nibabel as nib
recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
for r in recs:
    s = str(r.get("label_file","")).split("/")[-1].replace("_label.nii.gz","")
    if s == "0816":
        for k in ("config","match_type","prov_spine","prov_pelvis","lstv_label",
                  "pelvic_series_uid","spine_series_uid","manufacturer","kvp",
                  "convolution_kernel","slice_thickness","pelvic_bone_pct","spine_bone_pct"):
            print(f"  {k:22s} {r.get(k)}")
        break
lab = np.asanyarray(nib.load("data/v5_final/0816_label.nii.gz").dataobj)
ids, cnt = np.unique(lab[lab>0], return_counts=True)
print("  label ids present:", dict(zip(ids.tolist(), cnt.tolist())))
ct = np.asanyarray(nib.load("data/hf_export_v4/ct/0816_ct.nii.gz").dataobj).astype(np.float32)
print(f"  CT min={ct.min():.0f} max={ct.max():.0f} mean={ct.mean():.0f} "
      f"p1={np.percentile(ct,1):.0f} p50={np.percentile(ct,50):.0f} p99={np.percentile(ct,99):.0f}")
print(f"  frac >250HU: {(ct>250).mean():.3f}   frac <-500 (air): {(ct<-500).mean():.3f}")
