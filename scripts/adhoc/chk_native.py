import json, collections
recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
by = {}
for r in recs:
    s = str(r.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
    if s:
        by[s] = r
BATCH = "0033 0068 0167 0241 0344 0357 0383 0389 0409 0424 0428 0720 0730 1004".split()
print("  case   config          prov_spine  lstv_pelvic       lstv_vertebral")
n_native = 0
for c in BATCH:
    r = by.get(c, {})
    native = r.get("config") == "pelvic_native"
    n_native += native
    print(f"  {c}   {str(r.get('config')):15s} {str(r.get('prov_spine')):11s} "
          f"{str(r.get('lstv_pelvic')):17s} {r.get('lstv_vertebral')}")
print(f"\n  pelvic_native in this batch: {n_native}/{len(BATCH)}")
allnat = [s for s, r in by.items() if r.get("config") == "pelvic_native"]
print(f"  pelvic_native in the release: {len(allnat)}")
print("   ", " ".join(sorted(allnat)))
