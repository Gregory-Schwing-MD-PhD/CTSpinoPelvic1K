"""Per-structure coverage across the release — the MPDA inventory question.

The editorial requires the number of missing values per data type. 0816 turned out to have
no pelvis and no ribs at all, with prov_pelvis unset, so "how many more like it" is both a
data-quality question and a required table.
"""
import json, collections
recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
n = len(recs)
print(f"  {n} cases in manifest\n")
for field in ("config", "match_type", "prov_spine", "prov_pelvis", "partial_annotation"):
    c = collections.Counter(str(r.get(field)) for r in recs)
    print(f"  {field:20s} " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
print()
miss = [r for r in recs if r.get("prov_pelvis") in (None, "", "None")]
print(f"  prov_pelvis unset: {len(miss)} cases")
print("   ", " ".join(sorted(str(r.get('label_file','')).split('/')[-1]
      .replace('_label.nii.gz','') for r in miss))[:600])
