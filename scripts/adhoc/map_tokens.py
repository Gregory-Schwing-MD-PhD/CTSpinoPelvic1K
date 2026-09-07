import json
recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
want = {"22", "46", "103", "155", "248", "770"}
wl = {c["token"]: c.get("reason", "?") for c in
      json.load(open("class_mixing_worklist.json"))["cases"]}
for r in recs:
    t = str(r.get("token"))
    if t in want:
        s = str(r.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
        reason = wl.get(t, "?")
        print("  token %-5s = case %s   %s" % (t, s, reason[:70]))
