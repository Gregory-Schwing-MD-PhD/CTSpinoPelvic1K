"""Where does one case's LSTV status come from, and did CTPelvic1K flag it?

The pelvic side of the label inherits from CTPelvic1K, whose authors encoded the variant in
their MASK FILENAMES (`..._0006_4_260_sacralization_mask_4label.nii.gz`). So a case can be
checked against a human annotation from the source dataset rather than against our own
derived measures -- which matters when the derived measures are exactly what is in doubt.

    python chk_case_lstv.py 0344 [0008 ...]
"""
import json
import subprocess
import sys

recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
by_stem = {}
for r in recs:
    s = str(r.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
    if s:
        by_stem[s] = r

FIELDS = ("token", "config", "match_type", "prov_spine", "prov_pelvis",
          "lstv_label", "lstv_class", "lstv_pelvic", "lstv_vertebral",
          "lstv_agreement", "lstv_confusion_zone", "castellvi_type", "has_l6",
          "n_lumbar_labels", "pelvic_series_uid", "spine_series_uid")

for stem in sys.argv[1:] or ["0344"]:
    r = by_stem.get(stem)
    print(f"\n=== {stem}")
    if r is None:
        print("  not in manifest")
        continue
    for k in FIELDS:
        print(f"  {k:22s} {r.get(k)}")

    # the pelvic uid ends in the patient number CTPelvic1K zero-pads into its filenames
    uid = str(r.get("pelvic_series_uid") or "")
    tail = uid.rsplit(".", 1)[-1] if uid else ""
    pats = [tail, tail.zfill(4)] if tail else []
    print("  --- CTPelvic1K files matching this patient:")
    hits = 0
    for pat in dict.fromkeys(pats):
        try:
            out = subprocess.run(["bash", "-lc",
                                  f"find data/ctpelvic1k -name '*{pat}*' 2>/dev/null | head -5"],
                                 capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception:
            out = ""
        for line in filter(None, out.splitlines()):
            print("     ", line.split("/")[-1])
            hits += 1
    if not hits:
        print("      (none — pelvic mask is not from CTPelvic1K for this case)")
