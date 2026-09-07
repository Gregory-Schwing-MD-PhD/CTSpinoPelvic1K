import json
from pathlib import Path
p = Path("class_mixing_worklist.json")
wl = json.load(open(p))
n = 0
for c in wl["cases"]:
    if str(c["token"]) == "46":
        c["reason"] = ("DEFERRED - HAND ANNOTATION ONLY: interbody cage fuses L5-L6 "
                       "(implicit boundary, no gradient); 6 free lumbar so numbering "
                       "shifts to L1-L6; spine is pseudolabelled. See docs/DEFERRED_CASES.md. "
                       "Do NOT assign as routine review.")
        c["deferred"] = True
        n += 1
p.write_text(json.dumps(wl, indent=1))
print(f"flagged {n} entry; worklist has {len(wl['cases'])} cases")
