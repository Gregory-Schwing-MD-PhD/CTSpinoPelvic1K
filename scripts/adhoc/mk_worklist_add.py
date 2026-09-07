"""Add the cases needing spine work back onto the review worklist.

Reasons kept distinct so a reviewer knows what they are opening:
  no-thoracic        thoracic vertebrae ARE in the field but unlabelled. 0696 and 1106 are
                     deliberately excluded: their FOV starts at L1, so there is nothing to
                     add and their ribs are permanently unnameable under the naming rule.
  vertebra-speckle   vertebra labels carry stray fragments far from the body -- 0344's L5
                     has 112 connected components, and fragments sitting up among the ribs
                     are what made a 6th rib appear to articulate with L3.
  no-pelvis-no-ribs  0816 is the only case in the release with no pelvis, no femurs and no
                     rib labels at all; prov_pelvis never ran and it is the single case
                     flagged partial_annotation.

Merged into the existing worklist rather than replacing it: tokens already queued keep
their original reason, and anything already present is not duplicated.
"""
import json
from pathlib import Path

NO_THORACIC = "0033 0068 0167 0241 0344 0357 0383 0389 0409 0424 0428 0720 0730 1004".split()
SPECKLE = ["0344"]
NO_PELVIS = ["0816"]
REASONS = [(NO_THORACIC, "no-thoracic: segment every thoracic vertebra in the FOV, "
                         "numbering UP from T12 (the body above L1)"),
           (SPECKLE, "vertebra-speckle: stray label fragments away from the vertebral body"),
           (NO_PELVIS, "no-pelvis-no-ribs: sacrum, hips, femurs and all ribs absent")]

recs = json.load(open("data/hf_export_v4/manifest.json"))
recs = recs if isinstance(recs, list) else recs.get("records", [])
tok = {}
for r in recs:
    s = str(r.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
    if s and r.get("token"):
        tok[s] = str(r["token"])

wl = json.load(open("class_mixing_worklist.json"))
have = {c["token"]: c for c in wl["cases"]}
added, already, unmapped = [], [], []
for cases, reason in REASONS:
    for cs in cases:
        t = tok.get(cs)
        if t is None:
            unmapped.append(cs); continue
        if t in have:
            already.append(f"{cs}(tok {t}): {have[t]['reason']}")
            continue
        have[t] = {"token": t, "reason": reason, "case": cs}
        added.append(f"{cs} -> token {t}")

# tokens are mixed: plain integers and CTC-nnnnnnnn strings, so sort numerically only
# where that is meaningful and lexically otherwise
def _key(c):
    t = str(c["token"])
    return (0, int(t), "") if t.isdigit() else (1, 0, t)
wl["cases"] = sorted(have.values(), key=_key)
wl["tokens"] = [c["token"] for c in wl["cases"]]
Path("class_mixing_worklist.json").write_text(json.dumps(wl, indent=1))
print(f"  added {len(added)}:")
for a in added:
    print("    ", a)
if already:
    print(f"  already queued ({len(already)}):")
    for a in already:
        print("    ", a)
if unmapped:
    print("  NO TOKEN:", unmapped)
print(f"\n  worklist now {len(wl['cases'])} cases")
