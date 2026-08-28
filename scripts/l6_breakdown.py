"""Cross-tabulate the L6 and lumbar-rib records against their transitional label.

The manuscript states how the two annotation axes disagree -- that a record can carry six
lumbar-type vertebrae and still be read as a sacralisation, depending on where the reader
started counting -- and gives the breakdown as fourteen lumbarisation, two sacralisation,
one semi-sacralisation against a count of seventeen L6 records.

The count is now known to be eighteen, so the breakdown behind it cannot be right either, and
the sentence is making a point about disagreement between axes: it has to be exact or it
undercuts itself. Same for the lumbar-rib laterality, which the paper reports as twelve
bilateral and three unilateral out of fifteen against a measured sixteen.

Reads the voxel census rather than the manifest, since the manifest flags are what this whole
correction is about.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="data/l6_truth.json")
    ap.add_argument("--audit", default="data/l6_audit.json")
    ap.add_argument("--manifest", default="data/hf_export_v6/manifest.json")
    a = ap.parse_args()

    src = Path(a.truth) if Path(a.truth).exists() else Path(a.audit)
    j = json.loads(src.read_text(encoding="utf-8"))
    truth = j.get("truth", j)
    lat = j.get("laterality", {})
    print(f"  reading {src}")

    recs = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", list(recs.values()))
    lstv = {str(r["volume_id"]): (r.get("lstv_label") or "unlabelled") for r in recs}

    # the census and the repair script name these fields differently -- "l6"/"lumbar_rib"
    # in the audit, "has_l6"/"has_lumbar_rib" in the repaired manifest. Accept either
    # rather than silently reporting zero records, which is what looking for only one
    # spelling did.
    def flag(v, *names):
        for n in names:
            if n in v:
                return bool(v[n])
        return False

    l6 = sorted(c for c, v in truth.items() if flag(v, "has_l6", "l6"))
    rib = sorted(c for c, v in truth.items()
                 if flag(v, "has_lumbar_rib", "lumbar_rib"))
    if not l6 and not rib:
        print("  no records matched; the truth file's field names are unexpected: "
              f"{sorted(next(iter(truth.values())).keys()) if truth else 'empty'}")
    print(f"\n  L6 records: {len(l6)}")
    print(f"    {', '.join(l6)}")
    c = Counter(lstv.get(x, "?") for x in l6)
    for k, v in c.most_common():
        print(f"      {k:22} {v}")

    print(f"\n  lumbar-rib records: {len(rib)}")
    print(f"    {', '.join(rib)}")
    if lat:
        cl = Counter(lat.get(x, "?") for x in rib)
        for k, v in cl.most_common():
            print(f"      {k:22} {v}")
        uni = sorted(x for x in rib if lat.get(x) in ("left", "right"))
        if uni:
            print(f"      unilateral cases: "
                  + ", ".join(f"{x}({lat[x]})" for x in uni))
    cr = Counter(lstv.get(x, "?") for x in rib)
    print("    by transitional label:")
    for k, v in cr.most_common():
        print(f"      {k:22} {v}")

    both = sorted(set(l6) & set(rib))
    print(f"\n  carrying both an L6 and a lumbar rib: {len(both)}  {both}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
