"""Repair has_l6, has_lumbar_rib and n_lumbar_labels from the label volumes.

These three fields are what a user filters on to find the transitional cases this dataset
exists to document, and all three are wrong as shipped:

  has_l6 is true in one record, 0706, which contains no L6 at all -- a false positive -- and
    false in the eighteen that do;
  has_lumbar_rib is false everywhere, hiding all sixteen records that carry one;
  n_lumbar_labels reads 0 in 799 of 802 records, which cannot be true of a lumbar spine.

The failure mode is quiet and total: someone selecting the six-lumbar cases gets one record
that is not one, and someone selecting lumbar ribs gets an empty set and concludes the
dataset has none. Neither has any way to notice.

Rewrites the three fields from the voxel census, records the laterality of each lumbar rib
since the manuscript reports it, and prints what changed so the edit is auditable rather than
trusted.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

L6 = 25
LUMBAR = list(range(20, 26))
RIB_L, RIB_R = 74, 75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--also", nargs="*", default=[],
                    help="other manifest.json copies to repair identically")
    a = ap.parse_args()
    src = Path(a.src)

    truth, lat = {}, {}
    files = sorted((src / "labels").glob("*_label.nii.gz"))
    for i, p in enumerate(files, 1):
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        present = {int(v) for v in np.unique(arr) if v}
        cid = p.name[:4]
        left, right = RIB_L in present, RIB_R in present
        truth[cid] = {
            "has_l6": L6 in present,
            "has_lumbar_rib": left or right,
            "n_lumbar_labels": sum(1 for v in LUMBAR if v in present),
        }
        if left or right:
            lat[cid] = "bilateral" if (left and right) else ("left" if left else "right")
        if i % 200 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    n_l6 = sum(1 for v in truth.values() if v["has_l6"])
    n_rib = sum(1 for v in truth.values() if v["has_lumbar_rib"])
    print(f"\n  measured: {n_l6} with L6, {n_rib} with a lumbar rib")
    print(f"  lumbar rib laterality: {dict(Counter(lat.values()))}")
    for side in ("left", "right"):
        ones = sorted(c for c, s in lat.items() if s == side)
        if ones:
            print(f"    {side}-only: {ones}")
    print(f"  lumbar labels per record: {dict(sorted(Counter(v['n_lumbar_labels'] for v in truth.values()).items()))}")

    for target in [src / "manifest.json"] + [Path(x) for x in a.also]:
        if not target.exists():
            print(f"  {target}: absent, skipped")
            continue
        recs = json.loads(target.read_text(encoding="utf-8"))
        as_list = isinstance(recs, list)
        rows = recs if as_list else recs.get("records", list(recs.values()))
        changed = Counter()
        for r in rows:
            cid = str(r.get("volume_id"))
            t = truth.get(cid)
            if not t:
                continue
            for k, v in t.items():
                if r.get(k) != v:
                    changed[k] += 1
                    r[k] = v
            r["lumbar_rib_side"] = lat.get(cid)
        target.write_text(json.dumps(recs, indent=1) + "\n", encoding="utf-8")
        print(f"  {target}: corrected {dict(changed)}; added lumbar_rib_side")

    Path(src.parent / "l6_truth.json").write_text(
        json.dumps({"truth": truth, "laterality": lat}, indent=1, sort_keys=True),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
