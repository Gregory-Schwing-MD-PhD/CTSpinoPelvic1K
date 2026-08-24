"""scripts/clean_manifest.py — make every manifest field true, or remove it.

Three kinds of defect, all found by reading the released volumes rather than the manifest:

  WRONG IN BOTH DIRECTIONS. `has_l6` is true for exactly one record, and that record
  contains no L6, while all 17 records that do contain one are flagged false. `n_lumbar_labels`
  is 0 in 795 of 802 records including every LSTV case, so it is not a count of lumbar
  labels. Both are recomputed here from the label volumes.

  DECLARED AND ALWAYS EMPTY. `patient_size` and `postwrite_hip_bone_pct` are present in
  every record and populated in none. A field that is always empty is not metadata; it is a
  promise the data does not keep, and it reads as an annotation layer to anyone who lists
  the columns. This is exactly how `castellvi_type` sat null through a whole release cycle.
  They are removed.

  MISSING ENTIRELY. Nothing records that a record carries a lumbar rib, though 15 do, and
  the class exists precisely because that phenotype cannot be expressed by a rib number.
  `has_lumbar_rib` is added.

Every value written here comes from counting identifiers in the label volume itself, so the
manifest describes the release rather than the intent behind it.

    python scripts/clean_manifest.py --census morphometrics/tp_height.csv --check
    python scripts/clean_manifest.py --census morphometrics/tp_height.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

L6, SACRUM, S1 = 25, 26, 29
LUMBAR = range(20, 26)
LUM_RIB = (74, 75)

DROP = ["patient_size", "postwrite_hip_bone_pct"]


def rec_id(r):
    return Path(str(r.get("label_file", ""))).name.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/hf_export_v5/manifest.json")
    ap.add_argument("--census", default="morphometrics/tp_height.csv",
                    help="carries labels_present per record")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    doc = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = doc if isinstance(doc, list) else doc.get("records", list(doc.values()))

    present = {}
    for r in csv.DictReader(open(a.census)):
        if r.get("error"):
            continue
        present[r["case"]] = {int(v) for v in r["labels_present"].split()}
    print(f"  {len(recs)} record(s); label census covers {len(present)}")

    missing = [rec_id(r) for r in recs if rec_id(r) not in present]
    if missing:
        print(f"  ! no census for {len(missing)} record(s): {missing[:5]}")
        print("  Refusing to guess. Re-run measure_tp_height.py over the whole release.")
        return 1

    changed = Counter()
    for r in recs:
        ids = present[rec_id(r)]

        new_l6 = L6 in ids
        if r.get("has_l6") != new_l6:
            changed["has_l6"] += 1
        r["has_l6"] = new_l6

        n_lum = len([v for v in LUMBAR if v in ids])
        if r.get("n_lumbar_labels") != n_lum:
            changed["n_lumbar_labels"] += 1
        r["n_lumbar_labels"] = n_lum

        new_rib = any(v in ids for v in LUM_RIB)
        if "has_lumbar_rib" not in r:
            changed["has_lumbar_rib (added)"] += 1
        elif r.get("has_lumbar_rib") != new_rib:
            changed["has_lumbar_rib"] += 1
        r["has_lumbar_rib"] = new_rib

        for k in DROP:
            if k in r:
                del r[k]
                changed[f"{k} (removed)"] += 1

    print(f"  corrections: {dict(changed)}")
    print(f"  has_l6 true in {sum(1 for r in recs if r['has_l6'])} record(s)")
    print(f"  has_lumbar_rib true in {sum(1 for r in recs if r['has_lumbar_rib'])} record(s)")
    print(f"  n_lumbar_labels: {dict(Counter(r['n_lumbar_labels'] for r in recs))}")

    if not a.apply:
        print("\n  --check only; pass --apply to write")
        return 0

    shutil.copy(a.manifest, str(a.manifest) + ".bak")
    Path(a.manifest).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {a.manifest} (previous kept as .bak)")
    print("  The release itself is unchanged until this manifest is uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
