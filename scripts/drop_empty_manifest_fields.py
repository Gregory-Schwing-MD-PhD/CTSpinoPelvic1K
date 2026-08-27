"""drop_empty_manifest_fields.py — remove manifest fields that are null in every record.

The deposit check refuses a manifest that declares a field and never fills it, and it is
right to: a column of 802 nulls reads as an annotation layer to anyone who lists the schema,
and only somebody who inspects the values finds out it is empty. That is exactly how
`castellvi_type` sat unnoticed for a release cycle.

Two survive that test on v6: `patient_size`, a DICOM tag that was never extracted, and
`postwrite_hip_bone_pct`, an internal QC metric from a pipeline stage that no longer runs.
Neither has ever held a value in any release.

DROPPED RATHER THAN FILLED, because there is nothing to fill them with. Inventing a value
would be worse than either alternative. A field that is genuinely wanted later can be added
when it has data behind it.

Refuses to drop a field that holds a value anywhere, so it cannot quietly delete data.

    python scripts/drop_empty_manifest_fields.py --manifest data/hf_export_v6/manifest.json \\
        --fields patient_size,postwrite_hip_bone_pct --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fields", required=True, help="comma separated")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    p = Path(a.manifest)
    recs = json.loads(p.read_text(encoding="utf-8"))
    want = [f.strip() for f in a.fields.split(",") if f.strip()]
    print(f"  {len(recs)} record(s)")

    drop, keep = [], []
    for f in want:
        present = sum(1 for r in recs if f in r)
        filled = [r for r in recs
                  if r.get(f) not in (None, "", [], {}) and str(r.get(f)).lower() != "nan"]
        if not present:
            print(f"  {f}: not in the manifest at all")
            continue
        if filled:
            ex = filled[0]
            keep.append(f)
            print(f"  {f}: KEPT -- {len(filled)} record(s) hold a value, "
                  f"e.g. {ex.get('volume_id')} = {ex.get(f)!r}")
        else:
            drop.append(f)
            print(f"  {f}: null in all {present} record(s) -> drop")

    if keep:
        print(f"\n  refusing to drop {keep}: they hold data")
    if not drop:
        print("  nothing to drop")
        return 0
    if a.check:
        print(f"\n  --check: would drop {drop}, nothing written")
        return 0

    for r in recs:
        for f in drop:
            r.pop(f, None)
    p.write_text(json.dumps(recs, indent=1) + "\n", encoding="utf-8")
    print(f"\n  dropped {drop} from {len(recs)} record(s)")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
