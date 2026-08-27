"""Does the shipped manifest's hardware flag agree with the shipped label volumes?

The hardware search ran at 1800 HU and flagged 84 records. A radiologist then read every one
of them and kept 11. If `manifest.json` still carries the pre-read flag, then the deposit
ships a field that asserts instrumentation in 73 records that have none -- and KNOWN_ISSUES.md
tells the reader to filter on exactly that field before measuring the gap between two bones.
A flag that over-reports is not a conservative error here: it silently deletes 73 clean cases
from any analysis that honours it, and it tells 73 patients' worth of anatomy that it is
metal.

The label volume is the only thing that can settle it: an id in 76..82 is present or it is
not. This compares the two and prints every disagreement in both directions.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

HW = list(range(76, 83))
NAMES = {76: "hardware_screw", 77: "hardware_cage", 78: "hardware_rod",
         79: "hardware_plate", 80: "hardware_arthroplasty",
         81: "hardware_si_screw", 82: "hardware_osteosynthesis"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--out", default="hardware_truth.json")
    a = ap.parse_args()

    src = Path(a.src)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    flagged = {str(r["volume_id"]) for r in recs if r.get("hardware")}
    print(f"  manifest says hardware in {len(flagged)} record(s)")

    truth = {}
    labs = sorted((src / "labels").glob("*_label.nii.gz"))
    for i, p in enumerate(labs, 1):
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        present = {int(v): int((arr == v).sum()) for v in HW if (arr == v).any()}
        if present:
            truth[p.name[:4]] = present
        if i % 100 == 0:
            print(f"    scanned {i}/{len(labs)}", flush=True)

    print(f"  label volumes actually containing 76..82: {len(truth)}\n")
    for cid, ids in sorted(truth.items()):
        detail = ", ".join(f"{NAMES.get(k, k)} {v:,}" for k, v in sorted(ids.items()))
        print(f"   {cid}  {detail}")

    tot = Counter()
    for ids in truth.values():
        tot.update(ids)
    print("\n  per class:")
    for k, v in sorted(tot.items()):
        n = sum(1 for r in truth.values() if k in r)
        print(f"   {k:>3} {NAMES.get(k, ''):<24} {n:>3} case(s)  {v:>10,} voxels")

    over = sorted(flagged - set(truth))
    under = sorted(set(truth) - flagged)
    print(f"\n  flagged but no hardware in the volume : {len(over)}")
    if over:
        print(f"    {', '.join(over)}")
    print(f"  hardware in the volume but not flagged: {len(under)}")
    if under:
        print(f"    {', '.join(under)}")

    Path(a.out).write_text(json.dumps(truth, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
