"""manifest_add_hardware.py — record surgical instrumentation in the release manifest.

docs/DEFERRED_CASES.md names the gap: "there is currently no field in the manifest recording
surgical hardware ... until that is run, the instrumented population is unknown". It matters
for a specific reason rather than for completeness: AN IATROGENIC FUSION IS
INDISTINGUISHABLE FROM A CONGENITAL ONE TO A DISTANCE MEASUREMENT. The transitional-anatomy
result measures the gap between the lowest lumbar vertebra and the sacrum, and a
cage-bridged interspace reads as "no gap" exactly like a congenitally fused transitional
vertebra. Without a field, an instrumented case cannot be excluded, because nothing says
which cases they are.

TWO LEVELS OF DETAIL, deliberately kept apart:

  DETECTED   from qc_hardware/hardware_scan.csv -- a metal threshold inside a shell around
             the spine masks. Objective, reproducible, and available for all 802. It says
             metal is present and how much; it does not say what the implant is.
  READ       what the implant actually is, and at which level. That takes someone looking at
             it, so it is populated only where that has happened and is null everywhere
             else. A null here means "not yet read", never "no hardware".

THE MANIFEST IS NOT IN GIT (data/ is ignored), so this writes a timestamped backup beside it
first. There is no `git checkout` to fall back on.

    python scripts/manifest_add_hardware.py --manifest data/hf_export_v5/manifest.json \\
        --scan qc_hardware/hardware_scan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

# What a human has actually read off the images, case by case. Nothing is inferred into
# this table; a case appears here only after someone has looked.
READ = {
    "0068": {
        "hardware_type": "interbody_cage",
        "hardware_count": 2,
        "hardware_level": "L5-L6",
        "hardware_detail": (
            "Paired threaded cylindrical interbody cages, hollow, 27 x 14 x 14 mm each, "
            "screwed into the L5-L6 disc space side by side and symmetric about the "
            "midline; BAK/Ray type. No posterior instrumentation anywhere in the volume -- "
            "these are standalone, which is the technique this implant was used with."),
        "hardware_posterior_fixation": False,
        "fusion": "iatrogenic",
        "fusion_level": "L5-L6",
    },
}

# Corrections that follow from reading the case, and are wrong in the shipped manifest.
CORRECT = {
    "0068": {
        "has_l6": True,
        "n_lumbar_labels": 6,
        "note_labels": (
            "Six free lumbar bodies. The pseudolabel merged the top two under one label, so "
            "the shipped five lumbar labels were each named one level too high. Renumbered "
            "against the twelfth rib, which articulates with T12 at ~5 mm -- an anchor that "
            "is in the image, unlike the top of the spine."),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mp = Path(a.manifest)
    recs = json.loads(mp.read_text(encoding="utf-8"))
    if not isinstance(recs, list):
        print("  ! expected a list of records")
        return 2

    detected = {}
    with Path(a.scan).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            detected[row["case"]] = row
    print(f"  {len(recs)} records; metal detected in {len(detected)}")

    n_flag = n_read = n_fix = 0
    for r in recs:
        cid = str(r.get("volume_id", ""))
        d = detected.get(cid)
        r["hardware"] = bool(d)
        r["hardware_components"] = int(d["n_components"]) if d else 0
        r["hardware_mm3"] = float(d["mm3"]) if d else 0.0
        r["hardware_bridges_interspace"] = (bool(int(d["bridges_interspace"]))
                                            if d else False)
        # read-level fields: absent means NOT YET READ, not "no hardware"
        for k in ("hardware_type", "hardware_count", "hardware_level", "hardware_detail",
                  "hardware_posterior_fixation", "fusion", "fusion_level"):
            r.setdefault(k, None)
        if d:
            n_flag += 1
        if cid in READ:
            r.update(READ[cid])
            n_read += 1
        if cid in CORRECT:
            before = {k: r.get(k) for k in CORRECT[cid]}
            r.update(CORRECT[cid])
            n_fix += 1
            print(f"  {cid}: corrected {before} -> "
                  f"{ {k: CORRECT[cid][k] for k in ('has_l6', 'n_lumbar_labels')} }")

    print(f"  flagged {n_flag} instrumented, read {n_read}, corrected {n_fix}")
    print(f"  {n_flag - n_read} instrumented cases carry a detection but no read")

    if a.dry_run:
        print("  (dry run, nothing written)")
        return 0

    # data/ is gitignored, so there is no version control to fall back on
    bak = mp.with_suffix(".json.bak_prehardware")
    if not bak.exists():
        shutil.copy2(mp, bak)
        print(f"  backup: {bak.name}")
    mp.write_text(json.dumps(recs, indent=1) + "\n", encoding="utf-8")
    print(f"  wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
