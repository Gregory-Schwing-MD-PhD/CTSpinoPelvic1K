"""arthroplasty_attributes.py — total hip or hemiarthroplasty, and which side.

A reader looking at 0515 noticed the femoral ball has a socket component with it -- an
acetabular cup -- and that both hips are done. That distinction is recoverable from what the
implant was TAKEN FROM, without looking at the images again:

    a femoral stem and head displace FEMUR voxels
    an acetabular cup displaces HIP voxels

So a case with thousands of voxels out of the femur and essentially none out of the hip is a
HEMIARTHROPLASTY -- the femoral side replaced against the patient's own acetabulum. A case
losing voxels from both is a TOTAL hip arthroplasty. Doing it per side also names the
laterality, and one case here turns out to be a total hip on one side and a hemi on the
other, which no single label would have said.

WHY THIS IS AN ATTRIBUTE AND NOT A CLASS. Both are arthroplasty and both occupy the same
voxels; nothing about the segmentation changes. Splitting id 80 into 80a/80b would put a
clinical distinction into the voxel scheme, where it would have to be re-derived every time
someone merged the classes back. The manifest is where it belongs.

IT MATTERS FOR THE SPINOPELVIC PARAMETERS. Pelvic incidence and pelvic tilt are measured
from the femoral head centre. In a total hip the acetabulum is a metal cup and the head is a
metal ball; in a hemi the socket is still bone. Either way the head is an implant, but the
geometry differs, and a study using these cases needs to know which.

    python scripts/arthroplasty_attributes.py --applied hardware_review/applied.json \\
        --manifest hardware_review/hardware_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

STEM_MIN = 5000           # voxels out of a femur, above which that side has a stem
CUP_MIN = 3000            # voxels out of a hip, above which that side has a cup
CUP_MAYBE = 1000          # between this and CUP_MIN: too few to call, flag it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--applied", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    import json
    applied = {r["case"]: r for r in json.loads(Path(a.applied).read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))

    print(f"  {'case':<6} {'laterality':<12} {'left':<18} {'right':<18} note")
    print("  " + "-" * 78)
    for r in rows:
        r.setdefault("laterality", "")
        r.setdefault("arthroplasty_type", "")
        r.setdefault("attribute_note", "")
        ap_r = applied.get(r["case"])
        if not ap_r or ap_r.get("class") != "hardware_arthroplasty":
            continue
        t = ap_r["taken_from"]
        lf, rf = t.get("femur_left", 0), t.get("femur_right", 0)
        lh, rh = t.get("left_hip", 0), t.get("right_hip", 0)

        def side(fem, hip):
            if fem < STEM_MIN:
                return None
            if hip >= CUP_MIN:
                return "total"
            if hip >= CUP_MAYBE:
                return "total?"
            return "hemi"

        L, R = side(lf, lh), side(rf, rh)
        sides = [s for s in (("left", L), ("right", R)) if s[1]]
        lat = "bilateral" if len(sides) == 2 else (sides[0][0] if sides else "unclear")
        kinds = sorted({s[1] for s in sides})
        typ = kinds[0] if len(kinds) == 1 else " + ".join(f"{n} {k}" for n, k in sides)
        note = ""
        if "total?" in (L, R):
            note = "cup call is borderline -- few hip voxels, could be blooming"
        if len(kinds) > 1:
            note = ("different operations on the two sides"
                    + ("; " + note if note else ""))
        r["laterality"] = lat
        r["arthroplasty_type"] = typ
        r["attribute_note"] = note
        print(f"  {r['case']:<6} {lat:<12} "
              f"{(str(L) + f' ({lf:,}f/{lh:,}h)') if L else '-':<18} "
              f"{(str(R) + f' ({rf:,}f/{rh:,}h)') if R else '-':<18} {note}")

    keys = list(rows[0].keys())
    with open(a.manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n  wrote {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
