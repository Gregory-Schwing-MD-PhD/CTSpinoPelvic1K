"""make_hardware_review_set.py — order the hardware proposals for a human to work through.

Fifty-odd cases is too many to look at in a random order and too few to sample. What decides
the order is how much the geometry had to guess:

  UNNAMED FIRST     a component proposed as generic `hardware` (76) is one the shape rules
                    could not name. On this cohort that is mostly appendicular metal -- a
                    femoral stem is neither a cage nor a screw nor a plate, and the subtype
                    block has nowhere to put it.
  THEN DISAGREEMENT  components in one case proposed as different classes. Either the case
                    really carries two kinds of implant, or one of the calls is wrong.
  THEN THE EDGES    a piece sitting 8-15 mm from bone is near the cutoff that separates an
                    implant from a bolus of tagged stool, and these are colonography series
                    where the tagging saturates exactly as titanium does.
  THEN FRAGMENTS    many small pieces: a broken construct, several implants, or a threshold
                    catching blooming rather than metal.
  CONFIDENT LAST    one component, named by a rule that fired cleanly, sitting on bone.

The point of the order is that a reviewer who runs out of time has still seen everything
that was uncertain. A confident call skimmed late costs less than an unnamed implant never
looked at.

Writes review_index.csv and a contact sheet per page, so a case can be opened by name.

    python scripts/make_hardware_review_set.py --proposals data/hardware_fix \\
        --out hardware_review
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

HW_NAME = {76: "hardware (unnamed)", 77: "cage", 78: "screw/rod", 79: "plate"}


def rank(meta, geom):
    """(priority, [reasons]) -- lower priority is looked at first."""
    reasons = []
    comps = meta.get("components", [])
    classes = {c.get("v5_id") for c in comps}
    gcomps = (geom or {}).get("components", [])

    if 76 in classes:
        reasons.append("a component could not be named from its shape")
    if len(classes) > 1:
        reasons.append("components disagree: "
                       + ", ".join(sorted(HW_NAME.get(c, str(c)) for c in classes)))
    edge = [g for g in gcomps
            if g.get("dist_to_bone_mm") is not None and 8.0 <= g["dist_to_bone_mm"] <= 15.0]
    if edge:
        reasons.append(f"a piece sits {edge[0]['dist_to_bone_mm']:.0f} mm from bone, near "
                       f"the cutoff between an implant and tagged bowel")
    if len(comps) > 4:
        reasons.append(f"{len(comps)} separate pieces")
    mm3 = sum(c.get("mm3", 0.0) for c in comps)
    if mm3 < 150:
        reasons.append(f"only {mm3:.0f} mm3 of metal")
    if mm3 > 40000:
        reasons.append(f"{mm3:.0f} mm3 -- a very large construct")

    if 76 in classes:
        pri = 0
    elif len(classes) > 1:
        pri = 1
    elif edge:
        pri = 2
    elif len(comps) > 4:
        pri = 3
    elif reasons:
        pri = 4
    else:
        pri = 5
    return pri, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy the renders into the review folder as well")
    a = ap.parse_args()

    src = Path(a.proposals)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for mp in sorted(src.glob("*_hardware.json")):
        cid = mp.name[:4]
        meta = json.loads(mp.read_text(encoding="utf-8"))
        gp = src / f"{cid}_hardware_geometry.json"
        geom = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
        pri, reasons = rank(meta, geom)
        comps = meta.get("components", [])
        sizes = "; ".join(
            f"{g.get('length_mm', 0):.0f}x{g.get('width_mm', 0):.0f}x"
            f"{g.get('thick_mm', 0):.0f}mm" for g in geom.get("components", [])[:3])
        rows.append({
            "priority": pri,
            "case": cid,
            "proposed": " + ".join(sorted({HW_NAME.get(c.get("v5_id"), "?")
                                           for c in comps})),
            "pieces": len(comps),
            "mm3": round(sum(c.get("mm3", 0.0) for c in comps), 1),
            "touches": " ".join(sorted({t for c in comps for t in c.get("touches", [])}))[:60],
            "sizes": sizes,
            "taken_from_bone": meta.get("overlapping_existing_labels", 0),
            "why_review": "; ".join(reasons),
            "mip": f"{cid}_hardware_mip.png",
            "view3d": f"{cid}_hardware_3d.png"
            if (src / f"{cid}_hardware_3d.png").exists() else "",
        })

    rows.sort(key=lambda r: (r["priority"], -r["mm3"]))
    keys = ["priority", "case", "proposed", "pieces", "mm3", "touches", "sizes",
            "taken_from_bone", "why_review", "mip", "view3d"]
    with (out / "review_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if a.copy_images:
        img = out / "renders"
        img.mkdir(exist_ok=True)
        for r in rows:
            for k in ("mip", "view3d"):
                if r[k] and (src / r[k]).exists():
                    shutil.copy2(src / r[k], img / r[k])

    from collections import Counter
    print(f"  {len(rows)} case(s) with a proposal\n")
    band = {0: "unnamed component", 1: "components disagree", 2: "near the bone cutoff",
            3: "many fragments", 4: "other flag", 5: "confident"}
    for p, n in sorted(Counter(r["priority"] for r in rows).items()):
        print(f"    {n:>3}  {band[p]}")
    print(f"\n  by proposed class:")
    for k, n in Counter(r["proposed"] for r in rows).most_common():
        print(f"    {n:>3}  {k}")
    print(f"\n  wrote {out / 'review_index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
