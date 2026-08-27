"""organize_hardware_review.py — split the proposals into instrumentation and artefact.

A radiologist read the 52 proposals and said almost all are artefact: a handful of hip
arthroplasties, one case of sacroiliac screws (1035), one of femoral-neck fixation (0247),
and the interbody cages on 0068. This encodes the rule that reproduces that read, so the
split is reviewable rather than a list somebody typed.

TWO CONDITIONS, BOTH REQUIRED.

  A REAL SURGICAL SITE. An implant sits where an operation happened -- across a hip joint,
  across the sacroiliac joint, in or on the spine. The site comes from the labelled
  structures the metal actually touches. Most of the rejected proposals touch nothing at
  all, or only "sacrum", which is where rectal contrast and iliac calcification sit in a
  colonography series.

  VOLUME. Every confirmed implant here is 2,586 mm3 or more; every rejected proposal is
  1,768 mm3 or less. The threshold sits in that gap rather than in the middle of a cloud.

WHY SATURATION ALONE WAS NOT ENOUGH, having been the obvious answer. Nearly everything in
the rejected set reaches the 3071 HU ceiling too, and several go far past it -- 11,798 HU on
0878, 9,534 on 0763, 7,438 on 0027. Values above the scanner ceiling are the signature of
streak artefact and reconstruction overshoot around something dense, not of a denser
implant. A test built on "does it saturate" keeps all of them.

CLASSES THE SCHEME DOES NOT HAVE. 76-79 were written for spinal instrumentation and cannot
name what is actually in this cohort, so the block is extended:

    80  hardware_arthroplasty   hip or knee prosthesis: stem, head, acetabular cup
    81  hardware_si_screw       iliosacral / sacroiliac fixation, crossing the SI joint
    82  hardware_osteosynthesis metal holding parts of the SAME BONE together -- nails,
                                cannulated screws, a dynamic hip screw. The femoral-neck
                                literature is written as "arthroplasty vs osteosynthesis",
                                and the distinction is not cosmetic: fixation leaves the
                                patient's own femoral head, which is the landmark pelvic
                                incidence and pelvic tilt are measured from.

Each is an entity present in the data that the existing subtypes would name WRONGLY rather
than leave unnamed -- a femoral stem is long and thin, so the screw-or-rod rule claims it.
Adding a class costs one line; a wrong subtype has to be found and undone in every case that
used it.

    python scripts/organize_hardware_review.py --verdicts qc_hardware/verdicts.csv \\
        --renders hardware_review/renders --out hardware_review
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

MIN_IMPLANT_MM3 = 2000.0
REAL_SITES = {"hip joint", "sacroiliac joint", "spine", "femur"}

HW_CLASS = {
    76: "hardware", 77: "hardware_cage", 78: "hardware_screw_rod",
    79: "hardware_plate", 80: "hardware_arthroplasty", 81: "hardware_si_screw",
    82: "hardware_osteosynthesis",
}

# What the site implies, where the existing subtypes cannot say.
SITE_CLASS = {
    "hip joint": 80,
    "femur": 80,
    "sacroiliac joint": 81,
}

# Read off the images and confirmed by a reader, so it overrides the site rule.
CONFIRMED = {
    "0068": (77, "paired threaded cylindrical interbody cages, L5-L6, verified in ITK-SNAP"),
    "1035": (81, "sacroiliac screws crossing the SI joint"),
    # An order of magnitude smaller than the eight total hips, and read as fixation of a
    # femoral neck fracture rather than a replacement. Fixation is not arthroplasty: the
    # femoral head is still the patient's own, which matters because pelvic incidence and
    # pelvic tilt are measured from it. Calling this arthroplasty would have said the
    # opposite of the truth.
    "0247": (82, "three parallel cannulated screws, 7.2-7.7 mm across and 77-93 mm "
                 "long, inverted-triangle configuration in the left femoral neck"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mm3", type=float, default=MIN_IMPLANT_MM3)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.verdicts, encoding="utf-8")))
    ren = Path(a.renders)
    out = Path(a.out)
    inst_dir = out / "instrumentation"
    art_dir = out / "artefact"
    for d in (inst_dir, art_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    for r in rows:
        cid = r["case"]
        mm3 = float(r["total_mm3"])
        site = r["site"]
        is_impl = (site in REAL_SITES and mm3 >= a.min_mm3) or cid in CONFIRMED

        cls, why_cls = (None, "")
        if is_impl:
            if cid in CONFIRMED:
                cls, why_cls = CONFIRMED[cid]
                why_cls = "confirmed by a reader: " + why_cls
            elif site in SITE_CLASS:
                cls = SITE_CLASS[site]
                why_cls = f"metal spanning the {site}"
            else:
                cls = 76
                why_cls = "on the skeleton but the site does not name a class"

        manifest.append({
            "case": cid,
            "verdict": "instrumentation" if is_impl else "artefact",
            "class_id": cls or "",
            "class": HW_CLASS.get(cls, "") if cls else "",
            "site": site,
            "total_mm3": round(mm3, 1),
            "peak_HU": r["peak_HU"],
            "saturated_vox": r["saturated_vox"],
            "n_components": r["n_comp"],
            "touches": r["near"],
            "why": (why_cls if is_impl else
                    (f"{mm3:.0f} mm3, below the {a.min_mm3:.0f} mm3 implant floor"
                     if site in REAL_SITES else
                     f"no surgical site -- touches {r['near'] or 'nothing labelled'}")),
            "peak_above_ceiling": int(float(r["peak_HU"]) > 3100),
            "laterality": "", "arthroplasty_type": "", "cup_fixation": "",
            "attribute_note": "",
        })

        dst = inst_dir if is_impl else art_dir
        for suffix in ("_hardware_mip.png", "_hardware_3d.png"):
            src = ren / f"{cid}{suffix}"
            if src.exists():
                shutil.copy2(src, dst / src.name)

    manifest.sort(key=lambda r: (r["verdict"] != "instrumentation", -r["total_mm3"]))
    keys = ["case", "verdict", "class_id", "class", "site", "total_mm3", "peak_HU",
            "peak_above_ceiling", "saturated_vox", "n_components", "touches", "why",
            "laterality", "arthroplasty_type", "cup_fixation", "attribute_note"]
    with (out / "hardware_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in manifest:
            w.writerow({k: r.get(k, "") for k in keys})
    (out / "hardware_manifest.json").write_text(json.dumps({
        "rule": {
            "min_implant_mm3": a.min_mm3,
            "real_sites": sorted(REAL_SITES),
            "note": "instrumentation = a real surgical site AND volume above the floor; "
                    "saturation alone does not separate them, because streak artefact "
                    "reaches and exceeds the scanner ceiling too",
        },
        "classes_added": {"80": "hardware_arthroplasty", "81": "hardware_si_screw",
                          "82": "hardware_osteosynthesis"},
        "cases": manifest,
    }, indent=1) + "\n", encoding="utf-8")

    from collections import Counter
    inst = [r for r in manifest if r["verdict"] == "instrumentation"]
    print(f"  instrumentation: {len(inst)}    artefact: {len(manifest) - len(inst)}\n")
    for k, n in Counter(r["class"] for r in inst).most_common():
        print(f"    {n:>3}  {k}")
    print(f"\n  cases for review: {', '.join(r['case'] for r in inst)}")
    print(f"\n  wrote {out}/hardware_manifest.csv, hardware_manifest.json,")
    print(f"        {inst_dir}/ and {art_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
