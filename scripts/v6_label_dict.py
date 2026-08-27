"""v6_label_dict.py — put the hardware classes in the label dictionary that ships with v6.

dataset_labels.json is carried across from v5 unchanged, which is right for every id that
did not move and wrong for the seven that now appear in the voxels. Without this the tree
contains labels 76-82 and the file that names labels does not mention them: a loader would
render an unnamed class, and a reader inspecting a case would see a number.

Adds the whole block rather than only the three new ids. 76-79 were declared in the scheme
document but were never in this file either, because until v6 no record used them.

    python scripts/v6_label_dict.py --labels data/hf_export_v6/dataset_labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARDWARE = {
    "76": "hardware",
    "77": "hardware_cage",
    "78": "hardware_screw_rod",
    "79": "hardware_plate",
    "80": "hardware_arthroplasty",
    "81": "hardware_si_screw",
    "82": "hardware_osteosynthesis",
}

NOTE = (
    "Surgical instrumentation, populated from v6 onward (declared but empty in v1-v5). "
    "Metal outranks bone: where an implant lay inside a vertebra, hip or femur label, the "
    "voxel belongs to the implant. 80 (arthroplasty) REPLACES a joint; 82 (osteosynthesis) "
    "holds parts of the SAME bone together -- the distinction matters because pelvic "
    "incidence and pelvic tilt are measured from the femoral head, which a prosthesis "
    "replaces and fixation does not. Filter on manifest.hardware_labelled before any "
    "gap-based analysis: a cage-bridged interspace reads as 'no gap' exactly as a "
    "congenitally fused transitional vertebra does."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()

    p = Path(a.labels)
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("id_to_name")
    if m is None:
        print("  ! no id_to_name in this file")
        return 2

    added = [k for k in HARDWARE if k not in m]
    if not added:
        print("  hardware ids already declared")
        return 0
    m.update(HARDWARE)
    # keep it in numeric order so a reader scanning the file finds them where they belong
    d["id_to_name"] = {k: m[k] for k in sorted(m, key=lambda x: int(x))}
    d["hardware_note"] = NOTE
    d["scheme"] = str(d.get("scheme", "")).replace("v5", "v6") or "v6"

    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=1)
        fh.write("\n")
    print(f"  added {len(added)} hardware id(s): {', '.join(added)}")
    print(f"  scheme now: {d['scheme']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
