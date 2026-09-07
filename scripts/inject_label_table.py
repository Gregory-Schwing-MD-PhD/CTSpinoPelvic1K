"""inject_label_table.py -- build the card's label table from measured occupancy.

The table a reader trusts is the one nobody typed. This reads label_occupancy.json --
produced by scanning all 802 shipped label maps -- and renders the scheme grouped into
anatomical blocks, with the case count measured rather than declared. Repetitive runs (the
twelve ribs a side) collapse to one row showing the range; classes that carry a real
decision (the T12 anchor, the S1 carve, the lumbar rib, each hardware subtype) keep their
own row.

Replaces the <!--LABEL_TABLE--> marker in the card.

    python inject_label_table.py --occupancy label_occupancy.json --readme README.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (first_id, last_id, display name, source) -- last_id == first_id means a single row.
BLOCKS = [
    (1, 7, "C1–C7", "CTSpine1K (VerSe 1–7)"),
    (8, 18, "T1–T11", "CTSpine1K (VerSe 8–18)"),
    (19, 19, "**T12** — rostral counting anchor", "CTSpine1K (VerSe 19)"),
    (20, 20, "L1", "CTSpine1K (VerSe 20)"),
    (21, 21, "L2", "CTSpine1K (VerSe 21)"),
    (22, 22, "L3", "CTSpine1K (VerSe 22)"),
    (23, 23, "L4", "CTSpine1K (VerSe 23)"),
    (24, 24, "L5", "CTSpine1K (VerSe 24)"),
    (25, 25, "**L6 / LSTV**", "CTSpine1K (VerSe 25) — lumbarized S1"),
    (26, 26, "sacrum", "CTPelvic1K (dataset2 1 → 26)"),
    (27, 27, "coccyx", "CTSpine1K (VerSe 27)"),
    (28, 28, "T13 — supernumerary thoracic", "CTSpine1K (VerSe 28)"),
    (29, 29, "**S1** — caudal counting anchor", "(GT sacrum) ∩ (TS `vertebrae_S1`)"),
    (30, 31, "left_hip / right_hip", "CTPelvic1K (dataset2 2,3 → 30,31)"),
    (32, 33, "femur_left / femur_right", "TotalSegmentator"),
    (34, 45, "rib_left_1 … rib_left_12", "numbered off GT thoracic column"),
    (46, 57, "rib_right_1 … rib_right_12", "numbered off GT thoracic column"),
    (58, 59, "iliolumbar_left / _right", "ligament, annotated"),
    (60, 65, "nerve_L4/L5/S1_left, _right", "annotated"),
    (66, 67, "psoas_left / psoas_right", "TotalSegmentator"),
    (68, 69, "aorta, inferior_vena_cava", "TotalSegmentator"),
    (70, 73, "iliac_artery / iliac_vena, L+R", "TotalSegmentator"),
    (74, 75, "**rib_left_lumbar / rib_right_lumbar**", "a rib on a *lumbar* vertebra"),
    (76, 76, "hardware — subtype not distinguished", "radiologist-confirmed"),
    (77, 77, "hardware_cage", "radiologist-confirmed"),
    (78, 78, "hardware_screw_rod", "radiologist-confirmed"),
    (79, 79, "hardware_plate", "radiologist-confirmed"),
    (80, 80, "**hardware_arthroplasty** — replaces a joint", "radiologist-confirmed"),
    (81, 81, "hardware_si_screw", "radiologist-confirmed"),
    (82, 82, "**hardware_osteosynthesis** — same bone", "radiologist-confirmed"),
    (255, 255, "**ignore** — un-traced, NOT background", "partial-annotation sentinel"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--occupancy", required=True)
    ap.add_argument("--readme", required=True)
    a = ap.parse_args()

    occ = json.loads(Path(a.occupancy).read_text())
    n_vol = occ["n_volumes"]
    cases = {r["id"]: r["cases"] for r in occ["labels"]}

    lines = ["| id | class | source | cases (of %d) |" % n_vol,
             "|---:|---|---|---:|"]
    for lo, hi, name, src in BLOCKS:
        ids = f"{lo}" if lo == hi else f"{lo}–{hi}"
        counts = [cases.get(i, 0) for i in range(lo, hi + 1)]
        if not any(counts):
            cell = "**0 — declared, unused**"
        elif lo == hi:
            cell = f"{counts[0]}"
        elif min(counts) == max(counts):
            cell = f"{counts[0]}"
        else:
            cell = f"{min(counts)}–{max(counts)}"
        lines.append(f"| {ids} | {name} | {src} | {cell} |")

    table = "\n".join(lines)

    p = Path(a.readme)
    s = p.read_text(encoding="utf-8")
    marker = "<!--LABEL_TABLE-->"
    assert marker in s, "marker not found in card"
    s = s.replace(marker, table, 1)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)

    empty = [r["name"] for r in occ["labels"] if r["cases"] == 0]
    print(f"  injected table over {n_vol} volumes")
    print(f"  declared but unused ({len(empty)}): {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
