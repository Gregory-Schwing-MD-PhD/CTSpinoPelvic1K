"""
scripts/review/labels_descriptor.py — fixed ITK-SNAP label description file
for the canonical 10-class scheme.

Every reviewer must load the SAME palette so label values stay consistent —
ITK-SNAP lets a user renumber/recolour labels, and if reviewer A's "sacrum"
is value 7 but reviewer B paints it as 6, the corrected labels are silently
incompatible and the diff/IRR is meaningless. The reviewtool launches
ITK-SNAP with `-l labels.txt` pointing at the file this writes, locking
idx↔structure for everyone.

ITK-SNAP label-description format (whitespace-separated):
    IDX  R  G  B  A  VIS  MSH  "LABEL"
where R/G/B are 0–255, A is 0–1 alpha, VIS/MSH are 0/1 visibility flags.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

# RGB (0–255) per class. 0 = clear. Spine cool→warm, pelvis warm.
_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (38, 102, 204), 2: (64, 140, 217), 3: (89, 166, 230),
    4: (115, 191, 235), 5: (26, 204, 217), 6: (191, 217, 51),
    7: (217, 38, 38), 8: (242, 128, 26), 9: (242, 204, 13),
}


def descriptor_text() -> str:
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# CTSpinoPelvic1K canonical 10-class scheme — DO NOT renumber.",
        '# Fields: IDX  -R-  -G-  -B-  -A--  VIS MSH  "LABEL"',
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
    ]
    for idx in range(1, 10):
        r, g, b = _RGB[idx]
        name = schema.CLASS_NAMES[idx]
        lines.append(f'{idx:5d} {r:5d} {g:4d} {b:4d}'
                     f'        1  1  1    "{name}"')
    # IGNORE_LABEL: present in partial masks; show but de-emphasised.
    lines.append(f'{schema.IGNORE_LABEL:5d}   128  128  128'
                 f'        1  1  0    "IGNORE"')
    return "\n".join(lines) + "\n"


def write_label_descriptor(path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(descriptor_text())
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="labels.txt", type=Path)
    a = ap.parse_args()
    print("wrote", write_label_descriptor(a.out))
