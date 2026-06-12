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

# RGB (0–255) per class. 0 = clear. Sampled evenly from the JET colormap
# (idx 1→9 = dark-blue → blue → cyan → green → yellow → orange → red → dark-red)
# so adjacent vertebrae/structures are maximally distinguishable.
_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 0, 128), 2: (0, 0, 255), 3: (0, 128, 255),
    4: (21, 255, 226), 5: (123, 255, 123), 6: (226, 255, 21),
    7: (255, 151, 0), 8: (255, 33, 0), 9: (128, 0, 0),
    # v4 rib-anchor classes — off the JET ramp so the anchor + rib pop against
    # the vertebra/pelvis colours: last_rib_vertebra magenta, rib white.
    11: (255, 0, 255), 12: (255, 255, 255),
}


def descriptor_text() -> str:
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# CTSpinoPelvic1K canonical scheme — DO NOT renumber.",
        '# Fields: IDX  -R-  -G-  -B-  -A--  VIS MSH  "LABEL"',
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
    ]
    # Drive the palette off CLASS_NAMES so the v4 rib-anchor classes (11/12)
    # appear automatically; skip 0 (Clear, above) and IGNORE (added below).
    for idx in sorted(k for k in schema.CLASS_NAMES if k not in (0,)):
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
