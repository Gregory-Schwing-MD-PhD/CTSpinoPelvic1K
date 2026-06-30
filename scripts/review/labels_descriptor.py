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
    # v4 overlay: iliolumbar (51/52) + LS-nerve roots (53–58) — bright, off-ramp
    # colours so the overlay pops against the grey base context. Ribs (26–49) get
    # an auto JET ramp from _rib_rgb() (24 ids — not worth hardcoding).
    51: (255, 0, 128), 52: (200, 0, 128),                   # iliolumbar L/R
    53: (255, 255, 0), 54: (255, 200, 0),                   # nerve L4 L/R
    55: (0, 255, 0), 56: (0, 200, 0),                       # nerve L5 L/R
    57: (0, 255, 255), 58: (0, 200, 200),                   # nerve S1 L/R
}


def _rib_rgb(idx: int) -> Tuple[int, int, int]:
    """Auto colour for a rib id 26–49 — left ribs warm, right ribs cool, brightness
    ramped by rib number so adjacent ribs are distinguishable."""
    if 26 <= idx <= 37:          # rib_left_1..12 — magenta ramp
        t = (idx - 26) / 11
        return (255, int(40 + 160 * t), 255)
    t = (idx - 38) / 11          # rib_right_1..12 — cyan ramp
    return (int(40 + 160 * t), 255, 255)


def _rgb(idx: int) -> Tuple[int, int, int]:
    if idx in _RGB:
        return _RGB[idx]
    if 26 <= idx <= 49:
        return _rib_rgb(idx)
    return (200, 200, 200)       # safe fallback (shouldn't happen)


def descriptor_text(task: str | None = None) -> str:
    """ITK-SNAP label file. `task=None` → the canonical base scheme (back-compat).
    `task` in schema.OVERLAY_TASKS → base anatomical context (1–9) for spatial
    reference + IGNORE + ONLY that task's overlay palette, so a task's reviewers
    paint exactly its structures and nothing collides."""
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# CTSpinoPelvic1K canonical scheme — DO NOT renumber.",
        f'# task: {task or "canonical"}',
        '# Fields: IDX  -R-  -G-  -B-  -A--  VIS MSH  "LABEL"',
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
    ]
    if task is None:
        idxs = sorted(k for k in schema.CLASS_NAMES if k != 0)
        names = schema.CLASS_NAMES
    else:
        if task not in schema.OVERLAY_CLASSES:
            raise ValueError(f"unknown task {task!r}; expected one of "
                             f"{schema.OVERLAY_TASKS}")
        # base anatomical context (1–9) as read-only reference + the task overlay
        names = {k: schema.CLASS_NAMES[k] for k in range(1, 10)}
        names.update(schema.OVERLAY_CLASSES[task])
        idxs = sorted(names)
    for idx in idxs:
        r, g, b = _rgb(idx)
        lines.append(f'{idx:5d} {r:5d} {g:4d} {b:4d}'
                     f'        1  1  1    "{names[idx]}"')
    # IGNORE_LABEL: present in partial masks; show but de-emphasised.
    lines.append(f'{schema.IGNORE_LABEL:5d}   128  128  128'
                 f'        1  1  0    "IGNORE"')
    return "\n".join(lines) + "\n"


def _verse_rgb(idx: int) -> Tuple[int, int, int]:
    """A DISTINCT, high-contrast colour per id so ADJACENT vertebrae and ADJACENT ribs are
    obviously different at a glance. Earlier per-side ramps varied one channel slightly, so
    neighbours looked identical. Use a golden-angle hue rotation (consecutive ids land ~222°
    apart on the colour wheel) plus alternating value/saturation, so no two neighbours match.
    Identity still comes from the label name under the cursor — this is purely separation."""
    import colorsys
    if idx <= 0:
        return (0, 0, 0)
    hue = (idx * 0.6180339887498949) % 1.0         # golden-angle: big jump between adjacent ids
    sat = 1.0 if (idx % 3) else 0.72               # 3-cycle saturation -> extra neighbour contrast
    val = 1.0 if (idx % 2 == 0) else 0.74          # alternate brightness on consecutive ids
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def verse_native_descriptor_text() -> str:
    """ITK-SNAP palette for the ACTUAL VerSe-native dataset labels (scripts/
    label_scheme.py): spine 1-28, S1 29, hips 30/31, femurs 32/33, ribs 34-57,
    soft-tissue 58-73, ignore 255. Use this when reviewing real v3/v4 labels
    (reviewtool review-cases). The review-space LSTV palette (descriptor_text, a
    separate 1-9 scheme) is intentionally left untouched."""
    import sys as _sys
    _sd = str(Path(__file__).resolve().parents[1])     # scripts/ for label_scheme
    if _sd not in _sys.path:
        _sys.path.insert(0, _sd)
    import label_scheme as LS
    id2name = {v: k for k, v in LS.label_dict().items()}
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File — CTSpinoPelvic1K VerSe-native",
        '# Fields: IDX  -R-  -G-  -B-  -A--  VIS MSH  "LABEL"',
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
    ]
    for idx in sorted(i for i in id2name if i not in (0, LS.IGNORE_LABEL)):
        r, g, b = _verse_rgb(idx)
        lines.append(f'{idx:5d} {r:5d} {g:4d} {b:4d}        1  1  1    "{id2name[idx]}"')
    lines.append(f'{LS.IGNORE_LABEL:5d}   128  128  128        1  1  0    "ignore"')
    return "\n".join(lines) + "\n"


def write_label_descriptor(path, task: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(descriptor_text(task))
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="labels.txt", type=Path)
    ap.add_argument("--task", default=None,
                    help="overlay task palette (rib_anchor|ribs|ls_nerve|"
                         "iliolumbar); omit for the canonical base scheme")
    a = ap.parse_args()
    print("wrote", write_label_descriptor(a.out, a.task))
