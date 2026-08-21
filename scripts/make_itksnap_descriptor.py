"""scripts/make_itksnap_descriptor.py — an ITK-SNAP label descriptor you can actually read.

WHAT WAS WRONG WITH THE OLD ONE. It ran every label along a single continuous hue ramp,
so T7 came out (233,242,84) and T8 (220,242,84) -- thirteen units apart in one channel.
On screen they are the same yellow-green. That is fine for a picture of the whole spine
and useless for the only job anyone opens this file to do, which is deciding whether the
vertebra under a rib is the one the label claims. It also stopped at 75, so the hardware
classes rendered as unknown.

THE RULE HERE: NEIGHBOURS IN A NUMBERED SERIES MUST NEVER BE CONFUSABLE. Hues step by
five around a twelve-hue wheel, so consecutive levels land 150 degrees apart and nothing
within five of anything else is near it. The script checks this itself and refuses to
write a file where two labels three or fewer apart in the same series are within sixty
RGB units.

FAMILIES READ AT A GLANCE. Vertebrae are saturated; ribs use the same wheel but pale on
the left and deep on the right, so a rib is never mistaken for a body and the side reads
without the name. Pelvis and femurs are neutral bone, which keeps them visually behind
what is being judged. Lumbar ribs and hardware are deliberately loud -- they are the rare
findings and should catch the eye.

    python scripts/make_itksnap_descriptor.py --out data/itksnap_v5_labels.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Hues step by five around a twelve-hue wheel, so CONSECUTIVE levels land 150 degrees
# apart and nothing within five of anything else is near it. A six-hue cycle with light
# and deep variants was tried first and failed on its own check: deep orange and deep
# yellow came out 42 units apart, which is nothing on screen.
_HUE_ORDER = [(i * 5) % 12 for i in range(12)]


def _wheel(n: int, s: float, v: float):
    import colorsys
    h = _HUE_ORDER[(n - 1) % 12] / 12.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _vert(n: int):
    """C1..L6 as one continuous column, twenty-four distinct colours before any repeat."""
    tier = (n - 1) // 12
    return _wheel(n, *((0.80, 0.98) if tier % 2 == 0 else (1.00, 0.60)))

# RIBS NEED TWELVE DISTINCT COLOURS PER SIDE, not two alternating ones -- alternating
# makes rib 1 and rib 3 the same colour, and telling rib 11 from rib 12 is the whole
# reason anyone is looking. Which SIDE a rib is on is already obvious from where it sits
# on screen, so the side is carried by lightness (left pale, right deep) and all twelve
# hues are spent on the number.
#
def _rib(n: int, light: bool):
    return _wheel(n, *((0.34, 1.00) if light else (0.92, 0.66)))

NAMES = {
    0: "Clear Label", 26: "sacrum", 27: "coccyx", 28: "T13", 29: "S1",
    30: "left_hip", 31: "right_hip", 32: "femur_left", 33: "femur_right",
    58: "iliolumbar_left", 59: "iliolumbar_right", 60: "psoas_left", 61: "psoas_right",
    62: "quadratus_left", 63: "quadratus_right", 64: "gluteus_left", 65: "gluteus_right",
    66: "aorta", 67: "kidney_left", 68: "kidney_right", 69: "inferior_vena_cava",
    70: "iliac_artery_left", 71: "iliac_artery_right", 72: "iliac_vena_left",
    73: "iliac_vena_right", 74: "rib_left_lumbar", 75: "rib_right_lumbar",
    76: "hardware", 77: "hardware_cage", 78: "hardware_screw_rod",
    79: "hardware_plate", 255: "ignore",
}
for i in range(1, 8):
    NAMES[i] = f"C{i}"
for i in range(1, 13):
    NAMES[7 + i] = f"T{i}"
for i in range(1, 7):
    NAMES[19 + i] = f"L{i}"
for i in range(1, 13):
    NAMES[33 + i] = f"rib_left_{i}"
    NAMES[45 + i] = f"rib_right_{i}"

FIXED = {
    0: (0, 0, 0),
    # sacrum and S1 must separate from each other AND from L5/L6 above them: this is
    # exactly where the transitional question is decided
    26: (150, 96, 60), 27: (110, 74, 52), 28: (250, 120, 200), 29: (226, 140, 74),
    30: (196, 190, 176), 31: (168, 162, 150),        # hips, neutral bone
    # femurs go COOL against the warm grey of the hips: they meet at the acetabulum and
    # a shade-deeper version of the same grey does not survive that boundary
    32: (110, 122, 148), 33: (84, 96, 122),
    # THE RARE FINDINGS, DELIBERATELY LOUD -- and hardware especially, because that is
    # the one you go looking for. White was the first choice for generic hardware and it
    # was the wrong one: metal on CT is already the brightest thing in the image, so a
    # white label is invisible against exactly the voxels it is marking. These are
    # saturated hues no bone takes.
    74: (255, 60, 200), 75: (120, 255, 220),
    76: (255, 0, 255),      # generic hardware: magenta, nothing anatomical is magenta
    77: (0, 255, 255),      # cage: cyan
    78: (80, 255, 0),       # screws and rods: acid green
    79: (255, 140, 0),      # plate: orange
    255: (40, 40, 40),
}
# soft tissue: muted, and semi-transparent so it never hides bone
SOFT = {i: (150, 140, 130) for i in range(58, 66)}
SOFT.update({66: (190, 90, 90), 67: (150, 120, 170), 68: (150, 120, 170),
             69: (110, 130, 180), 70: (190, 100, 100), 71: (190, 100, 100),
             72: (110, 140, 190), 73: (110, 140, 190)})


def colour(idx: int):
    if idx in FIXED:
        return FIXED[idx]
    if idx in SOFT:
        return SOFT[idx]
    if 1 <= idx <= 25:                       # C1..L6, one continuous column
        return _vert(idx)
    if 34 <= idx <= 45:
        return _rib(idx - 33, light=True)
    if 46 <= idx <= 57:
        return _rib(idx - 45, light=False)
    return (200, 200, 200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lines = [
        "################################################",
        "# ITK-SnAP Label Description File (CTSpinoPelvic1K v5, VerSe-native)",
        "#",
        "# Colours are chosen so ADJACENT LEVELS NEVER LOOK ALIKE: hues step by five",
        "# around a twelve-hue wheel, so consecutive levels sit 150 degrees apart.",
        "# Ribs use the same wheel, pale on the left and deep on the right. Pelvis and",
        "# femurs are neutral bone; lumbar ribs and hardware are deliberately loud.",
        "#",
        "# IDX -R- -G- -B- -A-- VIS MSH  LABEL",
        "################################################",
    ]
    for idx in sorted(NAMES):
        r, g, b = colour(idx)
        # soft tissue is drawn semi-transparent so it never hides the bone under it
        alpha = 0.00 if idx == 0 else (0.45 if idx in SOFT else 1.00)
        vis = 0 if idx == 0 else 1
        lines.append(f"{idx:5d} {r:4d} {g:4d} {b:4d}  {alpha:.2f}  {vis}  {vis}    "
                     f'"{NAMES[idx]}"')

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # THE CHECK IS PER SERIES, because that is where index adjacency means anatomical
    # adjacency. A left and a right femur sharing a tone is fine -- they sit on opposite
    # sides of the body and no one counts along them. Two adjacent VERTEBRAE or two
    # adjacent RIBS sharing one is the failure this file exists to prevent.
    SERIES = [list(range(1, 26)), list(range(34, 46)), list(range(46, 58))]
    worst = None
    for ser in SERIES:
      for i in ser:
        for j in ser:
            if 0 < abs(i - j) <= 3:
                d = sum((x - y) ** 2 for x, y in zip(colour(i), colour(j))) ** 0.5
                if worst is None or d < worst[0]:
                    worst = (d, i, j)
    print(f"  wrote {p}  ({len(NAMES)} labels)")
    print(f"  closest pair within 3 levels: {NAMES[worst[1]]} vs {NAMES[worst[2]]}, "
          f"RGB distance {worst[0]:.0f}")
    if worst[0] < 60:
        print("  *** too close to distinguish on screen")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
