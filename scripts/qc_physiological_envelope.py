"""scripts/qc_physiological_envelope.py — values the anatomy does not allow.

WHY A SEPARATE CHECK. Every extractor already compares its MEDIAN against a published
value, and that catches a measurement that is wrong for everyone. It cannot catch a
measurement that is right for 799 cases and absurd for three, because three cases do not
move a median. Those three are exactly what a reader notices in the tail of a histogram,
and what a downstream model trains on without comment.

So this asks a different question of the same numbers: is any individual value outside
what the anatomy permits? A lumbar pedicle is not 27 mm wide. A spinal canal is not
0.7 mm across. A sacrum is not 22 cm tall. The bounds below are deliberately GENEROUS --
wide enough that a hit is a case to open rather than a distribution to argue about.

WHAT IT FOUND ON FIRST RUN. 391 values of roughly 75,000 (0.5%). They concentrate a
little in cases already listed as outstanding in the release checklist -- 0068, which
carries hardware and has not been hand-annotated, and the never-reviewed pelvic_native
cases 0090, 0196, 0419, 0877 are five of the eight worst -- but those eight account for
only 11%. The rest is diffuse: 230 cases with one or two failed measurements each, which
is what automated extraction across 802 records actually looks like.

The rate is the point. It belongs in the release documentation next to the QC gates,
because "0.5% of derived values are implausible and here is which" is a fact a user can
act on, and silence is not.

    python scripts/qc_physiological_envelope.py --morphometrics morphometrics
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

# (pattern, low, high, what it is). Generous: a hit means something is wrong, not that
# the patient is unusual.
BOUNDS = [
    (r"pedicle.*_mm",                 3.0,  20.0, "lumbar pedicle width"),
    (r"canal_width_.*_mm",           12.0,  40.0, "canal transverse diameter"),
    (r"canal_ap_mm.*",                8.0,  30.0, "canal AP diameter"),
    (r"body_height(_post)?_L\d_mm",  12.0,  45.0, "vertebral body height"),
    (r"endplate_width_.*_mm",        25.0,  70.0, "endplate width"),
    (r"tp_span_.*_mm",               30.0, 120.0, "transverse process span"),
    (r"tp_height_(max|left|right)_mm", 5.0,  60.0, "transverse process height"),
    (r"disc_height_.*_mm",            1.0,  20.0, "disc height"),
    (r"disc_(low|above)_mm",          1.0,  20.0, "disc height"),
    (r"femoral_head_diameter.*_mm",  30.0,  70.0, "femoral head diameter"),
    (r"neck_shaft_angle.*deg",      100.0, 155.0, "neck-shaft angle"),
    (r"hip_axis_length.*_mm",        70.0, 145.0, "hip axis length"),
    (r"pelvic_incidence_deg",        15.0, 100.0, "pelvic incidence"),
    (r"sacral_slope_deg",             5.0,  80.0, "sacral slope"),
    (r"pelvic_tilt_deg",            -15.0,  55.0, "pelvic tilt"),
    (r"ll_supine_deg",                5.0, 100.0, "lumbar lordosis"),
    (r"l\d_trabecular_hu",          -20.0, 450.0, "trabecular attenuation"),
    (r"femoral_neck_hu",            -20.0, 450.0, "femoral neck attenuation"),
    (r"sacrum_(width|height)_mm",    30.0, 180.0, "sacral dimension"),
]


def bounds_for(col):
    for pat, lo, hi, what in BOUNDS:
        if re.fullmatch(pat, col):
            return lo, hi, what
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphometrics", default="morphometrics")
    ap.add_argument("--out", default="qc_final/envelope_violations.csv")
    ap.add_argument("--max-rate", type=float, default=2.0,
                    help="fail if more than this percent of checked values are outside")
    a = ap.parse_args()

    rows_out = []
    checked = 0
    per_case = Counter()
    per_measure = Counter()

    for f in sorted(glob.glob(os.path.join(a.morphometrics, "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        for col in rows[0]:
            b = bounds_for(col)
            if not b:
                continue
            lo, hi, what = b
            for r in rows:
                try:
                    x = float(r[col])
                except (TypeError, ValueError):
                    continue
                checked += 1
                if x < lo or x > hi:
                    case = r.get("case", "?")
                    per_case[case] += 1
                    per_measure[col] += 1
                    rows_out.append({"file": os.path.basename(f), "case": case,
                                     "measure": col, "value": x, "low": lo, "high": hi,
                                     "what": what})

    rate = 100.0 * len(rows_out) / max(1, checked)
    print(f"  {checked} value(s) checked across {len(per_measure)} measures")
    print(f"  {len(rows_out)} outside a physiological envelope ({rate:.2f}%), "
          f"in {len(per_case)} case(s)\n")

    if per_measure:
        print("  worst measures:")
        for m, n in per_measure.most_common(8):
            print(f"    {m:30s} {n:4d}")
        print("\n  worst cases:")
        for c, n in per_case.most_common(8):
            print(f"    {c:6s} {n:4d}")

    if rows_out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n  wrote {a.out}")

    print(f"\n  These are NOT dropped from the CSVs. The extractors record what they")
    print("  measured; this names what should not be believed, so a user can exclude it")
    print("  knowingly. A value silently deleted is a value nobody can audit.")

    if rate > a.max_rate:
        print(f"\n  FAIL: {rate:.2f}% exceeds the {a.max_rate}% gate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
