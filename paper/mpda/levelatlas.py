"""levelatlas.py -- vertebral morphometry by level, in the format the textbooks use.

WHAT THIS REPLACES. Benzel's Biomechanics of Spine Stabilization plots four of its
foundational anatomy figures as measurement against spinal level:

    Fig. 1.1   vertebral body diameter (width, depth)   Berry, Panjabi, White & Panjabi
    Fig. 1.2   vertebral body height (ventral, dorsal)  Berry, Panjabi, White & Panjabi
    Fig. 1.9   spinal canal diameter (width, depth)     Berry, Panjabi, Reynolds, McCormack
    Fig. 1.11  transverse pedicle width                 Panjabi, Krag, Zindrick, Bernard

Each is a line through single values from cadaveric series of a few dozen specimens, drawn
without any indication of spread. A surgeon reading one cannot tell whether a patient in
front of them is unusual, because the figure never shows what usual looks like.

This panel answers the same four questions from 802 CT records and draws the spread. The
axes deliberately match the textbook convention -- level on the vertical, millimetres on the
horizontal -- so the two can be laid side by side.

WHAT THE INTERVALS ARE, AND ARE NOT. MPDA rules forbid hypothesis testing, and this figure
does none: there is no test, no p-value, and nothing here compares one level against
another. The bars are descriptive spread -- the interquartile range as a box, the 5th to
95th percentile as whiskers, the median as a rule. That is the "comprehensive descriptive
analysis" the policy asks for, and it is the part the textbook figures omit.

WHY MEDIAN AND NOT MEAN. The rest of the manuscript reports medians, and these
distributions are mildly skewed at L5 where transitional anatomy widens the lower tail.
A mean would move with the anatomy the dataset was built to study.

N IS DRAWN, NOT ASSUMED. Coverage is not uniform: body and canal-width measures rest on
~730-790 records per level, while canal depth and pedicle width fall to 331 at L1 because
the measurement needs an axial extent many field-limited abdominal series do not carry.
Reporting a tight interval from 331 records beside one from 773 without saying so would
misrepresent both, so every row is annotated with its own n.
"""
from __future__ import annotations

import numpy as np


LEVELS = ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]

# Plausibility windows. These reject measurement failures, not unusual patients: the bounds
# are far wider than any published range, so an outlier that is real survives and a value
# that could only come from a broken cut does not.
GATES = {
    "height":   (10.0, 60.0),
    "endplate": (20.0, 90.0),
    "canal_w":  (10.0, 50.0),
    "canal_ap": (6.0, 40.0),
    "pedicle":  (2.0, 40.0),
}


def series(rows, template, gate, levels=LEVELS):
    """Median, quartiles, 5-95 range and n for one measure across levels.

    Returns a dict keyed by level; a level with too few records to describe is absent
    rather than plotted as an empty row, so a gap in the figure means a gap in the data.
    """
    lo, hi = gate
    out = {}
    for lv in levels:
        key = template.format(l=lv)
        vals = []
        for r in rows:
            try:
                x = float(r[key])
            except (TypeError, ValueError, KeyError):
                continue
            if lo <= x <= hi:
                vals.append(x)
        v = np.asarray(vals, float)
        if v.size < 30:
            continue
        out[lv] = {
            "n": int(v.size),
            "med": float(np.median(v)),
            "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)),
            "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
        }
    return out


def draw(ax, stats, y_of, colour, marker="o", offset=0.0, label=None, lw=1.3):
    """One measure as a row per level: whiskers 5-95, box IQR, marker at the median.

    Drawn with explicit primitives rather than ax.boxplot because boxplot wants the raw
    samples and its own layout; here the row positions have to line up across panels that
    do not all carry the same levels.
    """
    first = True
    for lv, s in stats.items():
        y = y_of[lv] + offset
        ax.plot([s["p5"], s["p95"]], [y, y], color=colour, lw=lw * 0.55,
                solid_capstyle="butt", zorder=2)
        ax.plot([s["q1"], s["q3"]], [y, y], color=colour, lw=lw * 2.6,
                solid_capstyle="butt", alpha=0.45, zorder=3)
        ax.plot([s["med"]], [y], marker=marker, color=colour, ms=3.4,
                mew=0.9, mfc="white", zorder=4,
                label=(label if first else None))
        first = False
    return ax


def annotate_n(ax, stats, y_of, colour, offset=0.0, fontsize=5.5, x=0.995):
    """Print n for each row just inside the right spine, so unequal coverage is visible.

    x is an axes fraction and y is a data coordinate, via a blended transform. Placing
    these in data coordinates put them outside the panel's own x-limits, where they
    rendered in the gutter and read as though they belonged to the panel next door.
    """
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for lv, s in stats.items():
        ax.text(x, y_of[lv] + offset, f"{s['n']}", va="center", ha="right",
                fontsize=fontsize, color=colour, transform=tr)
