"""paper/mpda/make_priorart_figure.py — the prior-art figure, in two honest halves.

THE ARGUMENT THIS FIGURE HAS TO MAKE is not "ours is biggest", because it is not: three of
the five comparators carry more scans and one labels more ribs. Panel (a) says so outright
rather than leaving a reviewer to discover it. The claim is in panel (b): a scheme that has
no L6 class cannot record a six-lumbar spine and a scheme that numbers every rib 1-12 has
nowhere to put a thirteenth, whatever either does at segmentation time.

Every cell in (b) is a property of a published label scheme, checkable in a class list:

  CTSpine1K       C1-L6, 25 vertebral classes. No pelvis, no ribs.
  CTPelvic1K      four classes -- lumbar spine (as ONE undifferentiated object), sacrum,
                  left hip, right hip. No vertebral count is recoverable.
  VerSe           C1-L6 with L6 and T13 present, deliberately enriched for variants, and
                  transitional vertebrae ARE graded by Castellvi -- as an exclusion
                  criterion. Vertebrae partially fused to the sacrum (III/IV) were not
                  segmented, and the sacrum is not segmented.
  RibSeg v2       24 rib classes plus centrelines. Ribs only.
  TotalSegmentator 117 classes including vertebrae C1-L5, sacrum, a separate S1, both
                  hips, both femora and ribs 1-12 per side. No L6. No lumbar rib.
  CTSpinoPelvic1K all of the above plus L6, a lumbar-rib class, and a released Castellvi
                  layer on the 33 transitional records.

    python paper/mpda/make_priorart_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "figures" / "fig_priorart.pdf"

# (name, scans, is_ours)
SIZE = [
    ("TotalSegmentator", 1204, False),
    ("CTPelvic1K",       1184, False),
    ("CTSpine1K",        1005, False),
    ("CTSpinoPelvic1K",   802, True),
    ("RibSeg v2",         660, False),
    ("VerSe",             374, False),
]

CAPS = ["vertebral\ncount", "L6", "sacrum", "S1", "pelvis",
        "per-level\nribs", "lumbar\nrib", "femora", "transitional\ngrade"]

# 1 = the scheme has the class; 0 = it does not; 0.5 = present but qualified (see NOTES)
GRID = {
    "CTSpine1K":        [1, 1, 0, 0, 0, 0, 0, 0, 0],
    "CTPelvic1K":       [0, 0, 1, 0, 1, 0, 0, 0, 0],
    "VerSe":            [1, 1, 0, 0, 0, 0, 0, 0, 0.5],
    "RibSeg v2":        [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "TotalSegmentator": [1, 0, 1, 1, 1, 1, 0, 1, 0],
    "CTSpinoPelvic1K":  [1, 1, 1, 1, 1, 1, 1, 1, 1],
}
ORDER = ["CTSpine1K", "CTPelvic1K", "VerSe", "RibSeg v2", "TotalSegmentator",
         "CTSpinoPelvic1K"]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.0, 1.75]})

    # ---- (a) size, stated plainly -------------------------------------------------
    names = [s[0] for s in SIZE]
    vals = [s[1] for s in SIZE]
    ours = [s[2] for s in SIZE]
    y = np.arange(len(names))[::-1]
    cols = ["#c44e52" if o else "#b8b8b8" for o in ours]
    axl.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        axl.text(v + 18, yy, f"{v:,}", va="center", fontsize=8.5)
    axl.set_yticks(y)
    axl.set_yticklabels(names, fontsize=9)
    axl.set_xlim(0, 1420)
    axl.set_xlabel("CT scans in the collection", fontsize=9)
    axl.set_title("(a) this is not the largest collection", fontsize=9.5, loc="left")
    axl.tick_params(axis="x", labelsize=8)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)

    # ---- (b) what the label scheme can represent ----------------------------------
    n_r, n_c = len(ORDER), len(CAPS)
    axr.set_xlim(-0.5, n_c - 0.5)
    axr.set_ylim(-0.5, n_r - 0.5)
    for i, name in enumerate(ORDER):
        yy = n_r - 1 - i
        if name == "CTSpinoPelvic1K":
            axr.add_patch(Rectangle((-0.5, yy - 0.5), n_c, 1.0,
                                    facecolor="#c44e52", alpha=0.10, zorder=0))
        for j, v in enumerate(GRID[name]):
            if v == 1:
                axr.plot(j, yy, "o", ms=11,
                         color="#c44e52" if name == "CTSpinoPelvic1K" else "#4c72b0",
                         zorder=3)
            elif v == 0.5:
                axr.plot(j, yy, "o", ms=11, markerfacecolor="white",
                         markeredgecolor="#4c72b0", markeredgewidth=1.6, zorder=3)
                axr.plot(j, yy, "x", ms=6, color="#4c72b0", zorder=4)
            else:
                axr.plot(j, yy, "o", ms=11, markerfacecolor="white",
                         markeredgecolor="#cccccc", markeredgewidth=1.2, zorder=2)
    axr.set_xticks(range(n_c))
    axr.set_xticklabels(CAPS, fontsize=8)
    axr.set_yticks(range(n_r))
    axr.set_yticklabels(ORDER[::-1], fontsize=9)
    axr.set_title("(b) what the published label scheme can represent",
                  fontsize=9.5, loc="left")
    axr.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        axr.spines[sp].set_visible(False)
    axr.set_axisbelow(True)
    axr.grid(axis="x", color="#eeeeee", lw=0.8)

    axr.plot([], [], "o", ms=8, color="#4c72b0", label="class present")
    axr.plot([], [], "o", ms=8, markerfacecolor="white", markeredgecolor="#4c72b0",
             label="graded, but used to exclude")
    axr.plot([], [], "o", ms=8, markerfacecolor="white", markeredgecolor="#cccccc",
             label="no such class")
    axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
