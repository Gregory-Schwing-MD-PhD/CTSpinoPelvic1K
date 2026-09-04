"""paper/mpda/make_fov_fig.py — how many of the 802 records carry each level.

THE FIELD OF VIEW, AS A CURVE. Levels run cranial to caudal along the x axis, C1 to L6,
and the y axis is the number of records in which that level is labeled. The corpus is
abdominopelvic CT, so the curve is the field of view of the cohort seen from the labels:
flat at zero through the cervical spine, rising through the thoracic column as scans reach
higher, at 802 from T12 down.

Drawn as a filled line, the way the rib-length ratio in the count-free figure is, not as
bars: the quantity is a profile along the column, and a profile reads as a curve. Nothing
but vertebrae is on the figure. Ribs, sacrum, S1, pelvis, femora and hardware are in the
per-identifier census table in the supplement (scripts/make_census_table.py).

Counts come from morphometrics/label_census_v7.csv, which scripts/label_census_v7.py
regenerates by reading every released label, so the figure cannot disagree with the voxels.

    python paper/mpda/make_fov_fig.py --out paper/mpda/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MM = 1.0 / 25.4
COL1 = 80 * MM
BLACK, GRIDGREY = "#000000", "#B0B0B0"
TEAL, OCHRE, INK, FAINT = "#1c6b73", "#b8791f", "#22262b", "#8c9199"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.9, "axes.edgecolor": BLACK, "text.color": BLACK,
    "grid.color": GRIDGREY, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "legend.frameon": False, "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load_census(path):
    n = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            n[int(r["id"])] = int(r["records"])
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(ROOT / "morphometrics/label_census_v7.csv"))
    ap.add_argument("--out", default=str(ROOT / "paper/mpda/figures"))
    ap.add_argument("--total", type=int, default=802)
    a = ap.parse_args()
    n = load_census(a.census)
    N = a.total

    # the column, cranial to caudal: C1..C7 (1..7), T1..T12 (8..19), L1..L6 (20..25)
    levels = ([(f"C{i}", i) for i in range(1, 8)] + [(f"T{i}", 7 + i) for i in range(1, 13)]
              + [(f"L{i}", 19 + i) for i in range(1, 7)])
    x = np.arange(len(levels))
    vert = np.array([n.get(i, 0) for _, i in levels], float)
    fig, ax = plt.subplots(figsize=(COL1, 1.7))
    # FREQUENCY, ON A LOG AXIS. Linear, C1 (2 of 802) and T4 (9) sit on the zero line and
    # read as absent; the point of the figure is that they are not. The floor is one
    # tenth of a percent (below one record); levels with no record at all are drawn as
    # open markers on the floor and annotated 0.
    pct = 100.0 * vert / N
    floor = 0.1
    y = np.where(pct > 0, pct, floor)
    ax.fill_between(x, floor, y, color=TEAL, alpha=0.18, lw=0)
    ax.plot(x, y, color=TEAL, lw=1.4, marker="o", ms=2.2)
    z = vert == 0
    ax.plot(x[z], y[z], "o", ms=2.6, mfc="white", mec=TEAL, mew=0.8)
    for xi, v, pv in zip(x, vert, pct):
        if 0 < pv < 20:
            ax.annotate(f"{int(v)}", (xi, pv), xytext=(0, 3.5), textcoords="offset points",
                        ha="center", fontsize=5.4, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(floor, 160)
    ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0", "1", "10", "100"])
    ax.axhline(100, color=INK, lw=0.6, ls=(0, (2, 2)))

    ax.set_xticks(x); ax.set_xticklabels([nm for nm, _ in levels], rotation=90)
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylabel("records (%)")
    ax.set_xlabel("level, cranial to caudal")
    ax.grid(axis="y", which="major")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.2)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_fov.pdf"); fig.savefig(out / "fig_fov.png", dpi=200)
    lo = next((nm for (nm, i), v in zip(levels, vert) if v >= N * 0.5), "?")
    print(f"wrote {out / 'fig_fov.pdf'}  (first level in >=50% of records: {lo})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
