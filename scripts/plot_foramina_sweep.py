"""scripts/plot_foramina_sweep.py — show how unstable the foramina count is, per parameter.

A single "best setting" hides the thing worth knowing. If the count swings from 2 to 6 as
a threshold moves, then whatever setting happens to agree with the labels agreed by luck,
and the honest output is the swing itself rather than the winner.

So each panel holds three parameters fixed at the best-scoring setting and sweeps the
fourth, drawing the three groups side by side with their spread. Separation would look
like three bands that stay apart across the sweep. Overlap that persists everywhere is a
negative result, and a clean one.

The last panel is the individual-case view at the best setting, because a setting can
separate group medians while misclassifying half the cases in each group -- and only the
per-case distribution shows that.

    python scripts/plot_foramina_sweep.py --csv morphometrics/foramina_param_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

GROUND, INK, MUTE, GRID = "#FAFAF8", "#12181E", "#8A939B", "#DFE3E6"
COL = {"normal": "#7C8B98", "LUMBARIZATION": "#6B5B95", "SACRALIZATION": "#D2492A"}
TARGET = {"normal": 4, "LUMBARIZATION": 3, "SACRALIZATION": 5}

plt.rcParams.update({
    "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "axes.linewidth": .9, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 170,
})
PARAMS = ["slab_mm", "close_mm", "min_area", "max_dark"]
NICE = {"slab_mm": "projection slab (mm)", "close_mm": "closing radius (mm)",
        "min_area": "minimum hole area (mm²)", "max_dark": "CT darkness cut (HU)"}


def grp(lbl):
    if lbl == "normal":
        return "normal"
    return "SACRALIZATION" if "SACRAL" in lbl else "LUMBARIZATION"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/foramina_param_sweep.csv")
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.csv)):
        rows.append({**{k: float(r[k]) for k in PARAMS + ["pairs", "disagree"]},
                     "case": r["case"], "g": grp(r["label"])})
    print(f"  {len(rows)} rows, {len({r['case'] for r in rows})} cases")

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[tuple(r[p] for p in PARAMS)][r["g"]].append(r["pairs"])
    scored = []
    for key, g in by.items():
        if len(g) < 3:
            continue
        med = {k: float(np.median(v)) for k, v in g.items()}
        err = sum(abs(med[k] - TARGET[k]) for k in TARGET)
        hits = sum(1 for k, v in g.items() for x in v if x == TARGET[k])
        tot = sum(len(v) for v in g.values())
        scored.append((err, -hits / tot, key, hits, tot))
    scored.sort()
    best = scored[0][2]
    print(f"  best setting {dict(zip(PARAMS, best))}  "
          f"per-case correct {scored[0][3]}/{scored[0][4]}")

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, prm in zip(axes[:4], PARAMS):
        vals = sorted({r[prm] for r in rows})
        for gi, (gname, colour) in enumerate(COL.items()):
            xs, meds, los, his = [], [], [], []
            for j, v in enumerate(vals):
                sel = [r["pairs"] for r in rows
                       if r["g"] == gname and r[prm] == v
                       and all(r[q] == best[PARAMS.index(q)] for q in PARAMS if q != prm)]
                if not sel:
                    continue
                sel = np.array(sel)
                xs.append(j + (gi - 1) * .22)
                meds.append(np.median(sel))
                los.append(np.percentile(sel, 25))
                his.append(np.percentile(sel, 75))
            if xs:
                ax.errorbar(xs, meds,
                            yerr=[np.array(meds) - np.array(los),
                                  np.array(his) - np.array(meds)],
                            fmt="o", color=colour, capsize=3, lw=1.4, ms=5,
                            label=gname.lower())
                ax.axhline(TARGET[gname], color=colour, lw=.9, ls=(0, (4, 3)), alpha=.6)
        ax.set_xticks(range(len(vals)),
                      ["off" if (prm == "max_dark" and v > 1e8) else f"{v:g}" for v in vals])
        ax.set_xlabel(NICE[prm])
        ax.set_ylabel("foramina detected (larger side)")
        ax.grid(axis="y", alpha=.5, lw=.7)
        ax.set_title(NICE[prm], loc="left")
    axes[0].legend(fontsize=7.5, loc="upper left")

    ax = axes[4]
    g = by[best]
    width = .26
    allv = sorted({int(x) for v in g.values() for x in v})
    for gi, (gname, colour) in enumerate(COL.items()):
        c = Counter(int(x) for x in g.get(gname, []))
        n = max(1, sum(c.values()))
        ax.bar([v + (gi - 1) * width for v in allv],
               [100 * c.get(v, 0) / n for v in allv],
               width=width, color=colour, label=f"{gname.lower()} (n={n})")
        ax.axvline(TARGET[gname], color=colour, lw=.9, ls=(0, (4, 3)), alpha=.6)
    ax.set_xticks(allv)
    ax.set_xlabel("foramina detected at the best setting")
    ax.set_ylabel("% of group")
    ax.grid(axis="y", alpha=.5, lw=.7)
    ax.legend(fontsize=7.5)
    ax.set_title("Per-case, at the best setting", loc="left")

    s = dict(zip(PARAMS, best))
    ct_cut = "off" if s["max_dark"] > 1e8 else f"{s['max_dark']:g}HU"
    fig.suptitle("Can any setting separate the three groups?   "
                 f"best = slab {s['slab_mm']:g}mm · closing {s['close_mm']:g}mm · "
                 f"min area {s['min_area']:g}mm² · CT cut {ct_cut}"
                 "   —   dashed line = that group's target count",
                 fontsize=11, x=.006, ha="left", y=1.04)
    fig.tight_layout()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_foramina_sweep.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
