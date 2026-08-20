"""scripts/analyze_rib_incidence.py — read the incidence QC and show what the cohort does.

Six questions, one panel each. They are the questions the rib/vertebra numbering argument
keeps coming back to, and none of them is answerable from a pass/fail count:

  A  where the numbering breaks     bucket composition per rib number
  B  is it a shift or is it noise   distribution of the offset delta
  C  is 10 mm the right anchor      gap from each rib to the vertebra its number claims
  D  is the last rib symmetric      left rib count vs right rib count, per case
  E  how many ribs do people have   last rib number per side
  F  where do lumbar ribs attach    the phenotype, by level

B is the one that decides whether a case is misnumbered or merely imperfectly segmented:
a spike at a single delta is a renumber, a spread is segmentation noise. C is a check on
our own threshold rather than on the data -- if the gap distribution is bimodal with the
trough at 10 mm, the anchor is separating something real; if 10 mm cuts through a single
mode, the threshold is inventing the boundary it claims to find.

    python scripts/analyze_rib_incidence.py --qc qc_rib_incidence_v4 [--out DIR]

Writes fig_rib_incidence.{pdf,png} and, because two palette slots fall under 3:1 on the
light surface, the same numbers as a text table -- the figure is never the only copy.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import MaxNLocator                          # noqa: E402

# Validated categorical slots, in the documented fixed order (adjacent pairlist:
# worst CVD dE 9.1 light / 8.4 dark). Never cycled, never reordered per chart.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9a9992"          # "skipped" is missing data, not a category -> no hue
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"

# match first (the good case), then the defect, then the finding, then the gap.
BUCKETS = ["match", "offset", "lumbar", "no_contact", "skipped"]
BCOLOR = {"match": BLUE, "offset": ORANGE, "lumbar": AQUA,
          "no_contact": YELLOW, "skipped": GREY}
BLABEL = {"match": "on its own vertebra", "offset": "on a different thoracic level",
          "lumbar": "on a lumbar vertebra (phenotype)",
          "no_contact": "no vertebra within reach", "skipped": "not evaluable"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # real text in the PDF, not outlines
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def load(qc: Path):
    rows = list(csv.DictReader(open(qc / "rib_incidence.csv")))
    for r in rows:
        r["rib"] = int(r["rib"])
        r["gap_mm"] = float(r["gap_mm"]) if r["gap_mm"] else np.nan
        r["gap_own_mm"] = float(r["gap_own_mm"]) if r["gap_own_mm"] else np.nan
        r["delta"] = int(r["delta"]) if r["delta"] not in ("", None) else None
        r["truncated"] = int(r.get("truncated") or 0)
        r["voxels"] = int(r.get("voxels") or 0)
    summary = json.loads((qc / "summary.json").read_text())
    return rows, summary


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def panel_a(ax, rows):
    """Composition per rib number. Stacked, with a 2 px surface gap between segments."""
    per = defaultdict(Counter)
    for r in rows:
        per[r["rib"]][r["bucket"]] += 1
    ribs = sorted(per)
    bottom = np.zeros(len(ribs))
    for b in BUCKETS:
        v = np.array([per[n][b] for n in ribs], float)
        tot = np.array([sum(per[n].values()) for n in ribs], float)
        pct = 100 * v / np.maximum(tot, 1)
        ax.bar(ribs, pct, bottom=bottom, color=BCOLOR[b], width=0.72,
               label=BLABEL[b], linewidth=1.4, edgecolor=SURFACE, zorder=3)
        # relief rule: aqua and yellow sit under 3:1 on this surface, so the segments
        # that matter carry a visible number rather than relying on hue alone
        for x, p, bo in zip(ribs, pct, bottom):
            if p >= 7 and b != "match":
                ax.text(x, bo + p / 2, f"{p:.0f}", ha="center", va="center",
                        fontsize=6.5, color=INK, fontweight="bold")
        bottom += pct
    ax.set_xticks(ribs)
    ax.set_ylim(0, 100)
    style(ax, "A · Where the numbering breaks", "rib number", "% of ribs at this number")


def panel_b(ax, rows):
    d = Counter(r["delta"] for r in rows if r["bucket"] == "offset")
    if not d:
        ax.text(.5, .5, "no offset ribs", ha="center", va="center", color=INK2,
                transform=ax.transAxes)
        style(ax, "B · Shift, or noise?")
        return
    ks = sorted(d)
    ax.bar(ks, [d[k] for k in ks], color=ORANGE, width=0.62, zorder=3)
    for k in ks:
        ax.text(k, d[k], f"{d[k]}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style(ax, "B · Shift, or noise?", "levels between the rib and the vertebra it sits on",
          "ribs")


def panel_c(ax, rows, anchor=10.0):
    g = np.array([r["gap_own_mm"] for r in rows if np.isfinite(r["gap_own_mm"])])
    if not len(g):
        style(ax, "C · Is 10 mm the right anchor?"); return
    hi = float(np.percentile(g, 99))
    ax.hist(np.clip(g, 0, hi), bins=44, color=BLUE, zorder=3)
    ax.axvline(anchor, color=ORANGE, lw=2, zorder=4)
    ax.text(anchor, ax.get_ylim()[1] * .94, f"  anchor {anchor:.0f} mm",
            color=ORANGE, fontsize=7.5, fontweight="bold", va="top")
    style(ax, "C · Is 10 mm the right anchor?",
          "gap to the vertebra the number claims (mm)", "ribs")


def paired_counts(rows):
    """Left/right rib counts per case, over rib numbers where BOTH sides are evaluable.

    A rib cut by the edge of the scan says nothing about how many ribs the patient has --
    it says where the scan stopped: on v4, 228 of 236 ribs that made a case look
    asymmetric were touching the top slice.

    But dropping truncated ribs ONE AT A TIME manufactures the very asymmetry it is
    meant to remove. If left rib 5 is cut and right rib 5 is not, excluding only the
    left one leaves 'right has a rib 5, left does not' -- an artefact of the exclusion,
    not of the scan. So the unit of exclusion is the rib NUMBER: if either side's rib n
    is truncated, n is dropped from both sides and the comparison is made on the levels
    that are wholly inside the field. That took v4 from 112 apparent asymmetries to the
    number quoted in the report."""
    per = defaultdict(lambda: {"left": {}, "right": {}})
    for r in rows:
        per[r["case"]][r["side"]][r["rib"]] = r["truncated"]
    out = {}
    for case, sides in per.items():
        cut = {n for side in ("left", "right") for n, t in sides[side].items() if t}
        L = sum(1 for n in sides["left"] if n not in cut)
        R = sum(1 for n in sides["right"] if n not in cut)
        out[case] = {"left": L, "right": R, "dropped": len(cut)}
    return out


def panel_d(ax, rows):
    """Left vs right rib count per case, FOV-truncated ribs excluded. Integer x integer
    -> a count grid, not a scatter: overplotting would hide exactly the asymmetric cases
    being looked for."""
    per = paired_counts(rows)
    L = np.array([v["left"] for v in per.values()])
    R = np.array([v["right"] for v in per.values()])
    if not len(L):
        style(ax, "D · Is the last rib symmetric?"); return
    lo, hi = 0, int(max(L.max(), R.max()))
    grid = np.zeros((hi - lo + 1, hi - lo + 1), int)
    for a, b in zip(L, R):
        grid[b - lo, a - lo] += 1
    m = ax.imshow(np.where(grid > 0, grid, np.nan), origin="lower", cmap="Blues",
                  extent=(lo - .5, hi + .5, lo - .5, hi + .5), zorder=3)
    ax.plot([lo - .5, hi + .5], [lo - .5, hi + .5], color=INK2, lw=1, ls="--", zorder=4)
    n_asym = int((L != R).sum())
    raw = defaultdict(lambda: {"left": 0, "right": 0})
    for r in rows:
        raw[r["case"]][r["side"]] += 1
    n_raw = sum(1 for v in raw.values() if v["left"] != v["right"])
    for (yy, xx), v in np.ndenumerate(grid):
        if v:
            ax.text(xx + lo, yy + lo, str(v), ha="center", va="center", fontsize=6.5,
                    color=INK if v < grid.max() * .6 else SURFACE, zorder=5)
    ax.set_xticks(range(lo, hi + 1)); ax.set_yticks(range(lo, hi + 1))
    ax.grid(False)
    style(ax, f"D · Rib count symmetry  ({n_asym} asymmetric; {n_raw} before "
              f"excluding FOV-cut ribs)", "left ribs", "right ribs")
    plt.colorbar(m, ax=ax, fraction=.045, pad=.03).set_label("cases", color=INK2)


def panel_e(ax, rows):
    last = defaultdict(lambda: {"left": 0, "right": 0})
    for r in rows:
        s = last[r["case"]]
        s[r["side"]] = max(s[r["side"]], r["rib"])
    ls = Counter(v["left"] for v in last.values() if v["left"])
    rs = Counter(v["right"] for v in last.values() if v["right"])
    ks = sorted(set(ls) | set(rs))
    if not ks:
        style(ax, "E · How many ribs?"); return
    w = 0.38
    x = np.arange(len(ks))
    ax.bar(x - w / 2, [ls[k] for k in ks], w, color=BLUE, label="left", zorder=3)
    ax.bar(x + w / 2, [rs[k] for k in ks], w, color=ORANGE, label="right", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(ks)
    style(ax, "E · How many ribs?", "number of the last rib", "cases")
    ax.legend(frameon=False, fontsize=7.5)


def panel_f(ax, rows):
    lv = Counter(r["nearest"] for r in rows if r["bucket"] == "lumbar")
    if not lv:
        ax.text(.5, .5, "no lumbar ribs found", ha="center", va="center", color=INK2,
                transform=ax.transAxes)
        style(ax, "F · Lumbar ribs, by level"); return
    ks = sorted(lv, key=lambda s: int(s[1:]))
    ax.bar(ks, [lv[k] for k in ks], color=AQUA, width=0.58, zorder=3)
    for k in ks:
        ax.text(k, lv[k], str(lv[k]), ha="center", va="bottom", fontsize=7, color=INK)
    style(ax, "F · Lumbar ribs, by level", "vertebra the rib sits on", "ribs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True, help="output dir of qc_rib_vertebra_incidence.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    qc = Path(a.qc)
    out = Path(a.out or qc)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = load(qc)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    panel_a(axes[0, 0], rows)
    panel_b(axes[0, 1], rows)
    panel_c(axes[0, 2], rows)
    panel_d(axes[1, 0], rows)
    panel_e(axes[1, 1], rows)
    panel_f(axes[1, 2], rows)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, -.005))
    n_case = len({r["case"] for r in rows})
    fig.suptitle(f"Rib–vertebra incidence · {summary.get('source','')} · "
                 f"{n_case} cases, {len(rows)} ribs", x=.008, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, .045, 1, .96))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rib_incidence.{ext}", dpi=200)

    # The table view. Two light-mode slots sit under 3:1, so the figure is never the
    # only copy of these numbers.
    lines = [f"source            {summary.get('source','')}",
             f"cases             {n_case}",
             f"ribs evaluated    {len(rows)}", ""]
    tot = Counter(r["bucket"] for r in rows)
    for b in BUCKETS:
        lines.append(f"  {BLABEL[b]:38s} {tot[b]:6d}  {100*tot[b]/max(1,len(rows)):5.1f}%")
    d = Counter(r["delta"] for r in rows if r["bucket"] == "offset")
    lines += ["", f"offset deltas     {dict(sorted(d.items()))}",
              f"misnumbered cases {summary.get('misnumbered_cases')}",
              f"lumbar-rib cases  {summary.get('lumbar_rib_cases')}"]
    raw = defaultdict(lambda: {"left": 0, "right": 0})
    for r in rows:
        raw[r["case"]][r["side"]] += 1
    cnt = paired_counts(rows)
    n_raw = sum(1 for v in raw.values() if v["left"] != v["right"])
    n_ok = sum(1 for v in cnt.values() if v["left"] != v["right"])
    ntr = sum(1 for r in rows if r["truncated"])
    lines += ["",
              f"ribs cut by the scan edge   {ntr} of {len(rows)} "
              f"({100*ntr/max(1,len(rows)):.1f}%)",
              f"asymmetric cases  RAW       {n_raw} of {len(raw)}",
              f"asymmetric cases  FOV-fixed {n_ok} of {len(cnt)}   "
              f"<- the one to quote"]
    (out / "rib_incidence_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/fig_rib_incidence.pdf|.png and rib_incidence_report.txt")


if __name__ == "__main__":
    main()
