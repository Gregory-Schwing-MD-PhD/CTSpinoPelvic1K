"""paper/mpda/make_figures.py — the figures for the dataset article.

TEN PUBLISHED PAGES is the constraint that shapes every choice here. Each figure has to
carry a claim that the text would otherwise spend a paragraph on, so there are four and
each one is doing work:

  fig2  the anatomy the dataset exists for, four specimens rendered from the labels
  fig3  the count-free measures -- the interval count and the bimodal rib ratio
  fig4  validation: derived measures against published reference values, and the one
        parameter that does not change with age
  fig5  opportunistic measures, which is the reuse case the article is arguing for

MPDA RULES FORBID HYPOTHESIS TESTING and require comprehensive descriptive analysis.
So: no p-values, no significance marks, no error bars implying a test. Medians,
interquartile bands, published reference lines, and counts.

PRINT, NOT SCREEN. Vector PDF, a single serif face matching the journal body text, and
no colour that carries meaning on its own -- every series is also distinguishable by
position or line style, because the figures will be read in greyscale by someone.

    python paper/mpda/make_figures.py --out paper/mpda/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

M = "morphometrics"

# A restrained palette. Teal and ochre carry the two-group comparisons; the level
# gradient uses a single hue ramp so it reads in order and survives greyscale.
TEAL, OCHRE, INK, FAINT = "#1c6b73", "#b8791f", "#22262b", "#8c9199"
RAMP = ["#cfe3e5", "#9dc7cc", "#6aabb2", "#3f8f97", "#1c6b73"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name):
    p = Path(M) / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def col(rows, k, lo=None, hi=None):
    v = [num(r, k) for r in rows]
    v = [x for x in v if x is not None]
    if lo is not None:
        v = [x for x in v if lo <= x <= hi]
    return np.asarray(v, float)


def by_sex(rows, k, lo=None, hi=None):
    out = {}
    for want, lab in (("F", "female"), ("M", "male")):
        sel = [r for r in rows if (r.get("sex") or "").strip().upper().startswith(want)]
        out[lab] = col(sel, k, lo, hi)
    return out


def kde(v, lo, hi, n=200):
    """Gaussian KDE with a robust bandwidth, reflected at both bounds.

    The robust scale matters: on the bimodal rib ratio a plain standard deviation is
    inflated by the separation between the modes, and Silverman's rule then returns a
    bandwidth wide enough to smooth the two into one.
    """
    v = np.asarray(v, float)
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 8:
        return np.linspace(lo, hi, n), np.zeros(n)
    sd = v.std(ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = 0.85 * 0.9 * max(scale, 1e-6) * v.size ** -0.2
    xs = np.linspace(lo, hi, n)
    acc = np.zeros(n)
    for src in (v, 2 * lo - v, 2 * hi - v):        # reflect at both bounds
        z = (xs[:, None] - src[None, :]) / h
        acc += np.exp(-0.5 * z ** 2).sum(1)
    return xs, acc / (v.size * h * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- fig 3
def fig_countfree(out):
    tr = load("transition_morphometrics.csv")
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.34)

    # (a) the interval count
    ax = fig.add_subplot(gs[0])
    c = {}
    for r in tr:
        n = num(r, "n_non_rib_bearing")
        if n is not None:
            c[int(n)] = c.get(int(n), 0) + 1
    ks = sorted(c)
    ax.bar([str(k) for k in ks], [c[k] for k in ks], color=TEAL, width=0.62)
    for i, k in enumerate(ks):
        ax.text(i, c[k], f"{c[k]}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("rib-free vertebrae above the sacrum")
    ax.set_ylabel("cases (log)")
    ax.set_title("(a) an interval, not a level", loc="left", fontsize=8.5)

    # (b) the bimodal rib ratio
    ax = fig.add_subplot(gs[1])
    v = col(tr, "rib12_11_ratio_min", 0.05, 1.05)
    xs, ys = kde(v, 0.05, 1.05)
    ax.fill_between(xs, ys, color=TEAL, alpha=0.18, lw=0)
    ax.plot(xs, ys, color=TEAL, lw=1.3)
    rugv = v[:: max(1, len(v) // 250)]
    ax.plot(rugv, np.full(rugv.size, -0.06), "|", color=FAINT, ms=3, mew=0.5)
    for m in (0.33, 0.69):
        ax.axvline(m, color=FAINT, ls=":", lw=0.7)
    ax.set_xlabel("lowest rib length / rib above")
    ax.set_ylabel("density")
    ax.set_title(f"(b) two populations (n = {len(v)})", loc="left", fontsize=8.5)

    # (c) the Castellvi geometry
    ax = fig.add_subplot(gs[2])
    xs_, ys_, fl = [], [], []
    for r in tr:
        for side in ("left", "right"):
            s, g = num(r, f"ll_span_{side}_mm"), num(r, f"tp_gap_{side}_mm")
            if s is None or g is None or not (20 <= s <= 130) or not (0 < g <= 60):
                continue
            xs_.append(s); ys_.append(g)
            fl.append((r.get("lstv_label") or "normal") != "normal")
    xs_, ys_, fl = np.array(xs_), np.array(ys_), np.array(fl)
    ax.scatter(xs_[~fl], ys_[~fl], s=1.6, c=FAINT, alpha=0.35, lw=0)
    ax.scatter(xs_[fl], ys_[fl], s=7, c=OCHRE, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("transverse span, one side (mm)")
    ax.set_ylabel("gap to the ala (mm)")
    ax.set_title("(c) span against contact", loc="left", fontsize=8.5)

    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig_countfree.pdf")
    plt.close(fig)
    print("  fig_countfree.pdf")


# ---------------------------------------------------------------- fig 4
def fig_validation(out):
    sg = load("surgical_morphometrics.csv")
    lv = load("level_gradients.csv")
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.34)

    # (a-c) three spinopelvic measures against their published value
    # REFERENCE VALUES ARE SOURCED AND CARRY THEIR SPREAD. Round numbers drawn as a
    # single line made every distribution look off-centre against a figure nobody had
    # checked. These are Vialle 2005 (n = 260 asymptomatic adults, standing), quoted as
    # mean +- SD and drawn as a band, which is what a reference range actually is.
    #
    # Sacral slope is expected BELOW the standing reference: it is postural, and lying
    # down rotates the pelvis. Pelvic incidence is not postural, which is why it can be
    # compared to a standing cohort without apology -- and why it matching to a decimal
    # place is the strongest check in the figure.
    for i, (key, title, lo, hi, ref, sd) in enumerate([
        ("pelvic_incidence_deg", "pelvic incidence", 20, 90, 54.7, 10.6),
        ("sacral_slope_deg", "sacral slope", 10, 70, 41.0, 8.4),
        ("pelvic_tilt_deg", "pelvic tilt", -10, 45, 13.0, 6.0),
    ]):
        ax = fig.add_subplot(gs[0, i])
        v = col(sg, key, lo, hi)
        xs, ys = kde(v, lo, hi)
        ax.axvspan(ref - sd, ref + sd, color=OCHRE, alpha=0.13, lw=0)
        ax.axvline(ref, color=OCHRE, ls="--", lw=1.0)
        ax.fill_between(xs, ys, color=TEAL, alpha=0.18, lw=0)
        ax.plot(xs, ys, color=TEAL, lw=1.3)
        ax.set_xlabel(f"{title} (deg)")
        ax.set_ylabel("density" if i == 0 else "")
        ax.set_title(f"({'abc'[i]}) {np.median(v):.1f} vs {ref:g}" + r"$\pm$" + f"{sd:g}",
                     loc="left", fontsize=8.5)

    # (d) the level gradient
    ax = fig.add_subplot(gs[1, 0])
    levels = ["L1", "L2", "L3", "L4", "L5"]
    for i, lvl in enumerate(levels):
        v = col(lv, f"endplate_width_{lvl}_mm", 25, 75)
        if v.size < 30:
            continue
        xs, ys = kde(v, 25, 75)
        ax.plot(xs, ys, color=RAMP[i], lw=1.2, label=lvl)
    ax.set_xlabel("superior endplate width (mm)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.4, ncol=2, handlelength=1.1, columnspacing=0.8)
    ax.set_title("(d) bodies broaden caudally", loc="left", fontsize=8.5)

    # (e) what changes with age, and what does not
    ax = fig.add_subplot(gs[1, 1:])
    buckets = {}
    for r in sg:
        a = num(r, "age")
        if a is None or not (40 <= a <= 99):
            continue
        buckets.setdefault(int(a // 10) * 10, []).append(r)
    decs = [d for d in sorted(buckets) if len(buckets[d]) >= 25]
    style = {"pelvic_incidence_deg": ("pelvic incidence", TEAL, "-"),
             "ll_supine_deg": ("lumbar lordosis", OCHRE, "--"),
             "pelvic_tilt_deg": ("pelvic tilt", INK, ":")}
    for key, (lab, cc, ls) in style.items():
        med = [np.median(col(buckets[d], key)) for d in decs]
        q1 = [np.percentile(col(buckets[d], key), 25) for d in decs]
        q3 = [np.percentile(col(buckets[d], key), 75) for d in decs]
        x = np.arange(len(decs))
        ax.fill_between(x, q1, q3, color=cc, alpha=0.10, lw=0)
        ax.plot(x, med, color=cc, ls=ls, lw=1.4, marker="o", ms=3, label=lab)
    ax.set_xticks(np.arange(len(decs)))
    ax.set_xticklabels([f"{d}s" for d in decs])
    ax.set_xlabel("age")
    ax.set_ylabel("degrees")
    ax.legend(fontsize=6.8, ncol=3, handlelength=1.6)
    ax.set_title("(e) incidence holds still while the spine compensates",
                 loc="left", fontsize=8.5)

    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig_validation.pdf")
    plt.close(fig)
    print("  fig_validation.pdf")


# ---------------------------------------------------------------- fig 5
def fig_opportunistic(out):
    op = load("opportunistic.csv")
    lv = load("level_gradients.csv")
    if not op:
        print("  ! opportunistic.csv missing; fig5 skipped")
        return
    fig = plt.figure(figsize=(7.0, 2.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.34)

    # (a) the distribution, with the osteoporosis threshold
    ax = fig.add_subplot(gs[0])
    v = col(op, "l1_trabecular_hu", 40, 320)
    xs, ys = kde(v, 40, 320)
    ax.fill_between(xs, ys, color=TEAL, alpha=0.18, lw=0)
    ax.plot(xs, ys, color=TEAL, lw=1.3)
    ax.axvline(110, color=OCHRE, ls="--", lw=1.0)
    low = int((v < 110).sum())
    ax.text(112, ax.get_ylim()[1] * 0.94, f" 110 HU\n {100*low/len(v):.1f}% below",
            fontsize=6.6, color=OCHRE, va="top")
    ax.set_xlabel("L1 trabecular attenuation (HU)")
    ax.set_ylabel("density")
    ax.set_title(f"(a) median {np.median(v):.0f} HU", loc="left", fontsize=8.5)

    # (b) the crossover
    ax = fig.add_subplot(gs[1])
    buckets = {}
    for r in op:
        a = num(r, "age")
        sx = (r.get("sex") or "").strip().upper()[:1]
        if a is None or sx not in ("F", "M"):
            continue
        buckets.setdefault((int(a // 10) * 10, sx), []).append(r)
    decs = sorted({d for d, _ in buckets
                   if len(buckets.get((d, "F"), [])) >= 20
                   and len(buckets.get((d, "M"), [])) >= 20})
    for sx, lab, cc, ls in (("F", "women", OCHRE, "-"), ("M", "men", TEAL, "--")):
        med = [np.median(col(buckets[(d, sx)], "l1_trabecular_hu")) for d in decs]
        ax.plot(np.arange(len(decs)), med, color=cc, ls=ls, lw=1.4,
                marker="o", ms=3.5, label=lab)
    ax.set_xticks(np.arange(len(decs)))
    ax.set_xticklabels([f"{d}s" for d in decs])
    ax.set_xlabel("age")
    ax.set_ylabel("L1 attenuation (HU)")
    ax.legend(fontsize=6.8, handlelength=1.6)
    ax.set_title("(b) the lines cross", loc="left", fontsize=8.5)

    # (c) wedging
    ax = fig.add_subplot(gs[2])
    worst = []
    for r in lv:
        vv = [x for x in (num(r, f"wedge_ratio_{l}") for l in
                          ("L1", "L2", "L3", "L4", "L5")) if x is not None and 0.2 < x < 2]
        if vv:
            worst.append(min(vv))
    worst = np.asarray(worst)
    xs, ys = kde(worst, 0.4, 1.6)
    ax.fill_between(xs, ys, color=TEAL, alpha=0.18, lw=0)
    ax.plot(xs, ys, color=TEAL, lw=1.3)
    ax.axvline(0.80, color=OCHRE, ls="--", lw=1.0)
    lo = int((worst < 0.80).sum())
    ax.text(0.81, ax.get_ylim()[1] * 0.94, f" 0.80\n {100*lo/len(worst):.1f}% below",
            fontsize=6.6, color=OCHRE, va="top")
    ax.set_xlabel("anterior / posterior body height")
    ax.set_ylabel("density")
    ax.set_title("(c) vertebral wedging", loc="left", fontsize=8.5)

    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig_opportunistic.pdf")
    plt.close(fig)
    print("  fig_opportunistic.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/mpda/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_countfree(out)
    fig_validation(out)
    fig_opportunistic(out)
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    main()
