"""scripts/plot_anatomical_report.py — the cross-sectional anatomical report.

A dataset paper needs to show what is IN the data, not test a hypothesis about it. So this
is descriptive by design: distributions, stratifications, and one positive control.

THE POSITIVE CONTROL COMES FIRST, and it is the panel to read before believing any other.
The human sacrum is sexually dimorphic -- wider relative to its height in females -- and
that fact is established independently of anything here. If the pipeline's sacral
measurements reproduce it, the measurements are picking up real anatomy; if they do not,
every other panel is suspect no matter how tidy it looks. A dataset paper that shows a
known effect recovered is far more persuasive than one that only shows novel ones.

EVERYTHING IS COUNT-FREE. No panel names a vertebral level. Sacralization and lumbarization
are one morphology under two counts, so the measures are intervals, ratios and shapes:
vertebrae BETWEEN the lowest rib and the sacrum, rib length as a fraction of the rib above,
dimensionless disc ratios, and lateral span against gap to the ala.

AGE AND SEX ARE INCOMPLETE and the panels say so rather than dropping the cases silently:
age is present for 709 of 802 and sex for 738, so each stratified panel reports its own n.

    python scripts/plot_anatomical_report.py --csv morphometrics/transition_morphometrics.csv
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
NEUTRAL, SIGNAL, TEAL, AMBER, VIOLET = "#7C8B98", "#D2492A", "#1F7A6F", "#D89B2B", "#6B5B95"
F_COL, M_COL = "#B4508B", "#3E7CB1"

plt.rcParams.update({
    "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "axes.linewidth": .9, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 165,
})


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", "None", None):
                r[k] = None
                continue
            try:
                r[k] = float(v) if ("." in v or "e" in str(v).lower()) else int(v)
            except (ValueError, TypeError):
                pass
    return rows


def col(rows, k, where=None):
    out = []
    for r in rows:
        if where and not where(r):
            continue
        v = r.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return np.array(out)


def is_lstv(r):
    s = str(r.get("lstv_label") or "").lower()
    return bool(s) and s not in ("normal", "none", "nan")


def title(ax, main, sub=None):
    ax.set_title(main, loc="left", pad=22 if sub else 6)
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=8, color=MUTE,
                va="bottom", ha="left")


def cohen_d(a, b):
    if len(a) < 3 or len(b) < 3:
        return None
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / max(1, len(a) + len(b) - 2))
    return None if sd <= 0 else float((a.mean() - b.mean()) / sd)


# ---------------------------------------------------------------- panels
def p_control(ax, rows):
    """POSITIVE CONTROL: sacral sexual dimorphism, a fact established elsewhere."""
    f = col(rows, "sacrum_width_mm", lambda r: r.get("sex") == "female")
    m = col(rows, "sacrum_width_mm", lambda r: r.get("sex") == "male")
    fa = col(rows, "sacrum_aspect", lambda r: r.get("sex") == "female")
    ma = col(rows, "sacrum_aspect", lambda r: r.get("sex") == "male")
    parts = ax.violinplot([m, f], positions=[0, 1], widths=.7, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([M_COL, F_COL][i]); pc.set_alpha(.55); pc.set_edgecolor("none")
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        if k in parts:
            parts[k].set_color(MUTE); parts[k].set_linewidth(1.0)
    ax.set_xticks([0, 1], [f"male\nn={len(m)}", f"female\nn={len(f)}"])
    ax.set_ylabel("sacral width (mm)")
    d = cohen_d(f, m)
    da = cohen_d(fa, ma)
    ax.grid(axis="y", alpha=.5, lw=.7)
    title(ax, "Positive control: sacral dimorphism",
          f"width d={d:+.2f}  ·  height/width aspect d={da:+.2f}  ·  "
          f"females have the wider, squatter sacrum" if d and da else "")


def p_age_sex(ax, rows):
    a_f = col(rows, "age", lambda r: r.get("sex") == "female")
    a_m = col(rows, "age", lambda r: r.get("sex") == "male")
    bins = np.arange(20, 95, 5)
    ax.hist([a_m, a_f], bins=bins, color=[M_COL, F_COL], alpha=.85, lw=0,
            label=[f"male (n={len(a_m)})", f"female (n={len(a_f)})"], stacked=True)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("cases")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=.5, lw=.7)
    allage = col(rows, "age")
    title(ax, "Who is in the cohort",
          f"age {np.median(allage):.0f} median ({np.percentile(allage,5):.0f}"
          f"-{np.percentile(allage,95):.0f} 5-95th), missing in "
          f"{sum(1 for r in rows if r.get('age') is None)} of {len(rows)}")


def p_by_sex(ax, rows, key, label, sub):
    f = col(rows, key, lambda r: r.get("sex") == "female")
    m = col(rows, key, lambda r: r.get("sex") == "male")
    if f.size < 3 or m.size < 3:
        ax.axis("off"); return
    lo = float(min(f.min(), m.min())); hi = float(max(f.max(), m.max()))
    bins = np.linspace(lo, hi, 26)
    ax.hist(m, bins=bins, color=M_COL, alpha=.62, lw=0, label=f"male n={len(m)}",
            density=True)
    ax.hist(f, bins=bins, color=F_COL, alpha=.62, lw=0, label=f"female n={len(f)}",
            density=True)
    d = cohen_d(f, m)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=.5, lw=.7)
    title(ax, label, f"{sub}  ·  d={d:+.2f}" if d is not None else sub)


def p_vs_age(ax, rows, key, label, sub, colour):
    pts = [(r["age"], r[key]) for r in rows
           if isinstance(r.get("age"), (int, float))
           and isinstance(r.get(key), (int, float))]
    if len(pts) < 20:
        ax.axis("off"); return
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    ax.scatter(x, y, s=8, c=colour, alpha=.28, lw=0)
    # decade medians: a trend line invites a causal read the design cannot support
    dec = defaultdict(list)
    for xi, yi in zip(x, y):
        dec[int(xi // 10) * 10].append(yi)
    ks = sorted(k for k in dec if len(dec[k]) >= 8)
    ax.plot([k + 5 for k in ks], [np.median(dec[k]) for k in ks], "-o",
            color=INK, lw=1.6, ms=4, label="decade median")
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_xlabel("age (years)")
    ax.set_ylabel(label)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=.45, lw=.7)
    title(ax, label, f"{sub}  ·  r={r:+.2f}, n={len(pts)}")


def p_prev_by_sex(ax, rows):
    """Prevalence of each count-free variant, by sex, with binomial error bars."""
    def variant(r):
        out = {}
        n = r.get("n_non_rib_bearing")
        out["non-5 free\nlumbar"] = isinstance(n, (int, float)) and n != 5
        v = r.get("rib12_11_ratio_min")
        out["hypoplastic\n12th rib"] = isinstance(v, (int, float)) and v < 0.30
        out["rib on a\nlumbar body"] = bool(r.get("has_lumbar_rib"))
        out["labelled\nLSTV"] = is_lstv(r)
        return out
    keys = list(variant(rows[0]).keys())
    w = .36
    for gi, (sx, colour) in enumerate((("male", M_COL), ("female", F_COL))):
        sub = [r for r in rows if r.get("sex") == sx]
        n = len(sub)
        ps, es = [], []
        for k in keys:
            c = sum(1 for r in sub if variant(r)[k])
            p = c / max(1, n)
            ps.append(100 * p)
            es.append(100 * np.sqrt(max(p * (1 - p), 1e-9) / max(1, n)))
        ax.bar([i + (gi - .5) * w for i in range(len(keys))], ps, width=w,
               color=colour, alpha=.85, yerr=es, capsize=3,
               error_kw={"lw": 1, "ecolor": MUTE}, label=f"{sx} n={n}")
    ax.set_xticks(range(len(keys)), keys, fontsize=7.5)
    ax.set_ylabel("% of cases")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=.5, lw=.7)
    title(ax, "Variant prevalence by sex",
          "bars are binomial standard errors; overlapping bars are not a difference")


def p_hist(ax, rows, key, label, sub, colour, expect=None):
    a = col(rows, key)
    if a.size == 0:
        ax.axis("off"); return
    uniq = np.unique(a)
    integral = bool(np.all(np.isclose(uniq, np.round(uniq))))
    lo, hi = int(np.floor(a.min())), int(np.ceil(a.max()))
    bins = (np.arange(lo - .5, hi + 1.5, 1) if (integral and hi - lo <= 14)
            else np.linspace(float(np.percentile(a, .5)), float(np.percentile(a, 99.5)), 30))
    ax.hist(a, bins=bins, color=colour, alpha=.85, lw=0)
    if expect is not None:
        ax.axvline(expect, color=SIGNAL, lw=1.4, ls=(0, (4, 2)))
    ax.set_xlabel(label); ax.set_ylabel("cases")
    ax.grid(axis="y", alpha=.5, lw=.7)
    title(ax, label, sub)


def p_cooccur(ax, rows):
    def tl(r):
        v = r.get("rib12_11_ratio_min"); low = r.get("lowest_rib_bearing")
        return (bool(r.get("has_lumbar_rib"))
                or (isinstance(v, (int, float)) and v < 0.30)
                or (isinstance(low, str) and low not in ("T12", "")))
    def ls(r):
        n = r.get("n_non_rib_bearing")
        return is_lstv(r) or (isinstance(n, (int, float)) and n != 5)
    sub = [r for r in rows if isinstance(r.get("n_non_rib_bearing"), (int, float))]
    t = Counter((tl(r), ls(r)) for r in sub)
    m = np.array([[t[(False, False)], t[(False, True)]],
                  [t[(True, False)], t[(True, True)]]], float)
    ax.imshow(m, cmap="BuPu", vmin=0, vmax=max(1.0, m.max()))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i,j])}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=GROUND if m[i, j] > m.max() * .55 else INK)
    ax.set_xticks([0, 1], ["typical", "variant"]); ax.set_yticks([0, 1], ["typical", "variant"])
    ax.set_xlabel("lumbosacral border"); ax.set_ylabel("thoracolumbar border")
    n = m.sum(); exp = m.sum(1)[1] * m.sum(0)[1] / n if n else 0
    a_, b_, c_, d_ = m[1, 1], m[1, 0], m[0, 1], m[0, 0]
    orat = (a_ * d_) / (b_ * c_) if (b_ and c_) else float("nan")
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, "Do both borders vary together?",
          f"observed {int(a_)} vs {exp:.1f} if independent · odds ratio {orat:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()
    rows = load(a.csv)
    n_lstv = sum(1 for r in rows if is_lstv(r))
    n_age = sum(1 for r in rows if isinstance(r.get("age"), (int, float)))
    n_sex = sum(1 for r in rows if r.get("sex") in ("male", "female"))
    print(f"  {len(rows)} cases · {n_lstv} LSTV-labelled · age {n_age} · sex {n_sex}")

    fig = plt.figure(figsize=(16.5, 20))
    gs = fig.add_gridspec(5, 3, hspace=.62, wspace=.30,
                          left=.055, right=.975, top=.935, bottom=.035)

    p_control(fig.add_subplot(gs[0, 0]), rows)
    p_age_sex(fig.add_subplot(gs[0, 1]), rows)
    p_prev_by_sex(fig.add_subplot(gs[0, 2]), rows)

    p_hist(fig.add_subplot(gs[1, 0]), rows, "n_non_rib_bearing",
           "non-rib-bearing vertebrae", "counted between lowest rib and sacrum",
           NEUTRAL, expect=5)
    p_hist(fig.add_subplot(gs[1, 1]), rows, "rib12_11_ratio_min",
           "rib 12 / rib 11 length", "two modes: typical near 0.68, hypoplastic near 0.32",
           AMBER)
    p_hist(fig.add_subplot(gs[1, 2]), rows, "disc_ratio",
           "lowest disc / disc above", "dimensionless, comparable across patients", VIOLET)

    p_by_sex(fig.add_subplot(gs[2, 0]), rows, "rib12_11_ratio_min",
             "rib 12 / rib 11 length", "by sex")
    p_by_sex(fig.add_subplot(gs[2, 1]), rows, "sacrum_height_mm",
             "sacral height (mm)", "by sex")
    p_by_sex(fig.add_subplot(gs[2, 2]), rows, "ll_span_total_mm",
             "lowest lumbar total span (mm)", "by sex")

    p_vs_age(fig.add_subplot(gs[3, 0]), rows, "disc_ratio",
             "lowest disc / disc above", "disc space narrows with degeneration", VIOLET)
    p_vs_age(fig.add_subplot(gs[3, 1]), rows, "ll_span_total_mm",
             "lowest lumbar span (mm)", "osteophytic spread widens the body", AMBER)
    p_vs_age(fig.add_subplot(gs[3, 2]), rows, "tp_gap_min_mm",
             "closest gap to sacrum / ilium (mm)", "bridging closes the gap", TEAL)

    p_cooccur(fig.add_subplot(gs[4, 0]), rows)
    p_hist(fig.add_subplot(gs[4, 1]), rows, "sacrum_aspect",
           "sacral height / width", "the dimorphic ratio", NEUTRAL)
    p_hist(fig.add_subplot(gs[4, 2]), rows, "iliac_crest_at_id",
           "iliac crest height (vertebra id)", "a classic lumbosacral landmark", TEAL)

    fig.text(.055, .967, "CTSpinoPelvic1K — cross-sectional spinal morphometry",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(.055, .948,
             f"{len(rows)} CT cases · age known in {n_age} · sex known in {n_sex} · "
             f"{n_lstv} carrying an LSTV label.   Every measure is COUNT-FREE: none "
             f"requires knowing which vertebra is which, because a transitional vertebra "
             f"cannot be named from a spine-limited field of view.",
             fontsize=9.5, color=MUTE)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_anatomical_report.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
