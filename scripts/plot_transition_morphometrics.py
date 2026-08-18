"""scripts/plot_transition_morphometrics.py — the transitional-anatomy landscape.

THE FIGURE THIS IS BUILT AROUND. Sacral foramina against non-rib-bearing vertebrae, on
one axis pair. Four non-rib-bearing vertebrae is ambiguous on its own -- it is equally a
lumbar rib on L1 or a sacralized L5 -- and the foramina count is the one measure that
tells them apart without trusting a vertebra label. Plotting them together turns a
recurring judgement call into a position on a plane, and the known LSTV cases are drawn
over the top so the label can be checked against the anatomy rather than assumed.

Everything else supports that: Castellvi space (how far the lowest lumbar reaches, and how
close it comes to the ala, per side, because the asymmetry is the phenotype), the disc
ratio, and the thoracolumbar end. The last panel is the claim worth testing -- that T-L and
L-S border variants co-occur, i.e. one population rather than two.

Plain csv + numpy on purpose: this has to run inside the container without assuming pandas.

    python scripts/plot_transition_morphometrics.py --csv morphometrics/transition_morphometrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

# A radiograph's own palette: cool bone greys for the population, warm signal for the
# anomaly. One warm accent, spent only on the thing the figure is about.
GROUND = "#FAFAF8"
INK = "#12181E"
MUTE = "#8A939B"
GRID = "#DFE3E6"
NORMAL = "#7C8B98"
LSTV = "#D2492A"
LUMRIB = "#1F7A6F"
AMBER = "#D89B2B"
VIOLET = "#6B5B95"

plt.rcParams.update({
    "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.9, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 170,
})


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", "None", None):
                r[k] = None
                continue
            try:
                r[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
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
    lab = str(r.get("lstv_label") or "").lower()
    return bool(lab) and lab not in ("normal", "none", "nan")


def title(ax, main, sub=None):
    ax.set_title(main, loc="left", pad=10 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=MUTE,
                va="bottom", ha="left")


def jitter(n, w=0.16):
    return (np.random.RandomState(7).rand(n) - 0.5) * 2 * w


# ---------------------------------------------------------------- panels
def p_discriminator(ax, rows):
    """The one that matters: foramina vs non-rib-bearing count."""
    ok = [r for r in rows if isinstance(r.get("foramina_max_side"), (int, float))
          and isinstance(r.get("n_non_rib_bearing"), (int, float))]
    x = np.array([float(r["n_non_rib_bearing"]) for r in ok])
    y = np.array([float(r["foramina_max_side"]) for r in ok])
    flag = np.array([is_lstv(r) for r in ok])
    lr = np.array([bool(r.get("has_lumbar_rib")) for r in ok])

    ax.add_patch(Rectangle((4.5, 3.5), 1.0, 1.0, facecolor=NORMAL, alpha=0.09, zorder=0))
    ax.text(5.0, 4.52, "expected normal\n5 free lumbar · 4 foramina", ha="center",
            va="bottom", fontsize=7.5, color=MUTE)

    ax.scatter(x[~flag & ~lr] + jitter((~flag & ~lr).sum()),
               y[~flag & ~lr] + jitter((~flag & ~lr).sum()),
               s=14, c=NORMAL, alpha=.35, lw=0, label="no LSTV label", zorder=2)
    ax.scatter(x[lr] + jitter(lr.sum()), y[lr] + jitter(lr.sum()),
               s=34, facecolors="none", edgecolors=LUMRIB, lw=1.3,
               label="lumbar rib (class 74/75)", zorder=3)
    ax.scatter(x[flag] + jitter(flag.sum()), y[flag] + jitter(flag.sum()),
               s=30, c=LSTV, alpha=.85, lw=0, label="labelled LSTV", zorder=4)

    ax.set_xlabel("non-rib-bearing vertebrae above the sacrum")
    ax.set_ylabel("sacral foramina (larger side)")
    ax.grid(alpha=.5, lw=.7)
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0, -0.16), ncol=3)
    title(ax, "Sacralization, disentangled",
          "4 free lumbar is ambiguous alone — 5 foramina resolves it to a sacralized L5")


def p_hist(ax, rows, key, label, sub, colour, expect=None):
    a = col(rows, key)
    if a.size == 0:
        ax.axis("off"); return
    lo, hi = int(np.floor(a.min())), int(np.ceil(a.max()))
    bins = np.arange(lo - .5, hi + 1.5, 1) if (hi - lo) <= 14 else 24
    ax.hist(a, bins=bins, color=colour, alpha=.85, lw=0)
    if expect is not None:
        ax.axvline(expect, color=LSTV, lw=1.4, ls=(0, (4, 2)))
        ax.text(expect, ax.get_ylim()[1] * .96, f" normal = {expect}", color=LSTV,
                fontsize=7.5, va="top")
    ax.set_xlabel(label)
    ax.set_ylabel("cases")
    ax.grid(axis="y", alpha=.5, lw=.7)
    title(ax, label, sub)


def p_castellvi(ax, rows):
    """Span vs gap, per side — Castellvi's own two axes."""
    for side, mk, cl in (("left", "o", VIOLET), ("right", "^", AMBER)):
        ok = [r for r in rows
              if isinstance(r.get(f"ll_span_{side}_mm"), (int, float))
              and isinstance(r.get(f"tp_gap_{side}_mm"), (int, float))]
        x = np.array([float(r[f"ll_span_{side}_mm"]) for r in ok])
        y = np.array([float(r[f"tp_gap_{side}_mm"]) for r in ok])
        f = np.array([is_lstv(r) for r in ok])
        ax.scatter(x[~f], y[~f], s=11, c=cl, alpha=.28, lw=0, marker=mk,
                   label=f"{side}")
        ax.scatter(x[f], y[f], s=26, c=LSTV, alpha=.8, lw=0, marker=mk)
    ax.axhline(2.0, color=LSTV, lw=1.2, ls=(0, (4, 2)))
    ax.text(ax.get_xlim()[1], 2.2, "contact / fusion  (Castellvi III–IV) ", color=LSTV,
            fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("lowest lumbar lateral span, one side (mm)")
    ax.set_ylabel("gap to sacrum / ilium (mm)")
    ax.set_yscale("symlog", linthresh=2)
    ax.grid(alpha=.5, lw=.7)
    ax.legend(loc="upper right", fontsize=7.5)
    title(ax, "Castellvi space",
          "a long process that reaches the ala is the phenotype; red = labelled LSTV")


def p_cooccur(ax, rows):
    """Do the two ends of the spine go together?"""
    def tl_odd(r):
        v = r.get("rib12_11_ratio_min")
        return (bool(r.get("has_lumbar_rib"))
                or (isinstance(v, (int, float)) and v < 0.45)
                or (isinstance(r.get("n_rib_bearing"), (int, float))
                    and r["n_rib_bearing"] not in (0, 12)))

    def ls_odd(r):
        n = r.get("n_non_rib_bearing")
        f = r.get("foramina_max_side")
        return (is_lstv(r)
                or (isinstance(n, (int, float)) and n != 5)
                or (isinstance(f, (int, float)) and f >= 5))

    tab = Counter((tl_odd(r), ls_odd(r)) for r in rows)
    m = np.array([[tab[(False, False)], tab[(False, True)]],
                  [tab[(True, False)], tab[(True, True)]]], float)
    ax.imshow(m, cmap="BuPu", vmin=0, vmax=max(1.0, m.max()))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=GROUND if m[i, j] > m.max() * .55 else INK)
    ax.set_xticks([0, 1], ["typical", "variant"])
    ax.set_yticks([0, 1], ["typical", "variant"])
    ax.set_xlabel("lumbosacral border")
    ax.set_ylabel("thoracolumbar border")
    n = m.sum()
    exp = m.sum(1)[1] * m.sum(0)[1] / n if n else 0
    title(ax, "Do both ends vary together?",
          f"observed {int(m[1,1])} vs {exp:.1f} expected if independent")
    for s in ax.spines.values():
        s.set_visible(False)


def p_ribcount(ax, rows):
    l = col(rows, "n_ribs_left")
    r_ = col(rows, "n_ribs_right")
    b = np.arange(-.5, 14.5, 1)
    ax.hist([l, r_], bins=b, color=[VIOLET, AMBER], label=["left", "right"], lw=0)
    lr = sum(1 for r in rows if r.get("has_lumbar_rib"))
    ax.set_xlabel("ribs segmented per side")
    ax.set_ylabel("cases")
    ax.grid(axis="y", alpha=.5, lw=.7)
    ax.legend(fontsize=7.5)
    title(ax, "Rib count per side",
          f"{lr} case(s) additionally carry a rib on a lumbar body")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()
    rows = load(a.csv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_lstv = sum(1 for r in rows if is_lstv(r))
    print(f"  {len(rows)} cases, {n_lstv} with an LSTV label")

    fig = plt.figure(figsize=(15.5, 13.6))
    gs = fig.add_gridspec(3, 3, hspace=.52, wspace=.30,
                          left=.06, right=.97, top=.90, bottom=.07)

    p_discriminator(fig.add_subplot(gs[0, :2]), rows)
    p_hist(fig.add_subplot(gs[0, 2]), rows, "foramina_max_side",
           "sacral foramina (larger side)",
           "5 means the sacrum absorbed L5", NORMAL, expect=4)
    p_hist(fig.add_subplot(gs[1, 0]), rows, "n_non_rib_bearing",
           "non-rib-bearing vertebrae", "the classic count", NORMAL, expect=5)
    p_hist(fig.add_subplot(gs[1, 1]), rows, "disc_ratio",
           "lowest disc / disc above", "a rudimentary L5–S1 disc runs low", VIOLET)
    p_hist(fig.add_subplot(gs[1, 2]), rows, "rib12_11_ratio_min",
           "rib 12 / rib 11 length", "hypoplastic 12th rib runs low", AMBER)
    p_castellvi(fig.add_subplot(gs[2, 0]), rows)
    p_ribcount(fig.add_subplot(gs[2, 1]), rows)
    p_cooccur(fig.add_subplot(gs[2, 2]), rows)

    fig.text(.06, .965, "Transitional anatomy across CTSpinoPelvic1K",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(.06, .942,
             f"{len(rows)} densified cases · {n_lstv} carrying an LSTV label · "
             f"every measure derived from the segmentation, not from the vertebra numbering",
             fontsize=9.5, color=MUTE)
    p = out / "fig_transition_landscape.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
