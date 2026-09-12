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

# Resolved against the repository root, not the working directory: a relative
# path here made every input load as empty when the script was run from its own
# directory, and the failure surfaced as a numpy dtype error rather than a
# missing-file message.
M = str((Path(__file__).resolve().parents[2] / "morphometrics"))

# A restrained palette. Teal and ochre carry the two-group comparisons; the level
# gradient uses a single hue ramp so it reads in order and survives greyscale.
TEAL, OCHRE, INK, FAINT = "#1c6b73", "#b8791f", "#22262b", "#8c9199"
RAMP = ["#cfe3e5", "#9dc7cc", "#6aabb2", "#3f8f97", "#1c6b73"]

# --------------------------------------------------------------------------
# FIGURE STYLE -- Medical Physics author guidelines.
#
#   SANS SERIF: the guidelines name Arial, Helvetica or Calibri.
#
#   COLUMN WIDTHS ARE FIXED at 80 mm (single) and 180 mm (double), and "figure size
#   cannot be reduced in the typeset version". Figures are therefore authored AT the
#   final width and included at natural size, never rescaled afterwards -- rescaling
#   changes the printed type size and defeats any font choice made here.
#
#   TYPE SIZE: the guidelines ask for 20 pt or larger on the assumption that a figure
#   is drawn large and then shrunk to half a page. Authoring at the final width means
#   no shrinking occurs, so the equivalent is type that stays legible at 80-180 mm.
#
#   AXES bold black, GRIDLINES grey, MAJOR ticks outside and MINOR ticks inside.
#
#   RESOLUTION: 600 dpi for line art and charts; 300 dpi is for photographs only.
# --------------------------------------------------------------------------
MM = 1.0 / 25.4
COL1, COL2 = 80 * MM, 180 * MM          # single and double column, in inches
BLACK, GRIDGREY = "#000000", "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 1.1,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "text.color": BLACK,
    "axes.titleweight": "normal",
    "grid.color": GRIDGREY,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": BLACK, "ytick.color": BLACK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def mp_ticks(ax):
    """Major ticks outward, minor ticks inward, grey grid behind the data."""
    ax.tick_params(which="major", direction="out")
    ax.tick_params(which="minor", direction="in")
    ax.grid(True, which="major", axis="both")      # grey gridlines both ways (journal figure style)
    ax.set_axisbelow(True)
    return ax


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


def kde(v, lo, hi, n=200, bw=0.85):
    """Gaussian KDE with a robust bandwidth, reflected at both bounds.

    The robust scale matters: on the bimodal rib ratio a plain standard deviation is
    inflated by the separation between the modes, and Silverman's rule then returns a
    bandwidth wide enough to smooth the two into one.

    WHICH IS WHY `bw` IS A PARAMETER AND NOT A CONSTANT. Narrowing below Silverman buys
    resolution on a distribution that really has two modes and manufactures structure on one
    that does not. The spinopelvic panels are the second case: gated pelvic tilt is a single
    right-skewed hump (skew 0.34, kurtosis 0.05), and at 0.85 the curve showed an apparent
    notch near 19 degrees that is 0.8 Poisson sigma deep, moves with the bin width, and
    disappears entirely by 3-degree bins. A reader cannot tell that from the real
    bimodality this figure used to carry, so those panels are drawn at Silverman.
    """
    v = np.asarray(v, float)
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 8:
        return np.linspace(lo, hi, n), np.zeros(n)
    sd = v.std(ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = bw * 0.9 * max(scale, 1e-6) * v.size ** -0.2
    xs = np.linspace(lo, hi, n)
    acc = np.zeros(n)
    for src in (v, 2 * lo - v, 2 * hi - v):        # reflect at both bounds
        z = (xs[:, None] - src[None, :]) / h
        acc += np.exp(-0.5 * z ** 2).sum(1)
    return xs, acc / (v.size * h * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- fig 3
def fig_countfree(out):
    tr = load("transition_morphometrics.csv")
    fig = plt.figure(figsize=(COL2, 1.62), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

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
    from matplotlib.ticker import NullFormatter
    ax.set_yticks([30, 100, 300, 1000]); ax.set_yticklabels(["30", "100", "300", "1000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("rib-free vertebrae above the sacrum")
    ax.set_ylabel("cases")
    ax.set_title("(a) Rib-free vertebral count", loc="left", fontsize=8.5)

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
    ax.set_title(f"(b) Lowest rib length ratio (n = {len(v)})",
                 loc="left", fontsize=8.5)

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
    ax.set_title("(c) Span against gap to the ala", loc="left", fontsize=8.5)

    for ax in fig.axes:
        mp_ticks(ax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig_countfree.pdf")
    plt.close(fig)
    print("  fig_countfree.pdf")


# ---------------------------------------------------------------- fig 4
def _clean_rows(rows, flag_key="qc_flags"):
    """Rows whose every QC flag says ok.

    THE FILTER IS THE TOOLKIT'S OWN VERDICT, NOT A WINDOW ON THE VALUE. The previous
    version kept pelvic incidence between 20 and 90 degrees, which quietly dropped 68 of
    802 records -- and dropped them for having the wrong ANSWER rather than for any
    identified fault, which is the shape of a filter that flatters a distribution. Every
    excluded case now carries a reason: a violated PI = SS + PT identity, a geometry
    outside the anatomical envelope, or a landmark that could not be fitted.
    """
    out = []
    for r in rows:
        f = (r.get(flag_key) or "").strip()
        parts = [x for x in f.replace(",", ";").split(";") if x]
        if all(x in ("", "ok") for x in parts):
            out.append(r)
    return out


def fig_validation(out):
    # The spinopelvic parameters come from the toolkit, which reports PI, SS and PT
    # independently and flags what it could not measure; the per-level dimensions still
    # come from the release's own extraction code.
    # THE NUMBERS ARE THE RELEASE'S OWN, and the exclusion is the extractor's plate-tilt
    # rejection rather than a window on the answer. Compared head-to-head over 300 records
    # the release's extraction agreed with published supine CT better than the toolkit's
    # more elaborate estimator did -- pelvic incidence 49.2 against 45.8 for a published
    # 47.1-52.1, and 6.3% anatomically impossible against 16.0% -- so this reports the one
    # that measures better, not the one that is nicer to describe.
    # TWO EXCLUSIONS, BOTH ON GEOMETRY AND BOTH STATED.
    #
    #   s1_plate_rejected  the fitted sacral plate's normal lies more than 60 deg off the
    #     cranial axis, so it is the anterior face of the promontory and not an endplate.
    #     99 of 802. Their sacral slopes have a median of 67.5 deg and reach 89.9; the
    #     cases that survive have a median of 35.1 and a maximum of 59.8.
    #
    #   pelvic tilt below -15 deg  an anteversion steeper than any reported. 60 of the
    #     remainder. These cases have a NORMAL sacral plate (tilt 38 deg, slope 38.0) and
    #     a NORMAL femoral head separation (164 mm), yet return a pelvic incidence of 1 to
    #     17 deg, which would put the hip axis almost on the plate normal. The plate and
    #     the heads are each right and their relative geometry is not; the cause is not
    #     identified, so they are excluded and counted rather than explained away.
    #     Documented in docs/SPINOPELVIC_ESTIMATOR.md.
    def _keep(r):
        if (r.get("s1_plate_rejected") or "0") not in ("", "0"):
            return False
        try:
            # The tilt filter that used to live here is gone. It removed records by their
            # ANSWER -- "no pelvis is anteverted past 15 degrees" -- and it was standing in
            # for a defect in the plate fit: `_endplate` orients its normal by the superior
            # component alone, so a plane fitted to the ventral surface of the sacrum comes
            # back leaning posteriorly at a normal-looking angle from vertical, and pelvic
            # incidence collapses. That is now rejected upstream on the geometry, and it
            # leaves nothing behind: the released measures carry no pelvic tilt below -15
            # and no pelvic incidence below 22. See docs/SPINOPELVIC_ESTIMATOR.md.
            return True
        except (TypeError, ValueError, KeyError):
            return False
    _all = load("surgical_morphometrics.csv")
    sg = [r for r in _all if _keep(r)]
    KEYS = {k: k for k in ("pelvic_incidence_deg", "sacral_slope_deg", "pelvic_tilt_deg")}
    if _all:
        _plate = sum(1 for r in _all
                     if (r.get("s1_plate_rejected") or "0") not in ("", "0"))
        print(f"  fig_validation: {len(sg)}/{len(_all)} kept "
              f"({_plate} plate rejected, {len(_all) - len(sg) - _plate} tilt below -15 deg)")
    # ONE ROW. The level-by-level panels moved to the level atlas, which draws the
    # same measurements with their spread instead of as overlapping density curves.
    fig = plt.figure(figsize=(COL2, 1.28), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # (a-c) three spinopelvic measures against their published values
    #
    # TWO REFERENCES, BECAUSE THE MODALITY MATTERS MORE THAN THE POSTURE HERE. The band is
    # Vialle 2005 (n=300 asymptomatic adults, standing radiographs), quoted as mean +- SD,
    # which is what a reference range actually is. But this cohort is supine CT, and CT
    # reads LOWER than a standing radiograph in the same subjects -- 53 against 56 degrees
    # of pelvic incidence in Lee & Liu (Eur Spine J 2022;31:241). Comparing a CT cohort
    # against radiographic norms alone therefore builds in an offset that has nothing to
    # do with this dataset, and a reader cannot tell it from a real discrepancy.
    #
    # So the CT literature is drawn too, as a second line -- and it is ONE series for all
    # three panels: the 200-subject automated supine CT cohort of Veilleux (JBJS Am
    # 2020;102:e130), which is what Table II compares against. It was briefly two sources,
    # with pelvic incidence taken from Vrtovec (Spine 2012;37:E479, n=370, 47.1 deg) while
    # the table quoted Veilleux's 52.1, so the same paper carried two different published
    # CT values for the same parameter and the distribution looked five degrees high
    # against one of them. Mixing reference cohorts across panels of one figure is not
    # worth the extra citation.
    #
    # THE PELVIC INCIDENCE REFERENCE WAS ONCE 54.7, WHICH IS THIS COHORT'S OWN MEASURED
    # VALUE copied into the reference slot -- the same error a co-author caught in
    # Table II. Comparing a measurement against itself is not a check.
    # ONE CT SERIES WAS NOT ENOUGH, AND THE GAP WAS THE COHORT, NOT THE CODE.
    #
    # Sacral slope here reads about 3 degrees below Veilleux and pelvic tilt about 4 above,
    # with pelvic incidence unmoved -- which is exactly the signature of a rotated vertical
    # reference, since a rotation delta moves SS by -delta and PT by +delta and leaves PI
    # alone (Ohashi et al., Spine Surg Relat Res 8:61, who write it as aSS = SS - APPA and
    # aPT = PT + APPA). That reading was pursued and does not survive. Recumbency runs the
    # WRONG WAY for it: lying down rotates the pelvis anteriorly, RAISING sacral slope and
    # lowering tilt, by 0.9 degrees in 211 patients (Banitalebi, Clin Spine Surg 39:E104),
    # 3.9 in 15 volunteers (Chevillotte, OTSR 104:565) and 7.1 in 24 (Hasegawa below). An
    # anterior-pelvic-plane convention in Veilleux would push the same way. Neither
    # explains a cohort reading LOW on slope.
    #
    # What explains it is who is in the scanner. Veilleux's 200 were asymptomatic subjects
    # imaged for non-musculoskeletal reasons; this cohort is an abdominopelvic CT
    # population, older and symptomatic. The one published supine-CT series matched on
    # that -- Hasegawa et al., BMC Musculoskelet Disord 19:437, women of about 60 with
    # adult spinal deformity, supine CT-DRR -- reports PI 53.4, SS 34.1, PT 19.2 against
    # this cohort's 52.6, 33.1 and 19.6. That is agreement to within a degree on all
    # three, from a cohort resembling this one, using a scanner-axis vertical.
    #
    # So both are drawn, as a band between them rather than a line through one. A single
    # reference point invites the reading that any departure is a defect; two series that
    # differ by more than this cohort differs from either says what is actually true.
    #
    # Still not settled, and stated in docs/SPINOPELVIC_ESTIMATOR.md rather than guessed
    # at: whether Veilleux is anterior-pelvic-plane referenced. Its abstract lists ASIS and
    # pubic tubercles, which have no role in a scanner-axis slope, and the companion paper
    # on the same 200 subjects (Higgins, JBJS Am 96:1776) says the frame is the APP -- but
    # the sign of the published supine APP tilt is not consistent across cohorts, so the
    # correction cannot be applied in the right direction with any confidence.
    # ONE CT SERIES ON THE FIGURE, NOT TWO. Drawing Hasegawa beside Veilleux was tried and
    # reverted. The two differ by more than this cohort differs from either, which is the
    # honest picture -- but Hasegawa is 24 adult-spinal-deformity patients and Veilleux is
    # 200 asymptomatic subjects measured automatically, and a line on a figure carries no
    # n and no cohort. Giving them equal visual weight reads as having gone looking for the
    # series that agrees. A table can carry both with their n, so Table II does; the figure
    # keeps the stronger series and the text explains the gap.
    for i, (key, title, lo, hi, ref, sd, ct_refs, ct_lab) in enumerate([
        ("pelvic_incidence_deg", "pelvic incidence", 20, 90, 55.0, 10.6, (52.1,),
         "CT, Veilleux ($n$=200)"),
        ("sacral_slope_deg", "sacral slope", 10, 70, 41.0, 8.4, (36.5,),
         "CT, Veilleux ($n$=200)"),
        ("pelvic_tilt_deg", "pelvic tilt", -10, 45, 13.0, 6.0, (15.6,),
         "CT, Veilleux ($n$=200)"),
    ]):
        ax = fig.add_subplot(gs[0, i])
        v = col(sg, KEYS[key], lo, hi)
        # Silverman, not the narrowed bandwidth: see kde(). These three are unimodal, and
        # undersmoothing them draws sampling noise as a notch a reader will read as a
        # second population.
        xs, ys = kde(v, lo, hi, bw=1.0)
        ax.axvspan(ref - sd, ref + sd, color=OCHRE, alpha=0.13, lw=0)
        ax.axvline(ref, color=OCHRE, ls="--", lw=1.0,
                   label=None)
        for k, c in enumerate(ct_refs):
            ax.axvline(c, color=INK, ls=(0, (1.4, 1.2)), lw=1.0,
                       label=None)
        ax.fill_between(xs, ys, color=TEAL, alpha=0.18, lw=0)
        ax.plot(xs, ys, color=TEAL, lw=1.3)
        ax.set_xlabel(f"{title} (°)")
        ax.set_ylabel("density" if i == 0 else "")
        ax.set_title(f"({'abc'[i]}) {title.capitalize()}", loc="left", fontsize=8.5)
    for ax in fig.axes:
        mp_ticks(ax)
        ax.spines[["top", "right"]].set_visible(False)

    # ONE KEY UNDER ALL THREE PANELS, not a box inside panel (a). The two reference lines
    # mean the same thing in every panel, so a legend in one of them reads as though it
    # applied only there -- and inside the axes it sat on the distribution it was
    # explaining. Centred beneath the row, matching the anchors figure, and close to it:
    # a key that floats is a key the eye has to hunt for.
    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=OCHRE, ls="--", lw=1.0, label="standing radiograph"),
           Line2D([], [], color=INK, ls=(0, (1.4, 1.2)), lw=1.0, label="published CT")]
    fig.legend(handles=key, loc="lower center", ncol=2, frameon=False, fontsize=6.4,
               handlelength=1.6, columnspacing=2.0, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.20))
    fig.savefig(out / "fig_validation.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  fig_validation.pdf")


# ---------------------------------------------------------------- fig 5
def fig_opportunistic(out):
    op = load("opportunistic.csv")
    lv = load("level_gradients.csv")
    if not op:
        print("  ! opportunistic.csv missing; fig5 skipped")
        return
    fig = plt.figure(figsize=(COL2, 1.88), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

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
    # the median belongs in the caption: at this panel width the title overran into (b)
    ax.set_title("(a) L1 trabecular attenuation", loc="left", fontsize=8.5)

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
    ax.set_title("(b) Attenuation by decade and sex", loc="left", fontsize=8.5)

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
    ax.set_title("(c) Lowest lumbar wedge ratio", loc="left", fontsize=8.5)

    for ax in fig.axes:
        mp_ticks(ax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig_opportunistic.pdf")
    plt.close(fig)
    print("  fig_opportunistic.pdf")


def main():
    ap = argparse.ArgumentParser()
    # RESOLVED AGAINST THIS FILE, NOT THE WORKING DIRECTORY. As a bare relative path this
    # default silently created paper/mpda/paper/mpda/figures/ whenever the script was run
    # from its own directory, and the build then typeset the previous run's figures with
    # no error anywhere -- the worst kind of stale, because everything reports success.
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_countfree(out)
    fig_validation(out)
    fig_opportunistic(out)
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    main()
