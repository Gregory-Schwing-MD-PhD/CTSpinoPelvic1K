"""make_levelatlas_fig.py -- build fig_levelatlas.pdf and the CSV behind it.

Six panels, each answering a question a textbook figure answers from a few dozen cadaveric
specimens and no measure of spread:

  (a) body height, ventral and dorsal      Benzel Fig. 1.2
  (b) superior endplate width              Benzel Fig. 1.1 (width series)
  (c) canal width and depth                Benzel Fig. 1.9
  (d) transverse pedicle width             Benzel Fig. 1.11
  (e) disc height by interspace            quoted in the text as single values
  (f) the legend -- trabecular attenuation used to sit here, but no published series
      measures it, so it was the one panel a reader could not check

Imports the manuscript's figure style from make_figures so this panel cannot drift from the
others: same sans face, bold black axes, grey grid, 600 dpi, authored at the final 180 mm
double-column width so nothing is rescaled in typesetting.

    python paper/mpda/make_levelatlas_fig.py --out paper/mpda/figures
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import make_figures as MF          # style, palette, loaders
from levelatlas import LEVELS, GATES, series, draw, annotate_n

DISCS = ["L1L2", "L2L3", "L3L4", "L4L5", "L5S1"]
DISC_LABEL = {"L1L2": "L1–L2", "L2L3": "L2–L3", "L3L4": "L3–L4",
              "L4L5": "L4–L5", "L5S1": "L5–S1"}


SPREAD_CSV = (Path(__file__).resolve().parents[2] / "morphometrics"
              / "pedicle_width_references.csv")
LEVELREF_CSV = (Path(__file__).resolve().parents[2] / "morphometrics"
                / "level_references.csv")

# EVERY SERIES THE TABLES HOLD IS DRAWN. An earlier version showed Panjabi alone, on the
# argument that the manuscript's case runs through the textbook a surgeon consults and the
# textbook redraws Panjabi. That argument was half wrong, and checking it is what changed
# this figure: Benzel's Chapter 1 cites Panjabi's CERVICAL and THORACIC papers (refs 3 and
# 4, p. 15) and does not cite the 1992 lumbar paper at all, while the transverse-pedicle
# figure a surgeon actually reads (Fig. 1.11, p. 6) is drawn from Krag, Zindrick and
# Bernard. Showing one series and calling it the textbook's would have been wrong.
#
# So the panels now show the whole published record. They disagree -- by 21 to 69% -- and
# that disagreement is the point a reader is entitled to see rather than take on trust: it
# is the reason a single dashed line is not a reference range, and the reason a cohort of
# 802 with its own spread is worth having. Panjabi keeps a heavier black line because it is
# the series the manuscript's text cites; every other series is thin.
# ONLY PANJABI IS DRAWN. The machinery below handles all forty series in the two tables and
# was used that way for a while, but the manuscript cites Panjabi and a handful of others,
# so a legend naming twenty-eight studies put twenty-five names in the figure that appear
# nowhere in the bibliography. The full record stays in morphometrics/level_references.csv
# and morphometrics/pedicle_width_references.csv, and it is a paper of its own.
#
# One curve covers both Panjabi papers: the thoracic 1991 series supplies T11 and T12, the
# 1992 lumbar series L1 to L5, and SERIES_ALIAS merges them under one name so the line runs
# the whole span instead of breaking at the thoracolumbar junction.
#
# Set to None to draw them all again.
DRAW_SERIES = {"Panjabi 1992"}

DASHES = [(0, ()),                       # solid
          (0, (4.0, 1.4)),               # long dash
          (0, (2.4, 1.2)),               # medium dash
          (0, (1.2, 1.2)),               # fine dash
          (0, (5.0, 1.2, 1.0, 1.2)),     # dash-dot
          (0, (1.0, 1.8))]               # dotted
# Okabe-Ito, which stays separable under the common colour-vision deficiencies. Black is
# held back for Panjabi, and pure yellow is dropped -- it disappears on white.
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#E69F00", "#7A5195", "#4D7C3A"]

EMPHASIS = "Panjabi 1992"     # the series the manuscript's text cites

# The two tables name the same studies differently -- surname alone in the pedicle table,
# surname and year in the general one. Left unmerged, Panjabi's pedicle rows became two
# half-curves and only the thoracic half was drawn.
SERIES_ALIAS = {"Panjabi": "Panjabi 1992", "Zindrick": "Zindrick 1987",
                "Yu": "Yu 2015", "Arockiaraj": "Arockiaraj 2025",
                "Makino": "Makino 2012", "Krag": "Krag 1988",
                "Hou": "Hou 1993", "Olsewski": "Olsewski 1990",
                "Lien": "Lien 2007", "Mitra": "Mitra 2002",
                "Robertson": "Robertson 2000", "Otsuki": "Otsuki 2020",
                "Moncada-Habib": "Moncada-Habib 2023"}


def _style_table(names):
    """One unique (colour, dash) per series, assigned deterministically.

    Assigned by position in a SORTED list rather than written out by hand, because the
    hand-written version drifted: Bonczar, Shin and Yu were all given the same blue solid
    line and the legend could not tell a reader which curve was which. Colour cycles
    fastest so that neighbouring entries in the legend differ in colour as well as dash.
    """
    out = {EMPHASIS: ("#000000", DASHES[0], 1.25)}
    i = 0
    for n in sorted(x for x in names if x != EMPHASIS):
        out[n] = (PALETTE[i % len(PALETTE)], DASHES[(i // len(PALETTE)) % len(DASHES)], 0.75)
        i += 1
    assert len({(c, tuple(d) if isinstance(d, tuple) else d) for c, d, _ in out.values()})         == len(out), "two series would draw identically"
    return out


def load_reference_series():
    """({measure: {series: {level: mean}}}, {series: legend label}).

    The legend label is built from the table's own cohort/n/modality columns rather than
    written out per series, so adding a row to a CSV adds a correctly-labelled curve and
    the legend cannot fall out of step with the data behind it.
    """
    tally, meta, sdev = {}, {}, {}
    for path in (LEVELREF_CSV, SPREAD_CSV):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(rows):
            try:
                v = float(r["mean"])
            except (ValueError, KeyError, TypeError):
                continue
            # THE TWO TABLES NAME THE PEDICLE DIFFERENTLY. The pedicle-specific table calls
            # it PDW and the general one calls it "pedicle"; they are the same measure and
            # Panjabi's thoracic levels live in the second. Merging them is what lets one
            # curve run T11 to L5 instead of stopping at L1.
            meas = "PDW" if r["measure"] == "pedicle" else r["measure"]
            ser = SERIES_ALIAS.get(r["series"], r["series"])
            tally.setdefault((meas, ser), {}).setdefault(r["level"], []).append(v)
            meta.setdefault(ser, r)
            # POPULATION SD, not the standard error of the mean. Panjabi reports
            # "mean +/- SEM" with n=12, and SEM is about 3.5x narrower than the SD it
            # came from. Drawing that next to a percentile interval over 700 patients
            # would compare the precision of HIS MEAN against the spread of MY
            # POPULATION, which is not a comparison at all -- his band would look
            # implausibly tight and this cohort implausibly variable. Converted back.
            try:
                disp_v = float(r.get("sd") or "")
                nn = float(r.get("n") or "")
            except ValueError:
                continue
            kind = (r.get("dispersion") or "").upper()
            sd = disp_v * math.sqrt(nn) if "SEM" in kind or "SE" == kind.strip() else disp_v
            sdev.setdefault((meas, ser), {})[r["level"]] = (sd, nn)
    out = {}
    for (meas, ser), per_level in tally.items():
        for lv, vals in per_level.items():
            out.setdefault(meas, {}).setdefault(ser, {})[lv] = sum(vals) / len(vals)

    spread = {}
    for (meas, ser), per_level in sdev.items():
        spread.setdefault(meas, {})[ser] = per_level

    labels = {}
    for ser, r in meta.items():
        n = (r.get("n") or "").strip()
        coh = (r.get("cohort") or "").strip().lower()
        kind = ("cadaver" if "cadaver" in coh or "dry bone" in coh else
                "meta" if "meta" in coh or "pooled" in coh else
                "MRI" if "mri" in (r.get("modality") or "").lower() else
                "CT" if "ct" in (r.get("modality") or "").lower() else "")
        bits = ", ".join(x for x in (kind, (f"$n$={n}" if n else "")) if x)
        labels[ser] = f"{ser} ({bits})" if bits else ser
    return out, labels, spread


# WHAT INTERVAL THE REFERENCE BAND SHOWS, AND WHY IT IS NOT A CONFIDENCE INTERVAL.
#
# The question this figure asks is whether an individual patient is unusual, not whether
# two means differ. Hahn, Meeker & Escobar (Statistical Intervals, 2nd ed., Wiley 2017,
# ch. 1) put the choice plainly: a confidence interval bounds a PARAMETER, a prediction
# interval bounds ONE FUTURE OBSERVATION, a tolerance interval bounds A STATED PROPORTION
# of the population. Only the last two answer this figure's question. With n=12 the
# confidence interval on Panjabi's mean is sqrt(12) = 3.5x narrower than his population
# interval, so drawing it would make every cohort look aberrant for a reason that is
# arithmetic rather than anatomy -- the error the 2024 lumbar meta-analysis makes when it
# rules differences insignificant on overlapping confidence intervals (Bonczar et al.,
# Surg Radiol Anat 46:2097). Cumming, Stat Med 28:205, on why intervals should not be
# compared by eye at all.
#
# SO THE TWO LAYERS ARE MATCHED AS POPULATION INTERVALS. The cohort rows already show the
# 5th-95th percentile, which is a 95% population interval read off 700+ patients; the band
# is mean +/- 1.96 SD, the same interval for Panjabi's specimens under normality. They are
# the same quantity, which is what makes the widths comparable at all -- and they come out
# within a median 1.04x of each other.
#
# THE BAND ITSELF IS UNCERTAIN, AND THAT IS DRAWN TOO. The SD is recovered as SEM*sqrt(12)
# from twelve specimens, so it carries a chi-square(11) 95% interval of 0.71x to 1.70x.
# Drawing that as a second, wider band was tried and abandoned: at 1.70x the pedicle band
# covers most of its panel and the two canal bands merge into one wash, so the figure
# loses the comparison it exists to make. Bland & Altman's rule for limits of agreement
# (Stat Methods Med Res 8:135) is to publish the limits AND their uncertainty rather than
# retreat to the mean -- so the factor is stated in the text, where it costs no legibility.
#
# A 95/95 tolerance band would need k=3.16 rather than 1.96 at n=12 (Howe, JASA 64:610),
# i.e. 61% wider again. That is the honest figure for "where 95% of the population lies,
# with 95% confidence", and it is stated in the text rather than drawn, because at that
# width the reference stops discriminating anything.
REF_K = 1.96
# chi-square(11) upper 95% bound on an SD estimated from n=12
REF_SD_CI_HI = 1.70


def draw_reference_band(ax, spread, refs, measure, y_of, series=EMPHASIS, offset=0.0):
    """Panjabi's population interval, mean +/- REF_K SD, behind the cohort rows.

    Drawn only where the source reports a dispersion. The T11--T12 end-plate means are
    digitised from a figure, which carries no SEM, so the band stops at L1 there rather
    than being extended by assumption.
    """
    per_level = (refs.get(measure) or {}).get(series)
    sd_level = (spread.get(measure) or {}).get(series)
    if not per_level or not sd_level:
        return
    runs, cur = [], []
    for lv in LEVELS:                      # keep anatomical order, break where SD is absent
        if lv in per_level and lv in sd_level and lv in y_of:
            cur.append(lv)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        if len(run) < 2:
            continue
        ys = [y_of[l] + offset for l in run]
        lo = [per_level[l] - REF_K * sd_level[l][0] for l in run]
        hi = [per_level[l] + REF_K * sd_level[l][0] for l in run]
        ax.fill_betweenx(ys, lo, hi, color="#000000", alpha=0.085, lw=0, zorder=0.6)


def draw_reference_series(ax, refs, measure, y_of, styles, labels,
                          offset=0.0, drawn=None):
    """One line per series, through its own per-level means, over every level it has."""
    d = refs.get(measure)
    if not d:
        return
    for ser, per_level in sorted(d.items()):
        if DRAW_SERIES is not None and ser not in DRAW_SERIES:
            continue
        st = styles.get(ser)
        if st is None:
            continue
        colour, dash, lw = st
        lv = [l for l in LEVELS if l in per_level and l in y_of]
        if len(lv) < 2:                      # a two-point series is still a line worth
            continue                         # drawing; a one-point series is not
        ax.plot([per_level[l] for l in lv], [y_of[l] + offset for l in lv],
                color=colour, ls=dash, lw=lw,
                zorder=1.9 if ser == EMPHASIS else 1.5,
                alpha=0.95, solid_capstyle="round")
        if drawn is not None:
            drawn[labels.get(ser, ser)] = (colour, dash, lw)


def build(out: Path, reference: bool = True):
    lg = MF.load("level_gradients.csv")
    sm = MF.load("surgical_morphometrics.csv")
    dg = MF.load("degenerative.csv")
    op = MF.load("opportunistic.csv")
    refs, ref_labels, ref_spread = (load_reference_series() if reference
                                    else ({}, {}, {}))
    styles = _style_table({ser for d in refs.values() for ser in d})
    drawn = {}
    if not (lg and sm and dg and op):
        raise SystemExit("morphometrics CSVs not found")

    S = {
        "h_ant":    series(lg, "body_height_{l}_mm",      GATES["height"]),
        "h_post":   series(lg, "body_height_post_{l}_mm", GATES["height"]),
        "endplate": series(lg, "endplate_width_{l}_mm",   GATES["endplate"]),
        "canal_w":  series(lg, "canal_width_{l}_mm",      GATES["canal_w"]),
        "canal_ap": series(sm, "canal_ap_mm_{l}",         GATES["canal_ap"]),
        # PEDICLE WIDTH IS THE MEAN OF THE TWO SIDES HERE, not the minimum. pedicle_mm is
        # the narrower pedicle, which is the right number for a screw and the wrong one
        # for this figure: Panjabi and every other series report PDW per side, and their
        # own left-right gaps reach 1.3 mm at L4, so plotting our minimum against their
        # mean is biased low before any anatomy is involved. Measured over the cohort the
        # gap is 0.7 mm at L1 and 2.4 mm at L5 -- the wrong one of the two would have made
        # the caudal widening look shallower than it is.
        "pedicle":  series(sm, "pedicle_mean_mm_{l}",     GATES["pedicle"]),
        "disc":     series(dg, "disc_height_{l}_mm", (1.0, 25.0), levels=DISCS),
        "hu":       series(op, "{l}_trabecular_hu", (-50.0, 400.0),
                           levels=["l1", "l2", "l3", "l4"]),
    }

    # Level runs down the axis in anatomical order: T11 at the top, L5 at the bottom, the
    # way a surgeon reads a spine and the way the textbook figures are drawn.
    y_of = {lv: -i for i, lv in enumerate(LEVELS)}
    # A disc sits between its two vertebrae, so it is drawn at the midpoint of their rows
    # rather than given a row of its own.
    y_disc = {d: (y_of[d[:2]] + (y_of.get(d[2:], y_of["L5"] - 1))) / 2.0 for d in DISCS}
    y_hu = {k: y_of[k.upper()] for k in ["l1", "l2", "l3", "l4"]}

    # A legend band is kept for the multi-series case, but with one reference it is dead
    # page area: a single named curve belongs in the caption, and the panels get the height
    # back. The band collapses automatically when only one series is drawn.
    _multi = DRAW_SERIES is None or len(DRAW_SERIES) > 1
    fig = plt.figure(figsize=(MF.COL2, (49 if _multi else 36) * MF.MM))
    if _multi:
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.26], hspace=0.44, wspace=0.22)
        axes = np.array([fig.add_subplot(gs[0, i]) for i in range(3)])
        ax_key = fig.add_subplot(gs[1, :])
        ax_key.axis("off")
    else:
        gs = fig.add_gridspec(1, 3, wspace=0.22)
        axes = np.array([fig.add_subplot(gs[0, i]) for i in range(3)])
        ax_key = None
    TEAL, OCHRE, INK, FAINT = MF.TEAL, MF.OCHRE, MF.INK, MF.FAINT
    # BODY HEIGHT AND DISC HEIGHT ARE DEFERRED to the next paper: body height needs
    # the cohort-versus-method argument settled first, and neither is what the
    # textbook curve a surgeon consults is about.
    ax_b, ax_c, ax_d = axes.ravel()

    # (b) superior endplate width
    draw(ax_b, S["endplate"], y_of, TEAL, "o")
    annotate_n(ax_b, S["endplate"], y_of, FAINT)
    draw_reference_band(ax_b, ref_spread, refs, "endplate", y_of)
    draw_reference_series(ax_b, refs, "endplate", y_of, styles, ref_labels, drawn=drawn)
    ax_b.set_xlabel("upper end-plate width, EPWu (mm)")
    ax_b.set_title("(a) Upper end-plate width (EPWu)", loc="left", fontsize=8.0)
    # NAME THE STATISTIC ON THE FIGURE, not only in the caption, and name both n. A reader
    # who does not know the band is a population interval and not a confidence interval
    # will read its width as "the cadavers were less variable" (Cumming, Fidler & Vaux,
    # J Cell Biol 177:7, rule 1: say what the bars are and say n).
    ax_b.legend(handles=[Line2D([], [], color=MF.INK, lw=1.4,
                                label="Panjabi mean, $n$=12/level"),
                         Patch(facecolor="#000000", alpha=0.085, lw=0,
                               label="$\pm$1.96 SD (95% of specimens)"),
                         ],
                fontsize=5.6, handlelength=1.1, handleheight=0.9, labelspacing=0.32,
                borderpad=0.35, loc="upper right", framealpha=0.88, edgecolor="none")

    # (c) canal, width against depth
    draw(ax_c, S["canal_w"], y_of, TEAL, "o", offset=+0.17, label="width (SCW)")
    draw(ax_c, S["canal_ap"], y_of, INK, "^", offset=-0.17, label="depth (SCD)")
    draw_reference_band(ax_c, ref_spread, refs, "canal_w", y_of, offset=+0.17)
    draw_reference_series(ax_c, refs, "canal_w", y_of, styles, ref_labels,
                          offset=+0.17, drawn=drawn)
    draw_reference_band(ax_c, ref_spread, refs, "canal_ap", y_of, offset=-0.17)
    draw_reference_series(ax_c, refs, "canal_ap", y_of, styles, ref_labels,
                          offset=-0.17, drawn=drawn)
    ax_c.set_xlabel("spinal canal, SCW and SCD (mm)")
    ax_c.set_title("(b) Canal (SCW, SCD)", loc="left", fontsize=8.0)
    # upper right: the canal narrows upward, so the free space is to the right of the
    # T11-T12 rows. Lower left sits on the depth whiskers and lower right on the L5 width
    # marker, which is the widest canal in the panel.
    ax_c.legend(fontsize=6.3, handlelength=1.0, loc="upper right")

    # (d) transverse pedicle width
    draw(ax_d, S["pedicle"], y_of, OCHRE, "D")
    annotate_n(ax_d, S["pedicle"], y_of, FAINT)
    draw_reference_band(ax_d, ref_spread, refs, "PDW", y_of)
    draw_reference_series(ax_d, refs, "PDW", y_of, styles, ref_labels, drawn=drawn)
    ax_d.set_xlabel("transverse pedicle width, PDW (mm)")
    ax_d.set_title("(c) Pedicle width (PDW)", loc="left", fontsize=8.0)


    for ax in (ax_b, ax_c, ax_d):
        MF.mp_ticks(ax)
        ax.set_yticks([y_of[l] for l in LEVELS])
        ax.set_ylim(-len(LEVELS) + 0.4, 0.6)
        ax.grid(axis="both", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # Only the left column carries level labels; the discs panel names interspaces instead,
    # because labelling it with vertebral levels would put a disc on a vertebra.
    for ax in (ax_b,):
        ax.set_yticklabels(LEVELS)
        ax.set_ylabel("vertebral level")
    for ax in (ax_c, ax_d):
        ax.set_yticklabels(LEVELS)
        ax.tick_params(labelleft=True)
    # The key names every series actually drawn, in the order they were drawn, with
    # Panjabi first because it is the one the text cites.
    ordered = ([k for k in drawn if k.startswith(EMPHASIS)] +
               sorted(k for k in drawn if not k.startswith(EMPHASIS)))
    handles = [Line2D([0], [0], color=drawn[k][0], ls=drawn[k][1], lw=max(drawn[k][2], 1.0))
               for k in ordered]
    if handles and ax_key is not None:
        ncol = 3 if len(handles) <= 9 else (4 if len(handles) <= 16 else 6)
        ax_key.legend(handles, ordered, loc="upper center", ncol=ncol,
                      fontsize=4.4, handlelength=1.7, handletextpad=0.32,
                      columnspacing=0.6, labelspacing=0.24, frameon=False,
                      borderaxespad=0.0)
    fig.subplots_adjust(left=0.072, right=0.996, top=0.955, bottom=0.015)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_levelatlas.pdf")
    fig.savefig(out / "fig_levelatlas.png", dpi=200)   # for the website
    plt.close(fig)

    # The numbers behind the figure, released alongside it. A figure a reader cannot get
    # the values out of has the same defect as the textbook plots it replaces.
    csv_path = Path(MF.M) / "level_atlas.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "level", "n", "median", "q1", "q3",
                    "p5", "p95", "mean", "sd", "unit"])
        unit = {"hu": "HU"}
        for name, st in S.items():
            for lv, s in st.items():
                w.writerow([name, lv.upper(), s["n"]] +
                           [f"{s[k]:.2f}" for k in
                            ("med", "q1", "q3", "p5", "p95", "mean", "sd")] +
                           [unit.get(name, "mm")])
    print("  fig_levelatlas.pdf / .png")
    print(f"  {csv_path}")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/mpda/figures")
    ap.add_argument("--no-reference", dest="reference", action="store_false",
                    help="omit the published-range overlay")
    a = ap.parse_args()
    S = build(Path(a.out), reference=a.reference)
    for name, st in S.items():
        print(f"{name:9s} " + ", ".join(f"{lv.upper()}:{s['n']}" for lv, s in st.items()))


if __name__ == "__main__":
    main()
