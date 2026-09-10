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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures as MF          # style, palette, loaders
from levelatlas import LEVELS, GATES, series, draw, annotate_n

DISCS = ["L1L2", "L2L3", "L3L4", "L4L5", "L5S1"]
DISC_LABEL = {"L1L2": "L1–L2", "L2L3": "L2–L3", "L3L4": "L3–L4",
              "L4L5": "L4–L5", "L5S1": "L5–S1"}


SPREAD_CSV = (Path(__file__).resolve().parents[2] / "morphometrics"
              / "pedicle_width_references.csv")
LEVELREF_CSV = (Path(__file__).resolve().parents[2] / "morphometrics"
                / "level_references.csv")

# NAMED SERIES, DRAWN AS LINES. An unlabelled span with ticks showed that the references
# disagree but not WHICH reference is where, so a reader could not check our distribution
# against any particular one. Each series now gets its own line, its own colour and dash,
# and an entry in the legend.
#
# The set is chosen for coverage, not cherry-picked: these are the series that appear
# across the most panels, plus the largest cohort available for each measure. Panjabi is
# always drawn because it is the series the manuscript cites. Others exist in the tables
# and are deliberately not drawn -- eighteen lines on the canal panel would be unreadable,
# and the full record is in morphometrics/level_references.csv.
#
# Colours are Okabe-Ito, which stays distinguishable for the common colour-vision
# deficiencies, and every series also carries a distinct dash so the panels survive being
# printed in grey.
# ONLY PANJABI IS DRAWN HERE, deliberately. The full multi-series machinery below works and
# the tables carry ten series, but this manuscript's argument runs through the textbook a
# surgeon actually consults, and that textbook redraws Panjabi. Showing ten series that
# disagree by 21 to 69% is a different paper's argument, and a good one -- the tables, the
# styles and the drawing code are all kept for it. Set DRAW_SERIES to None to draw them all.
DRAW_SERIES = {"Panjabi 1992", "Panjabi"}

DASH_SOLID = "-"
DASH_LONG = (0, (4, 1.4))
DASH_MED = (0, (2.4, 1.2))
DASH_FINE = (0, (1.4, 1.2))
DASH_DOT = (0, (1, 1.6))
DASH_DASHDOT = (0, (5, 1.2, 1, 1.2))

# EVERY SERIES GETS A UNIQUE COLOUR-AND-DASH PAIR. A first version reused styles between
# the two reference tables, so the legend showed Bonczar, Shin and Yu as the same blue
# solid line and a reader could not tell which curve belonged to which study. The assert
# below is what keeps that from coming back.
#
# Colours are Okabe-Ito, which stays distinguishable for the common colour-vision
# deficiencies, and the dash carries the same information again so the panels survive
# being printed in grey.
SERIES_STYLE = {
    "Panjabi 1992":  ("#000000", DASH_LONG,    "Panjabi 1992 (cadaver, $n$=12)"),
    "Bonczar 2024":  ("#0072B2", DASH_SOLID,   "Bonczar 2024 (meta, $n$=1481)"),
    "Griffith 2016": ("#009E73", DASH_FINE,    "Griffith 2016 (CT, $n$=1080)"),
    "Duman 2026":    ("#D55E00", DASH_DASHDOT, "Duman 2026 (CT, $n$=517)"),
    "Yadav 2020":    ("#CC79A7", DASH_MED,     "Yadav 2020 (CT, $n$=302)"),
    "Tan 2004":      ("#56B4E9", DASH_DOT,     "Tan 2004 (cadaver, $n$=10)"),
    "Shin 2024":     ("#0072B2", DASH_FINE,    "Shin 2024 (CT, $n$=700)"),
}
# the pedicle table keys on surname alone
SERIES_STYLE.update({
    "Panjabi":    SERIES_STYLE["Panjabi 1992"],
    "Yu":         ("#56B4E9", DASH_DASHDOT, "Yu 2015 (cadaver, $n$=503)"),
    "Zindrick":   ("#D55E00", DASH_MED,     "Zindrick 1987 (cadaver)"),
    "Arockiaraj": ("#CC79A7", DASH_DOT,     "Arockiaraj 2025 (CT, $n$=300)"),
})
_seen = {}
for _k, _v in SERIES_STYLE.items():
    _sig = (_v[0], _v[1])
    assert _sig not in _seen or _seen[_sig] == _v[2], (
        f"style clash: {_v[2]!r} and {_seen[_sig]!r} would draw identically")
    _seen[_sig] = _v[2]


def load_reference_series():
    """{measure: {series: {level: mean}}}, sexes averaged into one line per series."""
    out, tally = {}, {}
    for path in (LEVELREF_CSV, SPREAD_CSV):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(rows):
            try:
                v = float(r["mean"])
            except (ValueError, KeyError):
                continue
            key = (r["measure"], r["series"], r["level"])
            tally.setdefault(key, []).append(v)
    for (meas, ser, lv), vals in tally.items():
        out.setdefault(meas, {}).setdefault(ser, {})[lv] = sum(vals) / len(vals)
    return out


def draw_reference_series(ax, refs, measure, y_of, offset=0.0, drawn=None):
    """One line per named series, through its own per-level means."""
    d = refs.get(measure)
    if not d:
        return
    for ser, per_level in sorted(d.items()):
        if DRAW_SERIES is not None and ser not in DRAW_SERIES:
            continue
        style = SERIES_STYLE.get(ser)
        if style is None:
            continue
        colour, dash, label = style
        lv = [l for l in LEVELS if l in per_level] or [l for l in per_level]
        lv = [l for l in lv if l in y_of]
        if len(lv) < 3:
            continue
        ax.plot([per_level[l] for l in lv], [y_of[l] + offset for l in lv],
                color=colour, ls=dash, lw=1.0, zorder=1.6, alpha=0.95,
                solid_capstyle="round")
        if drawn is not None:
            drawn[label] = (colour, dash)


def build(out: Path, reference: bool = True):
    lg = MF.load("level_gradients.csv")
    sm = MF.load("surgical_morphometrics.csv")
    dg = MF.load("degenerative.csv")
    op = MF.load("opportunistic.csv")
    refs = load_reference_series() if reference else {}
    drawn = {}
    if not (lg and sm and dg and op):
        raise SystemExit("morphometrics CSVs not found")

    S = {
        "h_ant":    series(lg, "body_height_{l}_mm",      GATES["height"]),
        "h_post":   series(lg, "body_height_post_{l}_mm", GATES["height"]),
        "endplate": series(lg, "endplate_width_{l}_mm",   GATES["endplate"]),
        "canal_w":  series(lg, "canal_width_{l}_mm",      GATES["canal_w"]),
        "canal_ap": series(sm, "canal_ap_mm_{l}",         GATES["canal_ap"]),
        "pedicle":  series(sm, "pedicle_mm_{l}",          GATES["pedicle"]),
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

    fig, axes = plt.subplots(1, 3, figsize=(MF.COL2, 52 * MF.MM))
    TEAL, OCHRE, INK, FAINT = MF.TEAL, MF.OCHRE, MF.INK, MF.FAINT
    # BODY HEIGHT AND DISC HEIGHT ARE DEFERRED to the next paper: body height needs
    # the cohort-versus-method argument settled first, and neither is what the
    # textbook curve a surgeon consults is about.
    ax_b, ax_c, ax_d = axes.ravel()

    # (b) superior endplate width
    draw(ax_b, S["endplate"], y_of, TEAL, "o")
    annotate_n(ax_b, S["endplate"], y_of, FAINT)
    draw_reference_series(ax_b, refs, "endplate", y_of, drawn=drawn)
    ax_b.set_xlabel("superior endplate width (mm)")
    ax_b.set_title("(a) Endplate width", loc="left", fontsize=8.0)

    # (c) canal, width against depth
    draw(ax_c, S["canal_w"], y_of, TEAL, "o", offset=+0.17, label="width")
    draw(ax_c, S["canal_ap"], y_of, INK, "^", offset=-0.17, label="depth (AP)")
    draw_reference_series(ax_c, refs, "canal_w", y_of, offset=+0.17, drawn=drawn)
    draw_reference_series(ax_c, refs, "canal_ap", y_of, offset=-0.17, drawn=drawn)
    ax_c.set_xlabel("spinal canal (mm)")
    ax_c.set_title("(b) Canal", loc="left", fontsize=8.0)
    # upper right: the canal narrows upward, so the free space is to the right of the
    # T11-T12 rows. Lower left sits on the depth whiskers and lower right on the L5 width
    # marker, which is the widest canal in the panel.
    ax_c.legend(fontsize=6.3, handlelength=1.0, loc="upper right")

    # (d) transverse pedicle width
    draw(ax_d, S["pedicle"], y_of, OCHRE, "D")
    annotate_n(ax_d, S["pedicle"], y_of, FAINT)
    draw_reference_series(ax_d, refs, "PDW", y_of, drawn=drawn)
    ax_d.set_xlabel("transverse pedicle width (mm)")
    ax_d.set_title("(c) Pedicle width", loc="left", fontsize=8.0)


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
    # THE LEGEND IS GONE with the panel that held it: one reference, named in the caption,
    # does not need a key.
    fig.tight_layout(pad=0.5, w_pad=1.4, h_pad=1.2)
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
