"""make_levelatlas_fig.py -- build fig_levelatlas.pdf and the CSV behind it.

Six panels, each answering a question a textbook figure answers from a few dozen cadaveric
specimens and no measure of spread:

  (a) body height, ventral and dorsal      Benzel Fig. 1.2
  (b) superior endplate width              Benzel Fig. 1.1 (width series)
  (c) canal width and depth                Benzel Fig. 1.9
  (d) transverse pedicle width             Benzel Fig. 1.11
  (e) disc height by interspace            quoted in the text as single values
  (f) trabecular attenuation by level      the in vivo stand-in for Benzel Fig. 1.4

Panel (f) needs care and its caption says so: Benzel Fig. 1.4 plots vertebral COMPRESSION
STRENGTH, measured by loading cadaveric specimens to failure. Attenuation is not that. It
is the in vivo quantity that stands in for it in living patients, it is measured here at
the published opportunistic-screening site, and it is what a CT can supply at no extra
dose -- but it is a different measurement and is not offered as the same number.

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


def build(out: Path):
    lg = MF.load("level_gradients.csv")
    sm = MF.load("surgical_morphometrics.csv")
    dg = MF.load("degenerative.csv")
    op = MF.load("opportunistic.csv")
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

    fig, axes = plt.subplots(2, 3, figsize=(MF.COL2, 108 * MF.MM))
    TEAL, OCHRE, INK, FAINT = MF.TEAL, MF.OCHRE, MF.INK, MF.FAINT
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # (a) body height, ventral against dorsal
    draw(ax_a, S["h_ant"], y_of, TEAL, "o", offset=+0.17, label="ventral")
    draw(ax_a, S["h_post"], y_of, OCHRE, "s", offset=-0.17, label="dorsal")
    ax_a.set_xlabel("vertebral body height (mm)")
    ax_a.set_title("(a) Body height", loc="left", fontsize=8.0)
    ax_a.legend(fontsize=6.3, handlelength=1.0, loc="upper left",
                bbox_to_anchor=(0.0, 0.30))

    # (b) superior endplate width
    draw(ax_b, S["endplate"], y_of, TEAL, "o")
    annotate_n(ax_b, S["endplate"], y_of, FAINT)
    ax_b.set_xlabel("superior endplate width (mm)")
    ax_b.set_title("(b) Endplate width", loc="left", fontsize=8.0)

    # (c) canal, width against depth
    draw(ax_c, S["canal_w"], y_of, TEAL, "o", offset=+0.17, label="width")
    draw(ax_c, S["canal_ap"], y_of, INK, "^", offset=-0.17, label="depth (AP)")
    ax_c.set_xlabel("spinal canal (mm)")
    ax_c.set_title("(c) Canal", loc="left", fontsize=8.0)
    # upper right: the canal narrows upward, so the free space is to the right of the
    # T11-T12 rows. Lower left sits on the depth whiskers and lower right on the L5 width
    # marker, which is the widest canal in the panel.
    ax_c.legend(fontsize=6.3, handlelength=1.0, loc="upper right")

    # (d) transverse pedicle width
    draw(ax_d, S["pedicle"], y_of, OCHRE, "D")
    annotate_n(ax_d, S["pedicle"], y_of, FAINT)
    ax_d.set_xlabel("transverse pedicle width (mm)")
    ax_d.set_title("(d) Pedicle width", loc="left", fontsize=8.0)

    # (e) disc height, drawn between the vertebrae it separates
    draw(ax_e, S["disc"], y_disc, TEAL, "o")
    annotate_n(ax_e, S["disc"], y_disc, FAINT)
    ax_e.set_xlabel("disc height (mm)")
    ax_e.set_title("(e) Disc height", loc="left", fontsize=8.0)

    # (f) trabecular attenuation
    draw(ax_f, S["hu"], y_hu, INK, "v")
    annotate_n(ax_f, S["hu"], y_hu, FAINT)
    ax_f.set_xlabel("trabecular attenuation (HU)")
    ax_f.set_title("(f) Bone density", loc="left", fontsize=8.0)

    for ax in axes.ravel():
        MF.mp_ticks(ax)
        ax.set_yticks([y_of[l] for l in LEVELS])
        ax.set_ylim(-len(LEVELS) + 0.4, 0.6)
        ax.grid(axis="x", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # Only the left column carries level labels; the discs panel names interspaces instead,
    # because labelling it with vertebral levels would put a disc on a vertebra.
    for ax in (ax_a, ax_d):
        ax.set_yticklabels(LEVELS)
        ax.set_ylabel("vertebral level")
    for ax in (ax_b, ax_c, ax_f):
        ax.set_yticklabels(LEVELS)
        ax.tick_params(labelleft=True)
    # The discs occupy the half-steps between vertebral rows, so the shared level limits
    # clip the lowest one against the bottom spine. This panel gets limits of its own.
    ax_e.set_yticks([y_disc[d] for d in DISCS])
    ax_e.set_yticklabels([DISC_LABEL[d] for d in DISCS])
    ax_e.set_ylim(min(y_disc.values()) - 0.6, max(y_disc.values()) + 0.6)

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
    a = ap.parse_args()
    S = build(Path(a.out))
    for name, st in S.items():
        print(f"{name:9s} " + ", ".join(f"{lv.upper()}:{s['n']}" for lv, s in st.items()))


if __name__ == "__main__":
    main()
