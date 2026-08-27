"""hardware_figure.py — the hardware result, as a figure and a table.

Four panels, each carrying one finding that a sentence alone would not settle:

  A  WHAT A THRESHOLD FINDS, AND WHAT SURVIVES A READER. 84 flagged at 1800 HU, 52 with a
     component above the floor at 2500, 11 confirmed. The attrition is the point: an
     unreviewed metal threshold over-counts instrumentation by a factor of seven.

  B  WHY SATURATION CANNOT SORT THEM. Peak attenuation for confirmed implants against
     rejected artefact, with the scanner ceiling drawn. Both groups sit on the ceiling and
     several artefacts go far past it, which is the signature of reconstruction overshoot
     rather than of denser metal. This is the panel that stops a reader proposing the
     obvious test.

  C  WHAT DOES SORT THEM. Volume against distance from bone, log scale, with the decision
     box drawn. The confirmed cases occupy a corner nothing else reaches.

  D  WHAT WAS TAKEN FROM WHAT. Per case, the voxels the implant reclaimed from each
     structure -- because in every one of these the metal was already inside a bone label,
     so naming it was a subtraction. The femur bars are the ones that matter: pelvic
     incidence and pelvic tilt are measured from the femoral head, and on nine cases that
     head is an implant.

Writes the figure at publication resolution plus a LaTeX table and a JSON of every number
quoted, so the manuscript and the website read from one source.

    python scripts/hardware_figure.py --manifest hardware_review/hardware_manifest.csv \\
        --verdicts qc_hardware/verdicts.csv --applied hardware_review/applied.json \\
        --out paper/mpda/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CEILING = 3071.0
FLOOR_MM3 = 2000.0

INK = "#1b1d1a"
MUTED = "#6f7268"
CONFIRM = "#1f6f5c"
REJECT = "#b4552d"
RULE = "#c9c6ba"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--applied", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scanned", type=int, default=802)
    ap.add_argument("--flagged-1800", type=int, default=84)
    a = ap.parse_args()

    man = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    ver = {r["case"]: r for r in csv.DictReader(open(a.verdicts, encoding="utf-8"))}
    applied = json.loads(Path(a.applied).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    conf = [r for r in man if r["verdict"] == "instrumentation"]
    rej = [r for r in man if r["verdict"] != "instrumentation"]
    print(f"  {len(man)} proposals: {len(conf)} confirmed, {len(rej)} artefact")

    fig = plt.figure(figsize=(11.0, 8.2), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.085, right=0.975, top=0.90, bottom=0.09)

    # ---- A: attrition ------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    stages = ["scanned", "flagged\n1800 HU", "above floor\n2500 HU", "confirmed\nby a reader"]
    vals = [a.scanned, a.flagged_1800, len(man), len(conf)]
    cols = [MUTED, REJECT, REJECT, CONFIRM]
    bars = ax.bar(range(4), vals, color=cols, width=0.62)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,}",
                ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.7, a.scanned * 4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("records", fontsize=9, color=MUTED)
    ax.set_title("A   a metal threshold over-counts instrumentation sevenfold",
                 fontsize=9.5, color=INK, loc="left", pad=8)

    # ---- B: saturation does not separate -------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    cp = [float(ver[r["case"]]["peak_HU"]) for r in conf if r["case"] in ver]
    rp = [float(ver[r["case"]]["peak_HU"]) for r in rej if r["case"] in ver]
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, .06, len(rp)), rp, s=18, color=REJECT, alpha=.75,
               edgecolor="none", label=f"artefact ({len(rp)})")
    ax.scatter(rng.normal(1, .06, len(cp)), cp, s=26, color=CONFIRM, alpha=.9,
               edgecolor="none", label=f"implant ({len(cp)})")
    ax.axhline(CEILING, color=INK, lw=1, ls="--")
    ax.text(1.42, CEILING * 1.03, "scanner ceiling 3071 HU", fontsize=7.5,
            color=INK, ha="right")
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["artefact", "implant"], fontsize=8.5)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("peak attenuation (HU, log)", fontsize=9, color=MUTED)
    ax.set_title("B   both groups saturate — several artefacts exceed the ceiling",
                 fontsize=9.5, color=INK, loc="left", pad=8)

    # ---- C: what does separate them --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    style(ax)

    def vol(r):
        return max(float(r["total_mm3"]), 1.0)

    site_ok = {"hip joint", "sacroiliac joint", "spine", "femur"}
    for grp, col, lab, mk in ((rej, REJECT, "artefact", "o"),
                              (conf, CONFIRM, "implant", "s")):
        x = [vol(r) for r in grp]
        y = [1 if r["site"] in site_ok else 0 for r in grp]
        y = [v + rng.normal(0, .07) for v in y]
        ax.scatter(x, y, s=[20 if mk == "o" else 34] * len(x), color=col,
                   alpha=.8, edgecolor="none", marker=mk, label=lab)
    ax.axvline(FLOOR_MM3, color=INK, lw=1, ls="--")
    ax.text(FLOOR_MM3 * 1.15, -0.34, "2,000 mm³", fontsize=7.5, color=INK)
    ax.add_patch(Rectangle((FLOOR_MM3, 0.62), 1e6, 0.62, facecolor=CONFIRM,
                           alpha=.07, edgecolor="none"))
    ax.set_xscale("log")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no surgical site", "on a surgical site"], fontsize=8)
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("metal volume (mm³, log)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("C   site and volume do — the shaded corner holds every implant",
                 fontsize=9.5, color=INK, loc="left", pad=8)

    # ---- D: what the implant took ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    order = [r for r in applied if r.get("taken_from")]
    order.sort(key=lambda r: -sum(r["taken_from"].values()))
    names = [r["case"] for r in order]
    fem = [sum(v for k, v in r["taken_from"].items() if k.startswith("femur")) / 1000
           for r in order]
    hip = [sum(v for k, v in r["taken_from"].items() if "hip" in k) / 1000 for r in order]
    oth = [sum(v for k, v in r["taken_from"].items()
               if not k.startswith("femur") and "hip" not in k) / 1000 for r in order]
    y = np.arange(len(names))
    ax.barh(y, fem, color=CONFIRM, height=.62, label="femur")
    ax.barh(y, hip, left=fem, color="#7fa8a0", height=.62, label="hip")
    ax.barh(y, oth, left=np.array(fem) + np.array(hip), color=MUTED, height=.62,
            label="spine / sacrum")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("voxels reclaimed from bone (thousands)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("D   every implant was already labelled as bone", fontsize=9.5,
                 color=INK, loc="left", pad=8)

    fig.suptitle("Surgical instrumentation in CTSpinoPelvic1K", fontsize=12,
                 color=INK, x=0.085, ha="left", y=0.965)
    dst = out / "fig_hardware.pdf"
    fig.savefig(dst, bbox_inches="tight")
    fig.savefig(out / "fig_hardware.png", bbox_inches="tight", dpi=200)
    print(f"  wrote {dst} and .png")

    # ---- the numbers, once, for everything that quotes them ----------------------------
    from collections import Counter
    cls = Counter(r["class"] for r in conf)
    stats = {
        "scanned": a.scanned,
        "flagged_1800HU": a.flagged_1800,
        "above_floor_2500HU": len(man),
        "confirmed": len(conf),
        "artefact": len(rej),
        "artefact_rate_of_proposals": round(100.0 * len(rej) / max(len(man), 1), 1),
        "prevalence_pct": round(100.0 * len(conf) / a.scanned, 2),
        "classes": dict(cls),
        "peak_HU_above_ceiling_artefacts": sum(
            1 for r in rej if r["case"] in ver and float(ver[r["case"]]["peak_HU"]) > 3100),
        "max_artefact_peak_HU": int(max(
            (float(ver[r["case"]]["peak_HU"]) for r in rej if r["case"] in ver), default=0)),
        "smallest_confirmed_mm3": round(min(float(r["total_mm3"]) for r in conf), 1),
        "largest_rejected_mm3": round(max(float(r["total_mm3"]) for r in rej), 1),
        "voxels_reclaimed_total": sum(sum(r["taken_from"].values()) for r in applied
                                      if r.get("taken_from")),
        "cases_with_femoral_head_replaced": sum(
            1 for r in applied if any(k.startswith("femur") for k in r.get("taken_from", {}))),
    }
    (out / "hardware_stats.json").write_text(json.dumps(stats, indent=1) + "\n",
                                             encoding="utf-8")

    tex = [r"\begin{tabular}{llrrl}", r"\hline",
           r"Case & Class & Metal (mm$^3$) & Reclaimed (vox) & Reading \\", r"\hline"]
    tk = {r["case"]: sum(r["taken_from"].values()) for r in applied if r.get("taken_from")}
    for r in sorted(conf, key=lambda x: -float(x["total_mm3"])):
        note = (r.get("arthroplasty_type") or "").replace("_", " ")
        if r.get("cup_fixation"):
            note = "total, screws into ilium"
        tex.append(f"{r['case']} & {r['class'].replace('hardware_', '').replace('_', ' ')} "
                   f"& {float(r['total_mm3']):,.0f} & {tk.get(r['case'], 0):,} "
                   f"& {note} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    (out / "table_hardware.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote hardware_stats.json and table_hardware.tex")
    for k, v in stats.items():
        print(f"    {k:<36} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
