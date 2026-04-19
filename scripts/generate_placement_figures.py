#!/usr/bin/env python3
"""
generate_placement_figures.py
─────────────────────────────
Reads placed_manifest.json and produces two publication-quality figures:

  fig1_bone_pct_histogram.{pdf,png}
      Overlapping histogram + KDE of bone-HU coverage for spine and pelvic
      masks, with a right-side inset showing match-type breakdown.

  fig2_bone_pct_dotplot.{pdf,png}
      Per-patient dot plot: each patient is two markers (spine ○, pelvic ▲)
      connected by a thin rule; x = patient ranked by spine bone-pct,
      y = bone-pct value.  Horizontal reference lines at clinical thresholds.

Usage
-----
    python scripts/generate_placement_figures.py \\
        --manifest data/placed/placed_manifest.json \\
        --out_dir  data/qc/publication

    python scripts/generate_placement_figures.py ... --tokens 69,75,149
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

mpl.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial",
                              "Liberation Sans", "DejaVu Sans"],
    "font.size":            8,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "axes.titleweight":     "bold",
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      7.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "0.75",
    "legend.handlelength":  1.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.75,
    "axes.grid":            False,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.0,
    "lines.markersize":     5,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.06,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

C_SPINE   = "#193C6E"
C_PELVIS  = "#501664"
C_FUSED   = "#1964B4"
C_SEP     = "#B96400"
C_SPINE_O  = "#1964B4"
C_PELV_O   = "#6A0F7A"

MATCH_COLORS = {
    "fused":        C_FUSED,
    "separate":     C_SEP,
    "spine_only":   C_SPINE_O,
    "pelvic_only":  C_PELV_O,
}
MATCH_LABELS = {
    "fused":        "Fused (same CT)",
    "separate":     "Separate (prone/supine)",
    "spine_only":   "Spine only",
    "pelvic_only":  "Pelvic only",
}

SPINE_THRESHOLD  = 40.0
PELVIC_THRESHOLD = 15.0


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())
    data["cases"] = cases
    return data


def extract_arrays(cases, token_filter=None):
    spine_bp, pelv_bp   = [], []
    spine_mt, pelv_mt   = [], []
    paired_s, paired_p, paired_mt, paired_tok = [], [], [], []

    for c in cases:
        tok = str(c.get("patient_token", ""))
        if token_filter and tok not in token_filter:
            continue
        mt = c.get("match_type", "unknown")
        sp = c.get("spine", None) or {}
        pv = c.get("pelvic", None) or {}
        sb = sp.get("bone_pct", None)
        pb = pv.get("bone_pct", None)

        if sb is not None:
            spine_bp.append(float(sb))
            spine_mt.append(mt)
        if pb is not None:
            pelv_bp.append(float(pb))
            pelv_mt.append(mt)
        if sb is not None and pb is not None:
            paired_s.append(float(sb))
            paired_p.append(float(pb))
            paired_mt.append(mt)
            paired_tok.append(tok)

    return (np.array(spine_bp), spine_mt,
            np.array(pelv_bp),  pelv_mt,
            paired_s, paired_p, paired_mt, paired_tok)


def kde_curve(values, x_grid):
    if len(values) < 4:
        return np.zeros_like(x_grid)
    try:
        k = gaussian_kde(values, bw_method="scott")
        return k(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def fig_histogram(spine_bp, spine_mt, pelv_bp, pelv_mt, out_dir):
    n_spine = len(spine_bp)
    n_pelv  = len(pelv_bp)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.0),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35},
    )

    ax = axes[0]
    bins = np.linspace(0, 100, 41)
    x_grid = np.linspace(0, 105, 500)

    for values, color, label, n in [
        (spine_bp, C_SPINE,  f"Spine  (n = {n_spine:,})",  n_spine),
        (pelv_bp,  C_PELVIS, f"Pelvic (n = {n_pelv:,})",   n_pelv),
    ]:
        if len(values) == 0:
            continue
        counts, edges = np.histogram(values, bins=bins)
        ax.bar(edges[:-1], counts, width=np.diff(edges),
               color=color, alpha=0.35, align="edge", linewidth=0)
        ky = kde_curve(values, x_grid)
        scale = len(values) * (bins[1] - bins[0])
        ax.plot(x_grid, ky * scale, color=color, lw=1.5, label=label)

    ax.axvline(SPINE_THRESHOLD,  color=C_SPINE,  lw=0.9,
               ls="--", alpha=0.7, label=f"Spine floor ({SPINE_THRESHOLD:.0f}%)")
    ax.axvline(PELVIC_THRESHOLD, color=C_PELVIS, lw=0.9,
               ls=":",  alpha=0.7, label=f"Pelvic floor ({PELVIC_THRESHOLD:.0f}%)")

    ax.set_xlabel("Bone-HU coverage $f_{\\mathrm{bone}}$ (%)")
    ax.set_ylabel("Mask count")
    ax.set_xlim(-1, 101)
    ax.set_title("Placement quality distribution", loc="left")

    leg = ax.legend(loc="upper left", frameon=True)
    leg.get_frame().set_linewidth(0.5)

    for values, color, yoff in [(spine_bp, C_SPINE, 0.93), (pelv_bp, C_PELVIS, 0.86)]:
        if len(values) == 0:
            continue
        med = np.median(values)
        ax.axvline(med, color=color, lw=0.6, ls="-", alpha=0.5)
        ax.text(med + 0.8, ax.get_ylim()[1] * yoff,
                f"med {med:.1f}%", color=color,
                fontsize=6.5, va="top", ha="left")

    ax2 = axes[1]
    all_mt = spine_mt + pelv_mt
    mt_types  = ["fused", "separate", "spine_only", "pelvic_only"]
    mt_counts = {k: all_mt.count(k) for k in mt_types}
    total     = sum(mt_counts.values()) or 1

    bar_bottom = 0.0
    for mt in mt_types:
        frac = mt_counts[mt] / total * 100
        if frac < 0.5:
            continue
        ax2.bar(0, frac, bottom=bar_bottom,
                color=MATCH_COLORS[mt], width=0.55,
                linewidth=0.4, edgecolor="white")
        if frac >= 3.0:
            ax2.text(0, bar_bottom + frac / 2,
                     f"{mt_counts[mt]}\n({frac:.0f}%)",
                     ha="center", va="center",
                     fontsize=6.5, color="white", fontweight="bold")
        bar_bottom += frac

    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(0, 100)
    ax2.set_xticks([])
    ax2.set_ylabel("Percentage of masks (%)")
    ax2.set_title("Match type", loc="left")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))

    patch_handles = [
        mpatches.Patch(color=MATCH_COLORS[mt], label=MATCH_LABELS[mt])
        for mt in mt_types if mt_counts[mt] > 0
    ]
    ax2.legend(handles=patch_handles, loc="upper right",
               bbox_to_anchor=(2.5, 1.0),
               frameon=True, fontsize=6.5)

    _save(fig, out_dir, "fig1_bone_pct_histogram")


def fig_dotplot(paired_s, paired_p, paired_mt, paired_tok,
                spine_bp, spine_mt, pelv_bp, pelv_mt, out_dir):
    order = np.argsort(paired_s)
    s_sorted  = np.array(paired_s)[order]
    p_sorted  = np.array(paired_p)[order]
    mt_sorted = [paired_mt[i] for i in order]
    n_paired  = len(s_sorted)

    fig, ax = plt.subplots(figsize=(7.1, 3.6))

    x = np.arange(n_paired)

    for xi, sv, pv, mt in zip(x, s_sorted, p_sorted, mt_sorted):
        lo, hi = sorted([sv, pv])
        ax.plot([xi, xi], [lo, hi],
                color=MATCH_COLORS.get(mt, "#888888"),
                lw=0.55, alpha=0.45, zorder=1)

    for mt in ["fused", "separate", "spine_only", "pelvic_only"]:
        mask = [i for i, m in enumerate(mt_sorted) if m == mt]
        if not mask:
            continue
        ax.scatter(x[mask], s_sorted[mask],
                   marker="o", s=18, linewidths=0.4,
                   facecolor=MATCH_COLORS[mt], edgecolor="white",
                   zorder=3, label=f"{MATCH_LABELS[mt]}")

    for mt in ["fused", "separate", "spine_only", "pelvic_only"]:
        mask = [i for i, m in enumerate(mt_sorted) if m == mt]
        if not mask:
            continue
        ax.scatter(x[mask], p_sorted[mask],
                   marker="^", s=18, linewidths=0.4,
                   facecolor=MATCH_COLORS[mt], edgecolor="white",
                   alpha=0.85, zorder=3)

    ax.axhline(SPINE_THRESHOLD,  color=C_SPINE,  lw=0.85, ls="--",
               alpha=0.65, zorder=0)
    ax.axhline(PELVIC_THRESHOLD, color=C_PELVIS, lw=0.85, ls=":",
               alpha=0.65, zorder=0)
    ax.text(n_paired * 1.005, SPINE_THRESHOLD,
            f"  spine ≥{SPINE_THRESHOLD:.0f}%",
            va="center", fontsize=6.5, color=C_SPINE)
    ax.text(n_paired * 1.005, PELVIC_THRESHOLD,
            f"  pelvic ≥{PELVIC_THRESHOLD:.0f}%",
            va="center", fontsize=6.5, color=C_PELVIS)

    ax.set_xlim(-1, n_paired + max(1, int(n_paired * 0.07)))
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Patient (ranked by spine bone-HU coverage)")
    ax.set_ylabel("Bone-HU coverage $f_{\\mathrm{bone}}$ (%)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.set_title("Per-patient placement quality", loc="left")

    anatomy_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=6, markeredgewidth=0.4, markeredgecolor="white",
               label="Spine"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#555",
               markersize=6, markeredgewidth=0.4, markeredgecolor="white",
               label="Pelvic"),
    ]
    present_mt = dict.fromkeys(mt_sorted)
    color_handles = [
        mpatches.Patch(color=MATCH_COLORS[mt], label=MATCH_LABELS[mt])
        for mt in present_mt
    ]
    first_legend = ax.legend(
        handles=anatomy_handles,
        title="Anatomy", title_fontsize=7,
        loc="lower right", frameon=True,
    )
    first_legend.get_frame().set_linewidth(0.5)
    ax.add_artist(first_legend)

    second_legend = ax.legend(
        handles=color_handles,
        title="Match type", title_fontsize=7,
        loc="upper left", frameon=True,
    )
    second_legend.get_frame().set_linewidth(0.5)

    n_s_fail = int(np.sum(s_sorted < SPINE_THRESHOLD))
    n_p_fail = int(np.sum(p_sorted < PELVIC_THRESHOLD))
    note = (f"n = {n_paired} paired patients"
            + (f" · {n_s_fail} spine below threshold" if n_s_fail else "")
            + (f" · {n_p_fail} pelvic below threshold" if n_p_fail else ""))
    ax.text(0.5, -0.13, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, color="#555555")

    _save(fig, out_dir, "fig2_bone_pct_dotplot")


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.06)
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Generate publication-quality placement figures."
    )
    ap.add_argument("--manifest", required=True,
                    help="Path to placed_manifest.json")
    ap.add_argument("--out_dir",  required=True,
                    help="Output directory for figures")
    ap.add_argument("--tokens",   default="",
                    help="Comma-separated patient tokens to include (default: all)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    token_filter = None
    if args.tokens.strip():
        token_filter = set(args.tokens.replace(",", " ").split())

    out_dir = Path(args.out_dir)

    print(f"\n generate_placement_figures.py")
    print(f"   manifest : {manifest_path}")
    print(f"   out_dir  : {out_dir}")
    if token_filter:
        print(f"   tokens   : {sorted(token_filter)}")

    data = load_manifest(manifest_path)
    cases = data["cases"]
    print(f"   cases in manifest: {len(cases)}")

    (spine_bp, spine_mt,
     pelv_bp,  pelv_mt,
     paired_s, paired_p, paired_mt, paired_tok) = extract_arrays(
        cases, token_filter)

    print(f"   spine masks  : {len(spine_bp)}")
    print(f"   pelvic masks : {len(pelv_bp)}")
    print(f"   paired       : {len(paired_s)}")

    if len(spine_bp) == 0 and len(pelv_bp) == 0:
        print("WARNING: no data found — nothing to plot.", file=sys.stderr)
        sys.exit(0)

    print("\n Generating Figure 1: histogram ...")
    fig_histogram(spine_bp, spine_mt, pelv_bp, pelv_mt, out_dir)

    print(" Generating Figure 2: dot plot ...")
    if len(paired_s) > 0:
        fig_dotplot(paired_s, paired_p, paired_mt, paired_tok,
                    spine_bp, spine_mt, pelv_bp, pelv_mt, out_dir)
    else:
        print("  (skipped — no paired patients in selection)")

    print(f"\n Done.  Figures written to: {out_dir}\n")


if __name__ == "__main__":
    main()
