"""scripts/analyze_vertebra_grouping.py — the rib-bearing / non-rib-bearing phenotype.

Stops asserting names and counts what the image shows. The headline quantity is the
number of NON-RIB-BEARING vertebrae between the lowest rib-bearing vertebra and the
sacrum: normally 5, and invariant to where you start numbering, so it survives exactly
the transitional anatomy that makes T12-vs-L1 undecidable.

Six panels:

  A  the phenotype              non-rib-bearing count between the anchors
  B  does it match a reader     automated count vs the radiologist's own column
  C  what kind of transition    count x Castellvi type
  D  rib-bearing burden         how many rib-bearing vertebrae are in the FOV
  E  who                        count by sex and age band
  F  did the review move it     v4 vs v5

B is the panel that decides whether any of the rest means anything: `Lumbosac.csv`
records "Non-rib bearing vertebra #" read by a human, so the automated count can be
scored against it rather than merely described. A method that disagrees with the reader
on transitional cases is measuring its own segmentation, not anatomy.

    python scripts/analyze_vertebra_grouping.py --v4 qc_group_v4 --v5 qc_group_v5 \
        --manifest MANIFEST.json --phenotypes _lstv_phenotypes.csv --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9a9992"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
NORMAL = 5                       # L1-L5

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def tok(case: str):
    m = re.match(r"(\d+)", case)
    return int(m.group(1)) if m else None


def load_group(d: Path):
    out = {}
    for r in csv.DictReader(open(d / "vertebra_grouping.csv")):
        t = tok(r["case"])
        r["token"] = t
        for k in ("n_between", "n_rib_bearing", "n_non_rib_bearing", "n_vert_labelled"):
            r[k] = int(r[k]) if r.get(k) not in ("", None) else None
        out[t] = r
    return out


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def bars(ax, counts, colour=BLUE, highlight=None):
    ks = sorted(counts)
    cs = [ORANGE if (highlight is not None and k != highlight) else colour for k in ks]
    ax.bar(ks, [counts[k] for k in ks], color=cs, width=0.62, zorder=3)
    tot = sum(counts.values()) or 1
    for k in ks:
        ax.text(k, counts[k], f"{counts[k]}\n{100*counts[k]/tot:.0f}%", ha="center",
                va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(counts.values()) * 1.22)


def panel_a(ax, g):
    det = [r for r in g.values() if r["n_between"] is not None]
    c = Counter(r["n_between"] for r in det)
    bars(ax, c, highlight=NORMAL)
    style(ax, f"A · Non-rib-bearing vertebrae between last rib and sacrum  "
              f"(n={len(det)})", "count", "cases")
    ax.text(.99, .95, f"blue = {NORMAL} (usual)\norange = transitional",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)


def panel_b(ax, g, pheno):
    pairs = [(int(p["non_rib_bearing_vertebrae"]), g[t]["n_between"])
             for t, p in pheno.items()
             if t in g and g[t]["n_between"] is not None
             and str(p.get("non_rib_bearing_vertebrae", "")).strip().isdigit()]
    if not pairs:
        ax.text(.5, .5, "no overlapping graded cases", ha="center", va="center",
                color=INK2, transform=ax.transAxes)
        style(ax, "B · Automated vs radiologist"); return
    human, auto = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    lo, hi = int(min(human.min(), auto.min())), int(max(human.max(), auto.max()))
    grid = np.zeros((hi - lo + 1, hi - lo + 1), int)
    for h, a in zip(human, auto):
        grid[a - lo, h - lo] += 1
    ax.imshow(np.where(grid > 0, grid, np.nan), origin="lower", cmap="Blues",
              extent=(lo - .5, hi + .5, lo - .5, hi + .5), zorder=3)
    ax.plot([lo - .5, hi + .5], [lo - .5, hi + .5], color=INK2, lw=1, ls="--", zorder=4)
    for (yy, xx), v in np.ndenumerate(grid):
        if v:
            ax.text(xx + lo, yy + lo, str(v), ha="center", va="center", fontsize=7,
                    color=INK, zorder=5)
    agree = int((human == auto).sum())
    ax.set_xticks(range(lo, hi + 1)); ax.set_yticks(range(lo, hi + 1))
    ax.grid(False)
    style(ax, f"B · Automated vs radiologist  ({agree}/{len(pairs)} exact, "
              f"{100*agree/len(pairs):.0f}%)", "radiologist's count", "automated count")


def panel_c(ax, g, pheno):
    by = defaultdict(Counter)
    for t, p in pheno.items():
        ct = (p.get("castellvi_type") or "").strip() or "none"
        if t in g and g[t]["n_between"] is not None:
            by[ct][g[t]["n_between"]] += 1
    if not by:
        style(ax, "C · Count x Castellvi type"); return
    types = sorted(by, key=lambda s: (s == "none", s))
    counts = sorted({k for c in by.values() for k in c})
    w = 0.8 / max(1, len(counts))
    x = np.arange(len(types))
    pal = [BLUE, ORANGE, AQUA, YELLOW, GREY]
    for i, k in enumerate(counts):
        ax.bar(x + i * w - 0.4 + w / 2, [by[t][k] for t in types], w,
               color=pal[i % len(pal)], label=f"{k} non-rib-bearing", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=7.5)
    style(ax, "C · Count × Castellvi type", "Castellvi", "cases")
    ax.legend(frameon=False, fontsize=6.8)


def panel_d(ax, g):
    c = Counter(r["n_rib_bearing"] for r in g.values()
                if r["n_rib_bearing"] is not None)
    bars(ax, c, colour=AQUA)
    style(ax, "D · Rib-bearing vertebrae in the FOV", "count", "cases")
    ax.text(.99, .95, "FOV-limited: a lumbosacral scan\nsees only the lowest thoracic levels",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)


def panel_e(ax, g, meta):
    by = defaultdict(Counter)
    for t, r in g.items():
        if r["n_between"] is None or t not in meta:
            continue
        s = (meta[t].get("sex") or "?").upper()[:1]
        by[s][r["n_between"]] += 1
    if not by:
        style(ax, "E · By sex"); return
    sexes = [s for s in ("F", "M") if s in by] + [s for s in by if s not in ("F", "M")]
    counts = sorted({k for c in by.values() for k in c})
    w = 0.8 / max(1, len(sexes))
    x = np.arange(len(counts))
    for i, s in enumerate(sexes):
        tot = sum(by[s].values()) or 1
        ax.bar(x + i * w - 0.4 + w / 2, [100 * by[s][k] / tot for k in counts], w,
               color=[BLUE, ORANGE, AQUA][i % 3], label=f"{s} (n={tot})", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(counts)
    style(ax, "E · Phenotype by sex", "non-rib-bearing count", "% within sex")
    ax.legend(frameon=False, fontsize=7.5)


def panel_f(ax, g4, g5):
    if not g4 or not g5:
        style(ax, "F · v4 vs v5"); return
    shared = [t for t in g5 if t in g4]
    c4 = Counter(g4[t]["n_between"] for t in shared if g4[t]["n_between"] is not None)
    c5 = Counter(g5[t]["n_between"] for t in shared if g5[t]["n_between"] is not None)
    ks = sorted(set(c4) | set(c5))
    w = 0.38
    x = np.arange(len(ks))
    ax.bar(x - w / 2, [c4[k] for k in ks], w, color=GREY, label="v4", zorder=3)
    ax.bar(x + w / 2, [c5[k] for k in ks], w, color=BLUE, label="v5 (merged)", zorder=3)
    moved = sum(1 for t in shared
                if g4[t]["n_between"] is not None and g5[t]["n_between"] is not None
                and g4[t]["n_between"] != g5[t]["n_between"])
    ax.set_xticks(x); ax.set_xticklabels(ks)
    style(ax, f"F · Did the review move the phenotype?  ({moved} cases changed)",
          "non-rib-bearing count", "cases")
    ax.legend(frameon=False, fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v5")
    ap.add_argument("--manifest")
    ap.add_argument("--phenotypes")
    ap.add_argument("--out", default="figs_grouping")
    a = ap.parse_args()

    g4 = load_group(Path(a.v4))
    g5 = load_group(Path(a.v5)) if a.v5 else {}
    g = g5 or g4

    meta = {}
    if a.manifest and Path(a.manifest).exists():
        raw = json.loads(Path(a.manifest).read_text())
        rows = raw if isinstance(raw, list) else list(raw.values())
        for r in rows:
            try:
                meta[int(r.get("token"))] = r
            except (TypeError, ValueError):
                continue

    pheno = {}
    if a.phenotypes and Path(a.phenotypes).exists():
        for r in csv.DictReader(open(a.phenotypes)):
            try:
                pheno[int(r["token"])] = r
            except (TypeError, ValueError, KeyError):
                continue

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    panel_a(axes[0, 0], g)
    panel_b(axes[0, 1], g, pheno)
    panel_c(axes[0, 2], g, pheno)
    panel_d(axes[1, 0], g)
    panel_e(axes[1, 1], g, meta)
    panel_f(axes[1, 2], g4, g5)
    det = [r for r in g.values() if r["n_between"] is not None]
    fig.suptitle(f"Rib-bearing vs non-rib-bearing phenotype · "
                 f"{len(g)} cases, {len(det)} with a determinate count · "
                 f"metadata {len(meta)} · graded {len(pheno)}",
                 x=.008, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_vertebra_grouping.{ext}", dpi=200)

    lines = [f"cases                 {len(g)}",
             f"determinate count     {len(det)}",
             ""]
    c = Counter(r["n_between"] for r in det)
    for k in sorted(c):
        tag = " (usual)" if k == NORMAL else " (transitional)"
        lines.append(f"  {k} non-rib-bearing   {c[k]:4d}  "
                     f"{100*c[k]/max(1,len(det)):5.1f}%{tag}")
    ind = Counter(r["indeterminate"] for r in g.values() if r["indeterminate"])
    if ind:
        lines += ["", "indeterminate reasons:"]
        for why, n in ind.most_common():
            lines.append(f"  {why:52s} {n}")
    (out / "grouping_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/fig_vertebra_grouping.pdf|.png, grouping_report.txt")


if __name__ == "__main__":
    main()
