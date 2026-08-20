"""scripts/build_gallery_data.py — the count-free distributions behind the website Gallery.

COUNT-FREE ON PURPOSE. Transitional vertebrae cannot be named from a spine-limited field of
view: sacralization and lumbarization are one morphology under two counts, and the same
holds for a lumbar rib versus a hypoplastic T12 rib. So nothing here reports a level name.
Every distribution is a measurement that survives not knowing which vertebra is which:

    non_rib_bearing   vertebrae BETWEEN the lowest rib and the sacrum -- an interval count,
                      not an absolute level assignment, valid whenever both ends are in view
    rib12_11_ratio    the lowest rib's length as a fraction of the one above it
    disc_ratio        lowest disc gap over the one above -- dimensionless, so comparable
    ll_span / tp_gap  how far the lowest lumbar reaches and how close it comes to the ala

Emits small JSON histograms rather than images: the page draws them as inline SVG, so the
figures stay legible on a phone, carry hover counts, and cost a few kB instead of a PNG.

    python scripts/build_gallery_data.py --csv morphometrics/transition_morphometrics.csv \\
        --out ../openspineconsortium.github.io/gallery/data/distributions.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def hist(vals, lo, hi, nbins):
    edges = [lo + (hi - lo) * i / nbins for i in range(nbins + 1)]
    counts = [0] * nbins
    for v in vals:
        if v is None or v < lo or v > hi:
            continue
        i = min(nbins - 1, int((v - lo) / (hi - lo) * nbins))
        counts[i] += 1
    return {"edges": [round(e, 3) for e in edges], "counts": counts,
            "n": sum(counts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    lstv = [r for r in rows if (r.get("lstv_label") or "normal") != "normal"]
    lumrib = [r for r in rows if r.get("has_lumbar_rib") == "1"]

    out = {
        "n_cases": len(rows),
        "n_lstv_labelled": len(lstv),
        "n_lumbar_rib": len(lumrib),
        "note": ("Every measure below is count-free: it does not require knowing which "
                 "vertebra is which. Level names cannot be assigned from a spine-limited "
                 "field of view, because the same bone is called a sacralised L5 or a "
                 "lumbarised S1 depending only on where the count starts."),
        "panels": [],
    }

    # --- interval count -------------------------------------------------------------
    c = Counter(int(v) for v in (num(r, "n_non_rib_bearing") for r in rows) if v is not None)
    out["panels"].append({
        "key": "non_rib_bearing",
        "title": "Non-rib-bearing vertebrae",
        "subtitle": "counted BETWEEN the lowest rib and the sacrum — an interval, not a level",
        "type": "categorical",
        "categories": [str(k) for k in sorted(c)],
        "counts": [c[k] for k in sorted(c)],
        "reference": "5",
        "caption": ("Five is typical. Four and six are where transitional anatomy sits — "
                    "and which of the two you see does not tell you what to call it."),
    })

    # --- the bimodal one ------------------------------------------------------------
    vals = [num(r, "rib12_11_ratio_min") for r in rows]
    out["panels"].append({
        "key": "rib_ratio",
        "title": "Lowest rib length, as a fraction of the rib above",
        "subtitle": "hypoplastic twelfth ribs form their own population",
        "type": "hist",
        **hist(vals, 0.05, 1.05, 25),
        "caption": ("Two modes, near 0.68 and near 0.32 — not one distribution with a "
                    "tail. That shape is what a discrete developmental variant looks like."),
    })

    # --- disc ratio -----------------------------------------------------------------
    vals = [num(r, "disc_ratio") for r in rows]
    out["panels"].append({
        "key": "disc_ratio",
        "title": "Lowest disc gap ÷ the disc above it",
        "subtitle": "dimensionless, so it compares across patients",
        "type": "hist",
        **hist(vals, 0.2, 4.0, 24),
        "caption": ("A rudimentary lowest disc runs low. Values near zero also arise from "
                    "fusion of any cause — congenital, surgical or bridging osteophyte — "
                    "which is why this measure alone cannot say why a gap closed."),
    })

    # --- Castellvi space ------------------------------------------------------------
    pts = []
    for r in rows:
        for side in ("left", "right"):
            s, g = num(r, f"ll_span_{side}_mm"), num(r, f"tp_gap_{side}_mm")
            if s is None or g is None or not (20 <= s <= 100) or not (0 < g <= 60):
                continue
            pts.append({"x": round(s, 1), "y": round(g, 2),
                        "f": int((r.get("lstv_label") or "normal") != "normal")})
    pts = pts[::2]           # thin for page weight; the shape is unchanged
    out["panels"].append({
        "key": "castellvi",
        "title": "How far the lowest lumbar reaches, and how close it comes to the ala",
        "subtitle": "one point per side — the asymmetry is the phenotype",
        "type": "scatter",
        "points": pts,
        "xlabel": "lateral span, one side (mm)",
        "ylabel": "gap to sacrum / ilium (mm)",
        "caption": ("Castellvi grades on a process that approaches, articulates with, or "
                    "fuses to the sacral ala. Span and gap carry most of that without "
                    "naming the level."),
    })

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    kb = p.stat().st_size / 1024
    print(f"  {len(rows)} cases -> {len(out['panels'])} panels, {kb:.0f} kB")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
