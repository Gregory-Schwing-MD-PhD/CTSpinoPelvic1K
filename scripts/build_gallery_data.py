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
import math
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


def density(vals, lo, hi, npts=220):
    """A fitted density curve rather than bars.

    WHY NOT BARS. A binned chart makes the reader see the bin edges as if they were in
    the data: the rib ratio drew two bars wide enough that the gap between the two modes
    read as an artefact of where the cuts fell. A continuous estimate has no edges to
    misread, and the shape is what these panels are for.

    BANDWIDTH. Silverman's rule with a ROBUST scale -- the smaller of the standard
    deviation and IQR/1.34. That distinction is the whole game here: on bimodal data the
    plain standard deviation is inflated by the separation between the modes, so the rule
    returns a bandwidth wide enough to smooth the two modes into one. The robust scale
    does not inflate, so the structure survives. Trimmed a further 15% because the point
    of the rib panel is that there ARE two modes, and a curve that merges them is not a
    conservative choice, it is a wrong one.

    A gaussian kernel puts mass outside [lo, hi] for values near an edge, so the curve is
    reflected back at both bounds -- these are ratios with hard floors, and letting the
    estimate leak past them would invent density where none can exist.
    """
    v = sorted(x for x in vals if x is not None and lo <= x <= hi)
    n = len(v)
    if n < 8:
        return {"x": [], "y": [], "n": n, "bandwidth": None}
    mean = sum(v) / n
    sd = (sum((x - mean) ** 2 for x in v) / max(1, n - 1)) ** 0.5
    q1 = v[int(0.25 * (n - 1))]
    q3 = v[int(0.75 * (n - 1))]
    scale = min(sd, (q3 - q1) / 1.34) if q3 > q1 else sd
    if scale <= 0:
        scale = sd or (hi - lo) / 50
    h = 0.85 * 0.9 * scale * n ** (-0.2)

    xs = [lo + (hi - lo) * i / (npts - 1) for i in range(npts)]
    ys = []
    c = 1.0 / (n * h * (2 * math.pi) ** 0.5)
    for x in xs:
        acc = 0.0
        for p in v:
            for q in (p, 2 * lo - p, 2 * hi - p):     # reflect at both bounds
                z = (x - q) / h
                if -5.0 < z < 5.0:
                    acc += math.exp(-0.5 * z * z)
        ys.append(acc * c)
    return {"x": [round(t, 4) for t in xs], "y": [round(t, 5) for t in ys],
            "n": n, "bandwidth": round(h, 4)}


def rug(vals, lo, hi, cap=260):
    """The observations themselves, thinned, drawn as ticks under the curve.

    A fitted curve is a model. Showing the data beneath it keeps the reader able to see
    where it is carrying many points and where it is carrying three.
    """
    v = [x for x in vals if x is not None and lo <= x <= hi]
    if len(v) <= cap:
        return [round(x, 4) for x in v]
    step = len(v) / cap
    return [round(v[int(i * step)], 4) for i in range(cap)]


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
        "type": "density",
        **density(vals, 0.05, 1.05),
        "rug": rug(vals, 0.05, 1.05),
        "caption": ("Two modes, near 0.68 and near 0.32 — not one distribution with a "
                    "tail. That shape is what a discrete developmental variant looks like. "
                    "The curve is a kernel density estimate; the ticks beneath it are the "
                    "cases themselves, thinned."),
    })

    # --- disc ratio -----------------------------------------------------------------
    vals = [num(r, "disc_ratio") for r in rows]
    out["panels"].append({
        "key": "disc_ratio",
        "title": "Lowest disc gap ÷ the disc above it",
        "subtitle": "dimensionless, so it compares across patients",
        "type": "density",
        **density(vals, 0.2, 4.0),
        "rug": rug(vals, 0.2, 4.0),
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
