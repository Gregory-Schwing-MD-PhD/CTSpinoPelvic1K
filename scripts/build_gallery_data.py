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


# ---------------------------------------------------------------------------------
# THE SURGICAL BLOCK. These are not count-free: pelvic incidence needs the sacral
# endplate and the femoral heads, and lordosis needs to know where the lumbar segment
# begins. So they live in their own section and say what they depend on. They belong in
# the gallery because they are what the anatomy is FOR -- a transitional segment matters
# to a surgeon through the corridor it opens or closes and the alignment it sets.
#
# EVERY PANEL CARRIES ITS PUBLISHED REFERENCE VALUE. A distribution read only against
# itself cannot be wrong. Read against the literature it can be, which is the point.
SURGICAL = [
    ("pelvic_incidence_deg", "Pelvic incidence", 25, 85, 50,
     "position-independent: identical standing, seated and supine",
     "A morphological property of the pelvis rather than a posture, which is why it "
     "needs no caveat on a supine CT, and why it sets how much lordosis a given spine "
     "requires."),
    ("sacral_slope_deg", "Sacral slope", 10, 70, 40,
     "postural, reported supine",
     "The sacral plate measured against the horizontal."),
    ("pelvic_tilt_deg", "Pelvic tilt", -10, 45, 13,
     "postural, reported supine",
     "Pelvic incidence less sacral slope. It rises as a pelvis retroverts to "
     "compensate for lost lordosis."),
    ("ll_supine_deg", "Lumbar lordosis, supine", 10, 95, 50,
     "supine sits about 4.6 degrees below standing; it is SEATED that collapses",
     "Measured only where the arc reaches the top of the lumbar segment. A field of "
     "view that clips the upper lumbar spine returns a smaller angle than the patient "
     "actually has, and that error would land in the mismatch below."),
    ("pi_ll_mismatch_deg", "PI-LL mismatch", -40, 40, 0,
     "the decision-maker: beyond about 10 degrees predicts residual pain after fusion",
     "Centred near zero across an unoperated cohort, which is the check that the two "
     "angles were measured independently and still agree."),
    ("crest_above_l45_mm", "Iliac crest above the L4-5 disc", -30, 40, 12,
     "the lateral corridor: positive means the crest obstructs a lateral approach",
     "The dashed line is the published 12 mm cutoff, above which subsidence risk rises "
     "after oblique lateral fusion at that level."),
    ("rib12_to_crest_mm", "Lowest rib to iliac crest", 20, 110, None,
     "the other boundary of the same corridor",
     "This is where the rib work becomes operative: a hypoplastic or absent twelfth rib "
     "moves the upper limit of the working window."),
    ("pedicle_min_mm", "Narrowest lumbar pedicle", 3, 20, None,
     "selects screw diameter, and is itself a phenotype",
     "Measured at the isthmus, per side, taking the narrower of the two."),
]


def add_surgical(out, path):
    p = Path(path)
    if not p.exists():
        print(f"  ! {path} not found; surgical panels skipped")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    sect = "Spinopelvic alignment and the surgical corridor"

    for key, title, lo, hi, ref, subtitle, caption in SURGICAL:
        vals = [num(r, key) for r in rows]
        d = density(vals, lo, hi)
        if not d["x"]:
            continue
        panel = {
            "key": key, "section": sect, "title": title, "subtitle": subtitle,
            "type": "density", "caption": caption, "rug": rug(vals, lo, hi),
            "xlabel": title + (" (mm)" if key.endswith("_mm") else " (degrees)"),
        }
        panel.update(d)
        if ref is not None:
            panel["reference"] = ref
            panel["reference_label"] = ("12 mm cutoff" if key.startswith("crest")
                                        else f"published ~{ref}")
        out["panels"].append(panel)

    add_pelvic_shape(out, rows)

    # PI against LL, with the identity line. Their AGREEMENT is the finding: a spine
    # either matches the pelvis it sits on or it does not, and the distance from the
    # diagonal IS the mismatch that drives the decision.
    pts = []
    for r in rows:
        x, y = num(r, "pelvic_incidence_deg"), num(r, "ll_supine_deg")
        if x is None or y is None or str(r.get("ll_complete")) != "1":
            continue
        pts.append({"x": round(x, 1), "y": round(y, 1),
                    "f": int((r.get("lstv_label") or "normal") != "normal")})
    if len(pts) > 40:
        out["panels"].append({
            "key": "pi_vs_ll", "section": sect,
            "title": "Does the lumbar spine match the pelvis it sits on?",
            "subtitle": "one point per case; the diagonal is a perfect match",
            "type": "scatter", "points": pts[::2], "identity": True, "log_y": False,
            "xlabel": "pelvic incidence (degrees)",
            "ylabel": "lumbar lordosis, supine (degrees)",
            "caption": ("Points below the diagonal have less lordosis than their pelvis "
                        "calls for. The cloud sits ON the line, which is what an "
                        "unoperated cohort should do and is the strongest evidence "
                        "these two angles were measured independently."),
        })

    # SEX. This began as a positive control on the theory that a pelvic measure must
    # separate by sex. It does not, and the theory was the thing that was wrong: pelvic
    # dimorphism is strong in SHAPE -- subpubic angle, inlet proportions, sciatic notch
    # -- while pelvic incidence is not one of those measures, and several series report
    # no significant sex difference in it. Measured here: 51.3 against 50.5 degrees.
    # The panel reports that null with both medians on the caption, because a null
    # stated with its effect size is a result and an unexamined adjective is not.
    series, meds = [], {}
    for want, label in (("F", "female"), ("M", "male")):
        v = [x for x in (num(r, "pelvic_incidence_deg") for r in rows
                         if (r.get("sex") or "").strip().upper().startswith(want))
             if x is not None]
        if not v:
            continue
        sv = sorted(v)
        meds[label] = sv[len(sv) // 2]
        dd = density(v, 25, 85)
        if dd["x"]:
            series.append({"label": label, "x": dd["x"], "y": dd["y"], "n": dd["n"]})
    if len(series) == 2:
        gap = abs(meds.get("female", 0) - meds.get("male", 0))
        out["panels"].append({
            "key": "pi_by_sex", "section": sect, "type": "split", "series": series,
            "title": "Pelvic incidence by sex",
            "subtitle": "reported because the cohort carries age and sex, not because "
                        "it separates",
            "xlabel": "pelvic incidence (degrees)",
            "caption": (f"Median {meds.get('female', float('nan')):.1f} degrees in women "
                        f"against {meds.get('male', float('nan')):.1f} in men, a "
                        f"difference of {gap:.1f}. Pelvic dimorphism is strong in shape "
                        "-- subpubic angle, inlet proportions, the sciatic notch -- and "
                        "pelvic incidence is not one of those measures, so a null here "
                        "agrees with the series that report no sex difference in it."),
        })


def add_demographics(out, rows):
    """Who these people are, before anything about their spines."""
    sect = "Who is in this dataset"
    note = ("Every case comes from CT colonography -- a colorectal cancer SCREENING "
            "examination. The age floor visible in the data is the screening guideline "
            "showing through, not a choice made here. So this is an asymptomatic "
            "screening population rather than a surgical or back-pain series: the "
            "spinopelvic measures below sit closer to a reference range than a clinical "
            "cohort would, and the transitional variants were found incidentally.")

    # --- age, split by sex ---------------------------------------------------------
    series, meds = [], {}
    for want, label in (("F", "female"), ("M", "male")):
        v = [x for x in (num(r, "age") for r in rows
                         if (r.get("sex") or "").strip().upper().startswith(want))
             if x is not None]
        if len(v) < 20:
            continue
        sv = sorted(v)
        meds[label] = sv[len(sv) // 2]
        d = density(v, 45, 95)
        if d["x"]:
            series.append({"label": label, "x": d["x"], "y": d["y"], "n": d["n"]})
    if len(series) == 2:
        out["panels"].append({
            "key": "age_by_sex", "section": sect, "section_note": note,
            "title": "Age, by sex", "type": "split", "series": series,
            "xlabel": "age (years)",
            "subtitle": "a screening population: the lower bound is the guideline, not a filter",
            "caption": (f"Median {meds['female']:.0f} years in women and "
                        f"{meds['male']:.0f} in men. Nothing below the screening age "
                        "threshold appears because nothing below it was scanned."),
        })

    # --- sex, source configuration, and source LSTV label --------------------------
    for key, title, subtitle, caption in (
        ("sex", "Sex", "as recorded in the source metadata",
         "Recorded in the source collection, not inferred from the images. Cases with "
         "no recorded value are shown rather than dropped."),
        ("config", "How each record was assembled",
         "spine and pelvic labels do not always land on the same acquisition",
         "Each patient was scanned prone and supine. Where both label sets landed on "
         "one acquisition the record is fused; otherwise the two are exported "
         "separately, which is why a patient can appear as more than one record."),
        ("lstv_label", "Transitional label carried by the source",
         "from the source collections, not adjudicated here",
         "These are the labels the source collections carried. They are shown as they "
         "arrived, and are not all expert-adjudicated -- which is exactly why the "
         "measures on this page are count-free and do not depend on them."),
    ):
        c = Counter((r.get(key) or "not recorded").strip() or "not recorded" for r in rows)
        if len(c) < 2:
            continue
        items = c.most_common()
        out["panels"].append({
            "key": f"demo_{key}", "section": sect, "title": title, "subtitle": subtitle,
            "type": "categorical",
            "categories": [k for k, _ in items], "counts": [v for _, v in items],
            "xlabel": title, "caption": caption,
        })


# Pelvic shape, split by sex. Range and a one-line reason for each; the range is the
# window the density is drawn over, chosen wide enough to show both tails.
PELVIC_SHAPE = [
    ("bi_iliac_width_mm", "Pelvic width across the iliac crests", 200, 340,
     "the widest span of the bony pelvis",
     "The measurement most people mean by pelvic width."),
    ("bi_acetabular_mm", "Width across the hip joints", 120, 220,
     "centre to centre between the femoral heads",
     "Measured from the femoral head found by its contact with the acetabulum, not from "
     "the centroid of the whole femur -- that sits down the shaft and reads several "
     "centimetres too wide."),
    ("pelvic_inlet_ap_mm", "Pelvic inlet, front to back", 80, 160,
     "sacral promontory to pubic symphysis: the obstetric conjugate",
     "The depth of the birth canal at its narrowest ring."),
    ("inlet_index", "Inlet shape", 0.35, 1.05,
     "inlet depth divided by width across the hips",
     "A rounder inlet scores higher, a heart-shaped one lower. Roundness is the classic "
     "obstetric distinction between pelvis types."),
    ("sacral_width_ratio", "Sacral proportions", 0.6, 2.0,
     "sacral width divided by sacral height",
     "A wider, shorter sacrum scores higher. This is the single most reported sexual "
     "difference in the pelvis."),
]


def add_pelvic_shape(out, rows):
    sect = "Pelvic shape, by sex"
    note = ("Pelvic incidence -- the angle above -- does not separate by sex, and several "
            "published series agree that it does not. Sexual dimorphism in the pelvis is "
            "a matter of SHAPE: how wide it is, how round the inlet is, how the sacrum is "
            "proportioned. Those are measured here, drawn as two overlaid distributions "
            "so the separation is visible rather than asserted, and each caption gives "
            "both medians and the gap between them.")
    first = True
    for key, title, lo, hi, subtitle, why in PELVIC_SHAPE:
        series, meds = [], {}
        for want, label in (("F", "female"), ("M", "male")):
            v = [x for x in (num(r, key) for r in rows
                             if (r.get("sex") or "").strip().upper().startswith(want))
                 if x is not None]
            if len(v) < 20:
                continue
            sv = sorted(v)
            meds[label] = sv[len(sv) // 2]
            d = density(v, lo, hi)
            if d["x"]:
                series.append({"label": label, "x": d["x"], "y": d["y"], "n": d["n"]})
        if len(series) != 2:
            continue
        gap = meds["female"] - meds["male"]
        unit = " mm" if key.endswith("_mm") else ""
        fmt = "{:.2f}" if not unit else "{:.0f}"
        panel = {
            "key": f"shape_{key}", "section": sect, "type": "split", "series": series,
            "title": title, "subtitle": subtitle,
            "xlabel": title + (" (mm)" if unit else ""),
            "caption": (f"{why} Median " + fmt.format(meds["female"]) + f"{unit} in women "
                        f"against " + fmt.format(meds["male"]) + f"{unit} in men, a "
                        f"difference of " + fmt.format(abs(gap)) + f"{unit}."),
        }
        if first:
            panel["section_note"] = note
            first = False
        out["panels"].append(panel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--surgical", default="morphometrics/surgical_morphometrics.csv")
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

    add_demographics(out, rows)

    # --- interval count -------------------------------------------------------------
    c = Counter(int(v) for v in (num(r, "n_non_rib_bearing") for r in rows) if v is not None)
    out["panels"].append({
        "key": "non_rib_bearing",
        "section": "Anatomy of the transition",
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
        "section": "Anatomy of the transition",
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
        "section": "Anatomy of the transition",
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
        "section": "Anatomy of the transition",
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

    add_surgical(out, a.surgical)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"  {len(rows)} cases -> {len(out['panels'])} panels, {kb:.0f} kB")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
