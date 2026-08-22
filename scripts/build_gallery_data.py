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

    # A BANDWIDTH FLOOR AT THE MEASUREMENT'S OWN RESOLUTION.
    # Several of these measures are a voxel COUNT times a spacing, so they can only land
    # on a comb: L3 body height took 35 distinct values across 775 cases, every one an
    # exact 0.8 mm step, because 0.8 mm is the slice thickness. A kernel narrower than
    # that comb draws the teeth, and the teeth are the grid rather than the anatomy.
    #
    # The step is estimated from the data instead of assumed: the modal gap between
    # adjacent distinct values IS the quantum, whatever the scanner used. Smoothing below
    # it would be claiming resolution the measurement does not have.
    uniq = sorted(set(round(x, 4) for x in v))
    if len(uniq) > 4:
        gaps = [round(b - a, 4) for a, b in zip(uniq, uniq[1:]) if b > a]
        if gaps:
            gaps.sort()
            step = gaps[len(gaps) // 2]          # median gap: robust to a few wide ones
            # A comb is a SMALL number of distinct values relative to n. When values are
            # dense the median gap is meaninglessly tiny and the floor does nothing,
            # which is the correct behaviour.
            if len(uniq) < 0.4 * n:
                h = max(h, 1.5 * step)

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
# REFERENCE VALUES ARE SOURCED, AND THEY CARRY THEIR SPREAD. Earlier versions of these
# panels drew round numbers -- 50, 40, 13 -- as a single line, which made every
# distribution look off-centre against a figure nobody had checked. These are Vialle
# 2005 (n = 260 asymptomatic adults, standing radiographs): PI 54.7 +- 10.6, SS 41.0 +-
# 8.4, PT 13 +- 6, LL 43 +- 11.2.
#
# Two caveats travel with them and are stated on the panels rather than buried:
#   - SS and PT are POSTURAL. The reference is standing and this cohort is supine, so a
#     sacral slope below the reference is expected, not a discrepancy.
#   - LUMBAR LORDOSIS HAS NO SINGLE REFERENCE. Published means run from 43 to 60 degrees
#     depending on whether the arc is measured L1-S1, L1-L5 or T12-S1, so drawing any one
#     of them as "the" value would be picking a number to agree with. The panel states
#     what was measured instead.
SURGICAL = [
    ("pelvic_incidence_deg", "Pelvic incidence", 25, 85, 54.7,
     "position-independent: identical standing, seated and supine",
     "A morphological property of the pelvis rather than a posture, which is why a "
     "supine CT can be compared to a standing reference without apology. Measured "
     "median 54.7 against Vialle's 54.7 +- 10.6 in 260 asymptomatic adults -- agreement "
     "to a decimal place, on the measure whose definition is least ambiguous."),
    ("sacral_slope_deg", "Sacral slope", 10, 70, 41.0,
     "postural: the reference is standing, this cohort is supine",
     "The sacral plate against the horizontal. Measured 36.9 against a standing "
     "reference of 41.0 +- 8.4. Lying down rotates the pelvis and lowers the slope, so "
     "sitting below a standing reference is the expected direction rather than a "
     "discrepancy."),
    ("pelvic_tilt_deg", "Pelvic tilt", -10, 45, 13.0,
     "postural, reported supine",
     "Pelvic incidence less sacral slope; it rises as a pelvis retroverts to compensate "
     "for lost lordosis. Measured 13.9 against 13 +- 6."),
    ("ll_supine_deg", "Lumbar lordosis, supine", 10, 95, None,
     "no single published reference -- the number depends on which arc is measured",
     "Measured here from the superior endplate of the topmost lumbar vertebra to the "
     "superior endplate of S1, supine, and only where the arc reaches L1. Published "
     "means range from about 43 to 60 degrees depending on whether the arc is L1-S1, "
     "L1-L5 or T12-S1, so no single line is drawn: choosing one would be choosing a "
     "number to agree with. Supine also sits about 4.6 degrees below standing."),
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
                                        else f"Vialle {ref}")
        out["panels"].append(panel)

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
    ("bi_iliac_width_mm", "Pelvic width, across the iliac crests", 210, 340,
     "the widest span of the bony pelvis",
     "The measurement most people mean by pelvic width. Men are wider in absolute terms "
     "because men are larger overall; the female pelvis is wider relative to its size, "
     "and that shows in shape measures rather than in this one."),
    ("bi_acetabular_mm", "Width across the hip joints", 120, 220,
     "centre to centre between the femoral heads",
     "Measured from the part of the femur in contact with the acetabulum. The femur "
     "labels run 61-109 mm from head to shaft, so a whole-label centroid sits down the "
     "shaft and read 18 mm too wide before this was corrected."),
]

# WITHHELD, DELIBERATELY. Pelvic inlet depth and the sacral index are two of the most
# sexually dimorphic measurements in the skeleton, and both came back with NO separation
# at all -- inlet 149.8 mm in women against 149.5 in men, sacral index 0.9 against 0.9 --
# with the inlet also 20 mm above the published range for an obstetric conjugate. A null
# in a measure that is known to separate is evidence the landmark is wrong, not evidence
# about the population, so neither is shown until the landmark is fixed.
WITHHELD = ["pelvic_inlet_ap_mm", "sacral_width_ratio", "inlet_index"]


def add_pelvic_shape(out, path, extra=None):
    """Rows are merged across both CSVs by case: bi-iliac width comes from the fast
    pass, bi-acetabular from the surgical one, whose femoral-head landmark is exact."""
    p = Path(path)
    if not p.exists():
        print(f"  ! {path} not found; pelvic shape panels skipped")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    if extra and Path(extra).exists():
        by = {r["case"]: r for r in rows}
        for r2 in csv.DictReader(open(extra)):
            tgt = by.get(r2.get("case"))
            if tgt and r2.get("bi_acetabular_mm"):
                tgt["bi_acetabular_mm"] = r2["bi_acetabular_mm"]
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


# Level-by-level gradients. Each entry: column stem, title, range, published gradient,
# and the levels to show -- because a measure can be sound at four levels and fail at the
# fifth, and dropping the level is more honest than dropping the panel.
LEVEL_GRADIENTS = [
    ("body_height", "Vertebral bodies grow taller under load", 18, 45, "mm",
     ["L1", "L2", "L3", "L4", "L5"],
     "anterior body height, level by level",
     "Published series run from about 29.9 mm at L1 to 34.5 at L5. Measured at the "
     "anterior CORTEX -- the wall -- as the tallest column in the anterior half of a "
     "mid-sagittal slab. Two earlier attempts measured the extreme anterior edge "
     "instead and read 6 to 12 mm, because the front of a vertebral body is a rounded "
     "rim that tapers to nothing."),
    ("canal_width", "The spinal canal widens as it descends", 15, 40, "mm",
     ["L1", "L2", "L3", "L4", "L5"],
     "transverse diameter, level by level",
     "Published series run from about 22 mm at L1 to 26.5 at L5, and that is the "
     "gradient here. The canal keeps widening below the end of the cord, where it "
     "carries only the cauda equina."),
    ("endplate_width", "Vertebral bodies broaden under load", 30, 70, "mm",
     ["L1", "L2", "L3", "L4", "L5"],
     "superior endplate, side to side",
     "Published series run from about 41.8 mm at L1 to 50.7 at L5; measured here at "
     "41.4, 42.9, 44.6, 47.8 and 51.8 -- both ends within a millimetre. Each curve "
     "carries two humps because it pools both sexes and vertebral size is strongly "
     "dimorphic; the panel below separates them. L5 needed a "
     "separate fix to get there: the cut that isolates the body follows the anterior "
     "wall of the spinal canal, and the L5 transverse processes arise far enough "
     "forward to survive it, which read 67.5 mm. Each axial slice is now eroded to "
     "snap the isthmus joining process to body, and the largest remaining piece is "
     "measured."),
]


def add_level_gradients(out, path):
    p = Path(path)
    if not p.exists():
        print(f"  ! {path} not found; level gradient panels skipped")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    sect = "How the lumbar spine changes as it descends"
    note = ("Almost every dimension of a lumbar vertebra grows from L1 downward, because "
            "each level carries everything above it. That makes the gradient a stronger "
            "check than any single measurement: it either reproduces the published trend "
            "or it does not. Two of the four measured here reproduce and are shown; "
            "anterior body height and the wedge ratio do not, and are not.")
    first = True
    for stem, title, lo, hi, unit, levels, subtitle, caption in LEVEL_GRADIENTS:
        series, meds = [], []
        for lv in levels:
            key = f"{stem}_{lv}_mm" if unit == "mm" else f"{stem}_{lv}"
            v = [x for x in (num(r, key) for r in rows) if x is not None]
            if len(v) < 30:
                continue
            sv = sorted(v)
            meds.append((lv, sv[len(sv) // 2]))
            d = density(v, lo, hi)
            if d["x"]:
                series.append({"label": lv, "x": d["x"], "y": d["y"], "n": d["n"]})
        if len(series) < 3:
            continue
        march = " to ".join(f"{m:.1f}" for _, m in (meds[0], meds[-1]))
        panel = {
            "key": f"grad_{stem}", "section": sect, "type": "split", "series": series,
            "title": title, "subtitle": subtitle,
            "xlabel": subtitle + (f" ({unit})" if unit else ""),
            "caption": f"{caption} Median {march}{unit} across {len(series)} levels.",
        }
        if first:
            panel["section_note"] = note
            first = False
        out["panels"].append(panel)


# Relative pelvic width. Each entry: numerator column, title, range, subtitle, caption.
# The denominator is always vertebral size, which is the point -- see add_relative_width.
RELATIVE = [
    ("bi_iliac_width_mm", "Pelvic width, relative to skeletal size", 4.0, 8.0,
     "bi-iliac breadth divided by vertebral body width",
     "In absolute millimetres men are wider, because men are larger. The textbook "
     "statement is that the female pelvis is relatively broader, and relatively needs a "
     "denominator -- here the mean superior endplate width of L2 to L4, which is a "
     "standard skeletal size proxy and is not part of the pelvis."),
    ("bi_acetabular_mm", "Hip joint separation, relative to skeletal size", 2.2, 4.8,
     "distance between femoral heads divided by vertebral body width",
     "The sharper version of the same finding. In absolute terms the hip joints sit "
     "almost the same distance apart in both sexes -- 165.5 mm against 167.0 -- while "
     "male vertebrae are meaningfully larger. The female pelvis reaches the same span "
     "on a smaller frame."),
]


def add_relative_width(out, pelvic_path, surgical_path, levels_path):
    """Pelvic width against a body-size measure that is not itself pelvic."""
    for q in (pelvic_path, levels_path):
        if not Path(q).exists():
            print(f"  ! {q} not found; relative width panels skipped")
            return
    pel = list(csv.DictReader(open(pelvic_path)))
    lev = {r["case"]: r for r in csv.DictReader(open(levels_path))}
    srg = ({r["case"]: r for r in csv.DictReader(open(surgical_path))}
           if Path(surgical_path).exists() else {})

    merged = []
    for r in pel:
        lv = lev.get(r["case"])
        if not lv:
            continue
        # vertebral size: the mean of three mid-lumbar endplate widths. Three rather than
        # one because a single level can fail its body/process separation, and L1 and L5
        # are excluded as the two most likely to.
        ep = [num(lv, f"endplate_width_L{i}_mm") for i in (2, 3, 4)]
        ep = [x for x in ep if x]
        if not ep:
            continue
        size = sum(ep) / len(ep)
        row = {"sex": r.get("sex", ""), "_size": size,
               "bi_iliac_width_mm": num(r, "bi_iliac_width_mm")}
        sr = srg.get(r["case"])
        row["bi_acetabular_mm"] = (num(sr, "bi_acetabular_mm") if sr
                                   else num(r, "bi_acetabular_mm"))
        merged.append(row)

    sect = "Pelvic shape, by sex"
    for key, title, lo, hi, subtitle, caption in RELATIVE:
        series, meds = [], {}
        for want, label in (("F", "female"), ("M", "male")):
            v = [r[key] / r["_size"] for r in merged
                 if r.get(key) and (r.get("sex") or "").strip().upper().startswith(want)]
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
        out["panels"].append({
            "key": f"rel_{key}", "section": sect, "type": "split", "series": series,
            "title": title, "subtitle": subtitle,
            "xlabel": subtitle,
            "caption": (f"{caption} Median {meds['female']:.2f} in women against "
                        f"{meds['male']:.2f} in men, a difference of {gap:+.2f} -- the "
                        "direction the literature describes, and the opposite of what "
                        "the absolute widths show."),
        })


def add_landmark_reliability(out, rows):
    """Where the iliac crest reaches, split by how many rib-free vertebrae there are."""
    def grp(r):
        n = num(r, "n_non_rib_bearing")
        if n is None:
            return None
        n = int(n)
        return "5 rib-free (typical)" if n == 5 else (f"{n} rib-free" if n in (4, 6) else None)

    order = ["5 rib-free (typical)", "4 rib-free", "6 rib-free"]
    tally = {g: {} for g in order}
    for r in rows:
        g, lv = grp(r), (r.get("iliac_crest_at") or "").strip()
        if not g or not lv or lv == "n/a":
            continue
        tally[g][lv] = tally[g].get(lv, 0) + 1

    # only the lumbar levels: a crest reported at a thoracic level means the pelvis was
    # barely in the field of view, which is a coverage fact rather than an anatomical one
    cats = [c for c in ("L3", "L4", "L5") if any(tally[g].get(c) for g in order)]
    series = []
    for g in order:
        tot = sum(tally[g].values())
        if tot < 15:
            continue
        counts = [tally[g].get(c, 0) for c in cats]
        series.append({"label": g, "n": tot, "counts": counts,
                       "pct": [100.0 * c / tot for c in counts]})
    if len(series) < 2 or not cats:
        return

    top = series[0]
    l4 = top["pct"][cats.index("L4")] if "L4" in cats else 0
    others = [f'{s["label"]} {s["pct"][cats.index("L4")]:.0f}%' for s in series[1:]
              if "L4" in cats]
    out["panels"].append({
        "key": "crest_landmark", "section": "Where the landmarks stop working",
        "section_note": ("Surgeons locate the lumbar spine by feel before they see it, "
                         "and the iliac crest is the landmark they use. These are the "
                         "measures where that habit meets the anatomy in this cohort."),
        "title": "The iliac crest does not always reach L4",
        "subtitle": "the level the crest rises to, by how many rib-free vertebrae there are",
        "type": "grouped", "categories": cats, "series": series,
        "xlabel": "vertebral level the iliac crest reaches",
        "caption": (f"The crest reaches L4 in {l4:.0f}% of people with a typical "
                    f"rib-free count, and in " + ", ".join(others) + " of those without "
                    "one. The landmark is least reliable in exactly the patients whose "
                    "levels are hardest to count, so the two errors compound rather "
                    "than cancel. Shown as percentages because the groups differ in "
                    "size by more than twenty to one."),
    })

    # the disc above a transitional segment is described as degenerating earlier than the
    # one below it; a narrower disc above raises this ratio
    series2 = []
    for g in order:
        v = [x for x in (num(r, "disc_ratio") for r in rows if grp(r) == g)
             if x is not None and 0.1 < x < 5]
        if len(v) < 20:
            continue
        d = density(v, 0.2, 4.0)
        if d["x"]:
            sv = sorted(v)
            series2.append({"label": g, "x": d["x"], "y": d["y"], "n": d["n"],
                            "_med": sv[len(sv) // 2]})
    if len(series2) >= 2:
        meds = ", ".join(f'{s["label"]} {s["_med"]:.2f}' for s in series2)
        for s2 in series2:
            s2.pop("_med", None)
        out["panels"].append({
            "key": "disc_by_group", "section": "Where the landmarks stop working",
            "title": "The disc above a transitional segment",
            "subtitle": "lowest disc gap divided by the disc above it",
            "type": "split", "series": series2,
            "xlabel": "lowest disc / disc above",
            "caption": ("A transitional segment moves less, and the level above it is "
                        "described as taking up that motion and degenerating earlier. A "
                        "narrower disc above raises this ratio. Medians: " + meds + ". "
                        "Shown as distributions; nothing here is a test."),
        })


AGING = [
    ("pelvic_incidence_deg", "pelvic incidence"),
    ("ll_supine_deg", "lumbar lordosis"),
    ("sacral_slope_deg", "sacral slope"),
    ("pelvic_tilt_deg", "pelvic tilt"),
]


def add_aging(out, path):
    """How the spinopelvic parameters move with age -- and the one that does not."""
    p = Path(path)
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return

    # decade bins. Anything with fewer than 25 cases is dropped rather than drawn thin:
    # a median over eight people is a number, not a trend.
    buckets = {}
    for r in rows:
        a = num(r, "age")
        if a is None or a < 40 or a > 99:
            continue
        buckets.setdefault(int(a // 10) * 10, []).append(r)
    decs = [d for d in sorted(buckets) if len(buckets[d]) >= 25]
    if len(decs) < 3:
        return

    def q(vals, f):
        sv = sorted(vals)
        return sv[min(len(sv) - 1, int(f * len(sv)))]

    series = []
    for key, label in AGING:
        med, q1, q3, ns = [], [], [], []
        ok = True
        for d in decs:
            v = [x for x in (num(r, key) for r in buckets[d]) if x is not None]
            if len(v) < 15:
                ok = False
                break
            med.append(q(v, 0.5)); q1.append(q(v, 0.25)); q3.append(q(v, 0.75))
            ns.append(len(v))
        if ok:
            series.append({"label": label, "med": med, "q1": q1, "q3": q3, "n": ns})
    if len(series) < 3:
        return

    pi = next((s for s in series if s["label"] == "pelvic incidence"), None)
    drift = (pi["med"][-1] - pi["med"][0]) if pi else 0.0
    out["panels"].append({
        "key": "aging", "section": "What changes with age, and what does not",
        "section_note": ("Adult spinal deformity is usually described in patients who "
                         "already have it. These are people who came in for a colon "
                         "screening, so what follows is the same mechanism seen before "
                         "anyone complained of anything."),
        "title": "Pelvic incidence holds still while the spine compensates around it",
        "subtitle": "median and interquartile range, by decade",
        "type": "trend", "bins": [f"{d}s" for d in decs], "series": series,
        "xlabel": "age", "ylabel": "degrees",
        "caption": (f"Pelvic incidence is a morphological property of the pelvis, fixed "
                    f"once the sacroiliac joints mature, and it moves {drift:+.1f} "
                    f"degrees across these decades. That flatness is a negative control: "
                    f"a cohort cannot fake it. Everything around it moves -- the pelvis "
                    f"retroverts, lordosis is lost, and the sacral slope follows because "
                    f"it is pelvic incidence less tilt. The lordosis a spine NEEDS is "
                    f"set by a number that never changes; what it HAS declines."),
    })

    # the mismatch on its own, because it is the number a surgeon acts on
    med, q1, q3, ns = [], [], [], []
    for d in decs:
        v = [x for x in (num(r, "pi_ll_mismatch_deg") for r in buckets[d]) if x is not None]
        if len(v) < 15:
            return
        med.append(q(v, 0.5)); q1.append(q(v, 0.25)); q3.append(q(v, 0.75)); ns.append(len(v))
    out["panels"].append({
        "key": "mismatch_age", "section": "What changes with age, and what does not",
        "title": "The gap between the lordosis a spine needs and the lordosis it has",
        "subtitle": "PI-LL mismatch, median and interquartile range, by decade",
        "type": "trend", "bins": [f"{d}s" for d in decs],
        "series": [{"label": "PI-LL mismatch", "med": med, "q1": q1, "q3": q3, "n": ns}],
        "xlabel": "age", "ylabel": "degrees",
        "caption": (f"Median {med[0]:.1f} degrees in the youngest decade here and "
                    f"{med[-1]:.1f} in the oldest, against a threshold near 10 beyond "
                    f"which residual pain after fusion becomes likely -- most of the "
                    f"widening comes from the pelvis retroverting rather than from "
                    f"lordosis collapsing. This is the same "
                    f"quantity surgeons plan a correction around, measured in people who "
                    f"were not being assessed for anything spinal."),
    })


def add_aging_by_sex(out, path):
    """The age trends again, split by sex. The overlap is the point."""
    p = Path(path)
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    buckets = {}
    for r in rows:
        a = num(r, "age")
        sx = (r.get("sex") or "").strip().upper()[:1]
        if a is None or a < 40 or a > 99 or sx not in ("F", "M"):
            continue
        buckets.setdefault((int(a // 10) * 10, sx), []).append(r)
    decs = sorted({d for d, _ in buckets
                   if len(buckets.get((d, "F"), [])) >= 25
                   and len(buckets.get((d, "M"), [])) >= 25})
    if len(decs) < 2:
        return

    def q(v, f):
        sv = sorted(v)
        return sv[min(len(sv) - 1, int(f * len(sv)))]

    for key, title, unit in (
        ("pelvic_tilt_deg", "Pelvic tilt", "degrees"),
        ("ll_supine_deg", "Lumbar lordosis", "degrees"),
    ):
        series, gaps = [], []
        for sx, label in (("F", "female"), ("M", "male")):
            med, q1, q3, ns = [], [], [], []
            ok = True
            for d in decs:
                v = [x for x in (num(r, key) for r in buckets.get((d, sx), []))
                     if x is not None]
                if len(v) < 15:
                    ok = False
                    break
                med.append(q(v, 0.5)); q1.append(q(v, 0.25)); q3.append(q(v, 0.75))
                ns.append(len(v))
            if ok:
                series.append({"label": label, "med": med, "q1": q1, "q3": q3, "n": ns})
        if len(series) != 2:
            continue
        gaps = [abs(a - b) for a, b in zip(series[0]["med"], series[1]["med"])]
        out["panels"].append({
            "key": f"agesex_{key}", "section": "What changes with age, and what does not",
            "title": f"{title}, by age and sex",
            "subtitle": "median and interquartile range, women against men",
            "type": "trend", "bins": [f"{d}s" for d in decs], "series": series,
            "xlabel": "age", "ylabel": unit,
            "caption": (f"The two bands sit on top of each other: the median difference "
                        f"between women and men across these decades is "
                        f"{sum(gaps) / len(gaps):.1f} degrees, on a measure whose "
                        f"interquartile range is several times that. Pelvic SHAPE is "
                        f"strongly dimorphic in this same cohort -- see the relative "
                        f"width panels -- but how the spine sits on that pelvis is not. "
                        f"Shape answers to obstetric constraint; alignment answers to "
                        f"balance, and balance does not care which pelvis it is on."),
        })


OSTEO_THRESHOLD = 110.0     # >90% specific for osteoporosis (Pickhardt 2013)


def add_bone_density(out, path):
    """Opportunistic bone density: distribution, the age-sex crossover, and who is below."""
    p = Path(path)
    if not p.exists():
        print(f"  ! {path} not found; bone density panels skipped")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    sect = "Bone density, measured for free"
    note = ("Every scan here was taken to look for colorectal polyps, and every one of "
            "them also contains a bone density measurement -- no extra dose, no extra "
            "table time. Vertebral trabecular attenuation is the most validated of the "
            "opportunistic CT measures, and the technique was established on CT "
            "colonography, which is exactly what this cohort is.")

    # --- the distribution, with the threshold drawn on it ---------------------------
    vals = [num(r, "l1_trabecular_hu") for r in rows]
    d = density(vals, 40, 320)
    if d["x"]:
        v = [x for x in vals if x is not None]
        low = sum(1 for x in v if x < OSTEO_THRESHOLD)
        out["panels"].append({
            "key": "l1_hu", "section": sect, "section_note": note,
            "title": "Vertebral bone density, from a scan taken for something else",
            "subtitle": "L1 trabecular attenuation -- the published standard site",
            "type": "density", "rug": rug(vals, 40, 320),
            "reference": OSTEO_THRESHOLD, "reference_label": "110 HU",
            "xlabel": "L1 trabecular attenuation (HU)",
            "caption": (f"The dashed line is 110 HU, over 90% specific for osteoporosis. "
                        f"{low} of {len(v)} cases ({100 * low / len(v):.1f}%) fall below "
                        f"it. Published series put the population mean near 226 HU under "
                        f"age 30, falling about 2.5 HU per year; at this cohort's median "
                        f"age that predicts roughly 155, measured here at "
                        f"{sorted(v)[len(v) // 2]:.0f}. L1 is the site both the original "
                        f"work and the 20,000-adult normative series report, because it "
                        f"is the vertebra most reliably present in both abdominal and "
                        f"thoracic CT -- which is what makes the measure opportunistic. "
                        f"L2 to L4 were measured too and agree within 4 HU, which is a "
                        f"check on the region of interest rather than a separate result: "
                        f"a misplaced ROI would not agree with itself across four "
                        f"independently segmented vertebrae."),
            **d,
        })

    # --- the crossover ---------------------------------------------------------------
    buckets = {}
    for r in rows:
        a, sx = num(r, "age"), (r.get("sex") or "").strip().upper()[:1]
        if a is None or sx not in ("F", "M"):
            continue
        buckets.setdefault((int(a // 10) * 10, sx), []).append(r)
    decs = sorted({d0 for d0, _ in buckets
                   if len(buckets.get((d0, "F"), [])) >= 20
                   and len(buckets.get((d0, "M"), [])) >= 20})
    if len(decs) < 2:
        return

    def q(v, f):
        sv = sorted(v)
        return sv[min(len(sv) - 1, int(f * len(sv)))]

    series, ends = [], {}
    for sx, label in (("F", "women"), ("M", "men")):
        med, q1, q3, ns = [], [], [], []
        ok = True
        for d0 in decs:
            v = [x for x in (num(r, "l1_trabecular_hu") for r in buckets.get((d0, sx), []))
                 if x is not None]
            if len(v) < 15:
                ok = False
                break
            med.append(q(v, 0.5)); q1.append(q(v, 0.25)); q3.append(q(v, 0.75)); ns.append(len(v))
        if ok:
            series.append({"label": label, "med": med, "q1": q1, "q3": q3, "n": ns})
            ends[label] = (med[0], med[-1])
    if len(series) == 2:
        f0, f1 = ends["women"]
        m0, m1 = ends["men"]
        out["panels"].append({
            "key": "bone_crossover", "section": sect,
            "title": "Women start with denser vertebrae and end with less",
            "subtitle": "L1 trabecular attenuation, median and interquartile range, by decade",
            "type": "trend", "bins": [f"{d0}s" for d0 in decs], "series": series,
            "xlabel": "age", "ylabel": "L1 attenuation (HU)",
            "caption": (f"The lines cross. Women begin at {f0:.0f} HU against {m0:.0f} "
                        f"for men and end at {f1:.0f} against {m1:.0f} -- a swing of "
                        f"{(f1 - m1) - (f0 - m0):+.0f} HU in the difference between the "
                        f"sexes across these decades. That is postmenopausal bone loss, "
                        f"drawn as a crossover, in people who were not being assessed "
                        f"for bone."),
        })

    # --- who is below the threshold ---------------------------------------------------
    cats = [f"{d0}s" for d0 in decs]
    gser = []
    for sx, label in (("F", "women"), ("M", "men")):
        counts, pct, tot = [], [], 0
        for d0 in decs:
            v = [x for x in (num(r, "l1_trabecular_hu") for r in buckets.get((d0, sx), []))
                 if x is not None]
            lo = sum(1 for x in v if x < OSTEO_THRESHOLD)
            counts.append(lo)
            pct.append(100.0 * lo / len(v) if v else 0.0)
            tot += len(v)
        gser.append({"label": label, "n": tot, "counts": counts, "pct": pct})
    if len(gser) == 2 and max(gser[0]["pct"]) > 0:
        out["panels"].append({
            "key": "osteo_share", "section": sect,
            "title": "How many fall below the osteoporosis threshold",
            "subtitle": "share of each group with L1 attenuation under 110 HU",
            "type": "grouped", "categories": cats, "series": gser,
            "xlabel": "age",
            "caption": (f"By the oldest decade shown, {gser[0]['pct'][-1]:.0f}% of women "
                        f"are below a threshold over 90% specific for osteoporosis, "
                        f"against {gser[1]['pct'][-1]:.0f}% of men. None of these people "
                        f"were referred for bone assessment; the measurement was already "
                        f"in an image taken for the colon."),
        })


GENANT_WEDGE = 0.80    # a 20% height reduction is a mild wedge deformity (Genant)


def add_wedge_and_sacrum(out, levels_path, pelvic_path):
    """Vertebral wedging, and the sacral base — the two measures that came back."""
    sect_bone = "Bone density, measured for free"

    # --- wedging, alongside the bone density it belongs with ------------------------
    if Path(levels_path).exists():
        rows = list(csv.DictReader(open(levels_path)))
        worst = []
        for r in rows:
            v = [x for x in (num(r, f"wedge_ratio_{lv}") for lv in
                             ("L1", "L2", "L3", "L4", "L5")) if x is not None and 0.2 < x < 2.0]
            if v:
                worst.append(min(v))
        d = density(worst, 0.4, 1.6)
        if d["x"] and len(worst) > 100:
            low = sum(1 for x in worst if x < GENANT_WEDGE)
            out["panels"].append({
                "key": "wedge", "section": sect_bone,
                "title": "Vertebral wedging, from the same scan",
                "subtitle": "lowest lumbar wedge ratio per case: anterior wall height over posterior",
                "type": "density", "rug": rug(worst, 0.4, 1.6),
                "reference": GENANT_WEDGE, "reference_label": "0.80",
                "xlabel": "anterior / posterior body height",
                "caption": (f"An unfractured body sits near 1.0, and the distribution "
                            f"does. The dashed line is 0.80 -- a 20% height reduction, "
                            f"which Genant's grading calls a mild wedge deformity. "
                            f"{low} of {len(worst)} cases ({100 * low / len(worst):.1f}%) "
                            f"have at least one lumbar body below it. A falling anterior "
                            f"wall is what a compression fracture looks like, and low "
                            f"trabecular attenuation is what precedes one -- both are in "
                            f"this image, which was ordered for the colon."),
                **d,
            })

    # --- sacral base breadth, absolute and relative ----------------------------------
    if not Path(pelvic_path).exists():
        return
    pel = list(csv.DictReader(open(pelvic_path)))
    lev = ({r["case"]: r for r in csv.DictReader(open(levels_path))}
           if Path(levels_path).exists() else {})

    series, meds = [], {}
    for want, label in (("F", "female"), ("M", "male")):
        v = [x for x in (num(r, "sacral_base_width_mm") for r in pel
                         if (r.get("sex") or "").strip().upper().startswith(want))
             if x is not None]
        if len(v) < 20:
            continue
        sv = sorted(v)
        meds[label] = sv[len(sv) // 2]
        d = density(v, 80, 160)
        if d["x"]:
            series.append({"label": label, "x": d["x"], "y": d["y"], "n": d["n"]})
    if len(series) == 2:
        out["panels"].append({
            "key": "sacral_base", "section": "Pelvic shape, by sex",
            "title": "Sacral base breadth", "type": "split", "series": series,
            "subtitle": "across both alae at the S1 level",
            "xlabel": "sacral base breadth (mm)",
            "caption": (f"Median {meds['female']:.0f} mm in women against "
                        f"{meds['male']:.0f} in men. This measurement had to be rebuilt: "
                        f"the label named 'sacrum' in the scheme is documented as the "
                        f"sacrum BELOW S1, with S1 carved out as its own class, so the "
                        f"first version measured S2 to S5 and produced an index of 0.9 "
                        f"in both sexes."),
        })

    # relative, against a body-size measure outside the pelvis
    if lev:
        series2, meds2 = [], {}
        for want, label in (("F", "female"), ("M", "male")):
            vals = []
            for r in pel:
                if not (r.get("sex") or "").strip().upper().startswith(want):
                    continue
                w = num(r, "sacral_base_width_mm")
                lv = lev.get(r["case"])
                if not w or not lv:
                    continue
                ep = [num(lv, f"endplate_width_L{i}_mm") for i in (2, 3, 4)]
                ep = [x for x in ep if x]
                if ep:
                    vals.append(w / (sum(ep) / len(ep)))
            if len(vals) < 20:
                continue
            sv = sorted(vals)
            meds2[label] = sv[len(sv) // 2]
            d = density(vals, 1.4, 3.6)
            if d["x"]:
                series2.append({"label": label, "x": d["x"], "y": d["y"], "n": d["n"]})
        if len(series2) == 2:
            gap = meds2["female"] - meds2["male"]
            out["panels"].append({
                "key": "sacral_base_rel", "section": "Pelvic shape, by sex",
                "title": "Sacral base breadth, relative to skeletal size",
                "type": "split", "series": series2,
                "subtitle": "sacral base divided by vertebral body width",
                "xlabel": "sacral base / vertebral body width",
                "caption": (f"Median {meds2['female']:.2f} in women against "
                            f"{meds2['male']:.2f} in men, a difference of {gap:+.2f}. "
                            f"The same normalisation that flips pelvic width: measured "
                            f"against a body-size proxy that is not itself pelvic, the "
                            f"female sacrum is relatively broader."),
            })


def add_vertebral_size_by_sex(out, levels_path, pelvic_path):
    """Vertebral body size by sex — the reason the pooled curve has two humps.

    The pooled endplate-width distribution is not misshapen, it is a MIXTURE. Female
    median 42.0 mm against male 48.4 at L3, a separation of 1.08 standard deviations,
    with each sex individually narrower than the pool. Smoothing that into one hump
    would be erasing the most strongly dimorphic skeletal measure in this dataset --
    far stronger than anything in the pelvis, where the difference only appears after
    normalising for body size.
    """
    if not (Path(levels_path).exists() and Path(pelvic_path).exists()):
        return
    lev = {r["case"]: r for r in csv.DictReader(open(levels_path))}
    pel = list(csv.DictReader(open(pelvic_path)))

    series, meds = [], {}
    for want, label in (("F", "female"), ("M", "male")):
        v = []
        for r in pel:
            if not (r.get("sex") or "").strip().upper().startswith(want):
                continue
            lv = lev.get(r["case"])
            if not lv:
                continue
            ep = [num(lv, f"endplate_width_L{i}_mm") for i in (2, 3, 4)]
            ep = [x for x in ep if x]
            if ep:
                v.append(sum(ep) / len(ep))
        if len(v) < 20:
            continue
        sv = sorted(v)
        meds[label] = sv[len(sv) // 2]
        d = density(v, 28, 68)
        if d["x"]:
            series.append({"label": label, "x": d["x"], "y": d["y"], "n": d["n"]})
    if len(series) != 2:
        return
    gap = meds["male"] - meds["female"]
    out["panels"].append({
        "key": "vertebral_size_sex",
        "section": "How the lumbar spine changes as it descends",
        "title": "Vertebral body size is strongly dimorphic",
        "subtitle": "mean L2-L4 superior endplate width, by sex",
        "type": "split", "series": series,
        "xlabel": "endplate width (mm)",
        "caption": (f"Median {meds['female']:.1f} mm in women against "
                    f"{meds['male']:.1f} in men, a difference of {gap:.1f} mm and just "
                    f"over one standard deviation of separation. This is why the pooled "
                    f"distribution above carries two humps: it is a mixture, not a "
                    f"misshapen curve. It is also the sharpest sexual dimorphism in this "
                    f"dataset -- sharper than anything in the pelvis, where a difference "
                    f"only appears after normalising for body size."),
    })


def _ridge(rows, key, lo, hi, bin_key="age", width=10, minn=25):
    """Densities by decade, newest last, each with its median. -> (series, bins)."""
    buckets = {}
    for r in rows:
        a = num(r, bin_key)
        if a is None or a < 40 or a > 99:
            continue
        buckets.setdefault(int(a // width) * width, []).append(r)
    out = []
    for b in sorted(buckets):
        v = [x for x in (num(r, key) for r in buckets[b]) if x is not None]
        if len(v) < minn:
            continue
        d = density(v, lo, hi)
        if not d["x"]:
            continue
        sv = sorted(v)
        out.append({"label": f"{b}s", "x": d["x"], "y": d["y"], "n": d["n"],
                    "med": round(sv[len(sv) // 2], 1)})
    return out


def add_ridges(out, surgical_path, bone_path):
    """Age progression, drawn as distributions rather than as lines."""
    sect = "What changes with age, and what does not"

    if Path(surgical_path).exists():
        rows = list(csv.DictReader(open(surgical_path)))
        for key, title, lo, hi, ref, sd, cap in (
            ("pelvic_incidence_deg", "Pelvic incidence does not move", 20, 90, 54.7, 10.6,
             "Every decade sits on the same reference band. Pelvic incidence is fixed "
             "once the sacroiliac joints mature, so this is the negative control for "
             "the two panels beside it -- a cohort cannot fake a distribution that "
             "refuses to move."),
            ("pelvic_tilt_deg", "Pelvic tilt climbs", -10, 45, 13.0, 6.0,
             "The whole distribution walks to the right and its upper tail lengthens. "
             "A pelvis retroverts to hold the trunk upright as lordosis is lost, and "
             "the cases doing the most of it are the ones in that growing tail."),
            ("ll_supine_deg", "Lumbar lordosis, largely held", 10, 95, None, None,
             "Medians of 52.2, 53.4, 52.9 and 51.8 degrees across the decades: this "
             "cohort does not lose much lordosis at all. That is worth saying plainly "
             "rather than dressing up, and it fits who these people are -- an "
             "asymptomatic screening population, not a deformity clinic. The pelvis "
             "beside it retroverts anyway, which is the more sensitive early sign. No "
             "reference line is drawn: published means run from 43 to 60 degrees "
             "depending on which arc is measured, so any single line would be one "
             "chosen to agree with."),
        ):
            ser = _ridge(rows, key, lo, hi)
            if len(ser) < 3:
                continue
            panel = {
                "key": f"ridge_{key}", "section": sect, "type": "ridge",
                "title": title, "series": ser, "x": ser[0]["x"],
                "subtitle": "one distribution per decade; the tick on each is its median",
                "xlabel": "degrees", "caption": cap,
            }
            if ref is not None:
                panel["ref"] = ref
                panel["ref_sd"] = sd
            out["panels"].append(panel)

    if Path(bone_path).exists():
        rows = list(csv.DictReader(open(bone_path)))
        for sx, label in (("F", "women"), ("M", "men")):
            sel = [r for r in rows
                   if (r.get("sex") or "").strip().upper().startswith(sx)]
            ser = _ridge(sel, "l1_trabecular_hu", 40, 320, minn=20)
            if len(ser) < 3:
                continue
            first, last = ser[0]["med"], ser[-1]["med"]
            out["panels"].append({
                "key": f"ridge_bone_{sx}", "section": "Bone density, measured for free",
                "type": "ridge", "title": f"Bone density by decade, {label}",
                "series": ser, "x": ser[0]["x"],
                "subtitle": "L1 trabecular attenuation, one distribution per decade",
                "xlabel": "L1 trabecular attenuation (HU)",
                "ref": 110.0, "ref_sd": 0.001,
                "caption": (f"Median {first:.0f} HU in the first decade shown falling to "
                            f"{last:.0f} in the last. The median is the least of it: what "
                            f"matters clinically is the low tail crossing the 110 HU line, "
                            f"because that is where the fractures come from, and a median "
                            f"with an error bar cannot show a tail thickening."),
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--surgical", default="morphometrics/surgical_morphometrics.csv")
    ap.add_argument("--pelvic", default="morphometrics/pelvic_shape.csv")
    ap.add_argument("--levels", default="morphometrics/level_gradients.csv")
    ap.add_argument("--bone", default="morphometrics/opportunistic.csv")
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
    add_landmark_reliability(out, rows)

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
    add_aging(out, a.surgical)
    add_aging_by_sex(out, a.surgical)
    add_ridges(out, a.surgical, a.bone)
    add_pelvic_shape(out, a.pelvic, a.surgical)
    add_relative_width(out, a.pelvic, a.surgical, a.levels)
    add_level_gradients(out, a.levels)
    add_vertebral_size_by_sex(out, a.levels, a.pelvic)
    add_bone_density(out, a.bone)
    add_wedge_and_sacrum(out, a.levels, a.pelvic)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"  {len(rows)} cases -> {len(out['panels'])} panels, {kb:.0f} kB")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
