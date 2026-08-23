"""scripts/screen_castellvi_type1.py — apply Castellvi's Type I criterion directly.

WHY THIS IS NOT THE OTHER SCREEN. `screen_missed_castellvi.py` learns a direction that
separates 33 labelled cases from the cohort and ranks by it. It is a ranking, it is
validated by leave-one-out, and it cannot state a grade. This file does something much
simpler and much stronger where it applies: Castellvi Type I is DEFINED by a measurement --
a dysplastic transverse process at least 19 mm in its rostrocaudal dimension, unilateral
(Ia) or bilateral (Ib) -- so on a corpus that measures transverse-process height there is
no model to fit. There is a threshold, and the only questions are whether the measurement
is the one Castellvi meant and whether the resulting prevalence is credible.

THE MEASUREMENT HAD TO BE FIXED FIRST, and the failure is worth stating because the number
looked fine. `tp_height` was the craniocaudal extent of a 12 mm slab at the lateral tip of
the lowest lumbar vertebra, computed as max(z) - min(z) over the slab. Two extreme voxels
set that, so a single detached speckle anywhere in the slab became the height. Case 0512
measured 43.2 mm against a true process of 16.0 mm and entered the re-read queue at rank 3
on the strength of it. Applying the 19 mm threshold to the unfixed measurement called
45.8% of the cohort Type I. The extractor now takes the largest connected component --
a transverse process is one bone -- and keeps the slab extent beside it as
`tp_height_slab_*` so the discrepancy stays visible.

THE PRIOR-ART CHECK IS THE VALIDATION, and it is external to this dataset. Hanhivaara
et al. read 3855 consecutive abdominal CTs and found LSTV in 29%, of which 68% were
Castellvi Type I: about 19.7% of an unselected adult population. That is the number this
screen has to land near. It is not a tuning target -- nothing here is fitted -- it is a
falsification test, and the unfixed measurement failed it by a factor of 2.3.

WHAT THIS STILL IS NOT. Castellvi measured on a coronal reformat at the widest point of
the process; this measures the craniocaudal extent of the largest component of a tip slab
in the released segmentation. Those agree in intent and are not the same operation, and a
process whose long axis is oblique will read differently. Every case this flags is a
request for a radiologist to look, not a grade.

    python scripts/screen_castellvi_type1.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Castellvi 1984: "at least 19 mm in its widest rostrocaudal dimension".
TYPE1_MM = 19.0

# External reference, not a fit: Hanhivaara et al., 3855 consecutive abdominal CTs --
# LSTV in 29%, Castellvi I in 68% of those.
REF_PREVALENCE = 0.29 * 0.68

# A tip slab whose largest component is much shorter than the slab itself is speckled, and
# the height of a speckled tip is not a measurement of anything. Such cases are reported
# separately rather than silently included or silently dropped.
SPECKLE_RATIO = 1.25


def num(r, k):
    try:
        v = float(r[k])
        return v if np.isfinite(v) else None
    except (TypeError, ValueError, KeyError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/tp_height.csv",
                    help="measure_tp_height.py output, or transition_morphometrics.csv")
    ap.add_argument("--counts", default="morphometrics/transition_morphometrics.csv",
                    help="optional, only to report the rib-free count beside a call")
    ap.add_argument("--grades", default="morphometrics/castellvi_grades.csv")
    ap.add_argument("--out", default="docs/castellvi_type1_queue.csv")
    ap.add_argument("--threshold", type=float, default=TYPE1_MM)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    rows = [r for r in rows if not (r.get("error") or "").strip()]

    # the rib-free count is context for a reviewer, not part of the criterion: Castellvi
    # Type I is defined on the process alone and occurs perfectly happily on a normal count
    counts = {}
    cp = Path(a.counts)
    if cp.exists():
        counts = {r["case"]: r.get("n_non_rib_bearing", "")
                  for r in csv.DictReader(open(cp))}
    if "tp_height_slab_left_mm" not in rows[0]:
        print("  ! this CSV predates the largest-component fix: it has no "
              "tp_height_slab_left_mm.")
        print("  Re-run extract_transition_morphometrics.py. Applying the 19 mm criterion")
        print("  to the old column called 45.8% of the cohort Type I, against ~19.7%")
        print("  expected, and that gap IS the bug.")
        return 1

    graded = {}
    gp = Path(a.grades)
    if gp.exists():
        graded = {r["case"]: r["castellvi_type"]
                  for r in csv.DictReader(open(gp)) if r.get("castellvi_type")}

    recs = []
    for r in rows:
        hl, hr = num(r, "tp_height_left_mm"), num(r, "tp_height_right_mm")
        sl, sr = num(r, "tp_height_slab_left_mm"), num(r, "tp_height_slab_right_mm")
        if hl is None or hr is None:
            continue
        speckled = ((sl or hl) > SPECKLE_RATIO * hl) or ((sr or hr) > SPECKLE_RATIO * hr)
        n_side = int(hl >= a.threshold) + int(hr >= a.threshold)
        recs.append({
            "case": r["case"],
            "tp_height_left_mm": hl,
            "tp_height_right_mm": hr,
            "tp_height_max_mm": max(hl, hr),
            "tp_height_min_mm": min(hl, hr),
            "sides_over_threshold": n_side,
            "type1_call": {0: "", 1: "Ia", 2: "Ib"}[n_side],
            "slab_left_mm": sl, "slab_right_mm": sr,
            "tip_speckled": int(speckled),
            "known_grade": graded.get(r["case"], ""),
            "n_non_rib_bearing": counts.get(r["case"], r.get("n_non_rib_bearing", "")),
        })

    clean = [x for x in recs if not x["tip_speckled"]]
    print(f"  {len(recs)} cases measured, {len(recs) - len(clean)} with a speckled tip "
          f"(largest component < 1/{SPECKLE_RATIO:g} of the slab)\n")

    # --- 1. does the measurement reproduce the known grades? ------------------------
    print("  KNOWN GRADES, measured height per side")
    print(f"    {'grade':<6}{'n':>3}  {'median max':>11}{'median min':>11}   "
          f"{'both sides >= threshold':>24}")
    by = {}
    for x in recs:
        if x["known_grade"]:
            by.setdefault(x["known_grade"], []).append(x)
    for g in sorted(by, key=lambda t: (len(t.rstrip("ab")), t)):
        v = by[g]
        mx = np.median([x["tp_height_max_mm"] for x in v])
        mn = np.median([x["tp_height_min_mm"] for x in v])
        both = sum(1 for x in v if x["sides_over_threshold"] == 2)
        print(f"    {g:<6}{len(v):>3}  {mx:>11.1f}{mn:>11.1f}   {both:>18}/{len(v)}")

    # --- 2. prevalence against the external reference -------------------------------
    ungraded = [x for x in clean if not x["known_grade"]]
    calls = Counter(x["type1_call"] for x in ungraded if x["type1_call"])
    n_any = sum(calls.values())
    prev = n_any / len(ungraded) if ungraded else 0.0
    print(f"\n  PREVALENCE among {len(ungraded)} ungraded, unspeckled cases")
    print(f"    Ia (one side  >= {a.threshold:.0f} mm): {calls.get('Ia', 0)}")
    print(f"    Ib (both sides >= {a.threshold:.0f} mm): {calls.get('Ib', 0)}")
    print(f"    any Type I                    : {n_any}  ({100 * prev:.1f}%)")
    print(f"    external reference            : {100 * REF_PREVALENCE:.1f}% "
          f"(Hanhivaara, 3855 consecutive abdominal CT)")
    ratio = prev / REF_PREVALENCE if REF_PREVALENCE else float("nan")
    print(f"    ratio to reference            : {ratio:.2f}x")
    if ratio > 1.6 or ratio < 0.5:
        print("    ! that is not a credible prevalence. The measurement, not the "
              "threshold, is the thing to doubt.")
    else:
        print("    consistent with the published rate, which is the only external "
              "check available.")

    # --- 3. the queue ----------------------------------------------------------------
    q = sorted((x for x in ungraded if x["type1_call"]),
               key=lambda x: -x["tp_height_min_mm"] if x["type1_call"] == "Ib"
               else -x["tp_height_max_mm"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(q)
    print(f"\n  wrote {len(q)} case(s) to {a.out}, bilateral first")
    print("  Every row is a request for a radiologist to look at a transverse process, "
          "not a grade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
