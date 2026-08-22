"""scripts/screen_missed_castellvi.py — find transitional cases the labels may have missed.

READ THIS BEFORE READING THE OUTPUT: THIS IS NOT VALIDATED AS A CASTELLVI SCREEN, AND IT
CANNOT BE, BECAUSE THE DATASET CONTAINS NO CASTELLVI GRADES. The castellvi_type column is
empty in all 802 records. What the 33 labelled cases carry is an LSTV label --
SACRALIZATION, LUMBARIZATION, SEMI_SACRALIZATION -- and those describe a vertebral COUNT
and its fusion, where Castellvi describes the MORPHOLOGY of a transverse process. The two
are different axes and are not interchangeable; the clinical paper's limitations section
says so and this script is bound by the same statement.

Every recovery figure printed below is therefore recovery of LSTV-LABELLED cases, not of
Castellvi grades, and the name of this file is aspirational. Grading the 33 labelled cases
(task A1 in docs/COLLABORATOR_TASKS.md) is the blocking prerequisite for the screen this
is supposed to be.


THIS IS A POSITIVE-UNLABELLED PROBLEM, not a classification problem, and the distinction
decides the whole method. 33 cases carry a transitional label. The other 769 are NOT
known negatives -- they are UNLABELLED. A transitional vertebra is easy to overlook when
the scan was read for colon polyps, so some unknown number of them are positives nobody
recorded. Training a classifier that treats unlabelled as negative would teach it to
reproduce the very omissions we are trying to find.

The standard treatment (Elkan and Noto 2008; Bekker and Davis 2020) is to fit
positive-versus-unlabelled and rank, rather than to classify. A high score on an
unlabelled case does not mean "positive" -- it means "resembles the labelled positives
more than the rest of the cohort does", which is precisely a re-read queue.

WHY THE FEATURES ARE CHOSEN BY MECHANISM AND NOT BY SEARCH. With 33 positives, a model
free to pick from dozens of features will find something that separates them and it will
be noise. So the features are fixed in advance to the ones Castellvi's classification is
literally about:

  tp_height_max       the craniocaudal height of the lowest lumbar transverse process.
                      THIS IS CASTELLVI'S TYPE I CRITERION LITERALLY -- a process of at
                      least 19 mm -- and the first version of this screen did not measure
                      it at all. Median 18.4 mm in unlabelled cases against 25.6 in the
                      labelled ones.
  tp_height_asym      left-right difference in that height. Types a and b are unilateral
                      versus bilateral, so the asymmetry IS part of the definition.
  tp_gap_min          how close the transverse process TIP comes to the sacral ala.
                      Castellvi grades on approach, articulation and fusion; this is that
                      distance. It was measured from the whole lateral third of the
                      vertebra, which contains the inferior articular process -- so it
                      returned the width of the L5-S1 FACET JOINT, about 3 mm, in
                      everybody, and the first version of this screen recovered 0% of
                      held-out positives on the strength of it.
  ll_span_total       how far the lowest lumbar vertebra reaches transversely.
  ll_span_asym        left-right asymmetry of that reach. Types a and b are unilateral
                      versus bilateral, so asymmetry IS part of the definition.
  disc_ratio          lowest disc height relative to the one above; a transitional
                      segment carries a rudimentary disc.
  n_non_rib_bearing   four or six rib-free vertebrae instead of five.

Every one of these is in the classification or its immediate mechanics. None was picked
because it happened to separate the 33.

VALIDATION IS LEAVE-ONE-OUT ON THE KNOWN POSITIVES. If the ranking cannot recover a
held-out positive it has no business proposing new ones, and the recovery rate is the
only honest statement of how much to trust the queue.

    python scripts/screen_missed_castellvi.py --csv morphometrics/transition_morphometrics.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# CASTELLVI IS A MORPHOLOGY CLASSIFICATION AND THE RIB-FREE COUNT IS NOT PART OF IT.
# A Castellvi I or II sits perfectly happily on a normal five-vertebra count -- that is
# the ordinary case, not the exception -- so a screen that uses the count is partly
# rediscovering something already recorded, and cannot by construction find the cases
# nobody would have flagged. The count is therefore excluded by default and available
# with --use-count for comparison. On this cohort it carries the single largest
# discriminant weight (+1.10, against +1.03 for transverse-process height), which is
# exactly why leaving it in would flatter the result.
MORPHOLOGY = ["tp_height_max_mm", "tp_height_asym_mm", "tp_gap_min",
              "ll_span_total_mm", "ll_span_asym_mm", "disc_ratio"]
FEATURES = list(MORPHOLOGY)


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def build(rows):
    """-> (X, y, cases). y = 1 for a labelled transitional case, 0 for UNLABELLED."""
    X, y, cases = [], [], []
    for r in rows:
        gaps = [num(r, "tp_gap_left_mm"), num(r, "tp_gap_right_mm")]
        gaps = [g for g in gaps if g is not None and 0 < g < 80]
        feats = {
            "tp_height_max_mm": num(r, "tp_height_max_mm"),
            "tp_height_asym_mm": abs(num(r, "tp_height_asym_mm") or 0.0),
            "tp_gap_min": min(gaps) if gaps else None,
            "ll_span_total_mm": num(r, "ll_span_total_mm"),
            "ll_span_asym_mm": abs(num(r, "ll_span_asym_mm") or 0.0),
            "disc_ratio": num(r, "disc_ratio"),
            "n_non_rib_bearing": num(r, "n_non_rib_bearing"),
        }
        if any(feats.get(k) is None for k in FEATURES):
            continue
        X.append([feats[k] for k in FEATURES])
        y.append(1 if (r.get("lstv_label") or "normal").strip().lower() != "normal" else 0)
        cases.append(r["case"])
    return np.asarray(X, float), np.asarray(y, int), cases


def fit_score(X, y, train_mask=None):
    """Rank by the direction along which positives differ from the rest (Fisher's).

    WHY NOT DISTANCE TO THE POSITIVE CENTROID, WHICH THIS USED TO DO. That score put the
    known positives at median rank 708 of 767 -- not merely uninformative but reliably
    ANTI-correlated with being a positive, which is a stronger signal that the method is
    wrong than a null result would be. The reason is structural and worth recording,
    because the mistake is an easy one: 33 positives in 7 dimensions are dispersed, so
    their centroid sits in the middle of a cloud none of them is especially near, while
    the 766 unlabelled cases form a dense mass right on top of it. Thousands of ordinary
    cases are then closer to the positive centroid than any individual positive is.
    Distance to a centroid measures typicality, and a rare phenotype is by construction
    not typical.

    What the question actually asks is "which direction separates the labelled cases from
    the cohort", and Fisher's linear discriminant answers exactly that with no free
    parameters: the pooled covariance, one mean difference, one projection. It does not
    care whether the positives are dispersed along that direction -- being far out on it
    is the point.

    The score is a projection, so it is ordinal only. It is not a probability and must
    not be read as one.
    """
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    m = np.ones(len(X), bool) if train_mask is None else train_mask
    P = Z[m & (y == 1)]
    U = Z[m & (y == 0)]
    if len(P) < 5 or len(U) < 5:
        return np.zeros(len(X))
    cov = np.cov(Z[m].T) + 1e-3 * np.eye(Z.shape[1])
    w = np.linalg.pinv(cov) @ (P.mean(0) - U.mean(0))
    return Z @ w                                 # higher = more like the positives


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--use-count", action="store_true",
                    help="include n_non_rib_bearing as a feature. Off by default: see "
                         "the note on MORPHOLOGY above.")
    ap.add_argument("--out", default="docs/castellvi_reread_queue.csv")
    a = ap.parse_args()

    if a.use_count:
        FEATURES.append("n_non_rib_bearing")
        print("  including rib-free count as a feature (--use-count)")

    rows = list(csv.DictReader(open(a.csv)))
    X, y, cases = build(rows)
    npos = int(y.sum())
    print(f"  {len(X)} cases with complete features, {npos} labelled transitional\n")
    if npos < 10:
        print("  too few labelled positives to rank against")
        return 1

    # --- leave-one-out recovery: can the ranking find a positive it was not shown? ---
    ranks = []
    for i in np.flatnonzero(y == 1):
        m = np.ones(len(X), bool)
        m[i] = False
        sc = fit_score(X, y, train_mask=m)
        unl = np.flatnonzero(y == 0)
        pool = np.concatenate([unl, [i]])
        order = pool[np.argsort(-sc[pool])]
        ranks.append(int(np.flatnonzero(order == i)[0]) + 1)
    ranks = np.asarray(ranks)
    npool = int((y == 0).sum()) + 1
    print("  leave-one-out recovery of the KNOWN positives")
    print(f"    median rank {np.median(ranks):.0f} of {npool}")
    for k in (10, 25, 50, 100):
        print(f"    in the top {k:3d}: {int((ranks <= k).sum())}/{len(ranks)} "
              f"({100 * (ranks <= k).mean():.0f}%)")
    hit25 = (ranks <= 25).mean()

    # --- rank the unlabelled -----------------------------------------------------------
    sc = fit_score(X, y)
    unl = np.flatnonzero(y == 0)
    order = unl[np.argsort(-sc[unl])][: a.top]
    by_case = {r["case"]: r for r in rows}

    print(f"\n  top {a.top} unlabelled cases by resemblance to the labelled positives:")
    print(f"    {'case':6} {'score':>7} {'tp_ht':>7} {'tp_gap':>7} {'span':>6} "
          f"{'disc':>6} {'ribfree':>8}")
    out_rows = []
    for i in order:
        c = cases[i]
        f = X[i]
        g = dict(zip(FEATURES, f))
        print(f"    {c:6} {sc[i]:7.2f} {g['tp_height_max_mm']:7.1f} "
              f"{g['tp_gap_min']:7.1f} {g['ll_span_total_mm']:6.1f} "
              f"{g['disc_ratio']:6.2f} {int(num(by_case[c], 'n_non_rib_bearing') or 0):8d}")
        out_rows.append({
            "case": c, "score": round(float(sc[i]), 3),
            "tp_height_max_mm": round(float(g["tp_height_max_mm"]), 1),
            "tp_height_asym_mm": round(float(g["tp_height_asym_mm"]), 1),
            "tp_gap_min_mm": round(float(g["tp_gap_min"]), 1),
            "ll_span_total_mm": round(float(g["ll_span_total_mm"]), 1),
            "ll_span_asym_mm": round(float(g["ll_span_asym_mm"]), 1),
            "disc_ratio": round(float(g["disc_ratio"]), 3),
            "n_non_rib_bearing": int(num(by_case[c], "n_non_rib_bearing") or 0),
            "source_label": (by_case[c].get("lstv_label") or "normal"),
            "reviewed": "", "castellvi_grade": "", "reviewer": "",
        })

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    print(f"\n  wrote {p}")
    print()
    print("  HOW TO READ THIS.")
    print(f"  Leave-one-out puts {100 * hit25:.0f}% of the LSTV-LABELLED cases in the "
          f"top {a.top}, against {100 * a.top / npool:.1f}% expected by chance.")
    print("  Enriched, and nowhere near decisive. A high score means the case RESEMBLES")
    print("  the labelled ones -- a re-read request, not a diagnosis, and a radiologist")
    print("  decides. Cases that come back negative are as informative as the rest.")
    print()
    if "n_non_rib_bearing" in FEATURES:
        print("  YOU RAN THIS WITH --use-count, so much of that recovery is the rib-free")
        print("  count re-finding cases that were labelled BECAUSE of their count. It is")
        print("  the flattering number, not the useful one.")
    else:
        print("  The rib-free count is excluded, so this queue can hold cases with a")
        print("  perfectly normal count and transitional morphology -- the only kind")
        print("  nobody would already have flagged, and the only kind worth a")
        print("  radiologist's time. It is also why the recovery rate is the lower one.")
    print()
    print("  AND IT IS NOT A CASTELLVI SCREEN until someone grades the 33 labelled")
    print("  cases. See the header, and task A1 in docs/COLLABORATOR_TASKS.md.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
