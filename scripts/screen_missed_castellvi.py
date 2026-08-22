"""scripts/screen_missed_castellvi.py — find transitional cases the labels may have missed.

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

  tp_gap_min          how close the transverse process comes to the sacral ala. Castellvi
                      grades on approach, articulation and fusion; this is that distance.
  ll_span_total       how far the lowest lumbar vertebra reaches transversely. Type I is
                      defined by a process at least 19 mm wide.
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

FEATURES = ["tp_gap_min", "ll_span_total_mm", "ll_span_asym_mm",
            "disc_ratio", "n_non_rib_bearing"]


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
            "tp_gap_min": min(gaps) if gaps else None,
            "ll_span_total_mm": num(r, "ll_span_total_mm"),
            "ll_span_asym_mm": abs(num(r, "ll_span_asym_mm") or 0.0),
            "disc_ratio": num(r, "disc_ratio"),
            "n_non_rib_bearing": num(r, "n_non_rib_bearing"),
        }
        if any(v is None for v in feats.values()):
            continue
        X.append([feats[k] for k in FEATURES])
        y.append(1 if (r.get("lstv_label") or "normal").strip().lower() != "normal" else 0)
        cases.append(r["case"])
    return np.asarray(X, float), np.asarray(y, int), cases


def fit_score(X, y, train_mask=None):
    """Mahalanobis-style score toward the positives, on standardised features.

    A logistic fit on 33 positives against 700 unlabelled would be dominated by the
    class ratio and would need regularisation choices that are themselves free
    parameters. A distance to the positive centroid under the POOLED covariance has no
    free parameters at all, which at this sample size is the honest choice: it says
    "resembles the labelled cases" and nothing more.
    """
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    m = np.ones(len(X), bool) if train_mask is None else train_mask
    P = Z[m & (y == 1)]
    if len(P) < 5:
        return np.zeros(len(X))
    cov = np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])
    inv = np.linalg.pinv(cov)
    c = P.mean(0)
    d = Z - c
    md = np.einsum("ij,jk,ik->i", d, inv, d)
    return -np.sqrt(np.maximum(md, 0))          # higher = more like the positives


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="docs/castellvi_reread_queue.csv")
    a = ap.parse_args()

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
    print(f"    {'case':6} {'score':>7} {'tp_gap':>7} {'span':>7} {'asym':>6} "
          f"{'disc':>6} {'ribfree':>8}")
    out_rows = []
    for i in order:
        c = cases[i]
        f = X[i]
        print(f"    {c:6} {sc[i]:7.2f} {f[0]:7.1f} {f[1]:7.1f} {f[2]:6.1f} "
              f"{f[3]:6.2f} {f[4]:8.0f}")
        out_rows.append({
            "case": c, "score": round(float(sc[i]), 3),
            "tp_gap_min_mm": round(float(f[0]), 1),
            "ll_span_total_mm": round(float(f[1]), 1),
            "ll_span_asym_mm": round(float(f[2]), 1),
            "disc_ratio": round(float(f[3]), 3),
            "n_non_rib_bearing": int(f[4]),
            "source_label": (by_case[c].get("lstv_label") or "normal"),
            "reviewed": "", "castellvi_grade": "", "reviewer": "",
        })

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    print(f"\n  wrote {p}")
    print(f"\n  HOW TO READ THIS. Leave-one-out puts {100 * hit25:.0f}% of known positives")
    print("  in the top 25, so the queue is enriched but not decisive. A high score means")
    print("  the case RESEMBLES the labelled transitional cases -- it is a re-read")
    print("  request, not a diagnosis, and a radiologist decides. Cases that come back")
    print("  negative are as informative as the ones that do not: they are the ones that")
    print("  tell you which features are doing the work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
