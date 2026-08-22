"""scripts/test_morphometric_separability.py — is the thoracolumbar boundary learnable?

WHY RUN THIS BEFORE TRAINING ANYTHING. The single-pass architecture argument rests on a
premise: that thoracic and lumbar vertebrae are morphologically distinct, so a network with
adequate receptive field can tell a hypoplastic twelfth rib from a lumbar rib by looking at
the vertebra rather than by counting. That premise is an empirical claim and the data to
test it already exists — every vertebra in this corpus carries per-level morphometrics AND
a ground-truth identity.

If hand-built features on a linear model separate the two cleanly, a CNN will do better and
the architecture argument is safe. If they do not, the premise is shaky and that is worth
knowing before a week of H200 time, not after.

THE CONFOUND THAT WOULD MAKE THIS MEANINGLESS. Lumbar vertebrae are simply BIGGER than
thoracic ones. A classifier given raw millimetres would separate them at once and would
have learned nothing about morphology — it would have learned about size, which a network
also gets for free from the coordinate of the patch and which tells you nothing at the
boundary where T12 and L1 are nearly the same size.

So every feature is normalised BY THE CASE'S OWN MEDIAN across its measured levels. That
removes patient size entirely and leaves shape. The question then becomes the one that
matters: is a T12 a different SHAPE from an L1, in the same patient?

Reported with grouped cross-validation by case, because two vertebrae from one patient are
not independent observations and a random split would leak.

    python scripts/test_morphometric_separability.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

THORACIC = ("T11", "T12")
LUMBAR = ("L1", "L2", "L3", "L4", "L5")

# per-level measures available for every level
BASE = ["endplate_width_{}_mm", "body_height_{}_mm", "body_height_post_{}_mm",
        "canal_width_{}_mm", "tp_span_{}_mm"]


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def build(path):
    rows = list(csv.DictReader(open(path)))
    levels = list(THORACIC) + list(LUMBAR)
    X, y, groups, names = [], [], [], None
    for r in rows:
        # the case's own median for each measure -- the size normaliser
        med = {}
        for b in BASE:
            v = [num(r, b.format(l)) for l in levels]
            v = [x for x in v if x is not None and x > 0]
            if len(v) >= 3:
                med[b] = float(np.median(v))
        if len(med) < len(BASE):
            continue
        for l in levels:
            vals = {b: num(r, b.format(l)) for b in BASE}
            if any(v is None or v <= 0 for v in vals.values()):
                continue
            # SHAPE ONLY: each measure as a fraction of this patient's own median, plus
            # ratios between measures, which are size-free by construction
            f, nm = [], []
            for b in BASE:
                f.append(vals[b] / med[b])
                nm.append(b.format("").replace("__", "_").strip("_") + "/case_median")
            ew, bh, bhp = (vals["endplate_width_{}_mm"], vals["body_height_{}_mm"],
                           vals["body_height_post_{}_mm"])
            cw, tp = vals["canal_width_{}_mm"], vals["tp_span_{}_mm"]
            for val, n2 in ((tp / ew, "tp_span / endplate_width"),
                            (cw / ew, "canal_width / endplate_width"),
                            (bh / ew, "body_height / endplate_width"),
                            (bh / bhp, "anterior / posterior height"),
                            (tp / bh, "tp_span / body_height"),
                            (cw / bh, "canal_width / body_height")):
                f.append(val)
                nm.append(n2)
            X.append(f)
            y.append(1 if l in THORACIC else 0)
            groups.append(r["case"])
            names = nm
    return np.asarray(X, float), np.asarray(y, int), np.asarray(groups), names


def fit_logreg(Xtr, ytr, iters=400, lr=0.5, l2=1e-3):
    """Plain logistic regression by gradient descent. No sklearn dependency, and the
    weights are the point -- a black box would answer the question less usefully."""
    w = np.zeros(Xtr.shape[1] + 1)
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A @ w))
        g = A.T @ (p - ytr) / len(ytr) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return w


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="morphometrics/level_gradients.csv")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    X, y, g, names = build(a.levels)
    if not len(X):
        print("  no complete rows")
        return 1
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd

    print(f"  {len(X)} vertebrae from {len(set(g))} cases: "
          f"{int(y.sum())} thoracic (T11-T12), {int((1 - y).sum())} lumbar (L1-L5)")
    print("  every feature is normalised by the CASE'S OWN median, so patient size is "
          "removed\n")

    # grouped k-fold: a case is never split across train and test
    cases = sorted(set(g))
    rng = np.random.default_rng(0)
    rng.shuffle(cases)
    fold_of = {c: i % a.folds for i, c in enumerate(cases)}
    fold = np.array([fold_of[c] for c in g])

    scores, truth = np.zeros(len(y)), y
    for k in range(a.folds):
        tr, te = fold != k, fold == k
        w = fit_logreg(Z[tr], y[tr])
        scores[te] = np.hstack([Z[te], np.ones((te.sum(), 1))]) @ w

    A = auc(truth, scores)
    pred = (scores > 0).astype(int)
    acc = float((pred == truth).mean())
    sens = float(pred[truth == 1].mean())
    spec = float(1 - pred[truth == 0].mean())
    print(f"  grouped {a.folds}-fold, thoracic vs lumbar from SHAPE alone")
    print(f"    AUC          {A:.3f}")
    print(f"    accuracy     {acc:.3f}")
    print(f"    sensitivity  {sens:.3f}   (thoracic correctly called thoracic)")
    print(f"    specificity  {spec:.3f}\n")

    # the boundary is the only part that matters: T12 against L1
    print("  and the pair that actually decides the phenotype, T12 versus L1:")
    idx = build_pair(a.levels, "T12", "L1")
    if idx is not None:
        Xp, yp, gp = idx
        mp, sp_ = Xp.mean(0), Xp.std(0)
        sp_[sp_ < 1e-9] = 1.0
        Zp = (Xp - mp) / sp_
        cs = sorted(set(gp))
        rng2 = np.random.default_rng(1)
        rng2.shuffle(cs)
        fo = {c: i % a.folds for i, c in enumerate(cs)}
        fp = np.array([fo[c] for c in gp])
        sc = np.zeros(len(yp))
        for k in range(a.folds):
            tr, te = fp != k, fp == k
            w = fit_logreg(Zp[tr], yp[tr])
            sc[te] = np.hstack([Zp[te], np.ones((te.sum(), 1))]) @ w
        pa = (sc > 0).astype(int)
        print(f"    n={len(yp)}  AUC {auc(yp, sc):.3f}   accuracy {(pa == yp).mean():.3f}")

    w = fit_logreg(Z, y)
    print("\n  what the model uses (standardised weights, largest first):")
    for n2, wt in sorted(zip(names, w[:-1]), key=lambda t: -abs(t[1]))[:8]:
        print(f"    {n2:34s} {wt:+.3f}")

    print("\n  HOW TO READ THIS. A high AUC on SHAPE ALONE means the thoracolumbar")
    print("  boundary carries morphological information a network can use, which is the")
    print("  premise the single-pass architecture rests on. A low one means the premise")
    print("  is weak and counting cannot be avoided. Either way it is settled here for")
    print("  the cost of a minute, not a week of GPU time.")
    return 0


def build_pair(path, la, lb):
    rows = list(csv.DictReader(open(path)))
    levels = list(THORACIC) + list(LUMBAR)
    X, y, g = [], [], []
    for r in rows:
        med = {}
        for b in BASE:
            v = [num(r, b.format(l)) for l in levels]
            v = [x for x in v if x is not None and x > 0]
            if len(v) >= 3:
                med[b] = float(np.median(v))
        if len(med) < len(BASE):
            continue
        for l, lab in ((la, 1), (lb, 0)):
            vals = {b: num(r, b.format(l)) for b in BASE}
            if any(v is None or v <= 0 for v in vals.values()):
                continue
            f = [vals[b] / med[b] for b in BASE]
            ew, bh = vals["endplate_width_{}_mm"], vals["body_height_{}_mm"]
            bhp, cw, tp = (vals["body_height_post_{}_mm"], vals["canal_width_{}_mm"],
                           vals["tp_span_{}_mm"])
            f += [tp / ew, cw / ew, bh / ew, bh / bhp, tp / bh, cw / bh]
            X.append(f)
            y.append(lab)
            g.append(r["case"])
    if not X:
        return None
    return np.asarray(X, float), np.asarray(y, int), np.asarray(g)


if __name__ == "__main__":
    sys.exit(main())
