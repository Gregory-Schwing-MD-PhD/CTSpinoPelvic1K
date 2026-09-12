"""scripts/compare_v10_v11_spinopelvic.py -- did dissolving S1 fix the angles, or move them?

THE ACCEPTANCE TEST. On the 580 records where the old carve was anatomically sound, the old
estimator already reproduced the published supine-CT values (PI 52.7, PT 19.6 against
Hasegawa's 53.4 and 19.2). The method was never the problem. So the whole-sacrum replacement
has to clear two bars, and only the pair of them is evidence:

  AGREEMENT WHERE THE OLD ONE WORKED. On those 580 the new value must track the old one
  case by case, not merely share a median. A median can match while every individual value
  is wrong, which is how two of this project's earlier estimators passed inspection.

  REPAIR WHERE IT DID NOT. On the 221 records with a degenerate carve the old values ran to
  PI 68.5 and PT 33.6. The new ones should fall back into the population, and the plate gate
  should stop firing on them.

Anything else -- new values that differ from the old everywhere, or that agree everywhere
including the broken cases -- means the replacement is not doing what it claims.

    python scripts/compare_v10_v11_spinopelvic.py --old OLD.csv --new NEW.csv --audit AUDIT.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

KEYS = ("pelvic_incidence_deg", "sacral_slope_deg", "pelvic_tilt_deg")
GOOD_LO, GOOD_HI = 0.15, 0.50


def load(p):
    with open(p) as fh:
        return {r["case"]: r for r in csv.DictReader(fh)}


def f(d, k):
    try:
        return float((d.get(k) or "").strip())
    except (ValueError, AttributeError):
        return None


def rejected(d):
    return (d.get("s1_plate_rejected") or "0") not in ("", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--audit", required=True)
    a = ap.parse_args()

    old, new, aud = load(a.old), load(a.new), load(a.audit)
    cases = sorted(set(old) & set(new))
    print(f"{len(cases)} cases in both runs\n")

    groups = {"carve was sound": [], "carve was degenerate": []}
    for c in cases:
        fr = f(aud.get(c, {}), "frac")
        if fr is None:
            continue
        key = ("carve was sound" if GOOD_LO <= fr <= GOOD_HI
               else "carve was degenerate")
        groups[key].append(c)

    for name, cs in groups.items():
        print("=" * 74)
        print(f"{name.upper()}   n={len(cs)}")
        print("=" * 74)
        print(f"{'':<22}{'old median':>12}{'new median':>12}{'median |diff|':>15}")
        for k in KEYS:
            o = np.array([f(old[c], k) for c in cs if f(old[c], k) is not None])
            n_ = np.array([f(new[c], k) for c in cs if f(new[c], k) is not None])
            both = [(f(old[c], k), f(new[c], k)) for c in cs
                    if f(old[c], k) is not None and f(new[c], k) is not None]
            d = np.array([abs(x - y) for x, y in both]) if both else np.array([np.nan])
            print(f"{k:<22}{np.median(o) if o.size else float('nan'):>12.1f}"
                  f"{np.median(n_) if n_.size else float('nan'):>12.1f}"
                  f"{np.median(d):>15.1f}")
        ro = sum(1 for c in cs if rejected(old[c]))
        rn = sum(1 for c in cs if rejected(new[c]))
        print(f"{'plate gate fired':<22}{ro:>12}{rn:>12}"
              f"{'':>15}   ({100.0*ro/max(len(cs),1):.0f}% -> {100.0*rn/max(len(cs),1):.0f}%)")

        # how many land outside anything anatomically defensible
        for k, lo, hi in (("pelvic_incidence_deg", 25.0, 85.0),
                          ("pelvic_tilt_deg", -5.0, 45.0),
                          ("sacral_slope_deg", 10.0, 70.0)):
            oo = sum(1 for c in cs
                     if f(old[c], k) is not None and not (lo <= f(old[c], k) <= hi))
            nn = sum(1 for c in cs
                     if f(new[c], k) is not None and not (lo <= f(new[c], k) <= hi))
            print(f"  outside [{lo:g},{hi:g}] {k.split('_')[1]:<10}{oo:>10}{nn:>12}")
        print()

    # the identity, which must hold in both
    print("=" * 74)
    print("PI = SS + PT, over every case")
    for tag, d in (("old", old), ("new", new)):
        bad, n = 0, 0
        for c in cases:
            pi, ss, pt = (f(d[c], k) for k in KEYS)
            if None in (pi, ss, pt):
                continue
            n += 1
            if abs(pi - (ss + pt)) > 1.0:
                bad += 1
        print(f"  {tag}: violated by more than 1 degree in {bad} of {n}")

    print("\nWHOLE COHORT, new run:")
    for k in KEYS:
        v = np.array([f(new[c], k) for c in cases if f(new[c], k) is not None])
        if v.size:
            print(f"  {k:<22} n={v.size:>4}  median {np.median(v):6.1f}  "
                  f"IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}")
    print("\n  published  Veilleux n=200 CT   PI 52.1  SS 36.5  PT 15.6")
    print("             Hasegawa n=24  CT   PI 53.4  SS 34.1  PT 19.2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
