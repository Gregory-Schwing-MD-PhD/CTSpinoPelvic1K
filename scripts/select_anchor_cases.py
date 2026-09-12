"""scripts/select_anchor_cases.py -- choose, by rule, the four cases Figure 2 renders.

WHY THIS EXISTS. The four panels of Figure 2 were rendered from four cases whose identifiers
were never written down -- not in a shell script, a note, or the commit message -- so the
figure could not be regenerated and a reader could not check it against the released labels.
A figure whose subjects cannot be named is not reproducible, and the repair is not to guess
which cases they were: it is to make the choice itself a function of the data, so that
anyone running this script on the release gets the same four cases and can say why.

THE RULE. Each panel is a phenotype defined by columns in transition_morphometrics.csv.
Within the records satisfying a panel's definition, the chosen case is the one closest to
that group's own median on the measurements the figure displays -- rib-free count, lowest-rib
ratio, transverse span -- with the case identifier breaking ties. So the figure shows the
most ORDINARY member of each phenotype rather than the most striking one, which is the
honest choice for a figure whose job is to define a category.

    python scripts/select_anchor_cases.py
    python scripts/select_anchor_cases.py --write morphometrics/anchor_cases.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "morphometrics" / "transition_morphometrics.csv"


def num(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def flag(r, k):
    return (r.get(k) or "").strip().lower() in ("1", "true", "yes")


def label(r):
    return (r.get("lstv_label") or "").strip().upper()


# Each panel: a title, a predicate, and the columns whose medians define "most ordinary".
PANELS = [
    ("five rib-free, last rib on T12",
     lambda r: (num(r, "n_non_rib_bearing") == 5
                and (r.get("lowest_rib_bearing") or "").strip() == "T12"
                and label(r) in ("NORMAL", "")
                and not flag(r, "has_lumbar_rib")),
     ("rib12_11_ratio_min", "ll_span_total_mm")),

    ("four, long rib on L1",
     lambda r: (num(r, "n_non_rib_bearing") == 4
                and flag(r, "has_lumbar_rib")),
     ("lumbar_rib_len_mm", "ll_span_total_mm")),

    ("four, stump ribs on T12, fused junction",
     lambda r: (num(r, "n_non_rib_bearing") == 4
                and (num(r, "rib12_11_ratio_min") or 1.0) < 0.33
                and (label(r) == "SACRALIZATION"
                     or (r.get("castellvi_type") or "").strip().startswith("III"))),
     ("rib12_11_ratio_min", "ll_span_total_mm")),

    ("six rib-free, last rib on T12",
     lambda r: (num(r, "n_non_rib_bearing") == 6
                and (flag(r, "has_l6")
                     or (r.get("lowest_lumbar") or "").strip() == "L6")),
     ("rib12_11_ratio_min", "ll_span_total_mm")),
]


S1_LABEL, SACRUM_LABEL = 29, 26          # v10, the released scheme
S1_FRAC_MIN, S1_FRAC_MAX = 0.15, 0.50


def s1_carve_ok(case: str, labels_dir: Path | None):
    """Is this case's S1 carve plausible, measured on the label itself?

    The carve is an automatic estimate of the S1--S2 boundary and it fails on about a
    hundred records, taking either most of the sacrum or almost none of it. A figure whose
    whole point is to show the caudal anchor must not define the category with a case where
    that anchor is wrong -- and the first selection did exactly that, picking a record whose
    "S1" was nearly the entire sacrum. Numbers alone did not reveal it; rendering did.

    Returns (ok, fraction) or (None, None) when the volume cannot be read.
    """
    if labels_dir is None:
        return None, None
    p = Path(labels_dir) / f"{case}_label.nii.gz"
    if not p.exists():
        return None, None
    try:
        import nibabel as nib
        import numpy as np
        # canonicalise: stored orientation is ('P','I','R'), so axis 2 is only the
        # craniocaudal axis after this call. See scripts/audit_s1_carve.py.
        a = np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj)
    except Exception:                                              # noqa: BLE001
        return None, None
    s1 = np.nonzero((a == S1_LABEL).any(axis=(0, 1)))[0]
    sac = np.nonzero((a == SACRUM_LABEL).any(axis=(0, 1)))[0]
    if s1.size == 0 or sac.size == 0:
        return False, 0.0
    lo, hi = min(s1[0], sac[0]), max(s1[-1], sac[-1])
    total = hi - lo + 1
    frac = (s1[-1] - s1[0] + 1) / total if total else 0.0
    return (S1_FRAC_MIN <= frac <= S1_FRAC_MAX), float(frac)


def pick(rows, pred, keys, labels_dir=None):
    pool = [r for r in rows if pred(r)]
    if not pool:
        return None, 0, {}
    meds = {}
    for k in keys:
        vals = [num(r, k) for r in pool if num(r, k) is not None]
        if vals:
            meds[k] = st.median(vals)

    def dist(r):
        d = 0.0
        for k, m in meds.items():
            v = num(r, k)
            if v is None:
                d += 1.0                      # a missing measure is never "most ordinary"
            elif m:
                d += abs(v - m) / (abs(m) or 1.0)
        return (d, (r.get("case") or ""))

    # Walk the ranking, not just its head: the most ordinary case on the plotted measures
    # can still carry a broken caudal anchor, and that disqualifies it for this figure.
    for r in sorted(pool, key=dist):
        cid = (r.get("case") or "").strip()
        ok, frac = s1_carve_ok(cid, labels_dir)
        if ok is None or ok:
            r = dict(r)
            r["_s1_frac"] = frac
            return r, len(pool), meds
        print(f"      (skipped {cid}: S1 carve implausible, "
              f"S1 is {frac:.0%} of the sacrum)")
    return None, len(pool), meds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--labels", default="data/zenodo_deposit/labels",
                    help="released labels, used to check each pick's S1 carve")
    ap.add_argument("--write", default=None,
                    help="also write the selection as JSON so the figure records its cases")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    print(f"{len(rows)} records from {a.csv}\n")

    out, spec = [], []
    for title, pred, keys in PANELS:
        r, n, meds = pick(rows, pred, keys, labels_dir=a.labels)
        if r is None:
            print(f"  !! NO CASE MATCHES: {title}")
            continue
        cid = (r.get("case") or "").strip()
        out.append(dict(case=cid, title=title, n_candidates=n,
                        n_non_rib_bearing=num(r, "n_non_rib_bearing"),
                        lowest_rib_bearing=(r.get("lowest_rib_bearing") or "").strip(),
                        lowest_lumbar=(r.get("lowest_lumbar") or "").strip(),
                        rib12_11_ratio_min=num(r, "rib12_11_ratio_min"),
                        has_lumbar_rib=flag(r, "has_lumbar_rib"),
                        lstv_label=label(r),
                        castellvi=(r.get("castellvi_type") or "").strip(),
                        s1_fraction_of_sacrum=r.get("_s1_frac")))
        spec.append(f"{cid}:{title}")
        print(f"  {cid}  {title}")
        print(f"      chosen from {n} matching record(s); rib-free "
              f"{num(r, 'n_non_rib_bearing'):.0f}, lowest rib-bearing "
              f"{(r.get('lowest_rib_bearing') or '?').strip()}, "
              f"rib12/11 {num(r, 'rib12_11_ratio_min')}, "
              f"LSTV {label(r) or 'normal'}"
              + (f", S1 is {r['_s1_frac']:.0%} of the sacrum"
                 if r.get("_s1_frac") is not None else ""))

    print("\nrender with:\n")
    print("  python scripts/render_anchors.py --labels <labels dir> \\")
    print(f"      --cases '{';'.join(spec)}'")

    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.write, "w"), indent=1)
        print(f"\nwrote {a.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
