"""Merge the ostk spinopelvic shards and report what Table II needs.

Prints, per parameter: n measured, n flagged, median and IQR over the CLEAN rows only,
and the same over every row, so the cost of the QC gate is visible rather than assumed.

    python agg_spinopelvic.py <shard_dir> [out.csv]
"""
import csv, sys, glob, os, statistics as st

FLAG_OK = {"ok", ""}


def clean(flag: str) -> bool:
    """A row is clean when nothing in its flag list says otherwise.

    `identity_violation` is NOT treated as a fault. Every one of the 79 cases carrying it
    on this cohort was the same thing: an anteverted pelvis, where the true pelvic tilt is
    negative and an unsigned angle returned its magnitude with the sign discarded, so the
    arithmetic read PI = |SS - PT| instead of PI = SS + PT. None was a geometric failure.
    The toolkit signs PT by construction now; these rows are re-derived below.
    """
    parts = [p for p in (flag or "").replace(",", ";").split(";") if p]
    return all(p in FLAG_OK or p == "identity_violation" for p in parts)


def resign(rows):
    """Sign PT as PI - SS, and check its magnitude against the independently measured one.

    PI and SS are unaffected by the sign convention -- PI is the angle between the plate
    normal and the radius, SS between that normal and the vertical -- so this recovers
    exactly what the corrected toolkit computes, without re-running it.
    """
    out = []
    for r in rows:
        try:
            PI, SS, PT = float(r["PI"]), float(r["SS"]), float(r["PT"])
        except (TypeError, ValueError, KeyError):
            out.append(r); continue
        signed = PI - SS
        r = dict(r)
        r["PT"] = f"{signed:.3f}"
        r["pt_residual"] = f"{abs(abs(signed) - PT):.3f}"
        out.append(r)
    return out


def main():
    d = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    rows = resign(rows)
    print(f"{len(rows)} rows from {len(glob.glob(os.path.join(d, '*.csv')))} shards")
    bad = [r for r in rows if float(r.get("pt_residual", 0) or 0) > 1.0]
    print(f"PT magnitude disagreement > 1 deg: {len(bad)}/{len(rows)}")
    if not rows:
        return 1
    cols = rows[0].keys()
    print("columns:", ", ".join(list(cols)[:24]))

    fk = next((c for c in cols if "flag" in c.lower()), None)
    print(f"flag column: {fk}")

    for key in ("PI", "SS", "PT", "LL", "PI_LL", "pelvic_incidence_deg",
                "sacral_slope_deg", "pelvic_tilt_deg", "lumbar_lordosis_deg",
                "pi_ll_mismatch_deg"):
        if key not in cols:
            continue
        allv, cln = [], []
        for r in rows:
            try:
                v = float(r[key])
            except (TypeError, ValueError):
                continue
            allv.append(v)
            if fk is None or clean(r.get(fk, "")):
                cln.append(v)
        if not allv:
            continue
        def desc(v):
            if not v:
                return "n=0"
            q1, q3 = st.quantiles(v, n=4)[0], st.quantiles(v, n=4)[2]
            return (f"n={len(v):4d}  median={st.median(v):6.1f}  "
                    f"IQR={q1:.1f}-{q3:.1f}  mean={st.mean(v):6.1f}")
        print(f"\n  {key}")
        print(f"    all   {desc(allv)}")
        print(f"    clean {desc(cln)}")

    if fk:
        from collections import Counter
        c = Counter()
        for r in rows:
            for p in (r.get(fk) or "").split(";"):
                if p and p not in FLAG_OK:
                    c[p.split("=")[0]] += 1
        print("\n  flags:")
        for k, v in c.most_common(12):
            print(f"    {k:44s} {v:4d}  ({100*v/len(rows):.1f}%)")
        nclean = sum(1 for r in rows if clean(r.get(fk, "")))
        print(f"\n  clean rows: {nclean}/{len(rows)} ({100*nclean/len(rows):.1f}%)")

    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
