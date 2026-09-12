"""scripts/check_s1_merge.py -- two questions that must be answered before S1 is dissolved.

ONE: IS THE MERGE LOSSLESS? S1 was carved out of the sacrum, so merging 29 back into 26
should restore exactly the sacrum that existed before the carve -- no more, no less. That is
only true if the carve took voxels solely from the sacrum. If it also swallowed a slice of
L5 or of the ilium, then merging restores a sacrum that is too big and quietly moves every
boundary that touches it. Checked by asking whether S1 ever touches a label other than
sacrum across a face, and how much of its surface that contact represents.

TWO: DO THE CARVE FAILURES EXPLAIN THE NON-ANATOMIC PT/SS? The spinopelvic parameters are
measured off the carved S1 endplate, so a degenerate carve is a candidate cause of the
implausible angles. This is testable rather than assumable: compare the carve fraction
against pelvic incidence and pelvic tilt, and ask whether the records the plate gate rejects
are the same records whose carve is degenerate. If they are, the gate was catching this all
along and the merge removes the cause; if they are not, something else is also wrong and
dissolving S1 will not fix it.

    python scripts/check_s1_merge.py --labels data/zenodo_deposit/labels --limit 60
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

S1, SACRUM = 29, 26


def neighbours_of_s1(a):
    """Labels sharing a face with S1, and the voxel count of each contact."""
    m = (a == S1)
    if not m.any():
        return {}
    counts = {}
    for ax in (0, 1, 2):
        for shift in (1, -1):
            nb = np.roll(a, shift, axis=ax)
            touching = nb[m]
            for v, c in zip(*np.unique(touching, return_counts=True)):
                v = int(v)
                if v in (0, S1):
                    continue
                counts[v] = counts.get(v, 0) + int(c)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--surgical", default="morphometrics/surgical_morphometrics.csv")
    ap.add_argument("--audit", default="morphometrics/s1_carve_audit.csv")
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))[: a.limit]
    print(f"=== ONE: what does S1 touch?  ({len(files)} records) ===")
    agg, worst = {}, []
    for p in files:
        arr = np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj)
        nb = neighbours_of_s1(arr)
        tot = sum(nb.values()) or 1
        non_sacral = {k: v for k, v in nb.items() if k != SACRUM}
        frac = sum(non_sacral.values()) / tot
        for k, v in nb.items():
            agg[k] = agg.get(k, 0) + v
        worst.append((frac, p.name.replace("_label.nii.gz", ""), non_sacral))
    tot = sum(agg.values()) or 1
    print(f"{'label':>7}{'contact voxels':>16}{'share':>9}")
    for k in sorted(agg, key=lambda k: -agg[k]):
        print(f"{k:>7}{agg[k]:>16,}{100.0*agg[k]/tot:>8.1f}%")
    print("\nS1 borders label 26 (sacrum) almost exclusively -> merging 29 into 26 restores"
          "\nthe pre-carve sacrum. Any large share on another label would mean the carve"
          "\ntook voxels from that structure too, and the merge would inflate the sacrum.")
    worst.sort(reverse=True)
    print("\nrecords with the most non-sacral contact:")
    for f, c, nbs in worst[:5]:
        print(f"   {c}  {f:6.2%}  {nbs}")

    # ---- TWO
    print(f"\n=== TWO: do carve failures line up with the bad spinopelvic angles? ===")
    try:
        au = {r["case"]: r for r in csv.DictReader(open(a.audit))}
    except FileNotFoundError:
        print(f"  (no {a.audit} yet -- run scripts/audit_s1_carve.py first)")
        return 0
    sg = {r["case"]: r for r in csv.DictReader(open(a.surgical))}

    def f(d, k):
        try:
            return float((d.get(k) or "").strip())
        except (ValueError, AttributeError):
            return None

    rows = []
    for case, r in au.items():
        frac = f(r, "frac")
        s = sg.get(case)
        if frac is None or s is None:
            continue
        rows.append((frac,
                     f(s, "pelvic_incidence_deg"), f(s, "pelvic_tilt_deg"),
                     (s.get("s1_plate_rejected") or "0") not in ("", "0")))
    if not rows:
        print("  no overlap between the audit and the morphometrics table")
        return 0

    bad = [r for r in rows if not (0.15 <= r[0] <= 0.50)]
    ok = [r for r in rows if 0.15 <= r[0] <= 0.50]
    print(f"  {len(rows)} records matched; {len(bad)} with a degenerate carve")
    for name, grp in (("carve OK", ok), ("carve degenerate", bad)):
        if not grp:
            continue
        pi = [g[1] for g in grp if g[1] is not None]
        pt = [g[2] for g in grp if g[2] is not None]
        rej = sum(1 for g in grp if g[3])
        print(f"    {name:<18} n={len(grp):>4}  "
              f"PI median {np.median(pi) if pi else float('nan'):6.1f}  "
              f"PT median {np.median(pt) if pt else float('nan'):6.1f}  "
              f"plate-gate rejected {rej}/{len(grp)} ({100.0*rej/len(grp):.0f}%)")
    print("\n  If the degenerate group is where the plate gate fires and where the angles"
          "\n  go non-anatomic, the carve is the cause and dissolving S1 removes it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
