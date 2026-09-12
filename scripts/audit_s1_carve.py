"""scripts/audit_s1_carve.py -- how often, and how badly, does the S1 carve fail?

WHY. The carve is an automatic estimate of the S1--S2 boundary, and the release ships it as
its own label. One case (0428) was found with nearly the whole sacrum labelled S1, which
raises the only question that matters for the release: is that a rare miss or the typical
result? A single bad render does not answer it and neither does a count of QC flags, since
the flag threshold is the thing under suspicion.

So this measures the carve on every record and reports the DISTRIBUTION, not a pass rate.
Two numbers per case:

  fraction -- S1's craniocaudal extent over the whole sacrum's. A real S1 is one segment of
              five, sitting at the top, so roughly 0.2-0.35 is anatomically right. Much more
              means the plane cut too low; much less means it cut into the endplate.

  height   -- S1's extent in millimetres, which catches the case where the sacrum itself is
              truncated by the field of view and the fraction looks fine for the wrong reason.

Reported as a histogram plus the named extremes, so the decision about whether to keep,
re-cut or withdraw the label is made against the shape of the distribution.

    python scripts/audit_s1_carve.py --labels data/zenodo_deposit/labels
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

S1, SACRUM = 29, 26                      # v10, the released scheme
GOOD_LO, GOOD_HI = 0.15, 0.50


def scan_one(p: Path):
    # CANONICALISE BEFORE MEASURING. These volumes are stored ('P','I','R'), so the
    # craniocaudal axis is axis 1, not axis 2. Reducing over (0,1) and calling the result a
    # height measures S1 left-to-right, where it spans almost the same extent as the whole
    # sacrum -- which is exactly how a first pass of this audit reported "S1 is 100% of the
    # sacrum" for a case whose render plainly showed a correct thin S1. Canonicalise for
    # ANALYSIS only; never write a reoriented label.
    img = nib.as_closest_canonical(nib.load(str(p)))
    a = np.asanyarray(img.dataobj)
    zoom_z = float(abs(img.header.get_zooms()[2]))
    s1 = np.nonzero((a == S1).any(axis=(0, 1)))[0]
    sac = np.nonzero((a == SACRUM).any(axis=(0, 1)))[0]
    n_s1 = int((a == S1).sum())
    n_sac = int((a == SACRUM).sum())
    if s1.size == 0:
        return dict(s1_slices=0, frac=None, s1_mm=0.0, whole_mm=None,
                    vox_frac=None, note="no S1 label")
    if sac.size == 0:
        return dict(s1_slices=int(s1.size), frac=1.0,
                    s1_mm=float(s1.size * zoom_z), whole_mm=float(s1.size * zoom_z),
                    vox_frac=1.0, note="no sacrum left below S1")
    lo, hi = min(s1[0], sac[0]), max(s1[-1], sac[-1])
    whole = hi - lo + 1
    return dict(s1_slices=int(s1.size), frac=float(s1.size / whole),
                s1_mm=float(s1.size * zoom_z), whole_mm=float(whole * zoom_z),
                vox_frac=float(n_s1 / (n_s1 + n_sac)) if (n_s1 + n_sac) else None,
                note="")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--out", default="morphometrics/s1_carve_audit.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} label volumes in {a.labels}", flush=True)

    rows = []
    for i, p in enumerate(files, 1):
        case = p.name.replace("_label.nii.gz", "")
        try:
            r = scan_one(p)
        except Exception as exc:                                    # noqa: BLE001
            r = dict(s1_slices=0, frac=None, s1_mm=0.0, whole_mm=None,
                     vox_frac=None, note=f"{type(exc).__name__}")
        r["case"] = case
        rows.append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "s1_slices", "frac", "s1_mm",
                                           "whole_mm", "vox_frac", "note"])
        w.writeheader()
        w.writerows(rows)

    vals = [r["frac"] for r in rows if r["frac"] is not None]
    print(f"\n{len(vals)} records with an S1 label; {len(rows) - len(vals)} without\n")
    if not vals:
        return 1
    v = np.array(vals)
    print(f"S1 as a fraction of sacral height:")
    for q in (0, 5, 25, 50, 75, 95, 100):
        print(f"   {q:>3}th percentile  {np.percentile(v, q):.3f}")

    print(f"\n{'band':<28}{'n':>6}{'%':>8}")
    bands = [("< 0.10  cut into endplate", v < 0.10),
             ("0.10-0.15  thin", (v >= 0.10) & (v < 0.15)),
             ("0.15-0.35  anatomically right", (v >= 0.15) & (v < 0.35)),
             ("0.35-0.50  low cut", (v >= 0.35) & (v < 0.50)),
             ("> 0.50  most of the sacrum", v >= 0.50)]
    for name, m in bands:
        print(f"{name:<28}{int(m.sum()):>6}{100.0*m.sum()/v.size:>7.1f}%")

    bad = sorted(((r["frac"], r["case"]) for r in rows
                  if r["frac"] is not None and not (GOOD_LO <= r["frac"] <= GOOD_HI)),
                 reverse=True)
    print(f"\n{len(bad)} outside [{GOOD_LO}, {GOOD_HI}]; worst 12:")
    for f_, c in bad[:12]:
        print(f"   {c}  {f_:.3f}")
    mm = [r["s1_mm"] for r in rows if r["frac"] is not None]
    print(f"\nS1 height in mm: median {np.median(mm):.1f}, "
          f"5th {np.percentile(mm,5):.1f}, 95th {np.percentile(mm,95):.1f} "
          f"(a real S1 body is roughly 30 mm tall)")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
