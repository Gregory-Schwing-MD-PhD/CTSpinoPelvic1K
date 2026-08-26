"""scripts/classify_detached.py — why each detached piece is detached. Read only.

Three causes, and they are not the same finding:

  the scan ends mid-vertebra, so body and arch survive as separate islands with the pedicle
  above the cut -- truncation, not a segmentation error, and nothing to repair;

  the rib leaves the reconstruction circle and returns, so the bone between was never
  imaged -- also not an error, and joining it would fabricate anatomy;

  the piece sits well inside the scanned region, and the label lost the bridge -- the only
  one of the three that is a defect.

The audit records the radial position of each gap, which separates the second from the other
two. This adds the axial test the audit never made: how far the structure sits below the top
of its own scan.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"
INV = {v: k for k, v in NAME.items()}

TOUCHING_MM = 3.0        # within this of the last slice carrying any bone


def one(job):
    case, labels_dir, wanted = job
    p = Path(labels_dir) / f"{case}_label.nii.gz"
    try:
        img = nib.as_closest_canonical(nib.load(str(p)))
        lab = np.asanyarray(img.dataobj)
        sp = np.array(img.header.get_zooms()[:3], float)
    except Exception:
        return []
    fg = np.nonzero((lab > 0).any(axis=(0, 1)))[0]
    if not len(fg):
        return []
    scan_top, scan_bot = int(fg.max()), int(fg.min())
    out = []
    for label in wanted:
        vid = INV.get(label)
        if vid is None:
            continue
        m = lab == vid
        if not m.any():
            continue
        z = np.nonzero(m.any(axis=(0, 1)))[0]
        d_top = float(scan_top - int(z.max())) * sp[2]
        d_bot = float(int(z.min()) - scan_bot) * sp[2]
        out.append({"case": case, "label": label,
                    "mm_below_scan_top": round(d_top, 1),
                    "mm_above_scan_bottom": round(d_bot, 1),
                    "truncated": int(d_top <= TOUCHING_MM or d_bot <= TOUCHING_MM)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--audit", default="qc_final/deposit_audit_full.csv")
    ap.add_argument("--out", default="qc_final/detached_causes.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    by_case = {}
    for r in csv.DictReader(open(a.audit, encoding="utf-8")):
        if r["verdict"] in ("inside_imaged_volume", "near_edge_uncertain",
                            "at_reconstruction_circle"):
            by_case.setdefault(r["case"], set()).add(r["label"])
    jobs = [(c, a.labels, sorted(v)) for c, v in sorted(by_case.items())]
    print(f"  {sum(len(v) for v in by_case.values())} detached labels in {len(jobs)} cases")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, got in enumerate(ex.map(one, jobs, chunksize=1), 1):
            rows += got
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    if not rows:
        print("  ! nothing measured")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    t = sum(r["truncated"] for r in rows)
    print(f"\n  {len(rows)} detached labels classified")
    print(f"    cut by the end of the scan : {t} ({100*t/len(rows):.0f}%)")
    print(f"    inside the scanned region  : {len(rows)-t} ({100*(len(rows)-t)/len(rows):.0f}%)")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
