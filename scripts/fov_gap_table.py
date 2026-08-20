"""scripts/fov_gap_table.py — millimetres of unlabelled spine above the labelled stack.

The worklist ordering for thoracic annotation. A case with 200mm of bone above its highest
labelled vertebra has a whole thoracic spine sitting in the scan unannotated; a case with
~0 has a field of view that stops at L1 and nothing to add. The second is not a defect to
fix -- it is a property of the acquisition, and under the naming convention those ribs are
permanently unnameable.

Largest connected component per vertebra, because vertebra labels here carry speckle and a
fragment sitting high among the ribs otherwise makes the highest labelled vertebra appear
to reach the top of the scan (0344 reported 0mm with an entire thoracic spine unlabelled).
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

BONE_HU = 250.0
VERT_IDS = list(range(8, 20)) + list(range(20, 26))


def one(args):
    stem, lp, cp = args
    try:
        li = nib.as_closest_canonical(nib.load(lp))
        lab = np.asanyarray(li.dataobj).astype(np.int16)
        zm = np.array(li.header.get_zooms()[:3], float)
        ci = nib.as_closest_canonical(nib.load(cp))
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "gap_mm": None, "note": type(exc).__name__}
    if ct.shape != lab.shape:
        return {"case": stem, "gap_mm": None, "note": "shape mismatch"}
    ys = np.nonzero((lab > 0).any(axis=(0, 2)))[0]
    if not len(ys):
        return {"case": stem, "gap_mm": None, "note": "no labels"}
    bone = (ct[:, int(ys.min()):int(ys.max()) + 1] > BONE_HU).max(axis=1)
    top, n_named = None, 0
    for vid in VERT_IDS:
        m = lab == vid
        if m.sum() < 500:
            continue
        n_named += 1
        cc, ncc = ndimage.label(m)
        if ncc > 1:
            sizes = ndimage.sum(m, cc, range(1, ncc + 1))
            m = cc == (int(np.argmax(sizes)) + 1)
        z = int(np.nonzero(m.any(axis=(0, 1)))[0].max())
        top = z if top is None else max(top, z)
    bz = np.nonzero(bone.any(axis=0))[0]
    if top is None or not len(bz):
        return {"case": stem, "gap_mm": None, "note": "no vertebra labelled"}
    return {"case": stem, "gap_mm": round(float(bz.max() - top) * zm[2], 1),
            "n_vertebrae": n_named, "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = []
    for s in [c.strip() for c in a.cases.split(",") if c.strip()]:
        lp, cp = Path(a.labels) / f"{s}_label.nii.gz", Path(a.ct) / f"{s}_ct.nii.gz"
        if lp.exists() and cp.exists():
            jobs.append((s, str(lp), str(cp)))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs))
    rows.sort(key=lambda r: (r["gap_mm"] is None, -(r["gap_mm"] or 0)))
    print(f"\n  {'case':6s} {'unlabelled spine above':>24s}   vertebrae   note")
    for r in rows:
        g = "n/a" if r["gap_mm"] is None else f"{r['gap_mm']:.0f} mm"
        print(f"  {r['case']:6s} {g:>24s}   {r.get('n_vertebrae','-'):>9}   {r['note']}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["case", "gap_mm", "n_vertebrae", "note"])
            w.writeheader(); w.writerows(rows)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
