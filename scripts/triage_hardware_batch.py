"""triage_hardware_batch.py — why did 39 of 84 instrumented cases produce no proposal?

scan_hardware.py flagged 84 cases using an 1800 HU threshold. seed_hardware.py now uses
2500 HU, the lower of the two values validated in the metal-segmentation literature, and on
39 of those cases it finds nothing above the 40-voxel floor and exits without writing.

Two very different explanations, and they need separating before anything is shipped:

  THE FLAG OVER-CALLED   1800 HU is below anything published. Dense sclerotic bone, cement
                         and contrast all reach it. A case whose brightest voxel near the
                         spine is around 2000 HU probably has no implant at all.
  THE IMPLANT IS SMALL   A single small screw, or one seen mostly in partial volume, can
                         saturate at its core and still leave fewer than 40 voxels above
                         2500. That case IS instrumented and the batch simply missed it.

Saturation separates them. The scanner clips at 3071 on this cohort, so a voxel at or near
that value is metal and nothing else -- bone does not get there. What matters is whether the
saturated voxels are NEAR THE SPINE or out in the colon, because these are colonography
series and the tagged stool saturates exactly as titanium does.

Writes a triage CSV: which cases have real metal that the batch missed, and which were never
instrumented to begin with.

    python scripts/triage_hardware_batch.py --scan qc_hardware/hardware_scan.csv \\
        --proposals data/hardware_fix --ct data/hf_export_v5/ct \\
        --labels data/v5_final --out qc_hardware/triage.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

BONE_IDS = list(range(8, 30)) + [30, 31, 32, 33]
NEAR_MM = 15.0            # further than this from labelled bone is not an implant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.scan, encoding="utf-8")))
    prop = Path(a.proposals)
    missing = [r for r in rows
               if not (prop / f"{r['case']}_hardware_only.nii.gz").exists()]
    print(f"  {len(rows)} flagged, {len(rows) - len(missing)} with a proposal, "
          f"{len(missing)} without\n")
    if a.limit:
        missing = missing[:a.limit]

    out = []
    for i, r in enumerate(missing, 1):
        cid = r["case"]
        cp = Path(a.ct) / f"{cid}_ct.nii.gz"
        lp = Path(a.labels) / f"{cid}_label.nii.gz"
        if not cp.exists() or not lp.exists():
            out.append({"case": cid, "verdict": "files missing"})
            continue
        img = nib.load(str(lp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        ct = np.asanyarray(nib.load(str(cp)).dataobj)
        sp = np.array(img.header.get_zooms()[:3], float)

        bone = np.isin(lab, BONE_IDS)
        # crop to the skeleton plus a margin: the whole-volume distance transform on a
        # 512-cubed grid costs far more than the question is worth
        idx = np.argwhere(bone)
        pad = np.ceil(np.array([40.0, 40.0, 40.0]) / sp).astype(int)
        lo = np.maximum(idx.min(0) - pad, 0)
        hi = np.minimum(idx.max(0) + pad + 1, np.array(lab.shape))
        sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
        bone_c, ct_c = bone[sl], ct[sl]
        dist = ndimage.distance_transform_edt(~bone_c, sampling=sp)

        near = dist <= NEAR_MM
        sat_near = int(((ct_c >= 3000) & near).sum())
        sat_far = int((ct_c >= 3000).sum()) - sat_near
        hi25_near = int(((ct_c > 2500) & near).sum())
        hi18_near = int(((ct_c > 1800) & near).sum())
        peak_near = int(ct_c[near].max()) if near.any() else 0

        if hi25_near >= 40:
            verdict = "METAL the batch missed"     # should have produced a proposal
        elif sat_near >= 5:
            verdict = "small implant, under the 40-voxel floor"
        elif peak_near < 2600:
            verdict = "no implant -- 1800 HU over-called it"
        else:
            verdict = "borderline, needs a look"
        out.append({"case": cid, "peak_HU_near_bone": peak_near,
                    "saturated_near_bone": sat_near, "saturated_far": sat_far,
                    "vox_gt2500_near": hi25_near, "vox_gt1800_near": hi18_near,
                    "scan_max_hu": r.get("max_hu"), "scan_voxels_1800": r.get("voxels"),
                    "verdict": verdict})
        print(f"  {cid}  peak {peak_near:>5} HU near bone | "
              f"sat near {sat_near:>6,} far {sat_far:>7,} | "
              f">2500 near {hi25_near:>6,} | {verdict}", flush=True)

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    keys = ["case", "peak_HU_near_bone", "saturated_near_bone", "saturated_far",
            "vox_gt2500_near", "vox_gt1800_near", "scan_max_hu", "scan_voxels_1800",
            "verdict"]
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in keys})

    from collections import Counter
    print("\n  " + "\n  ".join(f"{v:>4}  {k}" for k, v in
                               Counter(r.get("verdict", "?") for r in out).most_common()))
    print(f"\n  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
