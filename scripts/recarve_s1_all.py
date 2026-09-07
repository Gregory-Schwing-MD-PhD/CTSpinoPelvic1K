"""recarve_s1_all.py -- rebuild the S1/S2 split on every record, as a plane in the
patient's own frame.

WHAT WAS WRONG. v3 onward defines S1 as (GT sacrum) INTERSECT (TotalSegmentator
`vertebrae_S1`). Three consequences, all visible on 0816 and none specific to it:

  NOT A PLANE. The boundary is the edge of a network's blob, so it wanders. A radiologist
  reading it against the S1 foramina sees it run tangential to one and leave a gap at the
  other.

  NOT REPRODUCIBLE. Two identical runs on 0816 produced 190,451 and 103,467 S1 voxels. The
  S1 superior endplate is the landmark for sacral slope and pelvic incidence, so an unstable
  split moves a published measurement.

  NOT SQUARE TO THE PATIENT. Any scanner roll is baked in. 0816 is rolled 6.3 degrees.

WHAT THIS DOES. Keeps the LEVEL that TS chose -- it is anatomically informed and this is a
regularisation, not a re-detection -- and replaces the ORIENTATION with geometry measured
from the bone:

  1. Recombine sacrum + S1, so the bone's OUTER boundary stays exactly the radiologist's.
  2. Measure the patient's left-right axis as the normal of the sacrum's best mirror plane
     (max Dice between the bone and its reflection). This is what corrects scanner roll.
  3. Take the sacrum's cranio-caudal axis -- the principal component most aligned with
     superior-inferior, never assumed to be a voxel axis, because the sacrum is nearly as
     wide as it is tall -- and orthogonalise it against that LR axis.
  4. Place the cut so S1 keeps the VOLUME it already had.

WHY VOLUME AND NOT THE INTERFACE CENTROID. The first attempt put the plane at the centroid of
the existing S1/sacrum contact surface. That is a biased estimator: where the blob is
irregular the contact surface wraps around the sides of the bone, so its centroid sits caudal
to the true junction and the plane lands too low. Measured over the corpus it made things
worse -- S1 exceeded half the sacrum in 218 records against 99 before, 156 regressions
against 37 fixes. Preserving the voxel count instead changes ONLY the orientation, leaves
TS's judgement about HOW MUCH of the sacrum is S1 exactly where it was, and so cannot drift
the corpus in either direction.

Records where that inherited judgement is itself implausible -- S1 more than half the sacrum,
which no first sacral segment is -- are reported in the QC as `implausible_fraction` rather
than silently adjusted. That is a pre-existing problem and needs a radiologist, not a
heuristic.

The result is a true plane, square to the patient, carrying the sagittal tilt that follows
pelvic tilt, with equal clearance to the left and right foramen.

NOTHING IS REORIENTED: every label is written on its own grid with its own affine.

    python recarve_s1_all.py --in_dir data/hf_export_osc --out_dir data/s1_recarve \
        --report data/s1_recarve/s1_recarve_qc.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

S1_ID, SACRUM_ID = 29, 26
MIN_VOX = 2000          # below this the bone is FOV-clipped; leave the case alone
GRID = 3.0              # mm, occupancy grid for the symmetry score
OFF = 1024


def _rot(axis, deg):
    t = np.radians(deg)
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def _keys(pts):
    g = np.round(pts / GRID).astype(np.int64) + OFF
    return np.unique((g[:, 0] << 22) | (g[:, 1] << 11) | g[:, 2])


def lr_axis(P, ds=3):
    """The patient's left-right axis = normal of the sacrum's best mirror plane."""
    Pd = P[::ds]
    KA = _keys(Pd)

    def score(n):
        n = n / np.linalg.norm(n)
        KB = _keys(Pd - 2.0 * np.outer(Pd @ n, n))
        return 2.0 * np.intersect1d(KA, KB, assume_unique=True).size / (KA.size + KB.size)

    base = np.array([1.0, 0.0, 0.0])
    best_n, best_s = base, score(base)
    base_s = best_s
    for ay in np.arange(-14, 14.1, 2.0):
        Ry = _rot([0, 0, 1], ay)
        for az in np.arange(-14, 14.1, 2.0):
            n = Ry @ (_rot([0, 1, 0], az) @ base)
            s = score(n)
            if s > best_s:
                best_s, best_n = s, n / np.linalg.norm(n)
    return best_n, base_s, best_s


def one(args):
    src, dst = args
    cid = Path(src).name.split("_")[0]
    row = {"case": cid, "status": "", "roll_deg": "", "dice_before": "", "dice_after": "",
           "tilt_ap_deg": "", "s1_before": "", "s1_after": "", "delta": "",
           "s1_fraction": "", "implausible_fraction": "", "planarity": "", "bone_total": ""}
    try:
        li = nib.load(src)
        lab = np.asanyarray(li.dataobj).astype(np.uint8)
        aff = li.affine
    except Exception as exc:
        row["status"] = f"load_error:{type(exc).__name__}"
        return row

    s1, sac = lab == S1_ID, lab == SACRUM_ID
    whole = s1 | sac
    n_before = int(s1.sum())
    row["s1_before"] = n_before
    row["bone_total"] = int(whole.sum())

    if int(whole.sum()) < MIN_VOX or n_before == 0 or int(sac.sum()) == 0:
        # nothing to re-split (no sacrum, no S1, or a sliver clipped by the FOV)
        row["status"] = "skipped_no_split"
        nib.save(nib.Nifti1Image(lab, aff, li.header), dst)
        return row

    idx = np.argwhere(whole)
    ras = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
    centre = ras.mean(0)
    P = ras - centre

    lr, d0, d1 = lr_axis(P)
    row["dice_before"] = round(float(d0), 4)
    row["dice_after"] = round(float(d1), 4)
    row["roll_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(np.dot(lr, [1, 0, 0])))))), 2)

    _, _, vt = np.linalg.svd(P, full_matrices=False)
    cc = vt[int(np.argmax(np.abs(vt[:, 2])))]
    cc = -cc if cc[2] < 0 else cc
    cc = cc - np.dot(cc, lr) * lr
    nrm = np.linalg.norm(cc)
    if nrm < 1e-6:
        row["status"] = "degenerate_axis"
        nib.save(nib.Nifti1Image(lab, aff, li.header), dst)
        return row
    cc /= nrm
    row["tilt_ap_deg"] = round(float(np.degrees(np.arcsin(abs(cc[1])))), 2)

    # volume-preserving level: the cut sits where the top n_before voxels along the
    # axis end, so S1 keeps exactly the size TS gave it and only its SHAPE changes.
    proj = P @ cc
    order = np.argsort(-proj)
    keep = order[:n_before]
    out = lab.copy()
    out[tuple(idx.T)] = SACRUM_ID
    out[tuple(idx[keep].T)] = S1_ID

    n_after = int((out == S1_ID).sum())
    row["s1_after"] = n_after
    row["delta"] = n_after - n_before
    frac = n_after / max(int(whole.sum()), 1)
    row["s1_fraction"] = round(float(frac), 4)
    # no first sacral segment is more than half the sacrum; flag, do not adjust
    row["implausible_fraction"] = int(frac > 0.50 or frac < 0.15)

    ns1, nsac = out == S1_ID, out == SACRUM_ID
    ni = np.argwhere(ns1 & ndimage.binary_dilation(nsac, iterations=1))
    if len(ni) >= 20:
        nr = (aff @ np.c_[ni, np.ones(len(ni))].T).T[:, :3]
        _, ns, _ = np.linalg.svd(nr - nr.mean(0), full_matrices=False)
        row["planarity"] = round(float(ns[2] / ns.sum()), 5)

    img = nib.Nifti1Image(out, aff, li.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, dst)
    row["status"] = "recarved"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    src = Path(a.in_dir) / "labels"
    dst = Path(a.out_dir) / "labels"
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(dst / p.name)) for p in sorted(src.glob("*_label.nii.gz"))]
    print(f"  {len(jobs)} labels", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(one, jobs, chunksize=2), 1):
            rows.append(r)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    with open(a.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "recarved"]
    rolls = [float(r["roll_deg"]) for r in ok if r["roll_deg"] != ""]
    deltas = [int(r["delta"]) for r in ok if r["delta"] != ""]
    print(f"\n  recarved : {len(ok)}")
    for st in sorted({r['status'] for r in rows} - {"recarved"}):
        print(f"  {st:<18}: {sum(1 for r in rows if r['status'] == st)}")
    if rolls:
        rr = np.array(rolls)
        print(f"  scanner roll (deg): median {np.median(rr):.1f}  p90 {np.percentile(rr,90):.1f}  max {rr.max():.1f}")
        print(f"  cases rolled >3 deg: {int((rr > 3).sum())} of {len(rr)}")
    if deltas:
        dd = np.abs(np.array(deltas))
        print(f"  |S1 voxel change| : median {np.median(dd):,.0f}  p90 {np.percentile(dd,90):,.0f}  max {dd.max():,}")
    imp = [r for r in ok if r.get("implausible_fraction") == 1]
    print(f"  implausible S1 fraction (>50% or <15% of the sacrum): {len(imp)}"
          f"  <- pre-existing, inherited from TS, NOT introduced here")
    print(f"  wrote {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
