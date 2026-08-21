"""scripts/check_release_invariants.py — the things that must be true of every case.

WHY THIS EXISTS. A rib-renumbering pass loaded labels through as_closest_canonical and
wrote the reoriented array back under the canonical affine. Renumbering is pure id
arithmetic and never needed the reorientation, but the effect was to transpose one
label away from its CT -- 512x512x645 against a 512x645x512 scan. Nothing in the
pipeline noticed. It surfaced only when a human opened the case and ITK-SNAP refused
the pair, which is the worst possible detector: late, manual, and dependent on someone
happening to look at that one case.

Every check here is cheap, reads headers rather than voxels where it can, and answers a
question that has exactly one right answer for a finished release. A release that fails
any of them is not finished.

  geometry     label dimensions, affine and voxel spacing agree with the CT
  ids          no label id outside the published scheme
  emptiness    the label is not blank
  sidedness    left-side ids sit on one side of the midline and right-side ids the other

SIDEDNESS IS HERE BECAUSE IT IS FREE. A transposition that happens to preserve the shape
would pass the geometry check; ribs landing on the wrong side of the spine would not.

    python scripts/check_release_invariants.py --labels data/v5_final --ct data/hf_export/ct
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402

RIB_L = range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_LEFT_OFFSET + 13)
RIB_R = range(LS.RIB_RIGHT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13)


def allowed_ids() -> set:
    ok = {0, LS.IGNORE_LABEL}
    ok |= set(range(1, 34))                     # C1..L6, sacrum, coccyx, T13, S1, hips, femurs
    ok |= set(RIB_L) | set(RIB_R)
    ok |= set(range(58, 76))                    # soft tissue + lumbar ribs
    for name in ("HARDWARE", "HARDWARE_CAGE", "HARDWARE_SCREW_ROD", "HARDWARE_PLATE"):
        v = getattr(LS, name, None)
        if v is not None:
            ok.add(int(v))
    return ok


def one(args) -> dict:
    lab_p, ct_p = args
    stem = Path(lab_p).name.replace("_label.nii.gz", "")
    r = {"case": stem, "ok": 1, "problems": ""}
    bad = []
    try:
        li = nib.load(lab_p)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "ok": 0, "problems": f"unreadable:{type(exc).__name__}"}

    if ct_p and Path(ct_p).exists():
        ci = nib.load(ct_p)
        if li.shape != ci.shape:
            bad.append(f"shape {li.shape} vs ct {ci.shape}")
        else:
            # a shape match is not an orientation match: a cubic volume can be
            # transposed and still agree on dimensions
            if not np.allclose(li.affine, ci.affine, atol=1e-3):
                bad.append("affine differs from ct")
            lz = np.asarray(li.header.get_zooms()[:3], float)
            cz = np.asarray(ci.header.get_zooms()[:3], float)
            if not np.allclose(lz, cz, atol=1e-3):
                bad.append(f"spacing {tuple(round(v,3) for v in lz)} vs "
                           f"{tuple(round(v,3) for v in cz)}")

    lab = np.asanyarray(li.dataobj)
    present = {int(v) for v in np.unique(lab)}
    if present <= {0}:
        bad.append("label is empty")
    stray = sorted(present - allowed_ids())
    if stray:
        bad.append(f"ids outside the scheme: {stray[:6]}")

    # sidedness, in the file's own frame: work out which array axis is left-right from
    # the affine rather than assuming, then check the two rib families separate on it
    lr_axis = int(np.argmax(np.abs(li.affine[0, :3])))
    sign = np.sign(li.affine[0, lr_axis]) or 1.0
    lm = np.isin(lab, list(RIB_L))
    rm = np.isin(lab, list(RIB_R))
    if lm.any() and rm.any():
        lc = float(np.argwhere(lm)[:, lr_axis].mean()) * sign
        rc = float(np.argwhere(rm)[:, lr_axis].mean()) * sign
        # anatomical left is +x in RAS, so left ribs must sit at the larger coordinate
        if lc <= rc:
            bad.append("left and right ribs are on the wrong sides")

    r["ok"] = 0 if bad else 1
    r["problems"] = "; ".join(bad)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export/ct")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="qc_invariants.csv")
    a = ap.parse_args()

    labs = sorted(Path(a.labels).glob("*_label.nii.gz"))
    jobs = [(str(p), str(Path(a.ct) / p.name.replace("_label", "_ct"))) for p in labs]
    print(f"{len(jobs)} case(s)\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, jobs, chunksize=4), 1):
            res.append(x)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "ok", "problems"])
        w.writeheader()
        w.writerows(res)

    bad = [x for x in res if not x["ok"]]
    print(f"\n  {len(res) - len(bad)}/{len(res)} pass")
    for x in bad[:25]:
        print(f"    {x['case']}: {x['problems']}")
    if len(bad) > 25:
        print(f"    ... and {len(bad) - 25} more")
    print(f"\n  wrote {a.out}")
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
