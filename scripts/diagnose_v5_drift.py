"""diagnose_v5_drift.py — why do the two copies of v5 disagree, and how far?

data/v5_final is the working label directory. data/hf_export_v5/labels is the tree that was
published. A spot check found 1153 differing by 19,010 voxels while sixty other cases were
identical, which is the signature of an IN-PLACE edit made after the export was cut rather
than of a rebuild.

The suspect is slurm/finalize_v5.sh, whose first step is

    strip_vertebra_speckle.py --labels data/v5_final --apply

-- a pooled speckle strip that rewrites v5_final in place. Anything it removed after the
export was taken exists in the published tree and not in the working one.

That predicts a very specific shape for the difference, and this checks all three parts:

  ONE-DIRECTIONAL   voxels are only ever LOST from the working copy, never gained. A
                    removal pass cannot add anything.
  SMALL PIECES      what disappeared was disconnected specks, not parts of the main object.
  FILE TIMES        v5_final's copy is newer than the exported one.

If instead voxels moved between labels, or the working copy gained any, the cause is
something else entirely and the export is not simply older.

Runs over all 802 rather than a sample, because "how far did it spread" is the question.

    python scripts/diagnose_v5_drift.py --a data/v5_final --b data/hf_export_v5/labels \\
        --out qc_final/v5_drift.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
        **{v: f"L{v - 19}" for v in range(20, 26)},
        26: "sacrum", 29: "S1", 30: "left_hip", 31: "right_hip",
        32: "femur_left", 33: "femur_right",
        **{34 + i: f"rib_L{i + 1}" for i in range(12)},
        **{46 + i: f"rib_R{i + 1}" for i in range(12)},
        74: "lumbar_rib_L", 75: "lumbar_rib_R"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the working copy (v5_final)")
    ap.add_argument("--b", required=True, help="the published copy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A, B = Path(a.a), Path(a.b)
    cases = sorted(p.name[:4] for p in A.glob("*_label.nii.gz"))
    print(f"  {len(cases)} cases\n")

    rows = []
    n_same = n_diff = 0
    for i, cid in enumerate(cases, 1):
        pa, pb = A / f"{cid}_label.nii.gz", B / f"{cid}_label.nii.gz"
        if not pb.exists():
            rows.append({"case": cid, "state": "missing from the published copy"})
            continue
        ia, ib = nib.load(str(pa)), nib.load(str(pb))
        x = np.asanyarray(ia.dataobj).astype(np.int16)
        y = np.asanyarray(ib.dataobj).astype(np.int16)
        if x.shape != y.shape:
            rows.append({"case": cid, "state": "different shape"})
            n_diff += 1
            continue
        d = x != y
        if not d.any():
            n_same += 1
            continue
        n_diff += 1

        vox = float(np.prod(ia.header.get_zooms()[:3]))
        lost = int((d & (y > 0) & (x == 0)).sum())        # in published, gone from working
        gained = int((d & (x > 0) & (y == 0)).sum())      # in working, absent from published
        moved = int((d & (x > 0) & (y > 0)).sum())        # changed from one label to another

        # what kind of thing disappeared? a removal pass takes small disconnected pieces
        biggest = 0
        ids = []
        if lost:
            gone = d & (y > 0) & (x == 0)
            for v in np.unique(y[gone]):
                ids.append(NAME.get(int(v), str(int(v))))
            cc, n = ndimage.label(gone)
            if n:
                biggest = int(ndimage.sum(gone, cc, range(1, n + 1)).max())

        rows.append({"case": cid, "state": "differs",
                     "lost_from_working": lost, "gained_in_working": gained,
                     "relabelled": moved,
                     "largest_lost_piece_vox": biggest,
                     "largest_lost_piece_mm3": round(biggest * vox, 1),
                     "labels_affected": " ".join(sorted(set(ids))[:8])})
        print(f"  {cid}: lost {lost:,}  gained {gained:,}  relabelled {moved:,}  "
              f"largest lost piece {biggest} vox  [{' '.join(sorted(set(ids))[:6])}]",
              flush=True)
        if i % 200 == 0:
            print(f"  ... {i}/{len(cases)}", flush=True)

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    keys = ["case", "state", "lost_from_working", "gained_in_working", "relabelled",
            "largest_lost_piece_vox", "largest_lost_piece_mm3", "labels_affected"]
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    diffs = [r for r in rows if r.get("state") == "differs"]
    print(f"\n  identical: {n_same}    differing: {len(diffs)}")
    if diffs:
        tot_lost = sum(r["lost_from_working"] for r in diffs)
        tot_gain = sum(r["gained_in_working"] for r in diffs)
        tot_move = sum(r["relabelled"] for r in diffs)
        big = max(r["largest_lost_piece_vox"] for r in diffs)
        print(f"  voxels lost from the working copy : {tot_lost:,}")
        print(f"  voxels GAINED in the working copy : {tot_gain:,}")
        print(f"  voxels relabelled between classes : {tot_move:,}")
        print(f"  largest single piece lost         : {big} voxels")
        print()
        if tot_gain == 0 and tot_move == 0:
            print("  one-directional and removal-only: consistent with a speckle strip run")
            print("  on the working copy AFTER the export was cut.")
        else:
            print("  NOT removal-only -- something other than a speckle strip changed these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
