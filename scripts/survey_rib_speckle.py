"""scripts/survey_rib_speckle.py — how much of the rib layer is disconnected speckle.

0378 carries its left twelfth rib as 3765 voxels in 324 connected components: a 1459-voxel
core, about ten mid-sized pieces, and 313 fragments under twenty voxels each. The gallery
card for that case says the side was "recorded as an absence", which is wrong twice over --
the rib is present, and what is there is shattered.

Before deciding whether to clean the release, this establishes whether that is one bad case
or a property of the rib layer. It measures, per rib class per record: total voxels, number
of components, the size of the largest, and how much sits in components too small to be
bone at this resolution.

NOTHING IS MODIFIED. This only counts, so the decision to clean is made on numbers.

    python scripts/survey_rib_speckle.py --labels data/hf_export_v5/labels --workers 5
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

RIBS = list(range(34, 58)) + [74, 75]
SPECKLE_VOX = 20          # below this, at this resolution, it is not a piece of rib


def one(path: Path):
    case = path.name.split("_")[0]
    try:
        lab = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception as e:                                            # noqa: BLE001
        return [{"case": case, "error": f"{type(e).__name__}"}]
    counts = np.bincount(lab.reshape(-1))
    out = []
    for rid in RIBS:
        if rid >= len(counts) or not counts[rid]:
            continue
        m = lab == rid
        idx = np.nonzero(m)
        sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
        lbl, n = ndimage.label(m[sl])
        if n <= 1:
            continue                       # a single piece is the normal, uninteresting case
        sizes = ndimage.sum(m[sl], lbl, range(1, n + 1))
        out.append({
            "case": case, "rib_id": rid, "voxels": int(sizes.sum()),
            "components": int(n), "largest": int(sizes.max()),
            "speckle_components": int((sizes < SPECKLE_VOX).sum()),
            "speckle_voxels": int(sizes[sizes < SPECKLE_VOX].sum()),
            "error": "",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="qc_final/rib_speckle.csv")
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    print(f"  {len(files)} volume(s)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=2), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"    {i}/{len(files)}", flush=True)

    rows = [r for r in rows if not r.get("error")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    cases = {r["case"] for r in rows}
    sp_v = sum(r["speckle_voxels"] for r in rows)
    sp_c = sum(r["speckle_components"] for r in rows)
    tot = sum(r["voxels"] for r in rows)
    print(f"\n  {len(rows)} rib label(s) in more than one piece, across {len(cases)} record(s)")
    print(f"  speckle: {sp_c} component(s) under {SPECKLE_VOX} voxels, {sp_v} voxels total")
    print(f"  that is {100 * sp_v / max(tot, 1):.2f}% of the voxels in those labels")
    worst = sorted(rows, key=lambda r: -r["components"])[:8]
    print("\n  worst by component count:")
    for r in worst:
        print(f"    {r['case']} id={r['rib_id']:<3} {r['voxels']:>6} vox in "
              f"{r['components']:>4} pieces, largest {r['largest']:>5}, "
              f"{r['speckle_components']:>4} specks")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
