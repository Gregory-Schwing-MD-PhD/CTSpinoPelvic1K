"""label_occupancy_v6.py -- what is actually inside the v6 label maps.

The shipped card describes a scheme; this reports the tree. For every id in
dataset_labels.json it counts the cases that contain the id at all and the total
voxels carrying it, so the public card can state what a download really holds
rather than what the scheme reserves.

    python scripts/label_occupancy_v6.py --tree data/hf_export_v6 \
        --out data/hf_export_v6/label_occupancy.json --workers 12
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


def one(p: str):
    """Return {id: voxels} for a single label volume."""
    arr = np.asanyarray(nib.load(p).dataobj)
    ids, counts = np.unique(arr, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) != 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    tree = Path(a.tree)
    labels = sorted(str(p) for p in (tree / "labels").glob("*_label.nii.gz"))
    print(f"  {len(labels)} label volumes", flush=True)

    cases: Counter = Counter()   # id -> n cases containing it
    voxels: Counter = Counter()  # id -> total voxels
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, d in enumerate(ex.map(one, labels, chunksize=4), 1):
            for i, c in d.items():
                cases[i] += 1
                voxels[i] += c
            if n % 100 == 0:
                print(f"  {n}/{len(labels)}", flush=True)

    names = json.loads((tree / "dataset_labels.json").read_text())["id_to_name"]
    rows = []
    for i in sorted(set(cases) | {int(k) for k in names}):
        rows.append({
            "id": i,
            "name": names.get(str(i), f"UNDECLARED_{i}"),
            "cases": cases.get(i, 0),
            "voxels": voxels.get(i, 0),
        })

    out = {"n_volumes": len(labels), "labels": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")

    unpopulated = [r["name"] for r in rows if r["cases"] == 0]
    undeclared = [r["id"] for r in rows if r["name"].startswith("UNDECLARED_")]
    print(f"  declared-but-empty : {len(unpopulated)} -> {unpopulated}")
    print(f"  present-but-undeclared: {undeclared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
