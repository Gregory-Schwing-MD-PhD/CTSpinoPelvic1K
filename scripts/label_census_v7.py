"""label_census_v7.py -- which identifiers occur in how many released volumes.

Counted from the deposited v7 labels themselves (data/zenodo_deposit/labels, SHA-verified
against the published SHA256SUMS.txt), because morphometrics/label_census.csv predates the
hardware labels and the 0816 completion and disagrees with the release on both.

FAST PATH. np.unique on a 100-million-voxel volume sorts it; np.bincount on the same
uint8/uint16 data is a single pass and two orders of magnitude quicker. The first version
of this script used np.unique and was still running after ninety minutes.

Names come from label_scheme.label_dict(), the one map, so the census cannot carry a name
the scheme does not.

    python scripts/label_census_v7.py [--labels data/zenodo_deposit/labels]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--out", default="morphometrics/label_census_v7.csv")
    a = ap.parse_args()
    files = sorted(Path(a.labels).glob("*.nii.gz"))
    counts = np.zeros(256, np.int64)
    t0 = time.time()
    for i, f in enumerate(files, 1):
        arr = np.asanyarray(nib.load(str(f)).dataobj)
        arr = arr.astype(np.uint8, copy=False) if arr.max() < 256 else arr.astype(np.int64)
        present = np.bincount(arr.ravel(), minlength=256)[:256] > 0
        counts[:len(present)] += present
        if i % 50 == 0:
            print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    n = len(files)
    name = {v: k for k, v in LS.label_dict().items()}
    out = Path(a.out)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "name", "records", "pct"])
        for v in np.nonzero(counts)[0]:
            w.writerow([int(v), name.get(int(v), "UNNAMED"), int(counts[v]), f"{100 * counts[v] / n:.2f}"])
    print(f"wrote {out}: {int((counts > 0).sum())} ids across {n} volumes in {time.time() - t0:.0f}s")
    retired = [int(v) for v in np.nonzero(counts)[0] if 58 <= v <= 73]
    print("retired 58-73 present:", retired or "none", "| sentinel 255:", int(counts[255]))
    print("hardware:", {int(v): int(counts[v]) for v in range(76, 83) if counts[v]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
