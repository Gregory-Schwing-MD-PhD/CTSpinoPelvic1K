"""scripts/label_census.py — which identifiers actually occur, in how many records.

Several claims in the dataset article are about what the release contains, and every one
of them was written from intent rather than from the volumes: that soft-tissue (58-73) and
hardware (76-79) are declared but populated by no case, that 16 records carry a lumbar rib,
that L6 exists. The manifest cannot settle these -- its `has_l6` flag is true for exactly
one record, and that record has no L6 in it, while all 14 LUMBARIZATION records do.

A declared-but-empty class is not a defect as long as the paper says so. A class the paper
says is empty and is not, or a headline class resting on a broken flag, is a defect. This
counts them.

    python scripts/label_census.py --labels data/hf_export_v5/labels --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib


def ids_in(path):
    """-> (case, sorted unique ids). Reads the array only; no reorientation needed since
    the set of values present does not depend on axis order."""
    try:
        a = np.asanyarray(nib.load(str(path)).dataobj)
        return path.name.split("_")[0], sorted(int(v) for v in np.unique(a))
    except Exception as e:                                            # noqa: BLE001
        return path.name.split("_")[0], f"ERROR {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--scheme", default="data/hf_export_v5/dataset_labels.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="morphometrics/label_census.csv")
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    print(f"  {len(files)} label volume(s)")

    names = {}
    sp = Path(a.scheme)
    if sp.exists():
        names = {int(k): v for k, v in
                 json.loads(sp.read_text(encoding="utf-8"))["id_to_name"].items()}

    per_case, errs = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (case, ids) in enumerate(ex.map(ids_in, files, chunksize=4), 1):
            if isinstance(ids, str):
                errs.append((case, ids))
                continue
            per_case[case] = ids
            if i % 200 == 0:
                print(f"    {i}/{len(files)}")

    freq = Counter()
    for ids in per_case.values():
        freq.update(ids)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "records", "pct"])
        for i in sorted(freq):
            w.writerow([i, names.get(i, "?"), freq[i],
                        round(100 * freq[i] / max(1, len(per_case)), 2)])

    n = len(per_case)
    print(f"\n  {n} record(s) read, {len(errs)} unreadable")

    def block(lo, hi, title):
        present = {i: freq[i] for i in range(lo, hi + 1) if freq.get(i)}
        print(f"\n  {title} ({lo}--{hi})")
        if not present:
            print("    none present in any record")
            return
        for i in sorted(present):
            print(f"    {i:>3} {names.get(i,'?'):<22} {present[i]:>4} record(s) "
                  f"({100*present[i]/n:.1f}%)")

    block(20, 25, "lumbar")
    block(26, 29, "sacrum / coccyx / T13 / S1")
    block(30, 33, "hips and femora")
    block(74, 75, "lumbar ribs")
    block(58, 73, "soft tissue")
    block(76, 79, "hardware")
    print(f"\n  ignore (255): {freq.get(255, 0)} record(s)")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
