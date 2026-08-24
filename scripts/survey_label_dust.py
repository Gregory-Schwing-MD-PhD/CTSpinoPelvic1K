"""How much of the release is dust: labels made of many pieces none of which is a structure.

The existing connectivity check in qc_version_progression.py cannot see this. It counts a
label as fragmented only when TWO OR MORE pieces each clear MIN_VOX=200, so a label that is
233 voxels scattered over 138 fragments -- no piece larger than nine voxels -- passes as
intact. That is not a hypothetical: it is what case 1153 carries on rib_left_12, under a
gallery card asserting the opposite.

The measure here is the share of a label's voxels living in its largest piece. A real
structure is one piece and scores 1.0. A structure with a speck beside it scores 0.99. A
label that is only dust scores near zero however many voxels it has in total, which is the
case the old test lets through.

Reorientation is not performed: connected components do not care about axis order, and
as_closest_canonical would cost four seconds a volume to rewrite 160M voxels for nothing.

    python scripts/survey_label_dust.py --labels data/hf_export_v5/labels --out qc_final/label_dust.csv
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
from scipy import ndimage

VERT = set(range(8, 30)) - {27}
RIBS = set(range(34, 58))
LUMB = {74, 75}
CHECK = VERT | RIBS | LUMB

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"


def one(path: Path) -> list[dict]:
    case = path.name.split("_")[0]
    try:
        lab = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception as e:                                    # a read failure is a finding
        return [{"case": case, "label": "READ_ERROR", "id": -1, "voxels": 0,
                 "components": 0, "largest": 0, "largest_frac": 0.0, "note": str(e)[:80]}]
    counts = np.bincount(lab.reshape(-1), minlength=256)
    out = []
    for v in sorted(CHECK):
        tot = int(counts[v])
        if not tot:
            continue
        # CROP BEFORE LABELLING. ndimage.label on the whole 160M-voxel array, thirty times
        # per case, is what made the first run appear to hang. A label occupies a tiny box;
        # connected components inside that box are the same components.
        m = lab == v
        idx = np.argwhere(m)
        sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(idx.min(0), idx.max(0)))
        lt, n = ndimage.label(m[sl])
        if n == 1:
            big = tot
        else:
            big = int(ndimage.sum(m[sl], lt, range(1, n + 1)).max())
        out.append({"case": case, "label": NAME.get(v, str(v)), "id": v, "voxels": tot,
                    "components": n, "largest": big,
                    "largest_frac": round(big / tot, 4), "note": ""})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--out", default="qc_final/label_dust.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if not files:
        print(f"  ! no labels under {a.labels}")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    cols = ["case", "label", "id", "voxels", "components", "largest", "largest_frac", "note"]
    done = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for rows in ex.map(one, files, chunksize=2):
                w.writerows(rows)
                done += 1
                if done % 5 == 0:
                    fh.flush()
                    print(f"  {done}/{len(files)}", flush=True)
    print(f"  wrote {a.out}  ({done} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
