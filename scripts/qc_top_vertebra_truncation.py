"""qc_top_vertebra_truncation.py — does the release label a top vertebra the FOV cuts?

THE QUESTION THIS ANSWERS. 0068 images 48 mm of column above L1: one whole vertebra (T12)
and a second (T11) sliced through its body by the top of the scan. Whether T11 should be
labelled is not a judgement to make case by case -- it is a convention, and the convention
is whatever the other 801 records already do. Labelling a truncated level here when the rest
of the release drops them, or dropping it when the rest keep them, makes 0068 the odd record
either way.

Only the LABEL is read. A vertebra whose mask reaches the last slice of the array along the
superior axis is cut by the edge of the reconstruction; no CT is needed to see that, which
keeps this cheap enough to run over the whole release.

The superior axis comes off each affine rather than being assumed: these volumes sit on disk
as ('P','I','R'), so axis 2 is not superior and a hard-coded axis would report the bottom of
every scan.

    python scripts/qc_top_vertebra_truncation.py --labels data/v5_final --out qc_final
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

V5_T1, V5_T12 = 8, 19
V5_L1, V5_L6 = 20, 25
NAME = {**{v: f"T{v - V5_T1 + 1}" for v in range(V5_T1, V5_T12 + 1)},
        **{v: f"L{v - V5_L1 + 1}" for v in range(V5_L1, V5_L6 + 1)},
        26: "sacrum", 28: "T13", 29: "S1"}


def superior_axis(affine):
    col = affine[2, :3]
    ax = int(np.argmax(np.abs(col)))
    return ax, int(np.sign(col[ax]) or 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    print(f"{len(files)} labels in {a.labels}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, f in enumerate(files):
        img = nib.load(str(f))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        ax, sgn = superior_axis(img.affine)
        zmm = float(np.linalg.norm(img.affine[:3, ax]))
        n = arr.shape[ax]
        edge = (n - 1) if sgn > 0 else 0
        others = tuple(i for i in range(3) if i != ax)

        verts = [v for v in np.unique(arr) if V5_T1 <= v <= V5_L6 or v in (28,)]
        if not verts:
            rows.append({"case": f.name[:4], "top_vertebra": "", "truncated": "",
                         "note": "no vertebra label"})
            continue
        # the most superior vertebra present
        best, best_i = None, None
        for v in verts:
            idx = np.where((arr == v).any(axis=others))[0]
            i = int(idx.max() if sgn > 0 else idx.min())
            if best_i is None or (i > best_i if sgn > 0 else i < best_i):
                best, best_i = int(v), i
        m = (arr == best)
        idx = np.where(m.any(axis=others))[0]
        lo, hi = int(idx.min()), int(idx.max())
        touches = (hi == edge) if sgn > 0 else (lo == edge)
        # how much of the scan sits above that vertebra's top -- a level that ends exactly at
        # the edge and a level with 20 mm of air above it are different situations
        head = abs(edge - best_i) * zmm
        rows.append({"case": f.name[:4],
                     "top_vertebra": NAME.get(best, str(best)),
                     "height_mm": round((hi - lo + 1) * zmm, 1),
                     "mm_above_it": round(head, 1),
                     "truncated": int(touches),
                     "note": "cut by the edge of the reconstruction" if touches else ""})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    dst = out / "top_vertebra_truncation.csv"
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "top_vertebra", "height_mm",
                                           "mm_above_it", "truncated", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    trunc = [r for r in rows if r.get("truncated") == 1]
    print(f"\nwrote {dst}")
    print(f"top vertebra is CUT BY THE FOV in {len(trunc)} of {len(rows)} records "
          f"({100.0 * len(trunc) / max(1, len(rows)):.1f}%)")
    if trunc:
        hs = sorted(r["height_mm"] for r in trunc)
        print(f"  their labelled height: median {hs[len(hs)//2]:.1f} mm, "
              f"range {hs[0]:.1f}-{hs[-1]:.1f} mm")
    from collections import Counter
    print("  which level ends up on top:",
          ", ".join(f"{v} x{c}" for v, c in
                    Counter(r["top_vertebra"] for r in rows if r.get("top_vertebra")).most_common(8)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
