"""Did re-deriving hip laterality shred the labels it corrected?

relabel_hips_by_midline assigns EVERY hip voxel by which side of the midline it falls on.
That is a per-voxel decision with no connectivity constraint, so where the two hip bones
approach each other -- at the pubic symphysis in front, across the sacroiliac joints behind --
a ragged boundary can leave a spray of single-voxel islands of each label stranded inside the
other. The fix would then be correct in bulk and worse in detail.

The recount that just ran makes this a live question: right_hip and left_hip are the two
commonest labels among detached pieces, by a wide margin.

The backup taken before the relabel makes it answerable directly. Same eighteen records, same
component counting, before against after. Anything the fix introduced shows up as a
difference; anything that was already there cancels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

HIPS = (30, 31)


def pieces(path, ids):
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    out = {}
    for v in ids:
        m = arr == v
        if not m.any():
            continue
        lab, n = ndi.label(m)
        sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
        out[v] = {"n": int(n), "total": int(m.sum()),
                  "largest_frac": float(sizes[0] / m.sum()),
                  "tiny": int((sizes < 100).sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    a = ap.parse_args()

    tb = ta = nb = na = 0
    print(f"  {'case':<6} {'':>3} {'before: n / tiny / largest':<30} "
          f"{'after: n / tiny / largest':<30}")
    for cid in a.cases:
        b = pieces(Path(a.before) / f"{cid}_label.nii.gz", HIPS)
        c = pieces(Path(a.after) / "labels" / f"{cid}_label.nii.gz", HIPS)
        for v in HIPS:
            if v not in b and v not in c:
                continue
            bb, cc = b.get(v, {}), c.get(v, {})
            nb += bb.get("n", 0); na += cc.get("n", 0)
            tb += bb.get("tiny", 0); ta += cc.get("tiny", 0)
            print(f"  {cid:<6} {v:>3} "
                  f"{bb.get('n', 0):>6} /{bb.get('tiny', 0):>5} /"
                  f"{bb.get('largest_frac', 0):>7.1%}{'':<9}"
                  f"{cc.get('n', 0):>6} /{cc.get('tiny', 0):>5} /"
                  f"{cc.get('largest_frac', 0):>7.1%}")

    print(f"\n  hip components   before {nb:>7,}   after {na:>7,}   "
          f"change {na - nb:+,}")
    print(f"  pieces <100 vox  before {tb:>7,}   after {ta:>7,}   "
          f"change {ta - tb:+,}")
    print()
    if na > nb * 1.5:
        print("  THE FIX FRAGMENTED THE LABELS -- it needs a connectivity cleanup before it ships")
    elif na < nb:
        print("  the fix left the labels less fragmented than it found them")
    else:
        print("  fragmentation is essentially unchanged; the speckle predates the fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
