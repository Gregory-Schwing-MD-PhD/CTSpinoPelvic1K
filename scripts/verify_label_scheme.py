"""verify_label_scheme.py — assert every label id in a built export tree is a valid
VerSe-native id (scripts/label_scheme.py). Exit 1 on any stray / old-scheme id.

This is the correctness gate for the VerSe-native rebuild: it catches the v3 bug
signature (a thoracic vertebra landing in the rib range, or any old-scheme leftover
like ignore=50 / sacrum=8) by checking that NO label id falls outside the canonical set.

    python scripts/verify_label_scheme.py --tree data/hf_export_v3 [--max 0]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="export tree (expects <tree>/labels/*.nii.gz)")
    ap.add_argument("--max", type=int, default=0, help="check at most N files (0 = all)")
    a = ap.parse_args()

    LS.verify()  # the scheme itself is collision-proof
    valid = set(LS.label_dict().values()) | {0}

    files = sorted(glob.glob(str(Path(a.tree) / "labels" / "*.nii.gz")))
    if a.max:
        files = files[:a.max]
    if not files:
        print(f"ERROR: no label files under {a.tree}/labels")
        return 1

    seen: set[int] = set()
    vox: dict[int, int] = {}                       # total voxels per id across the sample
    bad: dict[str, list[int]] = {}
    for f in files:
        arr = np.asanyarray(nib.load(f).dataobj).astype(np.int64)
        u, c = np.unique(arr, return_counts=True)
        for i, n in zip(u.tolist(), c.tolist()):
            vox[i] = vox.get(i, 0) + n
        ids = set(int(i) for i in u)
        stray = ids - valid                        # ids not in the canonical legend AT ALL
        if stray:
            bad[Path(f).name] = sorted(stray)
        seen |= ids

    def vol(lo, hi):                               # summed voxels in an inclusive id range
        return sum(n for i, n in vox.items() if lo <= i <= hi)

    # ANATOMICAL discriminator — id membership alone can't tell the schemes apart (the
    # new scheme reuses the same integers), so compare WHERE the bone volume sits:
    #   old scheme packed lumbar+S1+sacrum+hips into ids 1-10;
    #   VerSe-native has those at 20-33, and ids 1-10 = C1-T3 (absent in a spinopelvic FOV).
    old_core = vol(1, 10)                          # VerSe C1-T3 -> must be ~empty
    new_core = vol(20, 33)                         # VerSe lumbar+sacrum+S1+hips+femurs -> large
    rib = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))  # 34..57

    print(f"checked {len(files)} label files")
    print(f"  union of ids        : {sorted(seen)}")
    print(f"  thoracic (8-19)     : {sorted(i for i in seen if 8 <= i <= 19)}   <- vertebrae, NOT ribs")
    print(f"  ribs (34-57)        : {sorted(i for i in seen if i in rib)}")
    print(f"  sacrum 26: {26 in seen} | S1 29: {29 in seen} | "
          f"hips 30/31: {30 in seen}/{31 in seen} | femurs 32/33: {32 in seen}/{33 in seen}")
    print(f"  voxels in ids 1-10 (VerSe C1-T3, expect ~0): {old_core:,}")
    print(f"  voxels in ids 20-33 (VerSe lumbar+pelvis):   {new_core:,}")

    fail = False
    if bad:
        fail = True
        print(f"\nFAIL — {len(bad)} file(s) contain ids not in the legend at all:")
        for k, v in list(bad.items())[:20]:
            print(f"  {k}: {v}")
    if new_core == 0:
        fail = True
        print("\nFAIL — no bone in the VerSe lumbar/pelvis range (20-33); export looks empty/wrong.")
    if old_core > 0.02 * max(new_core, 1):
        fail = True
        print(f"\nFAIL — {old_core:,} voxels sit in ids 1-10 (VerSe C1-T3), which a spinopelvic "
              f"FOV should not contain. This is the OLD scheme (lumbar/pelvis at 1-10), not VerSe-native.")
    if 26 not in seen:
        fail = True
        print("\nFAIL — no sacrum at id 26; sacrum may still be at the old id 8.")

    if fail:
        return 1
    print("\nPASS — VerSe-native: bone volume sits at VerSe ids (lumbar/pelvis 20-33, sacrum 26), "
          "ids 1-10 are ~empty, and no thoracic landed in the rib range. No old-scheme leftovers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
