"""scripts/fix_hip_sidedness.py — swap left_hip and right_hip where they are transposed.

THE DEFECT. Three released records -- 0027, 0107, 0935 -- carry `left_hip` (30) on the
patient's right and `right_hip` (31) on the patient's left. Their ribs and femora are
correctly sided, so within a single record the label called the left hip sits directly above
the correctly-labelled right femur.

WHY IT SURVIVED TO RELEASE. check_release_invariants.py tested the RIBS only, so it passed
all 802 and the paper claimed 802/802 on that basis. The version-progression QC pooled ribs,
hips and femora into one centroid per side and flagged these three, which is how they
surfaced -- but a pooled statistic cannot say WHICH pair is wrong, and can equally be
dragged wrong by a large correct structure. Both checks are fixed: the invariant check now
compares each sided pair separately.

WHAT THIS DOES, AND WHAT IT REFUSES TO DO. It swaps ids 30 and 31 in place, and only after
confirming from the affine that they really are transposed and that the femora on the same
record are NOT -- because if both pairs are reversed the volume is mirrored and swapping
labels would paper over that instead of fixing it. Nothing is written unless the check says
so. The array itself is never reoriented: only the two id values are exchanged, so the
geometry that was verified stays exactly as it was.

    python scripts/fix_hip_sidedness.py --labels data/hf_export_v5/labels --check
    python scripts/fix_hip_sidedness.py --labels data/hf_export_v5/labels --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33


def sided(lab, ax, codes, left_id, right_id):
    """-> (left centroid, right centroid, is_transposed) or None if either is absent."""
    lm = lab == left_id
    rm = lab == right_id
    if not (lm.any() and rm.any()):
        return None
    lc = float(np.argwhere(lm)[:, ax].mean())
    rc = float(np.argwhere(rm)[:, ax].mean())
    wrong = (lc >= rc) if codes[ax] == "R" else (lc <= rc)
    return lc, rc, wrong


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="default: scan every case")
    ap.add_argument("--apply", action="store_true", help="write the fix")
    ap.add_argument("--backup", default=None, help="directory for pre-fix copies")
    a = ap.parse_args()

    d = Path(a.labels)
    files = ([d / f"{c}_label.nii.gz" for c in a.cases] if a.cases
             else sorted(d.glob("*_label.nii.gz")))
    print(f"  scanning {len(files)} record(s)")

    to_fix, mirrored = [], []
    for f in files:
        img = nib.load(str(f))
        lab = np.asanyarray(img.dataobj)
        codes = nib.aff2axcodes(img.affine)
        lr = [i for i, k in enumerate(codes) if k in ("L", "R")]
        if not lr:
            continue
        ax = lr[0]
        hip = sided(lab, ax, codes, HIP_L, HIP_R)
        fem = sided(lab, ax, codes, FEM_L, FEM_R)
        if hip is None or not hip[2]:
            continue
        case = f.name.split("_")[0]
        if fem is not None and fem[2]:
            # both pairs reversed: the volume is mirrored, not mislabelled
            mirrored.append(case)
            print(f"  ! {case}: hips AND femora both reversed -- this is a mirrored volume, "
                  f"not a label swap. Refusing to touch it.")
            continue
        to_fix.append((f, case, hip, fem))
        print(f"  {case}: hips transposed (left {hip[0]:.0f}, right {hip[1]:.0f}; axis {ax} "
              f"grows toward {codes[ax]}), femora correct")

    print(f"\n  {len(to_fix)} record(s) to fix, {len(mirrored)} refused")
    if not a.apply:
        print("  --check only; pass --apply to write")
        return 0

    for f, case, _, _ in to_fix:
        if a.backup:
            Path(a.backup).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, Path(a.backup) / f.name)
        img = nib.load(str(f))
        lab = np.asanyarray(img.dataobj).copy()
        l_mask = lab == HIP_L
        r_mask = lab == HIP_R
        lab[l_mask] = HIP_R
        lab[r_mask] = HIP_L
        # NEVER REORIENT ON WRITE. Only the two id values change; the affine, the header
        # dtype and the array order are the ones that were already verified.
        out = nib.Nifti1Image(lab, img.affine, img.header)
        out.header.set_data_dtype(np.int16)
        nib.save(out, str(f))
        # re-read and confirm rather than trusting the write
        chk = nib.load(str(f))
        cl = np.asanyarray(chk.dataobj)
        cc = nib.aff2axcodes(chk.affine)
        cax = [i for i, k in enumerate(cc) if k in ("L", "R")][0]
        res = sided(cl, cax, cc, HIP_L, HIP_R)
        state = "still transposed" if (res and res[2]) else "correct"
        print(f"  {case}: written, hips now {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
