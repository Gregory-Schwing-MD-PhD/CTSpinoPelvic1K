"""Are the four flagged records really transposed, and if so, what says so besides centroids?

`transposed` is decided by the order of the two hip centroids along the left-right axis. That
is sound but it is one line of evidence, and relabelling a bone on one line of evidence is how
the original fault got in. Before touching anything, get a second opinion that does not share
the first one's assumptions.

The femurs are that second opinion. Each femur sits in its own hip's socket, so the femur on
the anatomical left is beside the hip on the anatomical left -- and the sidedness sweep passed
every femur pair in the release. If a record's hips disagree with its own femurs about which
side is which, that is not a threshold being crossed, it is the two labels being swapped, and
the femurs say which way round they belong.

Prints the evidence and, with --fix, swaps 30 and 31 in the volumes where both lines agree.
Writes in the original frame -- never canonicalised, never reoriented.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33


def centre(arr, v, axis):
    m = arr == v
    return float(np.argwhere(m)[:, axis].mean()) if m.any() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--backup", default=None)
    a = ap.parse_args()
    src = Path(a.src)

    agree, disagree = [], []
    for cid in a.cases:
        p = src / "labels" / f"{cid}_label.nii.gz"
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
        codes = nib.aff2axcodes(img.affine)
        ax = next((i for i, k in enumerate(codes) if k in ("L", "R")), None)
        if ax is None:
            print(f"  {cid}: no left-right axis in {codes}")
            continue
        # in an 'R' axis the index grows toward the patient's right
        toward_right = codes[ax] == "R"
        hl, hr = centre(arr, HIP_L, ax), centre(arr, HIP_R, ax)
        fl, fr = centre(arr, FEM_L, ax), centre(arr, FEM_R, ax)

        def sided_ok(left_c, right_c):
            if left_c is None or right_c is None:
                return None
            return (right_c > left_c) if toward_right else (right_c < left_c)

        hips_ok, fems_ok = sided_ok(hl, hr), sided_ok(fl, fr)
        # does each hip sit beside the femur that shares its name?
        beside = None
        if None not in (hl, hr, fl, fr):
            same = abs(hl - fl) + abs(hr - fr)
            crossed = abs(hl - fr) + abs(hr - fl)
            beside = "its own femur" if same < crossed else "THE OTHER femur"

        print(f"  {cid}  axes {codes}, left-right is axis {ax} ({codes[ax]})")
        print(f"    hip centres    left {hl!s:>8.8}  right {hr!s:>8.8}   "
              f"correctly ordered: {hips_ok}")
        print(f"    femur centres  left {fl!s:>8.8}  right {fr!s:>8.8}   "
              f"correctly ordered: {fems_ok}")
        print(f"    each hip lies beside {beside}")

        if hips_ok is False and fems_ok is True and beside == "THE OTHER femur":
            agree.append(cid)
            print("    VERDICT transposed -- centroid order and the femurs both say so")
        else:
            disagree.append(cid)
            print("    VERDICT not clear cut -- left alone")

    print(f"\n  transposed on both lines of evidence: {len(agree)}  {agree}")
    if disagree:
        print(f"  needing a look rather than a swap : {len(disagree)}  {disagree}")

    if a.fix and agree:
        for cid in agree:
            p = src / "labels" / f"{cid}_label.nii.gz"
            if a.backup:
                Path(a.backup).mkdir(parents=True, exist_ok=True)
                shutil.copy(p, Path(a.backup) / p.name)
            img = nib.load(str(p))
            arr = np.asanyarray(img.dataobj).copy()
            l, r = arr == HIP_L, arr == HIP_R
            arr[l], arr[r] = HIP_R, HIP_L
            # the original header and affine, untouched: reorienting a label silently
            # transposes it away from the CT it was drawn on
            out = nib.Nifti1Image(arr, img.affine, img.header)
            out.set_data_dtype(img.get_data_dtype())
            nib.save(out, str(p))
            print(f"    {cid}: swapped {l.sum():,} and {r.sum():,} voxels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
