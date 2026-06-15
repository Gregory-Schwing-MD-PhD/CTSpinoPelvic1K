"""
relabel_hips_by_midline.py — fix merged/swapped L/R hip classes geometrically.

The pelvis model frequently dumps both hips into one class (e.g. nearly all of the
hip bone becomes class 8/left), so the BONE is segmented correctly but the laterality
class is wrong. Because the two hips sit lateral to the midline spine/sacrum, we can
re-derive the side of every hip voxel from geometry alone: compute the midline world-X
(RAS) from the lumbar column (or sacrum) centroid, then relabel each hip voxel
(class 8 or 9) by which side of that midline it falls on:

    world-X > midline  ->  right_hip (9)     (RAS +X points to patient RIGHT)
    world-X < midline  ->  left_hip  (8)

Deterministic, orientation-robust (works through the affine, any stored orientation),
and idempotent: a case the model already lateralized correctly is unchanged. No
registration, no model. The write unlinks the target first so it never writes through
a hardlink into a sibling tree.

Usage:
  python scripts/relabel_hips_by_midline.py --labels_dir data/hf_export_v2/labels
  # test one case, see counts only: --tokens 0267 --dry_run
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

LEFT_HIP, RIGHT_HIP, SACRUM = 8, 9, 7
LUMBAR = (1, 2, 3, 4, 5, 6)


def _midline_world_x(lbl: np.ndarray, affine: np.ndarray):
    """World (RAS) X of the patient midline: lumbar-column centroid if present, else
    sacrum. Returns None if neither is available (cannot lateralize)."""
    for sel in (np.isin(lbl, LUMBAR), lbl == SACRUM):
        if sel.any():
            com_ijk = np.array(np.nonzero(sel)).mean(axis=1)   # (i,j,k) centroid
            return float(apply_affine(affine, com_ijk)[0])     # world X
    return None


def lateralize_hips(lbl: np.ndarray, affine: np.ndarray):
    """Return (relabelled, n_changed): every hip voxel reassigned to 8/9 by its side
    of the midline. Leaves the label untouched if there are no hips or no midline ref."""
    hips = (lbl == LEFT_HIP) | (lbl == RIGHT_HIP)
    if not hips.any():
        return lbl, 0
    mx = _midline_world_x(lbl, affine)
    if mx is None:
        return lbl, 0
    idx = np.array(np.nonzero(hips))                # (3, N) voxel coords
    world_x = apply_affine(affine, idx.T)[:, 0]     # (N,) world X per hip voxel
    new_vals = np.where(world_x > mx, RIGHT_HIP, LEFT_HIP).astype(lbl.dtype)
    out = lbl.copy()
    out[tuple(idx)] = new_vals
    return out, int((out != lbl).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_dir", required=True, type=Path)
    ap.add_argument("--tokens", default="", help="comma-separated subset (matched in filename)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    want = {t for t in re.split(r"[,:;\s]+", args.tokens.strip()) if t} or None
    files = sorted(args.labels_dir.glob("*.nii.gz"))
    n_changed_files = 0
    for f in files:
        if want is not None and not any(t in f.name for t in want):
            continue
        img = nib.load(str(f))
        lbl = np.asarray(img.dataobj).astype(np.int16)
        before = {c: int((lbl == c).sum()) for c in (LEFT_HIP, RIGHT_HIP)}
        out, n = lateralize_hips(lbl, img.affine)
        if n == 0:
            continue
        after = {c: int((out == c).sum()) for c in (LEFT_HIP, RIGHT_HIP)}
        n_changed_files += 1
        print(f"  {f.name}: {n} hip voxel(s) reassigned | "
              f"L {before[LEFT_HIP]}->{after[LEFT_HIP]}  "
              f"R {before[RIGHT_HIP]}->{after[RIGHT_HIP]}")
        if args.dry_run:
            continue
        f.unlink()                              # break any hardlink before writing
        nib.save(nib.Nifti1Image(out.astype(np.uint16), img.affine, img.header), str(f))

    verb = "would reassign" if args.dry_run else "reassigned"
    print(f"done: {verb} hips in {n_changed_files} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
