"""reconstruct_ct.py — rebuild the image half of CTSpinoPelvic1K from TCIA.

This deposit ships the LABELS, not the CT images. The images are already public in The
Cancer Imaging Archive, they are 193 GB against 1.8 GB of labels, and re-hosting them would
duplicate a primary archive for no benefit. What the source collections never published --
and what this deposit does -- is the mapping from each annotation to the CT series it
belongs on. `manifest.json` carries a `spine_series_uid` and, where one exists, a
`pelvic_series_uid` for every record: TCIA SeriesInstanceUIDs.

ONE STEP IS NOT OPTIONAL AND YOUR MASKS WILL NOT LINE UP WITHOUT IT. The released labels do
not sit on the grid of the DICOM series as you will download it. Each label was drawn on one
grid, and during export the CT was resampled onto THAT grid so that image and label share an
affine. Downloading the series and converting it to NIfTI gives you a volume with the right
anatomy on the wrong grid.

The fix is short, and everything it needs is already in the label file you downloaded: the
label's own affine IS the target. Resample your converted CT onto it, exactly as the export
did -- trilinear, and -1024 HU outside the original extent.

    python reconstruct_ct.py --label labels/0007_label.nii.gz \\
        --ct my_conversion/0007.nii.gz --out ct/0007_ct.nii.gz

Verify before you trust it: `--check` reports the fraction of bone voxels under the label,
which should be high. A low number means the wrong series, not a resampling problem, and no
amount of interpolation will fix it.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform

BONE_HU = 200.0


def resample_to(ct_img, ref_img):
    """CT resampled onto the reference grid. Same operation the export performed."""
    ref_shape = ref_img.shape[:3]
    ref_affine = ref_img.affine
    data = np.asarray(ct_img.dataobj, dtype=np.float32)
    if ct_img.shape[:3] == ref_shape and np.allclose(ct_img.affine, ref_affine, atol=1e-4):
        return nib.Nifti1Image(data, ref_affine)
    M = np.linalg.inv(ct_img.affine) @ ref_affine
    out = affine_transform(data, M[:3, :3], offset=M[:3, 3],
                           output_shape=ref_shape, order=1,
                           mode="constant", cval=-1024.0)
    return nib.Nifti1Image(out, ref_affine)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="released *_label.nii.gz")
    ap.add_argument("--ct", required=True, help="your NIfTI conversion of the TCIA series")
    ap.add_argument("--out", help="where to write the resampled CT")
    ap.add_argument("--check", action="store_true",
                    help="report how much of the label sits on bone")
    a = ap.parse_args()

    ref = nib.load(a.label)
    ct = nib.load(a.ct)
    res = resample_to(ct, ref)

    if a.check:
        lab = np.asanyarray(ref.dataobj)
        img = np.asarray(res.dataobj)
        m = (lab > 0) & (lab != 255)
        if not m.any():
            print("  ! the label is empty")
            return 1
        frac = float((img[m] >= BONE_HU).mean())
        print(f"  {frac * 100:.1f}% of labelled voxels are at or above {BONE_HU:.0f} HU")
        if frac < 0.5:
            print("  ! that is too low. This is almost certainly the wrong series for this")
            print("  ! record rather than a resampling error -- check the SeriesInstanceUID")
            print("  ! in manifest.json. Every patient here was scanned twice.")
        else:
            print("  consistent with the label sitting on this volume's bone")

    if a.out:
        nib.save(res, a.out)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
