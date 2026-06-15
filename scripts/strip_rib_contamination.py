"""
strip_rib_contamination.py — undo the v3-rib hardlink corruption of v2 labels.

A hardlink bug in build_v3_ribs let v3's rib merge write rib classes (10..33) into
v2 label files on disk. Ribs were merged onto BACKGROUND only, and the completed v2
labelmap is strictly 0..9, so every voxel with value > 9 is contamination and
resetting it to 0 restores the original label exactly. Idempotent; safe to re-run.

The write unlinks the target first, so even if a v2 label is still hardlinked to its
v3 twin, cleaning breaks the link (v2 gets a fresh, clean inode) instead of writing
through it.

Usage:
  python scripts/strip_rib_contamination.py --labels_dir data/hf_export_v2/labels
  # --max_valid defaults to 9 (the highest valid completed-v2 class, right_hip)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_dir", required=True, type=Path)
    ap.add_argument("--max_valid", type=int, default=9,
                    help="highest valid label in the completed v2 scheme (default 9)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    files = sorted(args.labels_dir.glob("*.nii.gz"))
    n_fixed = total_bad = 0
    for f in files:
        img = nib.load(str(f))
        a = np.asarray(img.dataobj)
        bad = a > args.max_valid
        nb = int(bad.sum())
        if nb == 0:
            continue
        total_bad += nb
        n_fixed += 1
        print(f"  {f.name}: {nb} contaminated voxel(s) "
              f"(values {sorted(int(v) for v in np.unique(a[bad]))})")
        if args.dry_run:
            continue
        cleaned = a.copy()
        cleaned[bad] = 0
        f.unlink()                       # break any hardlink to the v3 twin first
        nib.save(nib.Nifti1Image(cleaned.astype(np.uint16), img.affine, img.header),
                 str(f))

    verb = "would clean" if args.dry_run else "cleaned"
    print(f"done: {verb} {n_fixed}/{len(files)} file(s); {total_bad} contaminated voxel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
