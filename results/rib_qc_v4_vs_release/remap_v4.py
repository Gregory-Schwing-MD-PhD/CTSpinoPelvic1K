"""Rewrite the v4 pseudolabels in the release (v10) id scheme so the same QC code reads both.

v4 numbered right ribs 45+n (46..57) and lumbar ribs 58/59; v10 numbers them 46+n (47..59)
and 60/61 (rib 13 was inserted). Left ribs (33+n) and everything else are identical. The
affine and header are copied untouched (never reorient a label on write).

    python remap_v4.py --src ~/tmp/v4_labels/labels --dst ~/tmp/v4_labels_v10/labels --workers 32
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

LUT = np.arange(256, dtype=np.int16)
for n in range(1, 13):
    LUT[45 + n] = 46 + n            # right rib n
LUT[58], LUT[59] = 60, 61           # lumbar ribs


def one(args):
    src, dst = args
    im = nib.load(str(src)); lab = np.asanyarray(im.dataobj).astype(np.int16)
    out = LUT[np.clip(lab, 0, 255)].astype(lab.dtype)
    nib.save(nib.Nifti1Image(out, im.affine, im.header), str(dst))
    return src.name, int((lab != out).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    jobs = [(p, dst / p.name) for p in sorted(src.glob("*.nii.gz")) if not (dst / p.name).exists()]
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for name, changed in ex.map(one, jobs, chunksize=4):
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} {name} changed {changed}", flush=True)
    print(f"remapped {n} labels -> {dst}")


if __name__ == "__main__":
    main()
