"""scripts/compare_label_trees.py — are two label trees voxel-identical?

Byte checksums cannot answer this: a gzip stream carries a timestamp, so the same voxels
written twice give two different files. This loads each pair and compares shape, affine
and every voxel. Used to prove that a chain of remaps regenerates the released volumes.

    python scripts/compare_label_trees.py --a regenerated/ --b released/labels --workers 8 --out chain_check.txt
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


def one(pair):
    a, b = pair
    try:
        ia, ib = nib.load(str(a)), nib.load(str(b))
        if ia.shape != ib.shape:
            return a.name, f"shape {ia.shape} vs {ib.shape}"
        if not np.allclose(ia.affine, ib.affine, atol=1e-3):
            return a.name, "affine differs"
        va = np.asanyarray(ia.dataobj)
        vb = np.asanyarray(ib.dataobj)
        if not np.array_equal(va, vb):
            n = int((va != vb).sum())
            ids = sorted(set(np.unique(va[va != vb]).tolist()) | set(np.unique(vb[va != vb]).tolist()))
            return a.name, f"{n} voxels differ, identifiers {ids[:12]}"
        return a.name, "identical"
    except Exception as e:  # noqa: BLE001
        return a.name, f"error {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    fa = sorted(A.glob("*.nii.gz"))
    pairs, missing = [], []
    for f in fa:
        g = B / f.name
        (pairs if g.exists() else missing).append((f, g))
    only_b = sorted(p.name for p in B.glob("*.nii.gz") if not (A / p.name).exists())
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, pairs, chunksize=4))
    bad = [(n, r) for n, r in res if r != "identical"]
    lines = [f"compared {len(res)} pairs: {len(res) - len(bad)} identical, {len(bad)} differ",
             f"only in {A}: {len(missing)}    only in {B}: {len(only_b)}"]
    lines += [f"  {n}: {r}" for n, r in bad]
    lines += [f"  only in a: {p[0].name}" for p in missing[:20]]
    lines += [f"  only in b: {n}" for n in only_b[:20]]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if not bad and not missing and not only_b else 1


if __name__ == "__main__":
    sys.exit(main())
