"""scripts/merge_s1_into_sacrum.py -- dissolve the carved S1 back into the sacrum.

WHY THIS IS BEING DONE. The S1 label is an automatic estimate of the S1--S2 boundary, and
on a substantial share of records the plane fit is degenerate: instead of cutting across the
body it comes out nearly parallel to the sacral long axis and shaves the ventral cortex,
leaving "S1" as a thin anterior sliver plus detached fragments, and leaving the SACRUM label
missing its entire anterior portion. Case 0428 is the worked example, in
docs/qc/s1_carve_0428_vs_0704.png. A sacrum with its anterior cortex removed is wrong for
every downstream use, not only for the parameters derived from the endplate.

Merging restores the pre-carve sacrum exactly, because the carve only ever partitioned
sacral voxels -- which scripts/check_s1_merge.py verifies rather than assumes, by checking
what S1 borders.

WHAT IS DELIBERATELY NOT DONE: RENUMBERING. Removing id 29 leaves a hole in a scheme
documented as contiguous, and closing it by shifting 30-68 down one would rename the hips,
femora and all twenty-six ribs. Every consumer keyed to the released ids -- checkpoints, the
benchmark, ostk's scheme detection, the level tables -- would silently read the wrong
structure, which is precisely the failure mode this project keeps paying for. Id 29 is
retired instead: no voxel carries it, and nothing else moves.

    python scripts/merge_s1_into_sacrum.py --src data/zenodo_deposit/labels \\
        --dst data/v11_labels --write

Without --write it verifies and reports, touching nothing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

S1, SACRUM = 29, 26


def merge_one(src: Path, dst: Path, write: bool):
    """Merge in the STORED orientation. Never canonicalise on a path that writes.

    as_closest_canonical followed by nib.save transposes the label away from its image; the
    repository has shipped that bug once already. Relabelling is orientation-independent, so
    the array is edited exactly as stored and the original affine and header go back out
    untouched.
    """
    img = nib.load(str(src))
    a = np.asanyarray(img.dataobj)

    n_s1 = int((a == S1).sum())
    n_sac = int((a == SACRUM).sum())
    other_before = {int(v): int(c) for v, c in zip(*np.unique(a, return_counts=True))}

    if n_s1:
        a = a.copy()
        a[a == S1] = SACRUM

    n_sac_after = int((a == SACRUM).sum())
    ok = (n_sac_after == n_sac + n_s1) and int((a == S1).sum()) == 0

    # nothing but 26 and 29 may have changed
    after = {int(v): int(c) for v, c in zip(*np.unique(a, return_counts=True))}
    moved = []
    for v, c in other_before.items():
        if v in (S1, SACRUM):
            continue
        if after.get(v, 0) != c:
            moved.append(v)

    if write:
        out = nib.Nifti1Image(a.astype(img.get_data_dtype()), img.affine, img.header)
        out.set_data_dtype(img.get_data_dtype())
        nib.save(out, str(dst))

    return dict(n_s1=n_s1, n_sac=n_sac, n_sac_after=n_sac_after, ok=ok, moved=moved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/zenodo_deposit/labels")
    ap.add_argument("--dst", default="data/v11_labels")
    ap.add_argument("--scheme", default="data/zenodo_deposit/dataset_labels.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(Path(a.src).glob("*_label.nii.gz"))
    if a.limit:
        files = files[: a.limit]
    dst_dir = Path(a.dst)
    if a.write:
        dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} records: {a.src} -> {a.dst}"
          f"{'' if a.write else '   (dry run)'}\n")

    n_with_s1 = 0
    failures, moved_any = [], []
    tot_s1 = 0
    for i, p in enumerate(files, 1):
        case = p.name.replace("_label.nii.gz", "")
        r = merge_one(p, dst_dir / p.name, a.write)
        tot_s1 += r["n_s1"]
        if r["n_s1"]:
            n_with_s1 += 1
        if not r["ok"]:
            failures.append(case)
        if r["moved"]:
            moved_any.append((case, r["moved"]))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    print(f"\n{n_with_s1} of {len(files)} records carried an S1 label")
    print(f"{tot_s1:,} voxels returned to the sacrum")
    print(f"count check failed on: {failures or 'none'}")
    print(f"labels other than 26/29 changed in: {moved_any or 'none'}")

    if a.write:
        # the scheme file, with 29 retired rather than reused
        sp = Path(a.scheme)
        if sp.exists():
            d = json.loads(sp.read_text())
            d["scheme"] = d.get("scheme", "").replace("v10", "v11") + "; S1 (29) retired"
            d.setdefault("retired_ids", {})["29"] = (
                "S1 -- an automatic S1/S2 plane estimate. Withdrawn: the fit was degenerate "
                "on a substantial share of records, shaving the ventral sacral cortex and "
                "leaving the sacrum label anatomically incomplete. Merged back into 26.")
            d["id_to_name"] = {k: v for k, v in d["id_to_name"].items() if k != "29"}
            out = dst_dir.parent / "dataset_labels_v11.json"
            out.write_text(json.dumps(d, indent=1))
            print(f"\nwrote {out}")
    else:
        print("\n(dry run; pass --write to produce the merged labels)")
    return 1 if failures or moved_any else 0


if __name__ == "__main__":
    sys.exit(main())
