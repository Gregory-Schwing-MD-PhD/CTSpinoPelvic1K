"""update_manifest_0816.py -- make the manifest describe 0816 as it now is.

0816 was the release's only partial annotation: T11-L5 traced and 106,229,372 voxels of
`ignore` where every other record has sacrum, hips, femurs and ribs. It has now been
completed -- pelvis from the out-of-fold pseudolabeller, femurs/ribs/S1 from
TotalSegmentator with the Moller rib graft -- so four manifest fields are stale, and one
statement in the card is no longer true of any record at all.

Also recomputes the voxel counts, which read 0 for this record because they were written
before the fill.

    python update_manifest_0816.py --tree data/hf_export_osc --volume 0816
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

IGNORE = 255


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--volume", default="0816")
    a = ap.parse_args()

    tree = Path(a.tree)
    recs = json.loads((tree / "manifest.json").read_text())

    lab_path = tree / "labels" / f"{a.volume}_label.nii.gz"
    lab = np.asanyarray(nib.load(str(lab_path)).dataobj)
    ids = {int(v): int(n) for v, n in zip(*np.unique(lab, return_counts=True))}
    n_bg = ids.pop(0, 0)
    n_ign = ids.pop(IGNORE, 0)
    n_fg = int(sum(ids.values()))
    n_lumbar = sum(1 for i in ids if 20 <= i <= 25)
    has_anchor = 19 in ids                     # T12, the rostral counting anchor
    print(f"  {a.volume}: fg={n_fg:,}  bg={n_bg:,}  ignore={n_ign:,}  "
          f"lumbar={n_lumbar}  T12={has_anchor}")
    print(f"  ids: {sorted(ids)}")

    hit = 0
    for r in recs:
        if r.get("volume_id") != a.volume:
            continue
        hit += 1
        before = {k: r.get(k) for k in
                  ("partial_annotation", "prov_pelvis", "n_voxels_fg", "n_voxels_bg",
                   "n_voxels_ignore", "n_lumbar_labels", "has_anchor")}
        r["partial_annotation"] = False        # nothing is un-traced any more
        r["prov_pelvis"] = "pseudo"            # out-of-fold nnU-Net, not radiologist GT
        r["n_voxels_fg"] = n_fg
        r["n_voxels_bg"] = n_bg
        r["n_voxels_ignore"] = n_ign
        r["n_lumbar_labels"] = n_lumbar
        r["has_anchor"] = bool(has_anchor)
        after = {k: r.get(k) for k in before}
        for k in before:
            if before[k] != after[k]:
                print(f"    {k}: {before[k]!r} -> {after[k]!r}")

    if hit != 1:
        raise SystemExit(f"expected exactly 1 record for {a.volume}, found {hit}")

    # the claim that any record uses `ignore` must now be false everywhere
    still = [r["volume_id"] for r in recs if r.get("partial_annotation")]
    print(f"  records still marked partial_annotation: {still or 'none'}")

    (tree / "manifest.json").write_text(json.dumps(recs, indent=1))
    cols = []
    for r in recs:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(tree / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(f"  rewrote manifest.json + manifest.csv ({len(recs)} records, {len(cols)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
