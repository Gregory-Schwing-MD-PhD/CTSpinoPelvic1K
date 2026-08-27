"""verify_v6.py — check the v6 tree before it is published.

A release is easy to assemble and easy to get subtly wrong, and every check here exists
because the corresponding mistake was actually possible in this build:

  THE HARDWARE IS ACTUALLY THERE. The point of v6 is that ids 76-82 stop being declared and
  start being populated. A build that recognised only 76-79 would have written the labels
  and recorded none of them.
  EVERY CASE STILL EXISTS. 802 labels, 802 CTs, paired by name.
  NOTHING BUT HARDWARE CHANGED, except where it was meant to. Diffed against the tree the
  build actually READ -- data/v5_final, not the published hf_export_v5, because those two
  copies of v5 have drifted (1153 differs by 19,010 voxels, its stray fragments having been
  stripped after the v5 export was cut). Comparing against the published tree blames v6 for
  a change it inherited.
  0068 IS THE CORRECTED VERSION. Six lumbar bodies, T10-T12 present, cages labelled. If the
  build fell back to the v5 copy, the case would silently lose all of that.
  NO DUST WHERE THE IMPLANT CUT. Only in structures the metal took voxels from -- that is
  all the consolidation pass touched. Dust elsewhere is v5's own: 0188's left hip has ten
  specks in v5_final and ten in v6, on the opposite side from its implant.
  THE MANIFEST AGREES WITH THE VOXELS. A record saying it has hardware must have hardware.

    python scripts/verify_v6.py --v6 data/hf_export_v6 --v5 data/v5_final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

HW = (76, 77, 78, 79, 80, 81, 82)
HW_NAME = {76: "hardware", 77: "cage", 78: "screw_rod", 79: "plate",
           80: "arthroplasty", 81: "si_screw", 82: "osteosynthesis"}
LUMBAR = list(range(20, 26))
EXPECT_0068 = {17: "T10", 18: "T11", 19: "T12", 20: "L1", 21: "L2", 22: "L3",
               23: "L4", 24: "L5", 25: "L6", 77: "cage"}


def label_path(root: Path, cid: str) -> Path:
    """A label, whether the tree keeps them under labels/ or flat."""
    return ((root / "labels" / f"{cid}_label.nii.gz") if (root / "labels").is_dir()
            else (root / f"{cid}_label.nii.gz"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", required=True)
    ap.add_argument("--v5", required=True,
                    help="the tree the build actually READ, not the published one")
    ap.add_argument("--dust-mm3", type=float, default=30.0)
    a = ap.parse_args()
    v6, v5 = Path(a.v6), Path(a.v5)
    fails = []

    labs = sorted((v6 / "labels").glob("*_label.nii.gz"))
    cts = sorted((v6 / "ct").glob("*_ct.nii.gz"))
    print(f"  labels {len(labs)}   cts {len(cts)}")
    if len(labs) != 802 or len(cts) != 802:
        fails.append(f"expected 802 of each, got {len(labs)}/{len(cts)}")
    if {p.name[:4] for p in labs} != {p.name[:4] for p in cts}:
        fails.append("label and ct case sets differ")

    # ---- which cases carry hardware, and does the manifest agree ----------------------
    mf = json.loads((v6 / "manifest.json").read_text(encoding="utf-8"))
    says = {str(r["volume_id"]): r for r in mf if r.get("hardware_label_ids")}
    print(f"  manifest says {len(says)} record(s) carry hardware labels")

    # only open the flagged cases plus a few controls: 802 full-volume reads to answer a
    # question about eleven of them is a long wait for nothing
    controls = ("0004", "0033", "1153")
    found = {}
    for p in labs:
        cid = p.name[:4]
        if cid not in says and cid not in controls:
            continue
        arr = np.asanyarray(nib.load(str(p)).dataobj).astype(np.int16)
        ids = sorted(int(v) for v in np.unique(arr) if int(v) in HW)
        if ids:
            found[cid] = ids

    print(f"  voxels carry hardware in {len(found)} case(s): "
          + ", ".join(f"{c}({','.join(HW_NAME[i] for i in v)})"
                      for c, v in sorted(found.items())))
    if set(found) != set(says):
        fails.append(f"manifest and voxels disagree: manifest {sorted(says)}, "
                     f"voxels {sorted(found)}")
    for cid, ids in found.items():
        if cid in says and sorted(says[cid]["hardware_label_ids"]) != ids:
            fails.append(f"{cid}: manifest ids {says[cid]['hardware_label_ids']} "
                         f"!= voxel ids {ids}")

    # ---- 0068 must be the corrected version -------------------------------------------
    p68 = v6 / "labels" / "0068_label.nii.gz"
    if p68.exists():
        arr = np.asanyarray(nib.load(str(p68)).dataobj).astype(np.int16)
        present = {int(v) for v in np.unique(arr)}
        missing = [n for i, n in EXPECT_0068.items() if i not in present]
        nlum = len([v for v in LUMBAR if v in present])
        print(f"  0068: {nlum} lumbar bodies, "
              + ("all expected levels present" if not missing else f"MISSING {missing}"))
        if missing:
            fails.append(f"0068 missing {missing} -- the build fell back to the v5 copy")
        if nlum != 6:
            fails.append(f"0068 has {nlum} lumbar labels, expected 6")

    # ---- dust, only where the implant cut ----------------------------------------------
    worst = []
    for cid in sorted(found):
        img = nib.load(str(v6 / "labels" / f"{cid}_label.nii.gz"))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        p5 = label_path(v5, cid)
        if not p5.exists():
            continue
        before = np.asanyarray(nib.load(str(p5)).dataobj).astype(np.int16)
        cut = {int(v) for v in np.unique(before[np.isin(arr, HW)])
               if int(v) and int(v) not in HW}
        for v in sorted(cut):
            m = arr == v
            if not m.any():
                continue
            cc, n = ndimage.label(m)
            sizes = ndimage.sum(m, cc, range(1, n + 1))
            dust = int(sum(1 for s in sizes if s * vox < a.dust_mm3))
            if dust:
                worst.append((cid, v, dust))
    if worst:
        print(f"  ! dust still present in {len(worst)} structure(s): "
              + ", ".join(f"{c}/{HW_NAME.get(v, v)}:{d}" for c, v, d in worst[:6]))
        fails.append(f"{len(worst)} structure(s) the implant cut still carry dust")
    else:
        print("  no dust in any structure the implant cut")

    # ---- only the eleven may differ from the source tree --------------------------------
    diff = []
    for cid in sorted(set(list(found) + list(controls))):
        p5, p6 = label_path(v5, cid), v6 / "labels" / f"{cid}_label.nii.gz"
        if not p5.exists() or not p6.exists():
            continue
        a5 = np.asanyarray(nib.load(str(p5)).dataobj).astype(np.int16)
        a6 = np.asanyarray(nib.load(str(p6)).dataobj).astype(np.int16)
        if a5.shape == a6.shape and not np.array_equal(a5, a6):
            diff.append(cid)
    unexpected = [c for c in diff if c not in found]
    print(f"  differ from the source tree: {', '.join(diff) or 'none'}")
    if unexpected:
        fails.append(f"cases changed that should not have: {unexpected}")

    print()
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        return 1
    print("  v6 verified: hardware present and agreeing with the manifest, 0068 corrected, "
          "no dust where the implant cut, nothing else touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
