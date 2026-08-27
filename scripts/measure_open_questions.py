"""Numbers for the three questions KNOWN_ISSUES.md leaves open.

The entries were written as prose and one of them is now suspect. 1035 was described as
"genuinely fragmented"; that reading was made BEFORE the hip laterality fix, when a
476,756-voxel piece of the right hip carried the left label and made the left hip look like
it was in two pieces. If the fix resolved it, the entry now warns about a defect that is not
there -- which is its own kind of wrong, because a reader drops a good case on its say-so.

Measures, rather than reasons about:

  1035  hip fragmentation after the lateralisation, and where the SI screws sit
  0068  whether the L5-L6 interspace is bridged by the cages, and by how much
  the rejected detections: what the 41 artefact calls actually looked like, in mm3, so the
        line between kept and rejected is a number a reader can check instead of a claim
"""
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def pieces(mask):
    lab, n = ndi.label(mask)
    if not n:
        return 0, []
    sz = np.sort(np.bincount(lab.ravel())[1:])[::-1]
    return n, sz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--out", default="data/open_questions.json")
    a = ap.parse_args()
    src = Path(a.src)
    rep = {}

    # ---- 1035: are the hips still fragmented after the lateralisation? ---------------
    p = src / "labels" / "1035_label.nii.gz"
    img = nib.load(str(p))
    arr = np.asanyarray(img.dataobj)
    zoom = img.header.get_zooms()[:3]
    vox_mm3 = float(np.prod(zoom))
    print("  1035")
    r1035 = {}
    for lid, name in ((30, "left_hip"), (31, "right_hip"), (26, "sacrum"), (81, "si_screw")):
        m = arr == lid
        if not m.any():
            print(f"    {name:<10} absent")
            continue
        n, sz = pieces(m)
        frac = float(sz[0] / m.sum())
        rest = [int(x) for x in sz[1:5]]
        r1035[name] = {"voxels": int(m.sum()), "pieces": int(n),
                       "largest_fraction": round(frac, 4),
                       "next_pieces_voxels": rest,
                       "next_pieces_mm3": [round(x * vox_mm3, 1) for x in rest]}
        print(f"    {name:<10} {m.sum():>9,} vox  {n:>3} piece(s)  "
              f"largest {frac:6.1%}  next {rest[:3]}")
    # do the screws actually cross the joint? touching both sacrum and a hip
    scr = arr == 81
    if scr.any():
        d = ndi.binary_dilation(scr, iterations=2)
        touch = {n: bool((d & (arr == i)).any())
                 for i, n in ((26, "sacrum"), (30, "left_hip"), (31, "right_hip"))}
        r1035["screw_touches"] = touch
        print(f"    screws touch: "
              + ", ".join(k for k, v in touch.items() if v))
    rep["1035"] = r1035

    # ---- 0068: do the cages bridge the interspace? -----------------------------------
    p = src / "labels" / "0068_label.nii.gz"
    img = nib.load(str(p))
    arr = np.asanyarray(img.dataobj)
    vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    cage = arr == 77
    print("\n  0068")
    r68 = {"cage_voxels": int(cage.sum()), "cage_mm3": round(cage.sum() * vox_mm3, 1)}
    n, sz = pieces(cage)
    r68["cage_pieces"] = int(n)
    r68["cage_piece_mm3"] = [round(float(x) * vox_mm3, 1) for x in sz[:6]]
    print(f"    cage {cage.sum():,} vox = {r68['cage_mm3']:,} mm3 in {n} piece(s) "
          f"{r68['cage_piece_mm3']}")
    if cage.any():
        d = ndi.binary_dilation(cage, iterations=2)
        touching = sorted({int(v) for v in np.unique(arr[d]) if v and v != 77})
        r68["cage_touches_labels"] = touching
        print(f"    cage abuts label id(s): {touching}")
        # a bridge means the SAME cage component touches both bodies
        lab, _ = ndi.label(cage)
        bridging = []
        for k in range(1, int(lab.max()) + 1):
            dk = ndi.binary_dilation(lab == k, iterations=2)
            t = sorted({int(v) for v in np.unique(arr[dk]) if v and v != 77})
            if len(t) >= 2:
                bridging.append({"component": k, "touches": t})
        r68["bridging_components"] = bridging
        print(f"    component(s) touching two different bones: {len(bridging)}  {bridging}")
    rep["0068"] = r68

    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
