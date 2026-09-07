"""complete_0816.py -- finish the one record that never went through v2-v6.

0816 is the only partial annotation left in the release: T11-L5 traced by a radiologist,
and 106,229,372 voxels of `ignore` where every other case has sacrum, hips, femurs and
ribs. This brings it up to the v6 scheme so it can be corrected by hand rather than
re-traced from nothing.

ORDER OF PRECEDENCE, and it matters:
  1. HARDWARE outranks bone, as everywhere else in this pipeline -- a segmenter absorbs an
     implant it can see, so naming the metal is a subtraction from the bone label, not an
     addition.
  2. Existing labels are never overwritten. The spine is radiologist ground truth and the
     pelvis arrives from the out-of-fold pseudolabeller; TS additions land ONLY on
     background/ignore.
  3. What is left of `ignore` becomes background, because after this pass the record is
     densely labelled and a 255 that means "not traced" would now be a lie.

The metal is written as id 76 -- `hardware`, subtype NOT distinguished -- deliberately.
Which device it is, is a radiologist's call, and 77-82 are specific claims this script is
not entitled to make. Everything bone-hosted above the metal threshold is labelled so the
reviewer can see all of it; metal with no bone near it (surgical clips in the bowel or the
gallbladder fossa) is reported but NOT labelled, per the pipeline's own rule that an
orthopaedic implant is fixed to bone.

NOTHING IS REORIENTED. The label is written on the CT's own grid with the CT's own affine;
`as_closest_canonical` here would silently transpose the label away from its image.

    python complete_0816.py --ct ct/0816_ct.nii.gz --label labels/0816_label.nii.gz \
        --out 0816_label_completed.nii.gz --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                       # noqa: E402
from build_v3_totalseg import ts_femurs_and_s1  # noqa: E402  (the v3 pass, reused verbatim)

IGNORE = LS.IGNORE_LABEL          # 255
SACRUM, S1 = LS.SACRUM_ID, LS.S1_ID
HARDWARE = 76                     # subtype not distinguished -- the reviewer decides

# identical to scripts/detect_metal.py, so the two agree about what metal is
METAL_HU = 2200.0
MIN_VOX = 20
BONE_REACH_MM = 12.0


def label_metal(ct, lab, spacing):
    """Bone-hosted metal -> a mask; also returns what was rejected and why."""
    hot = ct >= METAL_HU
    out = np.zeros_like(hot)
    hosted, orphan = [], []
    if not hot.any():
        return out, hosted, orphan

    cc, n = ndimage.label(hot)
    reach = max(1, int(round(BONE_REACH_MM / float(min(spacing)))))
    for i in range(1, n + 1):
        m = cc == i
        v = int(m.sum())
        if v < MIN_VOX:
            continue
        # crop before dilating: the metal is a speck in a 100M-voxel volume
        idx = np.argwhere(m)
        lo = np.maximum(idx.min(0) - reach - 2, 0)
        hi = np.minimum(idx.max(0) + reach + 3, m.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        near = lab[sl][ndimage.binary_dilation(m[sl], iterations=reach)]
        near = near[(near > 0) & (near != IGNORE) & (near != HARDWARE)]
        host = int(Counter(near.tolist()).most_common(1)[0][0]) if near.size else 0
        mm3 = v * float(np.prod(spacing))
        if host > 0:
            out |= m
            hosted.append({"vox": v, "mm3": round(mm3, 1), "host": host})
        else:
            orphan.append({"vox": v, "mm3": round(mm3, 1)})
    return out, hosted, orphan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True, help="the pseudolabelled (pelvis-filled) label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default=None)
    ap.add_argument("--no_hardware", action="store_true",
                    help="do not label metal. The dense foci in this case were read as "
                         "seeds/clips, not an implant: discrete 3-7 mm objects, bilateral "
                         "and segmental, none entering bone. Labelling them `hardware` "
                         "would put a false implant in the release.")
    a = ap.parse_args()

    ci = nib.load(a.ct)                      # NATIVE orientation, deliberately
    li = nib.load(a.label)
    ct = np.asanyarray(ci.dataobj).astype(np.float32)
    lab = np.asanyarray(li.dataobj).astype(np.int16)
    sp = np.array(li.header.get_zooms()[:3], float)
    assert ct.shape == lab.shape, f"shape mismatch {ct.shape} vs {lab.shape}"
    print(f"  ct {ct.shape}  axcodes {nib.aff2axcodes(ci.affine)}  zooms {sp.round(3)}")
    before = {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True)) if k}
    print(f"  starting ids: {sorted(before)}")

    # ---- TotalSegmentator: femurs, ribs (TS's own numbering), S1 ----------------
    add, s1_mask, meta = ts_femurs_and_s1(Path(a.ct), li, device=a.device)
    free = (lab == 0) | (lab == IGNORE)
    placed = (add > 0) & free
    lab[placed] = add[placed].astype(np.int16)
    print(f"  TS added: femurs={meta.get('femurs')}  ribs={len(meta.get('ribs', []))} "
          f"-> {int(placed.sum()):,} voxels")

    # ---- carve S1 out of the sacrum (outer boundary stays as-is) ---------------
    n_s1 = 0
    if s1_mask is not None:
        carve = (lab == SACRUM) & s1_mask.astype(bool)
        lab[carve] = S1
        n_s1 = int(carve.sum())
    print(f"  S1 carved from sacrum: {n_s1:,} voxels")

    # ---- metal: hardware outranks bone -----------------------------------------
    hosted, orphan = [], []
    if a.no_hardware:
        print("  hardware: SKIPPED (--no_hardware) — metal left as bone/background")
        metal = np.zeros_like(lab, dtype=bool)
        stolen = {}
    else:
        metal, hosted, orphan = label_metal(ct, lab, sp)
        stolen = Counter(lab[metal].tolist())
        lab[metal] = HARDWARE
        print(f"  hardware (id {HARDWARE}): {int(metal.sum()):,} voxels in "
              f"{len(hosted)} bone-hosted component(s)")
        for h in sorted(hosted, key=lambda x: -x["mm3"]):
            print(f"      {h['mm3']:>8.1f} mm3  host={h['host']}")
        if orphan:
            print(f"  NOT labelled — metal with no bone near it ({len(orphan)}), "
                  f"most likely surgical clips:")
            for o in sorted(orphan, key=lambda x: -x["mm3"]):
                print(f"      {o['mm3']:>8.1f} mm3")
        print(f"  reclaimed from: { {int(k): int(v) for k, v in stolen.items()} }")

    # ---- ignore is no longer true --------------------------------------------
    n_ign = int((lab == IGNORE).sum())
    lab[lab == IGNORE] = 0
    print(f"  ignore -> background: {n_ign:,} voxels")

    out = nib.Nifti1Image(lab.astype(np.uint8), li.affine, li.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, a.out)
    after = {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True)) if k}
    print(f"  wrote {a.out}")
    print(f"  final ids: {sorted(after)}")

    if a.report:
        Path(a.report).write_text(json.dumps(
            {"before": before, "after": after, "hosted_metal": hosted,
             "orphan_metal": orphan, "s1_voxels": n_s1,
             "ts": {k: v for k, v in meta.items() if k != "ribs"},
             "ts_ribs": meta.get("ribs", [])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
