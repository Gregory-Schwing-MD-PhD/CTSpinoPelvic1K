"""Does the label sit proud of the bone at the end-plates?

This is the one candidate for the remaining gap that a mid-sagittal overlay cannot
settle: a mask that runs a voxel or two past the cortex adds height at BOTH ends and is
invisible at any sensible window. It is decidable from the CT itself, because bone and
disc differ by several hundred Hounsfield units and the cortex is the brightest thing in
the neighbourhood.

METHOD. At the posterior half of the body, step along the superior axis through the
end-plate boundary and record the CT value as a function of signed distance from the
label edge: negative inside the label, positive outside. Then read three things off the
mean profile:

  where the intensity crosses a bone threshold, relative to the label edge. If the
  label edge and the bone edge coincide the crossing sits at 0. If the label is proud,
  the bone has already ended before the label does, and the crossing is NEGATIVE by the
  amount of the overshoot.

  the intensity of the last voxel inside the label. Cortical bone is 300+ HU; if this
  reads like disc or marrow fat the mask has left the bone.

  the height of the cortical peak, as a check that the profile found a real end-plate
  rather than a smear.

Reported per level over many cases, at both the superior and the inferior plate, since
an overshoot at each end is what would double into the height.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}
BONE_HU = 200.0          # between disc/marrow and cortex; the crossing is insensitive to it
STEPS = np.arange(-4, 5)  # voxels, inside -> outside


def carve(whole):
    zz = np.nonzero(whole.any(axis=(0, 1)))[0]
    fronts = []
    for z in zz[len(zz) // 3: 2 * len(zz) // 3 + 1]:
        sl = whole[:, :, z]
        if sl.sum() < 40:
            continue
        h = ndimage.binary_fill_holes(sl) & ~sl
        if h.any():
            fronts.append(int(np.nonzero(h.any(axis=0))[0].max()))
    if not fronts:
        return None
    f = int(np.median(fronts))
    body = np.zeros_like(whole)
    body[:, f:, :] = whole[:, f:, :]
    return body


def profiles(ct, body, sp):
    """Mean CT profile across the superior and inferior label edge, in voxel steps."""
    out = {}
    idx = np.argwhere(body)
    if not len(idx):
        return out
    ylo, yhi = idx[:, 1].min(), idx[:, 1].max()
    y_post = ylo + int(0.35 * (yhi - ylo))          # posterior third of the body
    for which, sgn in (("sup", +1), ("inf", -1)):
        vals = []
        for x in np.unique(idx[:, 0]):
            for y in range(int(ylo), int(y_post) + 1):
                col = np.nonzero(body[x, y, :])[0]
                if len(col) < 4:
                    continue
                edge = col.max() if sgn > 0 else col.min()
                zs = edge + sgn * STEPS
                if zs.min() < 0 or zs.max() >= ct.shape[2]:
                    continue
                vals.append(ct[x, y, zs])
        if len(vals) >= 40:
            out[which] = np.asarray(vals, float).mean(axis=0)
    return out


def crossing(prof):
    """Signed distance in VOXELS from the label edge to where bone ends. 0 = coincident."""
    # STEPS runs inside (negative index positions) to outside; find the last index where
    # the profile is still bone
    above = prof >= BONE_HU
    if not above.any():
        return None
    last = int(np.max(np.nonzero(above)[0]))
    return float(STEPS[last])


def main():
    ct_dir, lab_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    cases = [p.name.split("_")[0] for p in sorted(lab_dir.glob("*_label.nii.gz"))[:n]]
    acc, prof_acc = {}, {}
    for i, c in enumerate(cases, 1):
        lp, cp = lab_dir / f"{c}_label.nii.gz", ct_dir / f"{c}_ct.nii.gz"
        if not (lp.exists() and cp.exists()):
            continue
        try:
            limg = nib.as_closest_canonical(nib.load(str(lp)))
            cimg = nib.as_closest_canonical(nib.load(str(cp)))
            lab = np.asanyarray(limg.dataobj)
            ct = np.asanyarray(cimg.dataobj).astype(np.float32)
            sp = np.asarray(limg.header.get_zooms()[:3], float)
            for vid, name in LUMBAR.items():
                whole = lab == vid
                if whole.sum() < 500:
                    continue
                body = carve(whole)
                if body is None:
                    continue
                for which, prof in profiles(ct, body, sp).items():
                    prof_acc.setdefault((name, which), []).append(prof)
                    x = crossing(prof)
                    if x is not None:
                        acc.setdefault((name, which), []).append(x * sp[2])
        except Exception as exc:
            print(f"  {c}: {type(exc).__name__}: {exc}", flush=True)
        if i % 10 == 0:
            print(f"  {i}/{len(cases)}", flush=True)

    print("\nmean CT profile across the label edge, HU. step 0 IS the last voxel inside.")
    print(f"{'level':6s} {'plate':5s} " + " ".join(f"{s:>7d}" for s in STEPS))
    for name in LUMBAR.values():
        for which in ("sup", "inf"):
            ps = prof_acc.get((name, which))
            if not ps:
                continue
            m = np.mean(ps, axis=0)
            print(f"{name:6s} {which:5s} " + " ".join(f"{v:7.0f}" for v in m))

    print(f"\nwhere bone ends, relative to the label edge, mm (0 = they coincide,")
    print(f"negative = the label extends past the bone). threshold {BONE_HU:.0f} HU.")
    print(f"{'level':6s} {'n':>5s} {'superior':>10s} {'inferior':>10s} {'sum':>8s}")
    for name in LUMBAR.values():
        s = acc.get((name, "sup"), [])
        i_ = acc.get((name, "inf"), [])
        if not s or not i_:
            continue
        ms, mi = float(np.median(s)), float(np.median(i_))
        print(f"{name:6s} {len(s):5d} {ms:10.2f} {mi:10.2f} {ms + mi:8.2f}")
    print("\n'sum' is the total height this would add if both plates overshoot.")


if __name__ == "__main__":
    main()
