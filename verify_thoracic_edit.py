"""Verify a hand-added thoracic annotation before the next thirteen are done the same way.

Checks the things that are cheap to get wrong and expensive to discover late:

  numbering runs UP from L1   the body directly above L1 must be T12. An AI segmenter
                              numbers thoracic vertebrae from its own reading of the whole
                              thorax, and if that numbering wins, the off-by-one this
                              project spent days unwinding comes straight back.
  contiguous                  no gaps in the thoracic run; a gap means a level was skipped
                              or misnumbered.
  nothing below L1 moved      the edit should be additive. A changed lumbar or sacral voxel
                              count means something was overwritten.
  each new label is one piece a vertebra is one bone; many components means stray paint.

Compares against a backup of the pre-edit label when one is given.

    python verify_thoracic_edit.py 0344_label.nii.gz [--before backup/0344_label.nii.gz]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
OTHER = {26: "sacrum", 29: "S1", 30: "hip_L", 31: "hip_R", 32: "fem_L", 33: "fem_R"}


def summarize(path):
    img = nib.as_closest_canonical(nib.load(path))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    return lab, np.array(img.header.get_zooms()[:3], float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--before", default="")
    a = ap.parse_args()

    lab, zoom = summarize(a.label)
    ok = True

    th = {n: int((lab == THORACIC_BASE + n).sum()) for n in range(1, 13)}
    th = {n: v for n, v in th.items() if v > 0}
    lm = {v: int((lab == v).sum()) for v in LUMBAR if (lab == v).any()}
    print(f"\n  {a.label}")
    print(f"  thoracic present: {['T%d' % n for n in sorted(th)] or 'NONE'}")
    print(f"  lumbar present:   {[LUMBAR[v] for v in sorted(lm)] or 'NONE'}")

    if not th:
        print("  ** no thoracic labels found — was the save written to this file? **")
        return 1

    # contiguity
    ns = sorted(th)
    if ns != list(range(ns[0], ns[-1] + 1)):
        print(f"  ** GAP in the thoracic run: {ns} **")
        ok = False
    else:
        print(f"  contiguous T{ns[0]}..T{ns[-1]}  ({len(ns)} levels)")

    # the rule: the body directly above L1 is T12
    l1 = lab == 20
    if l1.any() and 12 in th:
        z_l1 = np.nonzero(l1.any(axis=(0, 1)))[0]
        z_t12 = np.nonzero((lab == THORACIC_BASE + 12).any(axis=(0, 1)))[0]
        if z_t12.mean() > z_l1.mean():
            print(f"  T12 sits above L1  (T12 z~{z_t12.mean():.0f}, L1 z~{z_l1.mean():.0f})")
        else:
            print(f"  ** T12 is NOT above L1 (T12 z~{z_t12.mean():.0f} vs L1 z~{z_l1.mean():.0f}) **")
            ok = False
        # and it must be the LOWEST thoracic, i.e. nothing numbered higher sits lower
        for n in ns:
            if n == 12:
                continue
            zn = np.nonzero((lab == THORACIC_BASE + n).any(axis=(0, 1)))[0]
            if zn.mean() < z_t12.mean():
                print(f"  ** T{n} sits BELOW T12 — numbering is inverted **")
                ok = False
                break
    elif 12 not in th:
        print("  ** T12 absent: the body directly above L1 must be T12 **")
        ok = False

    # each vertebra should be one piece
    for n in ns:
        m = lab == THORACIC_BASE + n
        cc, ncc = ndimage.label(m)
        if ncc > 1:
            sizes = ndimage.sum(m, cc, range(1, ncc + 1))
            frac = sizes.max() / sizes.sum()
            flag = "  <-- stray paint" if frac < 0.98 else "  (specks only)"
            print(f"  T{n}: {ncc} components, largest {100*frac:.1f}%{flag}")

    if a.before:
        old, _ = summarize(a.before)
        changed = []
        for vid, nm in list(LUMBAR.items()) + list(OTHER.items()):
            b, n_ = int((old == vid).sum()), int((lab == vid).sum())
            if b != n_:
                changed.append(f"{nm} {b}->{n_}")
        if changed:
            print(f"  ** below-L1 structures CHANGED: {', '.join(changed)} **")
            ok = False
        else:
            print("  nothing below the thoracic spine was altered")

    print(f"\n  {'PASS' if ok else 'CHECK THE FLAGGED ITEMS'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
