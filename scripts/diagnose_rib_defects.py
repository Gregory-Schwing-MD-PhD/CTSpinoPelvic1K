"""scripts/diagnose_rib_defects.py — what is actually wrong with the residual rib cases?

Nineteen offset ribs survive in six cases and they are not the same defect, so a single
fix would be wrong for most of them. This reports the structure of each offending label so
the fix can be chosen per case rather than guessed:

  FRAGMENTED    one rib id split into several disconnected pieces. The pieces are usually
                speckle, and a stray piece far from the main body is what drags a rib's
                "nearest vertebra" onto something absurd -- a 6th rib cannot reach L3, but a
                handful of voxels carrying the rib-6 id sitting down at L3 certainly can.
  DISPLACED     one connected rib whose HEAD is nearer a different vertebra. A floating
                twelfth rib does this legitimately, and no renumber should follow.
  DISTANT       a rib whose nearest vertebra is beyond the contact threshold entirely.

For every component it prints size, its z-centre and its gap to the nearest vertebra, so a
speckle at the wrong level is visible as a tiny component with a huge gap from the main
one. The main body is whichever component holds the most voxels.

    python scripts/diagnose_rib_defects.py --labels data/v5_final \\
        --cases 0179,0344,0378,0412,0487,0720,0787,1153
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))
import label_scheme as LS                                          # noqa: E402

THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
MIN_VERT_VOX = 6000
SIDES = {"left": LS.RIB_LEFT_OFFSET, "right": LS.RIB_RIGHT_OFFSET}
EXTRA = {LS.LUMBAR_RIB_LEFT: "L-lumbar", LS.LUMBAR_RIB_RIGHT: "R-lumbar"}


def _pts(mask, cap=300):
    p = np.argwhere(mask)
    return p[:: max(1, len(p) // cap)] if len(p) else p


def _mindist(a, b, sp):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = (a[:, None, :] - b[None, :, :]) * sp
    return float(np.sqrt((d ** 2).sum(-1)).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--min-frag-vox", type=int, default=0,
                    help="only print components at or above this size")
    a = ap.parse_args()

    for stem in [c.strip() for c in a.cases.split(",") if c.strip()]:
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        img = nib.as_closest_canonical(nib.load(str(fp)))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)

        verts = {}
        for n in range(1, 13):
            m = lab == THORACIC_BASE + n
            if m.sum() >= MIN_VERT_VOX:
                verts[f"T{n}"] = _pts(m)
        for vid, nm in LUMBAR.items():
            m = lab == vid
            if m.sum() >= MIN_VERT_VOX:
                verts[nm] = _pts(m)

        print(f"\n=== {stem}   vertebrae labelled: "
              f"{', '.join(sorted(verts, key=lambda s: (s[0], int(s[1:])))) or 'NONE'}")

        targets = [(f"{s[0].upper()}{n}", base + n)
                   for s, base in SIDES.items() for n in range(1, 13)]
        targets += [(nm, cid) for cid, nm in EXTRA.items()]
        for name, rid in targets:
            m = lab == rid
            if not m.any():
                continue
            cc, ncc = ndimage.label(m)
            sizes = ndimage.sum(m, cc, range(1, ncc + 1))
            order = np.argsort(sizes)[::-1]
            flag = "FRAGMENTED" if ncc > 1 and sizes.min() > 0 and len(sizes) > 1 else ""
            main_gap, main_near = None, None
            lines = []
            for rank, ci in enumerate(order):
                vox = int(sizes[ci])
                if vox < a.min_frag_vox:
                    continue
                cm = cc == (ci + 1)
                zc = float(np.argwhere(cm)[:, 2].mean())
                cp = _pts(cm)
                gaps = {v: _mindist(cp, p, sp) for v, p in verts.items()}
                near, gap = (min(gaps.items(), key=lambda kv: kv[1])
                             if gaps else ("-", float("inf")))
                if rank == 0:
                    main_gap, main_near = gap, near
                tag = "main" if rank == 0 else f"frag{rank}"
                lines.append(f"        {tag:6s} {vox:7d} vox  z={zc:6.1f}  "
                             f"nearest {near:>4s} @ {gap:6.1f} mm")
            if not lines:
                continue
            head = f"    {name:9s} {ncc} component(s)"
            if ncc > 1:
                head += "   <== FRAGMENTED"
            elif main_near and not main_near.startswith(("T", "L")):
                head += ""
            print(head)
            for ln in lines:
                print(ln)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
