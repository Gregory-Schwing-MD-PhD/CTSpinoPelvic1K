"""scripts/which_rib_on_t12.py — ask the question from the vertebra's end, with no cap.

WHY THIS EXISTS. lumbar_rib_class_v5 asks, per rib, "what vertebra is nearest, within
ANCHOR_MM?" -- and on 0231, 0389 and 0720 the answer is "nothing thoracic", so it has no
evidence to re-anchor on and correctly declines. But that is the question asked from the
wrong end. A rib whose head is 20mm from T12 fails a 15mm cap while still being, by a wide
margin, the only rib anywhere near T12. Asking from the VERTEBRA -- which rib is closest to
T12? to T11? to T10? -- and reporting the distance instead of thresholding it recovers
exactly that case.

It answers, and does not decide. The output is the ranked evidence:

    T12 <- r11 at 21.4 mm   (next nearest r10 at 58.9 mm)

A single rib far closer than any other is a strong claim about which rib sits on T12; two
ribs at similar distance is not, and shows up as such. The renumber stays a decision made
on the numbers, not a threshold silently passing or failing.

    python scripts/which_rib_on_t12.py --labels data/v5_final --cases 0231,0389,0720
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))

import label_scheme as LS                                          # noqa: E402
from qc_rib_vertebra_incidence import _pts, _mindist, THORACIC_BASE  # noqa: E402
from review_anatomy_qc import MIN_VERT_VOX                         # noqa: E402

SIDES = {"left": LS.RIB_LEFT_OFFSET, "right": LS.RIB_RIGHT_OFFSET}
LUMBAR_CLASS = {"left": LS.LUMBAR_RIB_LEFT, "right": LS.LUMBAR_RIB_RIGHT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--levels", default="12,11,10")
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(",")]
    for stem in [c.strip() for c in a.cases.split(",") if c.strip()]:
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        img = nib.load(str(fp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
        print(f"\n  {stem}")

        for side, base in SIDES.items():
            ribs = {}
            for n in range(1, 13):
                m = lab == base + n
                if m.any():
                    ribs[n] = _pts(m)
            lc = lab == LUMBAR_CLASS[side]
            present = f"ribs {sorted(ribs)}" + ("  + lumbar-class rib" if lc.any() else "")
            print(f"    {side:5s}  {present}")
            if not ribs:
                continue
            for t in levels:
                vm = lab == THORACIC_BASE + t
                if vm.sum() < MIN_VERT_VOX:
                    print(f"      T{t:<3d} not labelled (or under {MIN_VERT_VOX} vox) "
                          f"-- no evidence from this level")
                    continue
                vp = _pts(vm)
                d = sorted(((_mindist(vp, p, spacing), n) for n, p in ribs.items()))
                best, second = d[0], (d[1] if len(d) > 1 else (float("inf"), None))
                # a claim is only as good as its margin over the runner-up
                mark = ("CLEAR" if second[0] > 2.5 * best[0] or second[1] is None
                        else "ambiguous")
                nxt = ("none" if second[1] is None
                       else f"r{second[1]} at {second[0]:.1f} mm")
                print(f"      T{t:<3d} <- r{best[1]} at {best[0]:6.1f} mm   "
                      f"(next {nxt})   {mark}")
                if mark == "CLEAR" and best[1] != t:
                    print(f"           -> implies rib {best[1]} should be rib {t} "
                          f"({t - best[1]:+d})")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
