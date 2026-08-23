"""scripts/render_tp_height_check.py -- look at what tp_height is actually measuring.

Castellvi Type I is a transverse process at least 19 mm in craniocaudal height. Applying
that threshold to `tp_height_max_mm` calls 45.8% of the ungraded cohort Type I, which is
not a plausible prevalence, so the measurement and the criterion are not the same quantity
and the number cannot be used for a Type I screen until it is understood.

Two candidate faults, and rendering separates them:
  * the 12 mm tip slab still catches something other than the process -- the pars, the
    accessory process, or on a wide L5 the lateral edge of the superior articular process;
  * the slab is right and the z-extent is the wrong statistic, because it spans the whole
    slab rather than the process at its widest point, which is what Castellvi measures.

Renders a coronal maximum-intensity projection of the lowest lumbar vertebra with the tip
slab overlaid and the measured extent drawn, so the failure is visible instead of inferred.
This is the same procedure that found the anterior sliver in the wedge measurement, where
two plausible explanations were both wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L1, L6, SACRUM = 20, 25, 26
TIP_MM = 12.0


def lowest_lumbar(lab):
    present = [v for v in range(L1, L6 + 1) if (lab == v).any()]
    return max(present) if present else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="directory of *_label.nii.gz")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default="morphometrics/tp_height_renders")
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    for case in a.cases:
        f = Path(a.labels) / f"{case}_label.nii.gz"
        if not f.exists():
            print(f"  ! {f} missing")
            continue
        raw = nib.load(str(f))
        # CANONICALISE TO ANALYSE, NEVER TO WRITE. These volumes sit on disk as
        # ('P','I','R'): axis 0 is anteroposterior and axis 2 is left-right. The first
        # version of this script read them raw while assuming axis 0 was lateral and
        # axis 2 craniocaudal, which is how it reported a 51.6 mm transverse process on
        # the left and 9.4 mm on the right in nearly every case -- it was slicing
        # anterior and posterior thirds and measuring their left-right width.
        # extract_transition_morphometrics.py canonicalises at load; so does this now,
        # or the render cannot check the thing it exists to check. Nothing is written
        # back, so the rule against reorienting a label being SAVED does not apply.
        img = nib.as_closest_canonical(raw)
        lab = np.asanyarray(img.dataobj)
        sp = img.header.get_zooms()[:3]
        axc = "".join(nib.aff2axcodes(raw.affine)) + " -> " + "".join(nib.aff2axcodes(img.affine))
        low = lowest_lumbar(lab)
        if low is None:
            print(f"  ! {case}: no lumbar label")
            continue
        m = lab == low
        mx = np.nonzero(m.any(axis=(1, 2)))[0]
        vmid = 0.5 * (float(mx.min()) + float(mx.max()))
        ax = np.arange(lab.shape[0])
        latL = int(vmid - 0.45 * (vmid - float(mx.min())))
        latR = int(vmid + 0.45 * (float(mx.max()) - vmid))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        info = []
        for k, (nm, sel, outward) in enumerate(
                (("left", ax < latL, -1), ("right", ax > latR, +1))):
            mm = m & sel[:, None, None]
            panel = axes[k]
            if not mm.any():
                panel.set_title(f"{nm}: empty"); panel.axis("off"); continue
            cols = np.nonzero(mm.any(axis=(1, 2)))[0]
            edge = int(cols.max()) if outward > 0 else int(cols.min())
            depth = max(1, int(round(TIP_MM / max(sp[0], 1e-6))))
            keep = ((ax <= edge) & (ax >= edge - depth) if outward > 0
                    else (ax >= edge) & (ax <= edge + depth))
            tip = mm & keep[:, None, None]
            zt = np.nonzero(tip.any(axis=(0, 1)))[0]
            h = float(zt.max() - zt.min() + 1) * sp[2]

            # coronal projection: collapse the anteroposterior axis
            whole = m.any(axis=1).T
            tipp = tip.any(axis=1).T
            panel.imshow(whole, cmap="Greys", origin="lower", aspect="auto")
            panel.imshow(np.ma.masked_where(~tipp, tipp), cmap="autumn",
                         origin="lower", alpha=0.75, aspect="auto")
            panel.axhline(zt.min(), color="tab:blue", lw=1)
            panel.axhline(zt.max(), color="tab:blue", lw=1)
            panel.set_title(f"{nm}  tp_height = {h:.1f} mm\n"
                            f"tip slab {TIP_MM:.0f} mm, {int(tip.sum())} vox")
            panel.set_xlabel("left-right (vox)"); panel.set_ylabel("cranio-caudal (vox)")
            info.append(f"{nm}={h:.1f}")
        fig.suptitle(f"{case}   lowest lumbar = label {low}   axcodes {axc}   "
                     f"spacing {sp[0]:.2f}/{sp[1]:.2f}/{sp[2]:.2f} mm   "
                     + "  ".join(info))
        fig.tight_layout()
        out = Path(a.out) / f"{case}_tp_height.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  {case}: label {low}, {' '.join(info)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
