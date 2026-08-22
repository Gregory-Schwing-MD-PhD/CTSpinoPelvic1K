"""scripts/render_wedge_check.py — look at the vertebrae the wedge ratio calls collapsed.

WHY THIS EXISTS. Six vertebrae came back with anterior heights of 8.8 to 14.4 mm against
posterior heights of 19 to 35 -- ratios from 0.27 to 0.54, which is 46 to 73 percent
anterior height loss. Two readings fit that equally well and the numbers cannot separate
them:

  a genuine severe compression fracture, Genant grade 3, which does occur in a cohort
  aged 50 to 89 at roughly one percent; or

  a failed measurement, where the mid-sagittal slab or the body mask landed somewhere
  that is not the anterior column.

A first guess -- that the body mask was contaminated by posterior elements -- was checked
and is wrong here: the posterior heights in these cases sit right on their neighbours'.
Whatever went wrong, if anything went wrong, is in the anterior half alone.

So this renders the mid-sagittal CT slice through the vertebra in question with the label
outlined and the two measured heights drawn where the code measured them. A wedge fracture
is unmistakable on a sagittal slice; so is a mask that has slipped off the bone. It is the
one question a histogram cannot answer.

    python scripts/render_wedge_check.py --labels data/v5_final --ct data/hf_export/ct \\
        --cases 0468:L4,0568:L3,0851:L2,0972:L3,0213:L2,0913:L3 --out wedge_renders
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LUMBAR = {"L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24, "L6": 25}


def _canal_front(mask):
    """Anterior wall of the canal, exactly as extract_level_gradients.py finds it."""
    zs = np.nonzero(mask.any(axis=(0, 1)))[0]
    if len(zs) < 5:
        return None
    fronts = []
    for z in range(int(np.percentile(zs, 30)), int(np.percentile(zs, 70)) + 1):
        sl = mask[:, :, z]
        if sl.sum() < 60:
            continue
        hole = ndimage.binary_fill_holes(sl) & ~sl
        if not hole.any():
            continue
        cc, n = ndimage.label(hole)
        if n == 0:
            continue
        sizes = ndimage.sum(hole, cc, range(1, n + 1))
        big = cc == (int(np.argmax(sizes)) + 1)
        if big.sum() < 20:
            continue
        fronts.append(int(np.nonzero(big.any(axis=0))[0].max()))
    return float(np.median(fronts)) if fronts else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export/ct")
    ap.add_argument("--cases", required=True, help="comma list of case:LEVEL")
    ap.add_argument("--out", default="wedge_renders")
    ap.add_argument("--pad_mm", type=float, default=25.0)
    a = ap.parse_args()

    spec = []
    for tok in a.cases.split(","):
        tok = tok.strip()
        if ":" in tok:
            c, lv = tok.split(":", 1)
            spec.append((c.strip(), lv.strip()))
    if not spec:
        print("nothing to render")
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(spec), figsize=(3.1 * len(spec), 4.0))
    if len(spec) == 1:
        axes = [axes]

    for ax, (case, lv) in zip(axes, spec):
        lp = Path(a.labels) / f"{case}_label.nii.gz"
        cp = Path(a.ct) / f"{case}_0000.nii.gz"
        if not cp.exists():
            alt = list(Path(a.ct).glob(f"{case}*.nii.gz"))
            cp = alt[0] if alt else cp
        if not lp.exists() or not cp.exists():
            ax.set_axis_off()
            ax.set_title(f"{case} {lv}\nmissing", fontsize=8)
            continue

        # ANALYSE in RAS; nothing is written back, so canonicalising is safe here
        li = nib.as_closest_canonical(nib.load(str(lp)))
        ci = nib.as_closest_canonical(nib.load(str(cp)))
        lab = np.asanyarray(li.dataobj)
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
        sp = li.header.get_zooms()[:3]

        vid = LUMBAR.get(lv)
        m = lab == vid

        # THE MASK THE MEASUREMENT ACTUALLY USES, not the whole vertebra. In a
        # mid-sagittal cut the body and the spinous process are naturally disconnected --
        # the pedicle joining them is lateral -- so outlining the whole label shows two
        # blobs on every normal vertebra and proves nothing. What matters is whether the
        # cut at the anterior wall of the canal removed the posterior elements, so the
        # cut is replicated here exactly as extract_level_gradients.py performs it.
        front = _canal_front(m)
        body = np.zeros_like(m)
        if front is not None:
            f = int(np.ceil(front))
            body[:, f:, :] = m[:, f:, :]
        if not m.any():
            ax.set_axis_off()
            ax.set_title(f"{case} {lv}\nlabel absent", fontsize=8)
            continue

        idx = np.argwhere(m)
        # the same mid-sagittal slab the measurement uses: centred on the body's own x
        xm = int(round(float(np.median(idx[:, 0]))))
        pad = [int(round(a.pad_mm / s)) for s in sp]
        y0, y1 = max(0, idx[:, 1].min() - pad[1]), min(lab.shape[1], idx[:, 1].max() + pad[1])
        z0, z1 = max(0, idx[:, 2].min() - pad[2]), min(lab.shape[2], idx[:, 2].max() + pad[2])

        sl_ct = ct[xm, y0:y1, z0:z1].T[::-1]
        sl_lb = m[xm, y0:y1, z0:z1].T[::-1]
        sl_bd = body[xm, y0:y1, z0:z1].T[::-1]
        ax.imshow(np.clip(sl_ct, -150, 900), cmap="gray",
                  aspect=sp[2] / sp[1], interpolation="bilinear")
        ax.contour(sl_lb, levels=[0.5], colors=["#ff3b3b"], linewidths=0.6)
        if sl_bd.any():
            ax.contour(sl_bd, levels=[0.5], colors=["#39ff88"], linewidths=1.3)

        # where the code measured: tallest column of each half of the slab
        # heights are taken from the BODY mask, which is what the code measures
        src = sl_bd if sl_bd.any() else sl_lb
        ys = np.nonzero(src.any(axis=0))[0]
        if len(ys):
            heights = {}
            for y in ys:
                zz = np.nonzero(src[:, y])[0]
                if len(zz):
                    heights[y] = (zz.min(), zz.max())
            if heights:
                ymid = (min(heights) + max(heights)) / 2.0
                for half, colour, tag in (
                        ([y for y in heights if y > ymid], "#31d0ff", "ant"),
                        ([y for y in heights if y <= ymid], "#ffd23b", "post")):
                    if not half:
                        continue
                    yb = max(half, key=lambda y: heights[y][1] - heights[y][0])
                    zlo, zhi = heights[yb]
                    ax.plot([yb, yb], [zlo, zhi], color=colour, lw=2.0)
                    ax.text(yb, zlo - 4, tag, color=colour, fontsize=7,
                            ha="center", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        for s2 in ax.spines.values():
            s2.set_visible(False)
        ax.set_title(f"{case}  {lv}", fontsize=9)

    fig.suptitle("mid-sagittal through the vertebra the wedge ratio calls collapsed\n"
                 "blue = anterior column measured, yellow = posterior",
                 fontsize=9)
    fig.tight_layout()
    p = out / "wedge_check.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
