"""Draw the body-height measurement on top of the CT it came from.

The numbers say our L3 posterior height is 30.4 mm and Panjabi's is 23.8, and that every
estimator we tried agrees with ours. Either the label is bigger than the bone, or the
bone really is that size and the cadaveric series is measuring a smaller population. A
mid-sagittal overlay separates those two in one look: if the label's posterior edge sits
on the cortex the measurement is right, and if it runs into the disc space it is not.

Per case and level, three panels:
  1. CT alone, bone window, so the anatomy can be read without a mask over it
  2. CT + the whole-vertebra label and the carved body, so the carve can be judged
  3. the body outline with the measurement drawn: the posterior wall chord that produces
     the reported number, and the ostk corner landmarks for the same plate

Run on the grid where the volumes are:
    python render_height.py <ct_dir> <label_dir> <out_dir> [case ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "OpenSpineToolkit"))
try:
    from ostk.morphometry import level_morphometry
except Exception:                                            # pragma: no cover
    level_morphometry = None

LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}
PAN = {"L1": 23.8, "L2": 24.3, "L3": 23.8, "L4": 24.1, "L5": 22.9}


def carve(whole):
    """The release's body carve: everything anterior of the median canal front."""
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
        return None, None
    f = int(np.median(fronts))
    body = np.zeros_like(whole)
    body[:, f:, :] = whole[:, f:, :]
    return body, f


def one(ct_p, lab_p, out_dir, vids=(22,)):
    cimg = nib.as_closest_canonical(nib.load(str(ct_p)))
    limg = nib.as_closest_canonical(nib.load(str(lab_p)))
    ct = np.asanyarray(cimg.dataobj).astype(np.float32)
    lab = np.asanyarray(limg.dataobj)
    sp = np.asarray(limg.header.get_zooms()[:3], float)
    aff = limg.affine
    case = Path(lab_p).name.split("_")[0]

    for vid in vids:
        name = LUMBAR[vid]
        whole = lab == vid
        if whole.sum() < 500:
            continue
        body, front = carve(whole)
        if body is None:
            continue
        idx = np.argwhere(body)
        bx = int(round(float(np.median(idx[:, 0]))))          # mid-sagittal column

        # the reported number, and where it is taken
        slab = body[max(0, int(round(bx - 5 / sp[0]))): int(round(bx + 5 / sp[0])) + 1]
        col = {}
        for y in np.nonzero(slab.any(axis=(0, 2)))[0]:
            c = slab[:, y, :]
            if c.sum() < 3:
                continue
            zc = np.nonzero(c.any(axis=0))[0]
            col[y] = (int(zc.min()), int(zc.max()))
        if len(col) < 6:
            continue
        yy = np.array(sorted(col))
        hh = np.array([(col[y][1] - col[y][0] + 1) * sp[2] for y in yy])
        post = yy <= (yy.min() + yy.max()) / 2.0
        y_at = int(yy[post][int(np.argmax(hh[post]))])         # the column that wins
        h_mm = float(hh[post].max())

        # sagittal images: array is (x, y, z) = (R, A, S); show y horizontal, z vertical
        ct_s = ct[bx].T[::-1]
        wh_s = whole[bx].T[::-1]
        bo_s = body[bx].T[::-1]
        nz = np.argwhere(wh_s)
        if not len(nz):
            continue
        r0, c0 = nz.min(0) - 18
        r1, c1 = nz.max(0) + 18
        r0, c0 = max(r0, 0), max(c0, 0)
        aspect = sp[2] / sp[1]

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))
        for ax in axes:
            ax.imshow(ct_s[r0:r1, c0:c1], cmap="gray", vmin=-200, vmax=1200,
                      interpolation="nearest", aspect=aspect)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_title(f"{case} {name}: CT, bone window", fontsize=9, loc="left")

        m1 = np.ma.masked_where(~wh_s[r0:r1, c0:c1], wh_s[r0:r1, c0:c1])
        m2 = np.ma.masked_where(~bo_s[r0:r1, c0:c1], bo_s[r0:r1, c0:c1])
        axes[1].imshow(m1, cmap="autumn", alpha=0.35, interpolation="nearest", aspect=aspect)
        axes[1].imshow(m2, cmap="winter", alpha=0.45, interpolation="nearest", aspect=aspect)
        axes[1].set_title("whole vertebra (warm) / carved body (cool)", fontsize=9, loc="left")

        axes[2].contour(bo_s[r0:r1, c0:c1].astype(float), levels=[0.5],
                        colors="#28c8ff", linewidths=1.1)
        zlo, zhi = col[y_at]
        # array row for a z index, after the vertical flip and the crop
        rr = lambda z: (ct_s.shape[0] - 1 - z) - r0
        cc = y_at - c0
        axes[2].plot([cc, cc], [rr(zlo), rr(zhi)], color="#ff4d4d", lw=2.0,
                     solid_capstyle="butt", zorder=5)
        axes[2].plot([cc], [rr(zlo)], marker="_", ms=13, color="#ff4d4d", zorder=5)
        axes[2].plot([cc], [rr(zhi)], marker="_", ms=13, color="#ff4d4d", zorder=5)
        txt = f"posterior wall chord = {h_mm:.1f} mm\nPanjabi VBHp {name} = {PAN[name]} mm"

        if level_morphometry is not None:
            try:
                r = level_morphometry(lab, aff, vid)
                if r and "VBHp" in r:
                    txt += f"\nostk VBHp = {r['VBHp']:.1f} mm"
            except Exception as exc:
                txt += f"\nostk: {type(exc).__name__}"
        axes[2].set_title("body outline + the chord that is reported", fontsize=9, loc="left")
        axes[2].text(0.02, 0.02, txt, transform=axes[2].transAxes, fontsize=8,
                     va="bottom", color="#ffe08a",
                     bbox=dict(fc="#000000cc", ec="none", pad=3))

        fig.tight_layout(pad=0.5)
        out = Path(out_dir) / f"{case}_{name}_height.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out.name}  chord={h_mm:.1f}mm", flush=True)


def main():
    ct_dir, lab_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sys.argv[4:] or [p.name.split("_")[0] for p in sorted(lab_dir.glob("*_label.nii.gz"))[:4]]
    for c in cases:
        lp = lab_dir / f"{c}_label.nii.gz"
        cp = ct_dir / f"{c}_ct.nii.gz"
        if not (lp.exists() and cp.exists()):
            print(f"  {c}: missing volume")
            continue
        one(cp, lp, out_dir, vids=(20, 22, 24))


if __name__ == "__main__":
    main()
