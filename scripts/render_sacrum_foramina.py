"""scripts/render_sacrum_foramina.py — count the sacral foramina, because counting up cannot.

THE AMBIGUITY THIS BREAKS. A case whose lowest rib sits on "L1" leaving four free lumbar
below has two readings that look identical from above: a real lumbar rib on L1, or a
sacralized L5 with the whole count shifted so that "L1" is really T12 and its rib is a
normal one. Counting vertebrae up from S1 cannot decide it, because where the count starts
is the very thing in question.

The sacrum can. A normal sacrum has FOUR pairs of anterior foramina; a sacrum that has
absorbed L5 has FIVE. That is a property of the bone itself and does not depend on any
vertebra label being right.

So this draws the sacrum alone, large, as a coronal MIP -- the foramina read as holes --
plus a couple of coronal slabs, since a MIP through the whole depth can smear the lowest
pair into the pelvic brim. The count is left to the reader: the number is the finding, and
an automatic hole-count on a segmentation mask would be a worse witness than an eye.

    python scripts/render_sacrum_foramina.py --labels data/v5_final --cases 0231
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

SACRUM, S1 = 26, 29
INK, SURFACE = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "font.size": 8})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="sacrum_sheets")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for stem in [c.strip() for c in a.cases.split(",") if c.strip()]:
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}"); continue
        img = nib.as_closest_canonical(nib.load(str(fp)))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        zoom = np.array(img.header.get_zooms()[:3], float)
        m = np.isin(lab, [SACRUM, S1])
        if not m.any():
            print(f"  ! no sacrum in {stem}"); continue

        xs = np.nonzero(m.any(axis=(1, 2)))[0]
        ys = np.nonzero(m.any(axis=(0, 2)))[0]
        zs = np.nonzero(m.any(axis=(0, 1)))[0]
        sub = m[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]

        # full-depth MIP, then anterior and mid slabs -- the anterior foramina are the
        # ones to count, and a whole-depth projection can fill them in from behind
        ny = sub.shape[1]
        views = [("full-depth MIP", sub.max(axis=1)),
                 ("anterior third", sub[:, :ny // 3].max(axis=1)),
                 ("middle third", sub[:, ny // 3:2 * ny // 3].max(axis=1))]

        fig, axes = plt.subplots(1, len(views), figsize=(4.0 * len(views), 7.0), dpi=190)
        for ax, (name, v) in zip(axes, views):
            ax.imshow(v.T, origin="lower", cmap="bone_r", vmin=0, vmax=1.35,
                      aspect=zoom[2] / zoom[0], interpolation="nearest")
            ax.set_title(name, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        h = (zs.max() - zs.min() + 1) * zoom[2]
        w = (xs.max() - xs.min() + 1) * zoom[0]
        fig.suptitle(f"{stem} — sacrum, {w:.0f} x {h:.0f} mm.  "
                     f"COUNT THE FORAMINA PAIRS: 4 = normal, 5 = L5 sacralized "
                     f"(image right = patient right)", fontsize=9)
        fig.tight_layout()
        p = out / f"{stem}_sacrum.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"  {stem}: sacrum {w:.0f} x {h:.0f} mm, {int(m.sum())} vox -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
