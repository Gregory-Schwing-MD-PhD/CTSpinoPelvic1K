"""render_hardware_gallery.py — one clean 3-D render per implant type, for the website.

The site should show what is in the cohort, not how it was found. Four implant types are
represented and each looks completely different, which is the whole point: a threaded
cylindrical cage, a femoral stem with an acetabular cup, three parallel cannulated screws,
and iliosacral screws crossing a joint. A reader who sees them side by side understands the
class list without reading a word of it.

RENDERED ALONE, NOT IN CONTEXT. In situ every one of these is buried inside bone -- that is
why they were invisible in the gallery in the first place. Isolated, each is legible from a
single angle.

LIT LIKE METAL. A cool grey with a tight specular highlight reads as titanium; the same
geometry in bone colour reads as a peculiar bone. The background is the site's paper tone so
the renders sit on the page rather than on a box of white.

    python scripts/render_hardware_gallery.py --verify hardware_review/verify \\
        --out ../openspineconsortium.github.io/assets/img/hardware
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render3d import surface_mesh, fit_camera, render          # noqa: E402

# id -> (case to render it from, label, the view that shows it best)
TYPES = [
    (80, "0515", "Hip arthroplasty", (0, -1, 0), (0, 0, 1)),
    (77, "0068", "Interbody cage", (0, -1, 0), (0, 0, 1)),
    (82, "0247", "Cannulated screws", (1, 0, 0), (0, 0, 1)),
    (81, "1035", "Sacroiliac screws", (0, -1, 0), (0, 0, 1)),
]

TITANIUM = (206, 210, 218)
PAPER = (247, 245, 239)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=900)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    verify = Path(a.verify)

    made = []
    for hid, case, name, direction, up in TYPES:
        f = verify / case / f"{case}_label_hw.nii.gz"
        if not f.exists():
            print(f"  ! {case}: {f} missing")
            continue
        img = nib.load(str(f))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
        m = lab == hid
        if not m.any():
            print(f"  ! {case}: no voxels with id {hid}")
            continue

        idx = np.argwhere(m)
        lo = np.maximum(idx.min(0) - 3, 0)
        hi = np.minimum(idx.max(0) + 4, np.array(m.shape))
        sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
        # smooth just enough to lose the voxel staircase without eating a thread
        verts, norms = surface_mesh(m[sl], sp, step=1, smooth=0.6)
        if verts is None or not len(verts):
            print(f"  ! {case}: mesh came out empty")
            continue

        cam = fit_camera(verts, direction, up, a.size, a.size, margin=1.18)
        im, cover = render([(verts, norms, TITANIUM)], cam, bg=PAPER, supersample=3)

        fig, ax = plt.subplots(figsize=(a.size / 200, a.size / 200), dpi=200)
        ax.imshow(im)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        dst = out / f"hw_{hid}.png"
        fig.savefig(dst, bbox_inches="tight", pad_inches=0,
                    facecolor=tuple(c / 255 for c in PAPER))
        plt.close(fig)

        mm3 = float(m.sum()) * float(np.prod(sp))
        made.append((hid, name, case, mm3, dst))
        print(f"  {name:<22} id {hid}  from {case}  {mm3:>9,.0f} mm3  "
              f"{len(verts):>7,} verts  -> {dst.name}")

    print(f"\n  {len(made)} render(s) in {out}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
