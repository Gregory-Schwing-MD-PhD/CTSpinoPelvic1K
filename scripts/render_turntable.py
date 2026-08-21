"""scripts/render_turntable.py — server-side 3-D stills of the label surfaces.

WHY STILLS. The WebGL viewer failed repeatedly in the browser for reasons that were hard to
see from here, and a gallery that does not render is worth less than one that cannot spin.
Rendering on the grid removes every browser variable at once: no GPU, no import map, no
typed-array alignment, no WebGL context limit. The page then shows PNGs, which cannot fail.

STILL INTERACTIVE. Frames are rendered on a turntable, so the page can swap them under a
drag and the object turns. That is an object movie -- the technique QuickTime VR used --
and it needs nothing but an <img> and a pointer handler.

HOW THE SURFACE IS FOUND. For each viewing angle the volume is rotated about the
superior-inferior axis, and along each viewing ray the FIRST non-zero voxel is taken. That
gives three things at once: which structure is visible (its colour), how far away it is
(depth shading), and the silhouette. It is a first-hit surface render, and being pure numpy
it is fast and has no dependencies beyond scipy's rotate.

Lighting is depth plus a cheap surface gradient: without a normal term the render reads as
a flat silhouette, and the shapes are the entire point.

    python scripts/render_turntable.py --labels data/v5_final --cases 0431 --frames 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
from matplotlib import image as mpimg

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402

BG = np.array([250, 250, 248], np.float32)
MIN_VOX = 150
DOWNSAMPLE = 2


def read_colours(path):
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(errors="replace").splitlines():
            t = line.split()
            if len(t) >= 8 and t[0].isdigit():
                out[int(t[0])] = np.array([int(t[1]), int(t[2]), int(t[3])], np.float32)
    return out


def render(lab, colours, angle_deg, ignore):
    """First-hit surface render looking along +y after rotating about the z axis."""
    if angle_deg:
        # order=0 so labels stay labels: any interpolation invents ids that do not exist
        vol = ndimage.rotate(lab, angle_deg, axes=(0, 1), reshape=False, order=0,
                             mode="constant", cval=0)
    else:
        vol = lab

    occ = (vol > 0) & (vol != ignore)
    if not occ.any():
        return None
    # first hit along the viewing axis
    hit = occ.argmax(axis=1)                       # (x, z)
    any_hit = occ.any(axis=1)
    xs, zs = np.nonzero(any_hit)
    ids = vol[xs, hit[xs, zs], zs]
    depth = hit[xs, zs].astype(np.float32)

    img = np.repeat(BG[None, None, :], vol.shape[0], 0).repeat(vol.shape[2], 1)
    base = np.zeros((len(xs), 3), np.float32)
    for uid in np.unique(ids):
        c = colours.get(int(uid))
        if c is None:
            c = np.array([180, 180, 180], np.float32)
        base[ids == uid] = c

    # depth cue: nearer is brighter. Normalised per frame so the whole object is used.
    d0, d1 = float(depth.min()), float(depth.max())
    near = 1.0 - (depth - d0) / max(1.0, d1 - d0)
    shade = 0.55 + 0.45 * near

    # a cheap normal term from the depth gradient, so the surface reads as curved rather
    # than as a flat silhouette in flat colour
    dmap = np.full(any_hit.shape, np.nan, np.float32)
    dmap[xs, zs] = depth
    gy, gx = np.gradient(np.nan_to_num(dmap, nan=float(d1)))
    n = np.clip(1.0 - 0.06 * np.hypot(gx, gy), 0.45, 1.0)
    shade = shade * n[xs, zs]

    img[xs, zs] = np.clip(base * shade[:, None], 0, 255)
    # transpose so superior is up, and flip so the view reads as a radiograph would
    out = np.transpose(img, (1, 0, 2))[::-1]
    return out.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--descriptor", default="data/itksnap_v5_labels.txt")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", default="gallery_stills")
    a = ap.parse_args()

    colours = read_colours(a.descriptor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []

    for stem in [c.strip() for c in a.cases.split(",") if c.strip()]:
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        img = nib.as_closest_canonical(nib.load(str(fp)))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        # drop specks so they do not pepper the silhouette, and shrink for speed
        for vid in np.unique(lab):
            if vid and (lab == vid).sum() < MIN_VOX:
                lab[lab == vid] = 0
        lab = lab[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
        # crop to the object: most of a CT volume is air
        occ = np.argwhere(lab > 0)
        if not len(occ):
            continue
        lo, hi = occ.min(0), occ.max(0) + 1
        lab = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        frames = []
        for i in range(a.frames):
            ang = 360.0 * i / a.frames
            fr = render(lab, colours, ang, LS.IGNORE_LABEL)
            if fr is None:
                continue
            p = out / f"{stem}_{i:02d}.png"
            mpimg.imsave(p, fr)
            frames.append(p.name)
        kb = sum((out / f).stat().st_size for f in frames) / 1024
        print(f"  {stem}: {len(frames)} frames, {kb:.0f} kB total")
        index.append({"case": stem, "frames": frames})

    (out / "index.json").write_text(json.dumps({"cases": index, "frames": a.frames},
                                               indent=1))
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
