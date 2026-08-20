#!/usr/bin/env python3
"""Render a rotating 3-D view of a spinopelvic segmentation to MP4, in parallel.

    python scripts/render_spinopelvic_video.py \
        --label scratchpad/case22/22_label.nii.gz --out spinopelvic_rotation.mp4 \
        --frames 180 --workers 16

WHY IT IS SHAPED THIS WAY
-------------------------
matplotlib's 3-D backend depth-sorts every polygon on every draw, so ~105k triangles cost
about 10 s per frame single-threaded -- 30 minutes for a 180-frame turntable. Frames are
independent, so they parallelise perfectly.

The mesh is built ONCE and cached to an .npz. Marching cubes over 29 labels takes ~25 s;
letting each of 16 workers repeat it would add ~7 minutes of duplicated work and dominate
the saving. Workers memory-map the cache, render their assigned frames to PNG, and ffmpeg
assembles them in order.

GEOMETRY
--------
Surfaces are transformed by the AFFINE into world millimetres, not left in voxel indices.
Index space bakes in the voxel anisotropy (0.78 x 0.80 x 0.78 here, worse on many scans),
which stretches the spine along one axis -- subtly enough to look fine and be wrong. When
the volume is decimated the affine is scaled to match; decimating the array alone renders
the anatomy at 1/d scale against unscaled axes, which also looks plausible.

Axis limits are the data bounds and `box_aspect` is set proportional to those ranges, so
a millimetre is the same length on every axis AND the subject fills the frame. Equal
limits on all three axes would preserve the scale but leave the spine adrift in whitespace
in a volume that is much taller than it is wide.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# v4 (VerSe-native) ids. Ribs occupy 34-57, hence ranges rather than a literal list.
GROUPS = [
    ("lumbar",   list(range(20, 26)), None),        # L1-L6, graded cranio-caudal
    ("sacrum",   [26],                "#8C6BB1"),
    ("S1",       [29],                "#D95F02"),   # the endplate driving PI/SS/PT
    ("hips",     [30, 31],            "#7FB3D5"),
    ("femurs",   [32, 33],            "#2C7FB8"),
    ("ribs",     list(range(34, 58)), "#D9D9D9"),
    ("thoracic", list(range(8, 20)),  "#FEE8C8"),
]
LUMBAR_RAMP = ["#FDD49E", "#FDBB84", "#FC8D59", "#EF6548", "#D7301F", "#990000"]


def _apply_affine(affine, v):
    """voxel (i,j,k) -> world mm; marching_cubes returns (row, col, slice) = (i, j, k)."""
    return (affine[:3, :3] @ v.T).T + affine[:3, 3]


def build_meshes(vol, affine, step, min_vox):
    """[(verts_mm, faces, colour, name)] per structure, in world millimetres."""
    from skimage import measure

    out = []
    for name, ids, colour in GROUPS:
        present = [i for i in ids if int((vol == i).sum()) >= min_vox]
        for n, i in enumerate(present):
            # pad so a structure touching the volume edge still closes into a surface
            mp = np.pad((vol == i).astype(np.uint8), 1)
            try:
                v, f, _, _ = measure.marching_cubes(mp, level=0.5, step_size=step)
            except (RuntimeError, ValueError):
                continue
            v -= 1.0                                       # undo the pad
            c = colour or LUMBAR_RAMP[min(n, len(LUMBAR_RAMP) - 1)]
            out.append((_apply_affine(affine, v), f, c, f"{name}:{i}"))
    return out


def _save_cache(path, meshes):
    """Flatten to one array pair + offsets: npz cannot hold a ragged list cheaply."""
    verts = np.vstack([m[0] for m in meshes]).astype(np.float32)
    vo = np.cumsum([0] + [len(m[0]) for m in meshes])
    faces = np.vstack([m[1] + vo[k] for k, m in enumerate(meshes)]).astype(np.int32)
    fo = np.cumsum([0] + [len(m[1]) for m in meshes])
    np.savez_compressed(path, verts=verts, faces=faces, vo=vo, fo=fo,
                        colours=np.array([m[2] for m in meshes]),
                        names=np.array([m[3] for m in meshes]))


def _load_cache(path):
    z = np.load(path, allow_pickle=False)
    verts, faces, fo = z["verts"], z["faces"], z["fo"]
    return [(verts, faces[fo[k]:fo[k + 1]], str(c))
            for k, c in enumerate(z["colours"])]


def _figure(meshes, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = meshes[0][0]
    lo, hi = verts.min(0), verts.max(0)
    pad = 0.02 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    rng = hi - lo

    fig = plt.figure(figsize=(6.0, 7.6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0E0E12")
    ax.set_facecolor("#0E0E12")
    for v, f, c in meshes:
        pc = Poly3DCollection(v[f], alpha=1.0, linewidths=0)
        pc.set_facecolor(c)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(rng / rng.max()))   # equal mm/axis AND fills the frame
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def render_chunk(args):
    """Worker: render an explicit list of frame indices to PNG. Top-level for spawn."""
    cache, frame_ids, n_frames, elev, dpi, tmpdir = args
    meshes = _load_cache(cache)
    fig, ax = _figure(meshes, dpi)
    import matplotlib.pyplot as plt
    written = []
    for k in frame_ids:
        ax.view_init(elev=elev, azim=360.0 * k / n_frames)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        p = os.path.join(tmpdir, f"f{k:05d}.png")
        import imageio.v2 as imageio
        imageio.imwrite(p, buf)
        written.append(p)
    plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--step", type=int, default=2,
                    help="marching-cubes step_size; higher = fewer triangles")
    ap.add_argument("--min_vox", type=int, default=25)
    ap.add_argument("--elev", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--count_only", action="store_true")
    a = ap.parse_args(argv)

    import multiprocessing as mp
    import tempfile
    import time

    import nibabel as nib

    t0 = time.time()
    img = nib.load(a.label)
    vol = np.asanyarray(img.dataobj)
    aff = img.affine.copy()
    if a.downsample > 1:
        d = a.downsample
        vol = vol[::d, ::d, ::d]
        aff[:3, :3] = aff[:3, :3] * d          # MUST scale with the data -- see docstring
    print(f"volume {vol.shape}  structures {len(np.unique(vol)) - 1}", flush=True)

    meshes = build_meshes(vol, aff, a.step, a.min_vox)
    tri = sum(len(f) for _, f, _, _ in meshes)
    print(f"{len(meshes)} surfaces, {tri:,} triangles  ({time.time() - t0:.0f}s)",
          flush=True)
    if a.count_only:
        for _, f, _, n in meshes:
            print(f"    {n:16s} {len(f):7,d} tri")
        return 0

    tmpdir = tempfile.mkdtemp(prefix="xrsp_render_")
    cache = os.path.join(tmpdir, "meshes.npz")
    _save_cache(cache, meshes)
    del meshes

    nw = max(1, min(a.workers, a.frames, mp.cpu_count()))
    # INTERLEAVED, not contiguous blocks. Cost per frame varies with how much geometry
    # faces the camera, so contiguous chunks make one worker inherit all the expensive
    # angles and everyone else waits on it.
    chunks = [(cache, list(range(w, a.frames, nw)), a.frames, a.elev, a.dpi, tmpdir)
              for w in range(nw)]
    print(f"rendering {a.frames} frames across {nw} workers "
          f"({len(chunks[0][1])} frames each)", flush=True)

    t1 = time.time()
    with mp.Pool(nw) as pool:
        done = 0
        for got in pool.imap_unordered(render_chunk, chunks):
            done += len(got)
            print(f"  {done}/{a.frames} frames", flush=True)
    dt = time.time() - t1
    print(f"rendered in {dt:.0f}s ({dt / a.frames:.2f}s/frame wall, "
          f"{nw}x parallel)", flush=True)

    import imageio_ffmpeg
    import subprocess
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(a.fps),
           "-i", os.path.join(tmpdir, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           # even dimensions are required by yuv420p; the pad is a no-op when already even
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out]
    subprocess.run(cmd, check=True)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB, "
          f"{a.frames} frames @ {a.fps} fps, {a.frames / a.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
