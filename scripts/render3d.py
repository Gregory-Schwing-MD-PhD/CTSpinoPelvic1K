"""scripts/render3d.py — a small software renderer, so a diagram can show real 3-D bone.

WHY THIS EXISTS. The first version of the measurement diagrams drew a first-hit depth map:
every voxel column reduced to the distance of its nearest surface, shaded light-to-dark. That
is a silhouette with a gradient on it. It reads as a smudge, and next to the gallery's lit
WebGL viewers it looks like a mistake.

WHAT IT DOES INSTEAD. Marching cubes turns the label into an actual triangulated surface with
vertex normals, the vertices are transformed by a real camera, and the surface is shaded with
a key light, a fill and a rim term. The result is a 3-D render of the specimen, not a picture
of its shadow.

WHY SPLATTING AND NOT TRIANGLE RASTERISATION. A lumbar vertebra is a couple of hundred
thousand triangles and there is no GPU here -- vtk, pyrender, moderngl and OpenGL are all
absent. Looping over triangles in Python is far too slow. Splatting the VERTICES is fully
vectorised: every vertex becomes a small disc, a z-buffer keeps the nearest, and because
marching cubes puts vertices roughly one voxel apart the surface closes at any sensible
output size. Supersampling then downsampling gives the edges back.

THE Z-BUFFER TRICK. numpy has minimum.at but no argmin.at, and shading needs to know WHICH
vertex won each pixel, not just how far away it was. Depth is quantised into the high bits of
an int64 and the vertex index packed into the low bits, so a single np.minimum.at resolves
depth and identity together and the index falls out by masking.

THE CAMERA IS THE POINT. project() applies exactly the transform the render used, so a
landmark in patient millimetres lands on the pixel where that anatomy was drawn. A diagram
whose annotation is computed by a different path than its picture is a diagram that can be
subtly, invisibly wrong.
"""
from __future__ import annotations

import numpy as np
from skimage import measure


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class Camera:
    """Orthographic look-at camera. Units in, pixels out, and it can be asked in reverse."""

    def __init__(self, eye, target, up, width, height, extent):
        eye = np.asarray(eye, float)
        target = np.asarray(target, float)
        fwd = _unit(target - eye)
        right = _unit(np.cross(fwd, np.asarray(up, float)))
        true_up = np.cross(right, fwd)
        self.R = np.stack([right, true_up, fwd])       # world -> camera
        self.eye = eye
        self.width, self.height = int(width), int(height)
        # extent is the half-width of the view in world units; the vertical follows aspect
        self.sx = (width / 2) / extent
        self.sy = self.sx

    def to_camera(self, pts):
        return (np.asarray(pts, float) - self.eye) @ self.R.T

    def project(self, pts):
        """World mm -> (x_px, y_px, depth). y is already flipped for image coordinates."""
        c = self.to_camera(pts)
        x = c[..., 0] * self.sx + self.width / 2
        y = self.height / 2 - c[..., 1] * self.sy
        return np.stack([x, y, c[..., 2]], axis=-1)


def surface_mesh(mask, spacing, step=1, level=0.5, smooth=1.5):
    """Marching cubes on a boolean mask -> (verts in world mm, vertex normals).

    THE MASK IS BLURRED FIRST, AND THAT IS THE WHOLE DIFFERENCE. Marching cubes on a binary
    volume can only ever place the surface on voxel faces, so every normal points along an
    axis and the shading comes out as a field of little facets flickering between a handful
    of brightness values -- which is what the first render looked like. Blurring turns the
    mask into a smooth field whose isosurface at 0.5 sits where the boundary really is, and
    the gradient there is a genuine surface normal. Half a voxel of blur is enough to fix the
    lighting and far too little to move the surface anywhere a reader would notice.
    """
    if mask.sum() < 64:
        return None, None
    from scipy import ndimage
    if smooth:
        # SMOOTH A DISTANCE FIELD, NOT THE MASK. Blurring the mask and taking the 0.5
        # isosurface erodes thin bone: a sheet three voxels thick peaks at 0.58 under a
        # 1.9-voxel Gaussian, so any local thinning drops below the threshold and opens a
        # hole. That is what put holes back in the iliac wing, which is about four voxels
        # thick here -- the blur made them, not the label, which has three cavities in six
        # hundred thousand voxels.
        #
        # A signed distance field is symmetric about the surface, so smoothing it moves the
        # boundary without thinning what is behind it. The isosurface is 0, and a sheet
        # survives as long as its centre stays positive.
        inside = ndimage.distance_transform_edt(mask)
        outside = ndimage.distance_transform_edt(~mask)
        vol = ndimage.gaussian_filter((inside - outside).astype(np.float32), smooth)
        level = 0.0
    else:
        vol = mask.astype(np.float32)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            vol, level=level, spacing=tuple(float(s) for s in spacing), step_size=step)
    except (RuntimeError, ValueError):
        return None, None
    # SKIMAGE ALREADY RETURNS OUTWARD NORMALS. Negating them here -- on the assumption that
    # they followed the gradient into the object -- pointed every normal away from the
    # lights, so both diffuse terms clipped to zero and the entire render came out at flat
    # ambient. Checked against a sphere of known orientation: raw normals dot the outward
    # radial direction at +0.998.
    return verts, _unit(normals)


def render(groups, cam, bg=(244, 242, 236), supersample=2, radius=None):
    """Shade a set of (verts, normals, rgb) groups into an RGB image plus a coverage mask.

    Groups are rendered into one shared z-buffer, so a structure genuinely occludes the ones
    behind it rather than being pasted over them in list order.
    """
    W, H = cam.width * supersample, cam.height * supersample
    big = Camera(cam.eye, cam.eye + cam.R[2], cam.R[1], W, H, cam.width / 2 / cam.sx)

    PACK = np.int64(1) << np.int64(24)
    zbuf = np.full(W * H, np.iinfo(np.int64).max, np.int64)

    allv, alln, allc = [], [], []
    for verts, normals, rgb in groups:
        if verts is None or not len(verts):
            continue
        allv.append(verts)
        alln.append(normals)
        allc.append(np.repeat(np.asarray(rgb, np.float32)[None, :], len(verts), axis=0))
    if not allv:
        return None, None
    V = np.concatenate(allv)
    N = np.concatenate(alln)
    C = np.concatenate(allc)
    # the vertex index shares an int64 with the quantised depth; if it ever outgrew its
    # field the z-buffer would silently return the wrong vertex and shade from it
    assert len(V) < PACK, f"{len(V)} vertices exceeds the {int(PACK)} index field"

    P = big.project(V)
    z = P[:, 2]
    finite = np.isfinite(z)
    z0, z1 = float(np.min(z[finite])), float(np.max(z[finite]))
    zq = np.clip(((z - z0) / max(1e-6, z1 - z0) * ((1 << 38) - 1)).astype(np.int64), 0, None)

    xs = np.rint(P[:, 0]).astype(np.int64)
    ys = np.rint(P[:, 1]).astype(np.int64)
    idx = np.arange(len(V), dtype=np.int64)

    # A SPLAT HAS TO BE AS BIG AS THE GAP BETWEEN VERTICES, and that gap depends on the
    # output size, the supersampling factor and how much of the frame the specimen fills --
    # none of which a fixed radius can know. A hard-coded 1 left the first render as a field
    # of separate dots. Estimate the spacing from the projected footprint: vertices lie on a
    # surface, so n of them spread over the silhouette's area sit about sqrt(area / n) apart.
    if radius is None:
        seen = finite & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if seen.sum() > 16:
            bw = max(1.0, float(xs[seen].max() - xs[seen].min()))
            bh = max(1.0, float(ys[seen].max() - ys[seen].min()))
            # roughly half a bounding box is ink for a shape like a vertebra
            spacing = np.sqrt(bw * bh * 0.55 / max(1, int(seen.sum())))
            # generous: a splat slightly too big loses a little crispness, one
            # slightly too small leaves the dither pattern the first attempt had
            radius = int(max(1, np.ceil(spacing * 1.45)))
        else:
            radius = 1
    offs = [(dx, dy) for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius]
    for dx, dy in offs:
        x = xs + dx
        y = ys + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & finite
        flat = y[ok] * W + x[ok]
        np.minimum.at(zbuf, flat, zq[ok] * PACK + idx[ok])

    hit = zbuf != np.iinfo(np.int64).max
    won = (zbuf[hit] % PACK).astype(np.int64)

    # ---- shading: key, fill, rim. Camera-space normals, so the light follows the view.
    n_cam = N @ big.R.T
    n = n_cam[won]
    key_dir = _unit(np.array([0.38, 0.66, -0.65]))
    fill_dir = _unit(np.array([-0.58, 0.20, -0.79]))
    lam = np.clip(n @ key_dir, 0, 1)
    fil = np.clip(n @ fill_dir, 0, 1)
    rim = np.power(np.clip(1.0 - np.abs(n[:, 2]), 0, 1), 2.6)
    half = _unit(key_dir + np.array([0, 0, -1.0]))
    spec = np.power(np.clip(n @ half, 0, 1), 60.0)

    base = C[won]
    shade = (0.34 + 0.74 * lam[:, None] + 0.24 * fil[:, None]) * base \
        + 26.0 * rim[:, None] + 185.0 * spec[:, None]

    img = np.empty((H * W, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    img[hit] = np.clip(shade, 0, 255)
    img = img.reshape(H, W, 3)
    cover = hit.reshape(H, W)

    # PINHOLES ARE A SPLATTING ARTEFACT, NOT ANATOMY. Even with the radius scaled to vertex
    # density a few background pixels survive inside the silhouette, and at full resolution
    # they still totalled a couple of thousand across a pelvis. A hole small enough to be
    # one of these is filled from the surface around it; anything larger is left alone,
    # because an obturator foramen is also an enclosed gap and it is supposed to be there.
    from scipy import ndimage as _nd
    enclosed = _nd.binary_fill_holes(cover) & ~cover
    if enclosed.any():
        lt, k = _nd.label(enclosed)
        if k:
            sizes = _nd.sum(enclosed, lt, range(1, k + 1))
            tiny = np.isin(lt, [i + 1 for i, sz in enumerate(sizes)
                                if sz <= 400 * supersample * supersample])
            if tiny.any():
                # nearest-surface colour: cheap, and correct for a hole a few pixels wide
                idx = _nd.distance_transform_edt(~cover, return_distances=False,
                                                 return_indices=True)
                img[tiny] = img[idx[0][tiny], idx[1][tiny]]
                cover = cover | tiny

    if supersample > 1:
        s = supersample
        img = img.reshape(H // s, s, W // s, s, 3).mean(axis=(1, 3))
        cover = cover.reshape(H // s, s, W // s, s).mean(axis=(1, 3)) > 0.5
    return np.clip(img, 0, 255).astype(np.uint8), cover


def fit_camera(points, direction, up, width, height, margin=1.12):
    """A camera looking along `direction` that frames `points` with a little air."""
    pts = np.asarray(points, float)
    centre = (pts.min(0) + pts.max(0)) / 2
    d = _unit(np.asarray(direction, float))
    span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    eye = centre - d * (span * 2.0 + 50.0)
    cam = Camera(eye, centre, up, width, height, 1.0)
    c = cam.to_camera(pts)
    half = max(float(np.abs(c[:, 0]).max()), float(np.abs(c[:, 1]).max()) * width / height)
    cam.sx = (width / 2) / (half * margin)
    cam.sy = cam.sx
    return cam
