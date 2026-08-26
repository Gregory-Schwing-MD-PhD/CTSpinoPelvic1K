"""scripts/make_measure_diagrams.py — the construction behind a plotted number, in 3-D.

A distribution says how a quantity varies and nothing about what it is. "Pelvic incidence"
is a phrase until you have seen the two points and the angle between them on somebody's
actual sacrum.

THESE ARE RENDERS, NOT SILHOUETTES. The first version reduced each voxel column to the depth
of its nearest surface and shaded it light to dark. That is a shadow with a gradient on it,
and beside the gallery's lit WebGL viewers it read as a mistake. render3d builds a real
surface with marching cubes and shades it with a key, fill and rim; these are pictures of
bone.

THE ANNOTATION USES THE RENDER'S OWN CAMERA. cam.project maps a landmark in patient
millimetres onto the pixel where that anatomy was drawn, so a line cannot drift from the
structure it points at. Everything downstream of the camera is in image pixels, which is
also the SVG's coordinate system -- there is no second mapping to get wrong.

THE CONSTRUCTION IS IMPORTED, NOT REIMPLEMENTED. _endplate, _femoral_head and body_of come
from the extractors that produced the numbers in the plots. A diagram that re-derived the
geometry could drift from the extractor and explain a measurement nobody made, which is
worse than no diagram because it would look right.

THEY ARE SIZED TO SIT INSIDE A PLOT. No caption block and no scale bar competing for room:
the panel around them supplies the context, and the diagram is an inset, not a figure.

    python scripts/make_measure_diagrams.py --out ../openspineconsortium.github.io/assets/gallery/measures
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render3d import surface_mesh, fit_camera, render                # noqa: E402
from extract_surgical_morphometrics import (                          # noqa: E402
    _endplate, _femoral_head, SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R, MIN_VOX,
)

INK = "#1A1C18"
MUTED = "#8d8d86"
ACCENT = "#c8452c"          # the quantity being measured
SECOND = "#2f6fb0"          # the reference it is measured against
THIRD = "#b8860b"           # a second measured quantity in the same picture
PAPER = "#f4f2ec"

BONE = (232, 220, 200)
BONE_FOCUS = (214, 196, 168)
ACCENT_BONE = (228, 190, 172)   # the structure the panel is actually about
FONT = "ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif"

# viewing directions in RAS: +x patient right, +y anterior, +z superior.
# `direction` is where the camera LOOKS, so an anterior view looks posteriorly.
VIEW_ANT = dict(direction=(0, -1, 0), up=(0, 0, 1))
VIEW_LAT = dict(direction=(1, 0, 0), up=(0, 0, 1))     # from the left; anterior to the left


def load(case, labels_dir):
    """Canonicalised to RAS for ANALYSIS AND DRAWING ONLY. Nothing here is written back."""
    img = nib.as_closest_canonical(nib.load(str(Path(labels_dir) / f"{case}_label.nii.gz")))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    sp = np.array(img.header.get_zooms()[:3], float)
    return lab, sp


def mesh_of(lab, sp, ids, step=1, clip=None):
    """Surface of the given ids, in world millimetres, cropped so marching cubes is cheap.

    `clip` is (axis, keep_from_index): everything below that index on that axis is dropped
    before the surface is built, cutting the specimen open. A measurement taken in a midline
    column is invisible on an intact lateral view -- the line lands on the outer wall, above
    and in front of where the number comes from -- so the volume is cut at the midline and
    the reader sees the plane the measurement is actually made in.
    """
    m = np.isin(lab, list(ids)) if hasattr(ids, "__iter__") else (lab == ids)
    if clip is not None:
        ax, keep_from = clip
        cut = np.zeros_like(m)
        sel = [slice(None)] * 3
        sel[ax] = slice(int(keep_from), None)
        cut[tuple(sel)] = m[tuple(sel)]
        m = cut
    if m.sum() < MIN_VOX // 8:
        return None, None
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, np.array(m.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    v, n = surface_mesh(m[sl], sp, step=step)
    if v is None:
        return None, None
    return v + lo * sp, n


def png_b64(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Overlay:
    """SVG drawn in the render's pixel space. Points arrive as world mm and are projected."""

    def __init__(self, img, cam):
        self.h, self.w = img.shape[:2]
        self.cam = cam
        self.b64 = png_b64(img)
        self.parts = []

    def px(self, world):
        p = self.cam.project(np.asarray(world, float)[None, :])[0]
        return float(p[0]), float(p[1])

    def line(self, a, b, colour, width=2.4, dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
            f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round" '
            f'opacity="{opacity}"{d}/>')

    def dot(self, p, colour, r=5.0):
        self.parts.append(
            f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{colour}" '
            f'stroke="{PAPER}" stroke-width="2"/>')

    def text(self, p, s, colour=INK, size=20, anchor="start", weight="700", dx=0, dy=0):
        """Halo then glyph, as two elements — no renderer honours paint-order reliably."""
        common = (f'x="{p[0] + dx:.1f}" y="{p[1] + dy:.1f}" font-size="{size}" '
                  f'font-family="{FONT}" font-weight="{weight}" text-anchor="{anchor}"')
        self.parts.append(
            f'<text {common} fill="none" stroke="{PAPER}" stroke-width="5" '
            f'stroke-opacity="0.95" stroke-linejoin="round">{s}</text>')
        self.parts.append(f'<text {common} fill="{colour}">{s}</text>')

    def arc(self, centre, v1, v2, radius, colour, width=2.6):
        a1 = float(np.arctan2(v1[1], v1[0]))
        a2 = float(np.arctan2(v2[1], v2[0]))
        d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        pts = [(centre[0] + radius * np.cos(a1 + d * i / 48),
                centre[1] + radius * np.sin(a1 + d * i / 48)) for i in range(49)]
        self.parts.append(
            '<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
            f'" fill="none" stroke="{colour}" stroke-width="{width}"/>')
        a = a1 + d / 2
        return (centre[0] + radius * 1.5 * np.cos(a), centre[1] + radius * 1.5 * np.sin(a))

    def svg(self, title):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'role="img" aria-label="{title}" preserveAspectRatio="xMidYMid meet" '
            f'width="100%">\n<title>{title}</title>\n'
            f'<rect width="100%" height="100%" fill="{PAPER}"/>\n'
            f'<image x="0" y="0" width="{self.w}" height="{self.h}" '
            f'href="data:image/png;base64,{self.b64}"/>\n'
            + "\n".join(self.parts) + '\n</svg>\n')


def _u2(v):
    n = float(np.hypot(v[0], v[1]))
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)


SIZE = 560


# ── constructions ───────────────────────────────────────────────────────────────────────
def diagram_pelvic_incidence(case, labels_dir):
    """PI, sacral slope and pelvic tilt: one picture, because they are one identity.

    Three separate panels could not say the thing a reader needs to carry away -- that PI is
    fixed by the shape of the pelvis, that SS and PT divide it according to posture, and that
    PI = SS + PT always.
    """
    lab, sp = load(case, labels_dir)
    have = {v: (lab == v) for v in (SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R)
            if (lab == v).sum() >= MIN_VOX}
    if not (FEM_L in have and FEM_R in have):
        return None, "no femora"
    cl = _femoral_head(have[FEM_L], have.get(HIP_L), sp)
    cr = _femoral_head(have[FEM_R], have.get(HIP_R), sp)
    if cl is None or cr is None:
        return None, "no femoral heads"
    fem = (cl + cr) / 2
    s1c, s1n = (_endplate(have[S1], sp, True) if S1 in have else (None, None))
    if s1c is None and SACRUM in have:
        s1c, s1n = _endplate(have[SACRUM], sp, True)
    if s1c is None or s1n is None:
        return None, "no S1 endplate"

    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(         # noqa: E731
        np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))))
    v = fem - s1c
    vs = np.array([0.0, v[1], v[2]])
    ns = np.array([0.0, s1n[1], s1n[2]])
    pi_deg = 180.0 - ang(ns, vs)
    ss_deg = ang(ns, np.array([0.0, 0.0, 1.0]))
    pt_deg = pi_deg - ss_deg

    # THE HIP BONES ARE NOT DRAWN. In a lateral view the near hemipelvis is a solid wall
    # in front of everything the diagram is about: it hid the sacrum completely, so the
    # S1 endplate -- one of the two landmarks -- sat on top of bone that was not it. The
    # femoral heads are what the construction needs from the pelvis, and the femurs carry
    # them. Sacrum at full resolution because its endplate is the measurement; femurs
    # decimated because they are only there to show where the axis is.
    groups, allv = [], []
    for ids, colour, step in (([SACRUM, S1], BONE_FOCUS, 1),
                              ([FEM_L, FEM_R], BONE, 1)):
        vv, nn = mesh_of(lab, sp, ids, step=step)
        if vv is not None:
            groups.append((vv, nn, colour))
            allv.append(vv)
    if not groups:
        return None, "nothing to render"
    cam = fit_camera(np.concatenate(allv), width=SIZE, height=SIZE, margin=1.16, **VIEW_LAT)
    img, _ = render(groups, cam, supersample=3)
    o = Overlay(img, cam)

    P_s1, P_fem = o.px(s1c), o.px(fem)
    n2 = _u2(np.subtract(o.px(s1c + s1n * 30.0), P_s1))
    plate = (-n2[1], n2[0])
    to_fem = _u2(np.subtract(P_fem, P_s1))

    o.line((P_s1[0] - plate[0] * 52, P_s1[1] - plate[1] * 52),
           (P_s1[0] + plate[0] * 52, P_s1[1] + plate[1] * 52), SECOND, 3.4)
    o.line(P_s1, (P_s1[0] + n2[0] * 96, P_s1[1] + n2[1] * 96), SECOND, 2.2, dash="7 6")
    o.line((P_s1[0] - 70, P_s1[1]), (P_s1[0] + 70, P_s1[1]), MUTED, 1.8, dash="5 6")
    o.line((P_fem[0], P_fem[1] - 20), (P_fem[0], P_fem[1] + 96), MUTED, 1.8, dash="5 6")
    o.line(P_s1, P_fem, ACCENT, 3.6)

    lp = o.arc(P_s1, n2, to_fem, 54, ACCENT, 3.0)
    o.text(lp, f"PI {pi_deg:.0f}&#176;", ACCENT, 25, "middle")
    lp = o.arc(P_s1, (np.sign(plate[0]) or 1.0, 0.0), plate, 30, THIRD, 2.4)
    # pushed clear of the S1 endplate caption, which sits immediately right of the vertex
    o.text(lp, f"SS {ss_deg:.0f}&#176;", THIRD, 20, "middle", dx=-26, dy=30)
    lp = o.arc(P_fem, (0.0, -1.0), (-to_fem[0], -to_fem[1]), 40, SECOND, 2.4)
    o.text(lp, f"PT {pt_deg:.0f}&#176;", SECOND, 20, "middle", dx=-8)

    o.dot(P_s1, SECOND, 6)
    o.dot(P_fem, ACCENT, 6)
    o.text(P_s1, "S1 endplate", INK, 17, "start", "600", dx=16, dy=-14)
    o.text(P_fem, "femoral head axis", INK, 17, "middle", "600", dy=26)
    return o.svg("Pelvic incidence, sacral slope and pelvic tilt"), None


def diagram_endplate_flare(case, labels_dir):
    """Endplate width against mid-body width, which the site plots as two separate things.

    A healthy body is an hourglass seen from the front. Spurs grow at the rim, so the rim
    widens while the waist does not and the ratio rises. One vertebra says that; two
    histograms cannot.
    """
    from extract_degenerative import body_of                        # noqa: E402
    lab, sp = load(case, labels_dir)
    vid, name = 23, "L4"
    m = lab == vid
    if m.sum() < MIN_VOX:
        return None, f"no {name}"
    b = body_of(m)
    if b is None:
        return None, "no body carved"

    idx = np.argwhere(b)
    zs = idx[:, 2]

    def span(p0, p1):
        sel = idx[(zs >= np.percentile(zs, p0)) & (zs <= np.percentile(zs, p1))]
        if len(sel) < 60:
            return None
        return (float(np.percentile(sel[:, 0], 1)), float(np.percentile(sel[:, 0], 99)),
                float(np.mean(sel[:, 2])))

    rim, waist = span(80, 100), span(38, 62)
    if rim is None or waist is None:
        return None, "band too thin"

    vv, nn = mesh_of(lab, sp, [vid], step=1)
    if vv is None:
        return None, "nothing to render"
    cam = fit_camera(vv, width=SIZE, height=SIZE, margin=1.2, **VIEW_ANT)
    img, _ = render([(vv, nn, BONE)], cam, supersample=3)
    o = Overlay(img, cam)

    # THE BAR FOLLOWS THE ENDPLATE, NOT THE PAGE. Drawn horizontally it cut across a
    # vertebra that is tilted in the coronal plane, so it crossed the body instead of lying
    # on the plate it measures. The rim's own slope is fitted from the top surface of each
    # column and both bars are drawn along it. The NUMBER is unchanged -- it is still the
    # extractor's axis-aligned width, which measurement showed is within a tenth of a
    # millimetre of the body's own axis anyway -- only the line is honest about the tilt.
    top_z = {}
    for xx, _yy, zz in idx[(zs >= np.percentile(zs, 80))]:
        top_z[xx] = max(top_z.get(xx, -1), zz)
    xs_fit = np.array(sorted(top_z))
    if len(xs_fit) >= 8:
        lo_q, hi_q = np.percentile(xs_fit, [10, 90])
        keep = xs_fit[(xs_fit >= lo_q) & (xs_fit <= hi_q)]
        slope = float(np.polyfit(keep * sp[0],
                                 np.array([top_z[x] for x in keep]) * sp[2], 1)[0])
    else:
        slope = 0.0

    out = {}
    ymed = float(np.median(idx[:, 1]))
    for key, (lo, hi, zc), colour, dy in (("rim", rim, ACCENT, -18),
                                          ("waist", waist, SECOND, 32)):
        xmid = (lo + hi) / 2 * sp[0]
        z0 = zc * sp[2]
        pa = np.array([lo * sp[0], ymed * sp[1], z0 + slope * (lo * sp[0] - xmid)])
        pb = np.array([hi * sp[0], ymed * sp[1], z0 + slope * (hi * sp[0] - xmid)])
        width_mm = abs(hi - lo) * sp[0]
        out[key] = width_mm
        label = ("superior endplate" if key == "rim" else "mid-body")
        _span(o, pa, pb, colour, f"{label} {width_mm:.0f} mm", dy=dy, tick=10)

    ratio = out["rim"] / out["waist"] if out["waist"] > 5 else float("nan")
    o.text((o.w / 2, 30), f"flare {ratio:.2f}", INK, 20, "middle", "600")
    return o.svg("Endplate width against mid-body width"), None


def diagram_disc_height(case, labels_dir):
    """The interbody gap, measured where a radiologist measures it.

    Endplates are concave, so the space is narrowest rim to rim and taking it there reads 4
    to 6 mm against a published 8 to 12. Each column through a midline box is measured
    separately and the median reported.
    """
    from extract_degenerative import body_of                        # noqa: E402
    lab, sp = load(case, labels_dir)
    upper, lower, name = 23, 24, "L4-5"
    bu = body_of(lab == upper) if (lab == upper).sum() >= MIN_VOX else None
    bl = body_of(lab == lower) if (lab == lower).sum() >= MIN_VOX else None
    if bu is None or bl is None:
        return None, f"no bodies for {name}"

    iu, il = np.argwhere(bu), np.argwhere(bl)
    cx = float(np.median(np.concatenate([iu[:, 0], il[:, 0]])))
    cy = float(np.median(np.concatenate([iu[:, 1], il[:, 1]])))
    rx = max(2, int(round(8.0 / sp[0])))
    ry = max(2, int(round(8.0 / sp[1])))
    su = iu[(np.abs(iu[:, 0] - cx) <= rx) & (np.abs(iu[:, 1] - cy) <= ry)]
    sl_ = il[(np.abs(il[:, 0] - cx) <= rx) & (np.abs(il[:, 1] - cy) <= ry)]
    if len(su) < 40 or len(sl_) < 40:
        return None, "midline column too thin"
    cols = {}
    for xx, yy, zz in su:
        cols.setdefault((xx, yy), [None, None])
        c0 = cols[(xx, yy)][0]
        cols[(xx, yy)][0] = zz if c0 is None else min(c0, zz)
    for xx, yy, zz in sl_:
        if (xx, yy) in cols:
            c1 = cols[(xx, yy)][1]
            cols[(xx, yy)][1] = zz if c1 is None else max(c1, zz)
    pairs = [(a, b2) for a, b2 in cols.values() if a is not None and b2 is not None]
    if len(pairs) < 15:
        return None, "too few columns"
    gap = float(np.median([a - b2 for a, b2 in pairs]))
    h_mm = max(0.0, gap * sp[2])
    z_up = float(np.median([a for a, _ in pairs]))
    z_lo = z_up - gap

    # cut at the midline column the number comes from, so the reader is looking at the
    # plane the measurement is made in rather than the wall in front of it
    groups, allv = [], []
    for vid, colour in ((upper, BONE_FOCUS), (lower, BONE)):
        vv, nn = mesh_of(lab, sp, [vid], step=1, clip=(0, int(cx)))
        if vv is not None:
            groups.append((vv, nn, colour))
            allv.append(vv)
    if not groups:
        return None, "nothing to render"
    cam = fit_camera(np.concatenate(allv), width=SIZE, height=SIZE, margin=1.14, **VIEW_LAT)
    img, _ = render(groups, cam, supersample=3)
    o = Overlay(img, cam)

    yb = cy * sp[1]
    p_up = o.px(np.array([cx * sp[0], yb, z_up * sp[2]]))
    p_lo = o.px(np.array([cx * sp[0], yb, z_lo * sp[2]]))
    o.line((p_up[0] - 84, p_up[1]), (p_up[0] + 84, p_up[1]), SECOND, 3.0)
    o.line((p_lo[0] - 84, p_lo[1]), (p_lo[0] + 84, p_lo[1]), SECOND, 3.0)
    o.line(p_up, p_lo, ACCENT, 4.2)
    o.text(((p_up[0] + p_lo[0]) / 2, (p_up[1] + p_lo[1]) / 2),
           f"{h_mm:.1f} mm", ACCENT, 25, "start", dx=16, dy=8)
    o.text((p_up[0], p_up[1]), "midline column", INK, 17, "middle", "600", dy=-18)
    return o.svg("Disc height at the midline"), None


VIEW_AXIAL = dict(direction=(0, 0, 1), up=(0, 1, 0))   # from below: patient left on the right


def _span(o, a_world, b_world, colour, label, dy=-14, tick=11):
    """A measured distance between two points in space, with end ticks and a label."""
    a, b = o.px(a_world), o.px(b_world)
    o.line(a, b, colour, 3.4)
    d = _u2((b[0] - a[0], b[1] - a[1]))
    perp = (-d[1], d[0])
    for pnt in (a, b):
        o.line((pnt[0] - perp[0] * tick, pnt[1] - perp[1] * tick),
               (pnt[0] + perp[0] * tick, pnt[1] + perp[1] * tick), colour, 3.0)
    o.text(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), label, colour, 22, "middle", dy=dy)
    return a, b


def _scene(lab, sp, spec, view, margin=1.16):
    """Render the listed (ids, colour, step) groups and return an Overlay."""
    groups, allv = [], []
    for ids, colour, step in spec:
        vv, nn = mesh_of(lab, sp, ids, step=step)
        if vv is not None:
            groups.append((vv, nn, colour))
            allv.append(vv)
    if not groups:
        return None
    cam = fit_camera(np.concatenate(allv), width=SIZE, height=SIZE, margin=margin, **view)
    img, _ = render(groups, cam, supersample=3)
    return Overlay(img, cam)


def _pelvis_landmarks(lab, sp):
    """The three points every spinopelvic angle is read from."""
    have = {v: (lab == v) for v in (SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R)
            if (lab == v).sum() >= MIN_VOX}
    if not (FEM_L in have and FEM_R in have):
        return None
    cl = _femoral_head(have[FEM_L], have.get(HIP_L), sp)
    cr = _femoral_head(have[FEM_R], have.get(HIP_R), sp)
    if cl is None or cr is None:
        return None
    s1c, s1n = (_endplate(have[S1], sp, True) if S1 in have else (None, None))
    if s1c is None and SACRUM in have:
        s1c, s1n = _endplate(have[SACRUM], sp, True)
    if s1c is None or s1n is None:
        return None
    return dict(fem=(cl + cr) / 2, cl=cl, cr=cr, s1c=s1c, s1n=s1n, have=have)


def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(
        np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))))


SPINE_SPEC = [([SACRUM, S1], BONE_FOCUS, 1), ([FEM_L, FEM_R], BONE, 1)]


def diagram_sacral_slope(case, labels_dir):
    """The S1 endplate against the horizontal. One angle, one plate, nothing else drawn."""
    lab, sp = load(case, labels_dir)
    L = _pelvis_landmarks(lab, sp)
    if L is None:
        return None, "landmarks unavailable"
    ss = _ang(np.array([0.0, L["s1n"][1], L["s1n"][2]]), np.array([0.0, 0.0, 1.0]))
    o = _scene(lab, sp, [([SACRUM, S1], BONE_FOCUS, 1)], VIEW_LAT, margin=1.2)
    if o is None:
        return None, "nothing to render"
    P = o.px(L["s1c"])
    n2 = _u2(np.subtract(o.px(L["s1c"] + L["s1n"] * 30.0), P))
    plate = (-n2[1], n2[0])
    o.line((P[0] - 96, P[1]), (P[0] + 96, P[1]), MUTED, 2.4, dash="7 6")
    o.line((P[0] - plate[0] * 96, P[1] - plate[1] * 96),
           (P[0] + plate[0] * 96, P[1] + plate[1] * 96), ACCENT, 4.0)
    lp = o.arc(P, (np.sign(plate[0]) or 1.0, 0.0), plate, 62, ACCENT, 3.0)
    o.text(lp, f"SS {ss:.0f}&#176;", ACCENT, 26, "middle")
    o.dot(P, ACCENT, 6)
    o.text((o.w / 2, o.h - 16), "S1 endplate against the horizontal", INK, 18, "middle", "600")
    return o.svg("Sacral slope"), None


def diagram_pelvic_tilt(case, labels_dir):
    """The line from the femoral head axis to the S1 midpoint, against the vertical."""
    lab, sp = load(case, labels_dir)
    L = _pelvis_landmarks(lab, sp)
    if L is None:
        return None, "landmarks unavailable"
    v = L["fem"] - L["s1c"]
    pi = 180.0 - _ang(np.array([0.0, L["s1n"][1], L["s1n"][2]]),
                      np.array([0.0, v[1], v[2]]))
    ss = _ang(np.array([0.0, L["s1n"][1], L["s1n"][2]]), np.array([0.0, 0.0, 1.0]))
    pt = pi - ss
    o = _scene(lab, sp, SPINE_SPEC, VIEW_LAT)
    if o is None:
        return None, "nothing to render"
    P_s1, P_fem = o.px(L["s1c"]), o.px(L["fem"])
    to_s1 = _u2((P_s1[0] - P_fem[0], P_s1[1] - P_fem[1]))
    o.line((P_fem[0], P_fem[1] - 130), (P_fem[0], P_fem[1] + 40), MUTED, 2.4, dash="7 6")
    o.line(P_fem, P_s1, ACCENT, 4.0)
    lp = o.arc(P_fem, (0.0, -1.0), to_s1, 66, ACCENT, 3.0)
    o.text(lp, f"PT {pt:.0f}&#176;", ACCENT, 26, "middle")
    o.dot(P_fem, ACCENT, 6)
    o.dot(P_s1, SECOND, 6)
    o.text(P_fem, "femoral head axis", INK, 17, "middle", "600", dy=28)
    o.text(P_s1, "S1 midpoint", INK, 17, "start", "600", dx=14, dy=-12)
    return o.svg("Pelvic tilt"), None


def diagram_lumbar_lordosis(case, labels_dir):
    """L1 superior endplate against S1 superior endplate — not the pelvic-incidence figure.

    These panels were previously showing the PI construction, which shares two of its points
    with nothing here: lordosis is an angle between two ENDPLATES and involves no femur.
    """
    lab, sp = load(case, labels_dir)
    have = {v: (lab == v) for v in (20, SACRUM, S1) if (lab == v).sum() >= MIN_VOX}
    if 20 not in have:
        return None, "no L1"
    s1c, s1n = (_endplate(have[S1], sp, True) if S1 in have else (None, None))
    if s1c is None and SACRUM in have:
        s1c, s1n = _endplate(have[SACRUM], sp, True)
    if s1n is None:
        return None, "no S1 endplate"
    l1c, l1n = _endplate(have[20], sp, True)
    if l1n is None:
        return None, "no L1 endplate"
    ll = _ang(l1n, s1n)

    ids = [v for v in range(20, 26) if (lab == v).sum() >= MIN_VOX] + [SACRUM, S1]
    o = _scene(lab, sp, [(ids, BONE, 1)], VIEW_LAT, margin=1.3)
    if o is None:
        return None, "nothing to render"
    for c, n, colour, name in ((l1c, l1n, ACCENT, "L1 superior"),
                               (s1c, s1n, SECOND, "S1 superior")):
        P = o.px(c)
        d2 = _u2(np.subtract(o.px(c + n * 30.0), P))
        plate = (-d2[1], d2[0])
        o.line((P[0] - plate[0] * 62, P[1] - plate[1] * 62),
               (P[0] + plate[0] * 62, P[1] + plate[1] * 62), colour, 4.0)
        o.dot(P, colour, 6)
        o.text(P, name, INK, 17, "start", "600", dx=16, dy=-10)
    # along the bottom: at the top it collided with the first endplate label and, on a
    # tall lumbar spine, with the frame itself
    o.text((o.w / 2, o.h - 18), f"lumbar lordosis {ll:.0f}&#176;", ACCENT, 26, "middle")
    return o.svg("Lumbar lordosis"), None


def diagram_rib_ratio(case, labels_dir):
    """The lowest rib against the one above it, both drawn, in the view that shows length."""
    lab, sp = load(case, labels_dir)
    pairs = []
    for lo, hi, side in ((45, 44, "left"), (57, 56, "right")):
        if (lab == lo).sum() > 200 and (lab == hi).sum() > 200:
            pairs.append((lo, hi, side))
    if not pairs:
        return None, "no rib 11/12 pair"
    lo, hi, side = pairs[0]

    def chord(vid):
        idx = np.argwhere(lab == vid).astype(float) * sp
        d = idx[:, 0][:, None] - idx[:, 0][None, :] if False else None
        # longest chord via the extremes of the principal axis: exact enough for a picture
        c = idx - idx.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        t = c @ u
        return idx[int(np.argmin(t))], idx[int(np.argmax(t))], float(t.max() - t.min())

    a0, a1, len_lo = chord(lo)
    b0, b1, len_hi = chord(hi)
    # THE VERTEBRAE ARE PART OF THE PICTURE, and both of them. A rib ending in mid-air
    # does not read as a rib, and the pair is what shows the two ribs are consecutive
    # levels rather than any two bones. Which vertebrae they are follows from the rib
    # numbers rather than being hard-coded, so a case whose lowest rib is not the twelfth
    # still gets the right two.
    n_lo = (lo - 34) % 12 + 1
    want = [7 + n_lo, 7 + n_lo - 1]                     # rib n articulates with T-n
    vert_ids = [v for v in want if (lab == v).sum() > 2000]
    o = _scene(lab, sp, [([lo], ACCENT_BONE, 1), ([hi], BONE, 1),
                         (vert_ids, BONE_FOCUS, 1)], VIEW_ANT, margin=1.24)
    if o is None:
        return None, "nothing to render"
    _span(o, b0, b1, SECOND, f"rib above {len_hi:.0f} mm", dy=-16)
    _span(o, a0, a1, ACCENT, f"lowest rib {len_lo:.0f} mm", dy=26)
    o.text((o.w / 2, o.h - 18),
           f"ratio {len_lo / max(len_hi, 1e-6):.2f}", INK, 24, "middle")
    return o.svg("Lowest rib length as a fraction of the rib above"), None


def _pelvic_span(case, labels_dir, which):
    """Three widths across the bony pelvis, each drawn alone on the same coronal view."""
    lab, sp = load(case, labels_dir)
    have = {v: (lab == v) for v in (SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R)
            if (lab == v).sum() >= MIN_VOX}
    # the sacrum and the lowest lumbar levels give the pelvis a reference the eye can use
    lumbar = [v for v in (23, 24, 25) if (lab == v).sum() >= MIN_VOX]
    spec = [([HIP_L, HIP_R], BONE, 1), ([SACRUM, S1], BONE_FOCUS, 1)]
    if lumbar:
        spec.append((lumbar, BONE_FOCUS, 1))
    if which == "acetabular":
        spec.append(([FEM_L, FEM_R], BONE, 1))
    o = _scene(lab, sp, spec, VIEW_ANT, margin=1.1)
    if o is None:
        return None, "nothing to render"

    if which == "iliac":
        if HIP_L not in have or HIP_R not in have:
            return None, "no hips"
        allidx = np.argwhere(have[HIP_L] | have[HIP_R])
        zc = float(np.median(allidx[:, 2]))
        xlo, xhi = float(allidx[:, 0].min()), float(allidx[:, 0].max())
        # AT THE HEIGHT OF THE POINTS IT SPANS. Drawing the bar at a fixed percentile of
        # pelvic height put it across the ilia at a level that had nothing to do with the
        # widest points, so it read as tilted against the anatomy. The two extreme columns
        # decide the height.
        ytop = allidx[allidx[:, 2] > np.percentile(allidx[:, 2], 80)]
        yy = float(np.median(ytop[:, 1])) if len(ytop) else float(np.median(allidx[:, 1]))
        z_lo = float(np.median(allidx[allidx[:, 0] <= xlo + 1][:, 2]))
        z_hi = float(np.median(allidx[allidx[:, 0] >= xhi - 1][:, 2]))
        zz = (z_lo + z_hi) / 2
        p0 = np.array([xlo * sp[0], yy * sp[1], zz * sp[2]])
        p1 = np.array([xhi * sp[0], yy * sp[1], zz * sp[2]])
        _span(o, p0, p1, ACCENT, f"{(xhi - xlo) * sp[0]:.0f} mm", dy=-16)
        o.text((o.w / 2, o.h - 16), "widest span across the iliac crests",
               INK, 18, "middle", "600")
        return o.svg("Pelvic width across the iliac crests"), None

    if which == "acetabular":
        if FEM_L not in have or FEM_R not in have:
            return None, "no femora"
        cl = _femoral_head(have[FEM_L], have.get(HIP_L), sp)
        cr = _femoral_head(have[FEM_R], have.get(HIP_R), sp)
        if cl is None or cr is None:
            return None, "no femoral heads"
        _span(o, cl, cr, ACCENT, f"{float(np.linalg.norm(cl - cr)):.0f} mm", dy=-16)
        for c in (cl, cr):
            o.dot(o.px(c), ACCENT, 7)
        o.text((o.w / 2, o.h - 16), "centre to centre between the femoral heads",
               INK, 18, "middle", "600")
        return o.svg("Width across the hip joints"), None

    sac = np.zeros_like(lab, bool)
    for v in (SACRUM, S1):
        if v in have:
            sac |= have[v]
    if not sac.any():
        return None, "no sacrum"
    sidx = np.argwhere(sac)
    ztop = np.percentile(sidx[:, 2], 85)
    base = sidx[sidx[:, 2] >= ztop]
    if len(base) < 100:
        return None, "sacral base too small"
    xlo, xhi = float(base[:, 0].min()), float(base[:, 0].max())
    yy, zz = float(np.median(base[:, 1])), float(np.median(base[:, 2]))
    p0 = np.array([xlo * sp[0], yy * sp[1], zz * sp[2]])
    p1 = np.array([xhi * sp[0], yy * sp[1], zz * sp[2]])
    _span(o, p0, p1, ACCENT, f"{(xhi - xlo) * sp[0]:.0f} mm", dy=-16)
    o.text((o.w / 2, o.h - 16), "across both alae at the S1 level", INK, 18, "middle", "600")
    return o.svg("Sacral base breadth"), None


def diagram_bi_iliac(case, labels_dir):
    return _pelvic_span(case, labels_dir, "iliac")


def diagram_bi_acetabular(case, labels_dir):
    return _pelvic_span(case, labels_dir, "acetabular")


def diagram_sacral_base(case, labels_dir):
    return _pelvic_span(case, labels_dir, "base")


def diagram_tp_span(case, labels_dir):
    """Tip to tip across the transverse processes — from ABOVE.

    The previous version drew this on a coronal view, where the transverse processes point
    towards the camera and their tips are the part you cannot see. Seen from above they are
    the widest thing in the picture, which is what the measurement says.
    """
    lab, sp = load(case, labels_dir)
    vid = 22 if (lab == 22).sum() >= MIN_VOX else 23
    m = lab == vid
    if m.sum() < MIN_VOX:
        return None, "no mid-lumbar vertebra"
    idx = np.argwhere(m)
    xlo, xhi = float(idx[:, 0].min()), float(idx[:, 0].max())
    lo_pts = idx[idx[:, 0] <= xlo + 1]
    hi_pts = idx[idx[:, 0] >= xhi - 1]
    p0 = np.array([xlo * sp[0], float(np.median(lo_pts[:, 1])) * sp[1],
                   float(np.median(lo_pts[:, 2])) * sp[2]])
    p1 = np.array([xhi * sp[0], float(np.median(hi_pts[:, 1])) * sp[1],
                   float(np.median(hi_pts[:, 2])) * sp[2]])
    o = _scene(lab, sp, [([vid], BONE, 1)], VIEW_AXIAL, margin=1.18)
    if o is None:
        return None, "nothing to render"
    _span(o, p0, p1, ACCENT, f"{(xhi - xlo) * sp[0]:.0f} mm", dy=-16)
    o.text((o.w / 2, o.h - 16), "tip to tip across the transverse processes",
           INK, 18, "middle", "600")
    return o.svg("Transverse process span"), None


def diagram_body_height(case, labels_dir):
    """Anterior body height — the tallest column in the anterior half, seen from the side.

    Height is a sagittal measurement and the previous figure was not a sagittal view of it.
    """
    from extract_degenerative import body_of                        # noqa: E402
    lab, sp = load(case, labels_dir)
    vid = 23
    m = lab == vid
    if m.sum() < MIN_VOX:
        return None, "no L4"
    body = body_of(m)
    if body is None:
        return None, "no body carved"
    bidx = np.argwhere(body)
    bx = float(np.median(bidx[:, 0]))
    xlo = max(0, int(round(bx - 5.0 / sp[0])))
    xhi = int(round(bx + 5.0 / sp[0])) + 1
    slab = body[xlo:xhi]
    ys = np.nonzero(slab.any(axis=(0, 2)))[0]
    col = {}
    for y in ys:
        c = slab[:, y, :]
        if c.sum() < 3:
            continue
        zc = np.nonzero(c.any(axis=0))[0]
        col[y] = (int(zc.min()), int(zc.max()))
    if len(col) < 6:
        return None, "body too thin to measure"
    ymid = (min(col) + max(col)) / 2.0
    ant = {y: v for y, v in col.items() if y > ymid}
    post = {y: v for y, v in col.items() if y <= ymid}
    if not ant or not post:
        return None, "no anterior/posterior split"
    ya = max(ant, key=lambda y: ant[y][1] - ant[y][0])
    yp = max(post, key=lambda y: post[y][1] - post[y][0])
    o = _scene(lab, sp, [([vid], BONE, 1)], VIEW_LAT, margin=1.22)
    if o is None:
        return None, "nothing to render"
    # ON THE NEAR LIP, NOT DOWN THE MIDLINE. The heights are measured in a mid-sagittal
    # slab, and drawing them at that slab's x buries both lines inside the bone in a lateral
    # view -- visible only as a smudge over the surface in front of them. The camera looks
    # along +x, so the nearest surface is the body's minimum x, and a line drawn a couple of
    # millimetres proud of it sits on the lip the measurement describes.
    # THE BAR'S ENDS MUST BE ON THE SURFACE THE BAR IS DRAWN ON. Taking the height from
    # the mid-sagittal slab and drawing it out at the near wall put one end on the deep
    # endplate edge and the other on the superficial one, which is the skew Greg saw. The
    # column is re-read at the near wall so both ends sit on the same lip; the NUMBER stays
    # the mid-sagittal one the release reports, and is labelled as such.
    # EACH WALL, BETWEEN ITS OWN ENDPLATE CORNERS, ON THE SURFACE FACING THE CAMERA.
    # Two things were wrong: the height came from the mid-sagittal slab but was drawn out at
    # the near wall, so one end sat on the deep endplate edge and the other on the
    # superficial one; and the posterior bar was hidden inside the bone because it was drawn
    # at the same x as the anterior one, behind the arch. Both bars are now taken from the
    # near-surface slab at their own y, which is where a reader measures them.
    bidx_all = np.argwhere(body)
    near_i = int(bidx_all[:, 0].min())
    near_x = float(near_i)
    thick = max(2, int(round(3.0 / sp[0])))
    surf = body[near_i:near_i + thick, :, :]
    for y, colour, name, dx in ((ya, ACCENT, "anterior", 0), (yp, SECOND, "posterior", 0)):
        z0, z1 = col[y]
        zc_surf = np.nonzero(surf[:, y, :].any(axis=0))[0]
        if len(zc_surf) >= 2:
            z0, z1 = int(zc_surf.min()), int(zc_surf.max())
        h = (z1 - z0 + 1) * sp[2]
        p0 = np.array([near_x * sp[0], y * sp[1], z0 * sp[2]])
        p1 = np.array([near_x * sp[0], y * sp[1], z1 * sp[2]])
        a, b = o.px(p0), o.px(p1)
        o.line(a, b, colour, 4.0)
        for pnt in (a, b):
            o.line((pnt[0] - 13, pnt[1]), (pnt[0] + 13, pnt[1]), colour, 3.2)
        o.text(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), f"{name} {h:.0f} mm",
               colour, 20, "middle", dx=dx, dy=4)
    o.text((o.w / 2, o.h - 16), "tallest column in each half of the body",
           INK, 17, "middle", "600")
    return o.svg("Anterior body height"), None


def _pedicle_of(m, sp):
    """-> (narrower side width mm, side, slice info) for one vertebra, or None.

    Reproduces extract_surgical_morphometrics: the narrowest bone run outboard of the canal
    in the FRONT THIRD of it per slice, the median of those across slices per side, and the
    narrower of the two sides.
    """
    idx = np.argwhere(m)
    if not len(idx):
        return None
    zlo, zhi = int(np.percentile(idx[:, 2], 20)), int(np.percentile(idx[:, 2], 80))
    per_side = {"l": [], "r": []}
    for z in range(zlo, zhi + 1):
        s2 = m[:, :, z]
        if s2.sum() < 60:
            continue
        h2 = ndimage.binary_fill_holes(s2) & ~s2
        if not h2.any():
            continue
        c2, n2 = ndimage.label(h2)
        sz = ndimage.sum(h2, c2, range(1, n2 + 1))
        cn = c2 == (int(np.argmax(sz)) + 1)
        if cn.sum() < 20:
            continue
        cy = np.nonzero(cn.any(axis=0))[0]
        y_front = cy[cy >= cy.min() + 0.66 * (cy.max() - cy.min())]
        if len(y_front) < 2:
            continue
        for side in ("l", "r"):
            runs = []
            for y in y_front:
                cx = np.nonzero(cn[:, y])[0]
                bx = np.nonzero(s2[:, y])[0]
                if len(cx) < 2 or len(bx) < 2:
                    continue
                out = bx[bx < cx.min()] if side == "l" else bx[bx > cx.max()]
                if len(out) < 2:
                    continue
                d = np.diff(out)
                brk = np.nonzero(d > 1)[0]
                seg = (out[brk[-1] + 1:] if side == "l" and len(brk)
                       else out[:brk[0] + 1] if side == "r" and len(brk)
                       else out)
                if len(seg) >= 2:
                    runs.append((float(seg.max() - seg.min() + 1) * sp[0], z, y, seg))
            if runs:
                per_side[side].append(min(runs, key=lambda t: t[0]))
    scored = {}
    for side, v in per_side.items():
        if not v:
            continue
        med = float(np.median([t[0] for t in v]))
        rep = min(v, key=lambda t: abs(t[0] - med))
        scored[side] = (med, side, rep)
    if not scored:
        return None
    return min(scored.values(), key=lambda t: t[0])


def diagram_pedicle_width(case, labels_dir):
    """The pedicle isthmus, by the extractor's own search, seen from above.

    A first attempt measured all bone outboard of the canal over the canal's whole depth and
    read 15-20 mm against a released 2.8-7.0. At the back of the canal that bone is lamina,
    which is thinner than any pedicle, and at the front it is the transverse process merging
    into the arch, which is far wider. The isthmus is a waist in the front third, and the
    honest statistic across slices is the median, not the extreme.
    """
    lab, sp = load(case, labels_dir)
    # THE RELEASE REPORTS A MINIMUM OVER LEVELS, not a measurement of one level, and it
    # discards any level outside 3.5-20 mm before taking it. Measuring L4 alone drew 9.1 mm
    # against a published 7.0 that came from L3 -- the right number for the wrong vertebra.
    PLAUSIBLE = (3.5, 20.0)
    per_level = {}
    for vid in (20, 21, 22, 23, 24, 25):
        if (lab == vid).sum() >= MIN_VOX:
            got = _pedicle_of(lab == vid, sp)
            if got:
                per_level[vid] = got
    ok = {v: g for v, g in per_level.items() if PLAUSIBLE[0] <= g[0] <= PLAUSIBLE[1]}
    if not ok:
        return None, "no plausible pedicle at any level"
    best = min(ok, key=lambda v: ok[v][0])
    m = lab == best
    idx = np.argwhere(m)
    width_mm, side, (_, z, y, seg) = ok[best]
    o = _scene(lab, sp, [([best], BONE, 1)], VIEW_AXIAL, margin=1.2)
    if o is None:
        return None, "nothing to render"
    p0 = np.array([float(seg.min()) * sp[0], float(y) * sp[1], float(z) * sp[2]])
    p1 = np.array([float(seg.max()) * sp[0], float(y) * sp[1], float(z) * sp[2]])
    _span(o, p0, p1, ACCENT, f"{width_mm:.1f} mm", dy=-18, tick=9)
    o.text((o.w / 2, o.h - 18), "pedicle isthmus; the narrower side selects the screw",
           INK, 18, "middle", "600")
    return o.svg("Narrowest lumbar pedicle"), None


def diagram_hip_width(case, labels_dir):
    """Centre to centre across the femoral heads, by the method the plotted panel uses.

    extract_pelvic_shape approximates each head as the centroid of the superior quarter of
    the femur label, which is what the histogram beside this picture is built from. The
    contact-based fit in extract_surgical_morphometrics is more exact and gives a number
    23 mm smaller on this case; the diagram follows the panel, and the disagreement between
    the two release columns is recorded separately.
    """
    from extract_pelvic_shape import _head_centre                # noqa: E402
    lab, sp = load(case, labels_dir)
    have = {v: (lab == v) for v in (SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R)
            if (lab == v).sum() >= MIN_VOX}
    if FEM_L not in have or FEM_R not in have:
        return None, "no femora"
    cl = _head_centre(have[FEM_L], sp)
    cr = _head_centre(have[FEM_R], sp)
    if cl is None or cr is None:
        return None, "no head centres"
    o = _scene(lab, sp, [([HIP_L, HIP_R], BONE, 1), ([FEM_L, FEM_R], ACCENT_BONE, 1),
                         ([SACRUM, S1], BONE_FOCUS, 1)], VIEW_ANT, margin=1.1)
    if o is None:
        return None, "nothing to render"
    _span(o, cl, cr, ACCENT, f"{float(np.linalg.norm(cl - cr)):.0f} mm", dy=-18)
    for c in (cl, cr):
        o.dot(o.px(c), ACCENT, 8)
    o.text((o.w / 2, o.h - 18), "centre to centre between the femoral heads",
           INK, 18, "middle", "600")
    return o.svg("Width across the hip joints"), None


def _corridor(case, labels_dir, which):
    """The two boundaries of the lateral working window, each drawn on its own."""
    from extract_surgical_morphometrics import _body_mask          # noqa: E402
    lab, sp = load(case, labels_dir)
    have = {v: (lab == v) for v in (HIP_L, HIP_R, 23, 24, SACRUM, S1)
            if (lab == v).sum() >= MIN_VOX}
    if HIP_L not in have and HIP_R not in have:
        return None, "no hip bone"
    hips = np.zeros_like(lab, bool)
    for h in (HIP_L, HIP_R):
        if h in have:
            hips |= have[h]
    crest_k = int(np.nonzero(hips.any(axis=(0, 1)))[0].max())
    crest = crest_k * sp[2]
    hidx = np.argwhere(hips[:, :, crest_k])
    cx_h = float(np.median(hidx[:, 0])) if len(hidx) else float(lab.shape[0] / 2)
    cy_h = float(np.median(hidx[:, 1])) if len(hidx) else float(lab.shape[1] / 2)
    crest_pt = np.array([cx_h * sp[0], cy_h * sp[1], crest])

    ids = [HIP_L, HIP_R, 23, 24, SACRUM, S1]
    if which == "rib":
        lowest, low_pt = None, None
        for base in (34, 46):
            for n in range(1, 13):
                mm = lab == base + n
                if mm.sum() > 200:
                    zz = np.nonzero(mm.any(axis=(0, 1)))[0]
                    z0 = int(zz.min())
                    if lowest is None or z0 * sp[2] < lowest:
                        lowest = z0 * sp[2]
                        ri = np.argwhere(mm[:, :, z0])
                        low_pt = np.array([float(np.median(ri[:, 0])) * sp[0],
                                           float(np.median(ri[:, 1])) * sp[1], z0 * sp[2]])
                        ids = ids + [base + n]
        if low_pt is None:
            return None, "no rib"
        o = _scene(lab, sp, [(list(set(ids)), BONE, 1)], VIEW_ANT, margin=1.08)
        if o is None:
            return None, "nothing to render"
        a = np.array([low_pt[0], low_pt[1], crest])
        _span(o, low_pt, a, ACCENT, f"{lowest - crest:.0f} mm", dy=0)
        # the crest reference sits under the measurement rather than across the midline,
        # so the eye reads the two as one construction
        o.line(o.px(np.array([low_pt[0] - 70, low_pt[1], crest])),
               o.px(np.array([low_pt[0] + 70, low_pt[1], crest])), SECOND, 3.0, dash="8 6")
        o.text((o.w / 2, o.h - 18), "lowest rib down to the iliac crest",
               INK, 18, "middle", "600")
        return o.svg("Lowest rib to iliac crest"), None

    if 23 not in have or 24 not in have:
        return None, "no L4/L5"
    z4 = np.nonzero(_body_mask(have[23], sp).any(axis=(0, 1)))[0]
    z5 = np.nonzero(_body_mask(have[24], sp).any(axis=(0, 1)))[0]
    disc = (z4.min() + z5.max()) / 2 * sp[2]
    # THE SACRUM BELONGS IN THIS PICTURE. The measurement is the height of the iliac crest
    # above the L4-5 disc, and without the sacrum between the two ilia the reader has no
    # anatomy connecting the pelvis to the spine -- the vertebrae float above a ring.
    sac = [v for v in (SACRUM, S1) if (lab == v).sum() >= MIN_VOX]
    o = _scene(lab, sp, [([HIP_L, HIP_R], BONE, 1), (sac, BONE, 1),
                         ([23, 24], BONE_FOCUS, 1)], VIEW_ANT, margin=1.08)
    if o is None:
        return None, "nothing to render"
    top = np.array([crest_pt[0], crest_pt[1], crest])
    bot = np.array([crest_pt[0], crest_pt[1], disc])
    _span(o, bot, top, ACCENT, f"{crest - disc:+.0f} mm", dy=0)
    o.line(o.px(np.array([crest_pt[0] - 100, crest_pt[1], disc])),
           o.px(np.array([crest_pt[0] + 100, crest_pt[1], disc])), SECOND, 3.0, dash="8 6")
    o.text((o.w / 2, o.h - 18),
           "positive means the crest rises above the L4-5 disc", INK, 18, "middle", "600")
    return o.svg("Iliac crest above the L4-5 disc"), None


def diagram_rib_to_crest(case, labels_dir):
    return _corridor(case, labels_dir, "rib")


def diagram_crest_above_l45(case, labels_dir):
    return _corridor(case, labels_dir, "disc")


def diagram_canal_width(case, labels_dir):
    """Transverse diameter of the canal, at the slice the extractor measures it on."""
    from extract_level_gradients import _canal                     # noqa: E402
    lab, sp = load(case, labels_dir)
    vid = next((v for v in (23, 22, 24) if (lab == v).sum() >= MIN_VOX), None)
    if vid is None:
        return None, "no lumbar vertebra"
    m = lab == vid
    front, cw = _canal(m)
    if cw is None:
        return None, "no canal"
    idx = np.argwhere(m)
    # SCAN FOR THE RING RATHER THAN ASSUMING THE MIDDLE SLICE HAS IT. The median z of a
    # whole vertebra is wherever the body's bulk puts it, which is often below the arch --
    # fill_holes then finds nothing and the diagram fails outright.
    zc, big, area = None, None, 0
    for z in range(int(np.percentile(idx[:, 2], 20)), int(np.percentile(idx[:, 2], 80)) + 1):
        sl_z = m[:, :, z]
        if sl_z.sum() < 60:
            continue
        h = ndimage.binary_fill_holes(sl_z) & ~sl_z
        if not h.any():
            continue
        cc, ncc = ndimage.label(h)
        sizes = ndimage.sum(h, cc, range(1, ncc + 1))
        k = int(np.argmax(sizes))
        if sizes[k] > area:
            area, zc, big = float(sizes[k]), z, (cc == k + 1)
    if zc is None:
        return None, "no canal anywhere in the vertebra"
    sl = m[:, :, zc]
    xs = np.nonzero(big.any(axis=1))[0]
    ys = np.nonzero(big.any(axis=0))[0]
    yy = float(np.median(ys))
    p0 = np.array([float(xs.min()) * sp[0], yy * sp[1], zc * sp[2]])
    p1 = np.array([float(xs.max()) * sp[0], yy * sp[1], zc * sp[2]])
    o = _scene(lab, sp, [([vid], BONE, 1)], VIEW_AXIAL, margin=1.2)
    if o is None:
        return None, "nothing to render"
    _span(o, p0, p1, ACCENT, f"{cw * sp[0]:.0f} mm", dy=-16)
    o.text((o.w / 2, o.h - 18), "transverse diameter of the spinal canal",
           INK, 18, "middle", "600")
    return o.svg("Spinal canal width"), None


def diagram_endplate_width(case, labels_dir):
    """Superior endplate width, by the eroded-core method the gradient panel plots."""
    # `largest` and ER live inside extract_level_gradients.one() and cannot be imported;
    # they are reproduced here verbatim so the diagram measures what the panel plots.
    ER = 2

    def largest(mask):
        cc, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(mask, cc, range(1, n + 1))
        return cc == (int(np.argmax(sizes)) + 1)

    lab, sp = load(case, labels_dir)
    vid = next((v for v in (23, 22, 24) if (lab == v).sum() >= MIN_VOX), None)
    if vid is None:
        return None, "no lumbar vertebra"
    # THE BODY CUT AND THE SLICE RANGE BOTH COME FROM extract_level_gradients, not from
    # extract_degenerative. The two carve the body differently and they sample different
    # slices: taking the top five slices of the degenerative carve read 34.0 mm against a
    # published 44.9. The gradient panel measures from the 80th percentile of body height
    # to its top, on its own cut.
    from extract_level_gradients import _canal                     # noqa: E402
    m = lab == vid
    front, _cw = _canal(m)
    if front is None:
        ys = np.nonzero(m.any(axis=(0, 2)))[0]
        if len(ys) < 4:
            return None, "no body"
        front = ys.min() + 0.45 * (ys.max() - ys.min())
    body = np.zeros_like(m)
    body[:, int(np.ceil(front)):, :] = m[:, int(np.ceil(front)):, :]
    if body.sum() < 300:
        return None, "body cut failed"
    bidx = np.argwhere(body)
    ztop = int(np.percentile(bidx[:, 2], 80))
    zmax = int(bidx[:, 2].max())
    widths, cands = [], []
    for z in range(ztop, zmax + 1):
        sl = body[:, :, z]
        if sl.sum() < 40:
            continue
        core = ndimage.binary_erosion(sl, iterations=ER)
        big = largest(core) if core.any() else None
        if big is None or big.sum() < 0.35 * sl.sum():
            big = largest(sl)
            margin = 0
        else:
            margin = 2 * ER
        if big is None:
            continue
        xs = np.nonzero(big.any(axis=1))[0]
        if len(xs) < 3:
            continue
        lo_x, hi_x = np.percentile(xs, [1, 99])
        widths.append((hi_x - lo_x + 1 + margin) * sp[0])
        cands.append((widths[-1], lo_x, hi_x, z,
                      float(np.median(np.nonzero(big.any(axis=0))[0])), margin))
    if not widths or not cands:
        return None, "no endplate slices"
    w = float(np.median(widths))
    # DRAW THE SLICE THAT IS THE REPORTED NUMBER. `drawn` was reassigned every iteration, so
    # the bar ended up on the LAST slice of the loop -- the very top of the vertebra, where
    # the body has narrowed to a cap. That is why the line spanned about half the endplate
    # while the label read the full width.
    _, lo_x, hi_x, z, yy, margin = min(cands, key=lambda t: abs(t[0] - w))
    # the reported width adds the eroded margin back, so the bar has to as well
    half = margin / 2.0
    o = _scene(lab, sp, [([vid], BONE, 1)], VIEW_ANT, margin=1.2)
    if o is None:
        return None, "nothing to render"
    # FOLLOW THE PLATE. A horizontal bar across a body that is tilted in the coronal plane
    # cuts through it instead of lying on it -- the same fault the flare figure had. The
    # slope is fitted from the top voxel of each column of the endplate itself.
    top = {}
    for xx, _yy, zz in np.argwhere(body[:, :, max(0, z - 2):z + 1]):
        top[xx] = max(top.get(xx, -1), zz)
    xs_f = np.array(sorted(top))
    slope = 0.0
    if len(xs_f) >= 8:
        q0, q1 = np.percentile(xs_f, [10, 90])
        keep = xs_f[(xs_f >= q0) & (xs_f <= q1)]
        if len(keep) >= 4:
            slope = float(np.polyfit(keep * sp[0],
                                     np.array([top[x] for x in keep]) * sp[2], 1)[0])
    xmid = (lo_x + hi_x) / 2 * sp[0]
    zc = z * sp[2]
    p0 = np.array([(lo_x - half) * sp[0], yy * sp[1],
                   zc + slope * ((lo_x - half) * sp[0] - xmid)])
    p1 = np.array([(hi_x + half) * sp[0], yy * sp[1],
                   zc + slope * ((hi_x + half) * sp[0] - xmid)])
    _span(o, p0, p1, ACCENT, f"{w:.0f} mm", dy=-18)
    o.text((o.w / 2, o.h - 18), "superior endplate, side to side", INK, 18, "middle", "600")
    return o.svg("Superior endplate width"), None


def diagram_ala_reach(case, labels_dir):
    """Transverse process height, and its gap to the ala — the two axes Castellvi uses.

    Type I is a dysplastic process nineteen millimetres or more in CRANIOCAUDAL height. An
    earlier version of this figure drew the process's lateral reach, which is a different
    measurement and does not carry the grade. This reproduces measure_tp_height: the outer
    12 mm of the process, its largest connected component, measured top to bottom.
    """
    TIP_MM, LATERAL_FRAC = 12.0, 0.45
    lab, sp = load(case, labels_dir)
    low = None
    for v in (25, 24, 23):
        if (lab == v).sum() >= MIN_VOX:
            low = v
            break
    if low is None:
        return None, "no lowest lumbar"
    m = lab == low
    idx = np.argwhere(m)
    lo_x, hi_x = int(idx[:, 0].min()), int(idx[:, 0].max())
    vmid = 0.5 * (lo_x + hi_x)
    latL = int(vmid - LATERAL_FRAC * (vmid - lo_x))
    latR = int(vmid + LATERAL_FRAC * (hi_x - vmid))
    depth = max(1, int(round(TIP_MM / max(sp[0], 1e-6))))

    best = None
    for nm, outward in (("left", -1), ("right", +1)):
        sel = np.zeros(m.shape[0], bool)
        if outward > 0:
            sel[latR + 1:] = True
        else:
            sel[:latL] = True
        if not sel.any():
            continue
        band = np.zeros_like(m)
        band[sel] = m[sel]
        bcols = np.nonzero(band.any(axis=(1, 2)))[0]
        if not len(bcols):
            continue
        edge = int(bcols.max()) if outward > 0 else int(bcols.min())
        keep = np.zeros(m.shape[0], bool)
        if outward > 0:
            keep[max(0, edge - depth):edge + 1] = True
        else:
            keep[edge:edge + depth + 1] = True
        tip = np.zeros_like(m)
        tip[keep] = band[keep]
        if not tip.any():
            continue
        lt, n = ndimage.label(tip)
        if n > 1:
            sizes = ndimage.sum(tip, lt, range(1, n + 1))
            core = lt == int(np.argmax(sizes)) + 1
        else:
            core = tip
        ci = np.argwhere(core)
        h = float(ci[:, 2].max() - ci[:, 2].min() + 1) * sp[2]
        if best is None or h > best[0]:
            best = (h, nm, core, ci)
    if best is None:
        return None, "no transverse process tip"
    height, side, core, ci = best

    gap = None
    if (lab == SACRUM).sum() >= MIN_VOX:
        sac = lab == SACRUM
        span = []
        for a in range(3):
            hit = np.nonzero(sac.any(axis=tuple(i for i in range(3) if i != a)))[0]
            span.append(slice(max(0, int(hit[0]) - 10), min(sac.shape[a], int(hit[-1]) + 11)))
        sl = tuple(span)
        d = ndimage.distance_transform_edt(~sac[sl], sampling=sp)
        sub_core = core[sl]
        if sub_core.any():
            gap = float(d[sub_core].min())

    ids = [low] + [v for v in (SACRUM, S1) if (lab == v).sum() >= MIN_VOX]
    o = _scene(lab, sp, [([low], ACCENT_BONE, 1),
                         ([v for v in (SACRUM, S1) if (lab == v).sum() >= MIN_VOX],
                          BONE, 1)], VIEW_ANT, margin=1.16)
    if o is None:
        return None, "nothing to render"

    # the measured extent, drawn where it was taken: at the tip, top to bottom
    x_tip = float(np.median(ci[:, 0])) * sp[0]
    y_tip = float(np.median(ci[:, 1])) * sp[1]
    z0 = float(ci[:, 2].min()) * sp[2]
    z1 = float(ci[:, 2].max()) * sp[2]
    p0 = np.array([x_tip, y_tip, z0])
    p1 = np.array([x_tip, y_tip, z1])
    a, b = o.px(p0), o.px(p1)
    o.line(a, b, ACCENT, 4.0)
    for pnt in (a, b):
        o.line((pnt[0] - 14, pnt[1]), (pnt[0] + 14, pnt[1]), ACCENT, 3.4)
    o.text(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
           f"{height:.0f} mm", ACCENT, 24, "start", dx=20, dy=6)
    o.text(((a[0] + b[0]) / 2, min(a[1], b[1])),
           f"{side} process, craniocaudal", INK, 17, "middle", "600", dy=-16)
    if gap is not None:
        o.text((o.w / 2, o.h - 40), f"{gap:.1f} mm to the ala", SECOND, 20, "middle", "600")
    o.text((o.w / 2, o.h - 16),
           "Castellvi I is a dysplastic process 19 mm or taller", INK, 17, "middle", "600")
    return o.svg("Transverse process height, and its gap to the ala"), None


# A CONSTRUCTION NEEDS A SPECIMEN THAT SHOWS IT. 0498 is at the cohort median for the
# spinopelvic measures, which is why it is the default -- but its T11 is 3,566 voxels,
# clipped at the top of the field of view, so the rib-ratio figure had a rib articulating
# with a sliver. These cases were chosen by measuring which volumes carry both vertebrae and
# both ribs at full size.
CASE_FOR = {
    "rib_ratio": "0578",
}

BUILDERS = {
    "pelvic_incidence": diagram_pelvic_incidence,
    "sacral_slope": diagram_sacral_slope,
    "pelvic_tilt": diagram_pelvic_tilt,
    "lumbar_lordosis": diagram_lumbar_lordosis,
    "endplate_flare": diagram_endplate_flare,
    "disc_height": diagram_disc_height,
    "rib_ratio": diagram_rib_ratio,
    "bi_iliac": diagram_bi_iliac,
    "sacral_base": diagram_sacral_base,
    "tp_span": diagram_tp_span,
    "body_height": diagram_body_height,
    "pedicle_width": diagram_pedicle_width,
    "hip_width": diagram_hip_width,
    "rib_to_crest": diagram_rib_to_crest,
    "crest_above_l45": diagram_crest_above_l45,
    "canal_width": diagram_canal_width,
    "endplate_width": diagram_endplate_width,
    "ala_reach": diagram_ala_reach,
}

# POPULATION PANELS ONLY. A ridge plot of pelvic tilt by decade is about change with age,
# not about how tilt is defined, so an inset there is clutter -- every trend, ridge and
# by-decade panel is deliberately absent from this map.
PANEL_MAP = {
    # PI ONLY ON THE PLAIN DISTRIBUTION. pi_by_sex, pi_vs_ll and pi_ll_mismatch_deg are
    # comparisons, not definitions -- a reader there already knows what pelvic incidence is
    # and is being shown how it relates to something else, so the construction is clutter.
    "pelvic_incidence": ["pelvic_incidence_deg"],
    "sacral_slope": ["sacral_slope_deg"],
    "pelvic_tilt": ["pelvic_tilt_deg"],
    "lumbar_lordosis": ["ll_supine_deg"],
    # only the flare RATIO here. The absolute endplate width plotted by
    # grad_endplate_width and vertebral_size_sex comes from extract_level_gradients, which
    # erodes the slice and trims differently and reads 44.9 mm where this draws 49 -- so
    # those two panels are deliberately unmapped rather than given a figure that disagrees.
    "endplate_flare": ["osteophyte"],
    "disc_height": ["disc_height", "disc_ratio", "disc_by_group", "vacuum"],
    "rib_ratio": ["rib_ratio"],
    "bi_iliac": ["shape_bi_iliac_width_mm"],
    "sacral_base": ["sacral_base"],
    "tp_span": ["grad_tp_span"],
    "body_height": ["grad_body_height", "wedge"],
    "pedicle_width": ["pedicle_min_mm"],
    "hip_width": ["shape_bi_acetabular_mm"],
    "rib_to_crest": ["rib12_to_crest_mm"],
    "crest_above_l45": ["crest_above_l45_mm", "crest_landmark"],
    "canal_width": ["grad_canal_width"],
    "endplate_width": ["grad_endplate_width", "vertebral_size_sex"],
    "ala_reach": ["castellvi"],
}


# A DIAGRAM THAT DISAGREES WITH ITS PLOT IS WORSE THAN NO DIAGRAM, because it looks
# authoritative. Each builder may declare the number it draws and the released column that
# number must equal; the build then checks, and refuses to write a diagram that has drifted.
#
# This is not hypothetical. A first pedicle diagram measured bone outboard of the canal over
# the canal's whole depth and read 15-20 mm, which is the exact mistake extract_surgical_
# morphometrics documents having fixed -- at the back of the canal that bone is lamina. It
# would have shipped under a histogram centred near 7 mm.
CHECKS = {
    "tp_span": ("morphometrics/level_gradients.csv", "tp_span_L3_mm", 1.0),
    "body_height": ("morphometrics/level_gradients.csv", "body_height_L4_mm", 1.0),
    "disc_height": ("morphometrics/degenerative.csv", "disc_height_L4-5_mm", 0.6),
    "bi_iliac": ("morphometrics/pelvic_shape.csv", "bi_iliac_width_mm", 2.0),
    "bi_acetabular": ("morphometrics/pelvic_shape.csv", "bi_acetabular_mm", 2.0),
    "sacral_base": ("morphometrics/pelvic_shape.csv", "sacral_base_width_mm", 2.0),
    "pelvic_incidence": ("morphometrics/surgical_morphometrics.csv", "pelvic_incidence_deg", 1.0),
    "sacral_slope": ("morphometrics/surgical_morphometrics.csv", "sacral_slope_deg", 1.0),
    "pelvic_tilt": ("morphometrics/surgical_morphometrics.csv", "pelvic_tilt_deg", 1.0),
    "lumbar_lordosis": ("morphometrics/surgical_morphometrics.csv", "ll_supine_deg", 1.5),
    "rib_ratio": ("morphometrics/transition_morphometrics.csv", "rib12_11_ratio_left", 0.05),
    "pedicle_width": ("morphometrics/surgical_morphometrics.csv", "pedicle_min_mm", 0.8),
    "hip_width": ("morphometrics/pelvic_shape.csv", "bi_acetabular_mm", 2.0),
    "rib_to_crest": ("morphometrics/surgical_morphometrics.csv", "rib12_to_crest_mm", 2.0),
    "crest_above_l45": ("morphometrics/surgical_morphometrics.csv", "crest_above_l45_mm", 2.0),
    "canal_width": ("morphometrics/level_gradients.csv", "canal_width_L4_mm", 1.5),
    "endplate_width": ("morphometrics/level_gradients.csv", "endplate_width_L4_mm", 1.5),
    "ala_reach": ("morphometrics/tp_height.csv",
                  ("tp_height_left_mm", "tp_height_right_mm"), 1.0),
}


def _drawn_number(svg, patterns):
    """Pull the headline figure back out of the finished SVG, so the check sees what a
    reader sees rather than an intermediate the builder happened to keep."""
    import re
    for pat in patterns:
        m = re.search(pat, svg)
        if m:
            return float(m.group(1))
    return None


def verify(name, svg, case):
    """-> (ok, message). Unknown or missing columns pass, with the reason stated."""
    import csv as _csv
    spec = CHECKS.get(name)
    if not spec:
        return True, "no check declared"
    path, col, tol = spec
    fp = Path(path)
    if not fp.exists():
        return True, f"{path} absent"
    rows = [r for r in _csv.DictReader(open(fp, encoding="utf-8")) if r.get("case") == case]
    # A COLUMN SPEC MAY NAME SEVERAL. The transverse-process figure draws whichever side is
    # taller, because that is the side a grade is read from, so pinning the check to the left
    # column made it pass or fail on how asymmetric the patient happened to be -- it passed
    # here by 1.2 mm against a left value the figure was not drawing.
    cols = (col,) if isinstance(col, str) else tuple(col)
    wants = [float(rows[0][c]) for c in cols
             if rows and (rows[0].get(c) or "").strip()]
    if not wants:
        return True, f"{cols} empty for {case}"
    want = wants[0]
    got = _drawn_number(svg, [
        r"(?:PI|SS|PT|lordosis)\s+([-\d.]+)&#176;",
        r"ratio\s+([\d.]+)<",
        r">([\d.]+)\s*mm<",
        r"(?:anterior)\s+([\d.]+)\s*mm",
    ])
    if got is None:
        return True, "no number found in the drawing"
    near = min(wants, key=lambda w: abs(got - w))
    if abs(got - near) > tol:
        return False, (f"draws {got} but {cols} are "
                       f"{', '.join(str(w) for w in wants)} (tolerance {tol})")
    return True, f"{got} matches {near}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--out", default="../openspineconsortium.github.io/assets/gallery/measures")
    ap.add_argument("--case", default="0498",
                    help="a case at the cohort median, so the picture is typical")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ([a.only] if a.only else list(BUILDERS)):
        case = CASE_FOR.get(name, a.case)
        svg, err = BUILDERS[name](case, a.labels)
        if svg is None:
            print(f"  ! {name}: {err}")
            continue
        ok, why = verify(name, svg, case)
        if not ok:
            print(f"  ! {name}: NOT WRITTEN -- {why}")
            continue
        (out / f"{name}.svg").write_text(svg, encoding="utf-8")
        written[name] = PANEL_MAP.get(name, [])
        print(f"  {out / (name + '.svg')}  ({len(svg) / 1024:.0f} kB)")

    if written:
        idx = out / "index.json"
        prev = {}
        if idx.exists():
            try:
                prev = json.loads(idx.read_text(encoding="utf-8")).get("panels", {})
            except ValueError:
                prev = {}
        # merge, so building one construction cannot delete the others -- the same failure
        # that emptied distributions.json and the mesh index
        for name, panels in written.items():
            for k in panels:
                prev[k] = f"{name}.svg"
        # A BUILD STAMP, because the figures are fetched by plain URL. tools/stamp_assets
        # versions the scripts and stylesheets, but nothing versioned these -- so a rebuilt
        # diagram was served from cache and the change reached nobody. The stamp is the hash
        # of the figures themselves, so it moves exactly when they do and not otherwise.
        import hashlib
        h = hashlib.sha256()
        for q in sorted(out.glob("*.svg")):
            h.update(q.read_bytes())
        build = h.hexdigest()[:8]
        idx.write_text(json.dumps({"build": build, "panels": prev}, indent=1) + "\n",
                       encoding="utf-8")
        print(f"  {idx}  ({len(prev)} panel(s) mapped, build {build})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
