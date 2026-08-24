"""scripts/make_measure_diagrams.py — draw the construction behind each plotted number.

A distribution tells the reader how a quantity varies and nothing at all about what it is.
"Pelvic incidence" is a phrase until you have seen the two points and the angle between them
on somebody's actual sacrum. These are those pictures: real released labels, the anatomy
rendered from the voxels, and the geometry drawn on top in the same construction the
extractor uses.

THE CONSTRUCTION IS IMPORTED, NOT REIMPLEMENTED. _endplate, _femoral_head and _body_mask
come from extract_surgical_morphometrics, which is what produced the numbers in the plots.
A diagram that re-derived the geometry could drift from the extractor and then explain a
measurement nobody made -- which is worse than no diagram, because it would look right.

OUTPUT IS ONE SELF-CONTAINED SVG PER CONSTRUCTION. The anatomy is a base64 PNG inside the
file and the geometry is vector, so lines and type stay crisp at any width and the whole
asset has no external references, which is what the site's CSP requires.

USER UNITS ARE MILLIMETRES OF PATIENT. The viewBox is set from the crop's physical extent,
so a point computed in world millimetres is drawn by passing it straight to the canvas with
no scale factor to get wrong. It also means the scale bar is honest by construction.

    python scripts/make_measure_diagrams.py --out ../openspineconsortium.github.io/assets/gallery/measures
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from extract_surgical_morphometrics import (                        # noqa: E402
    _endplate, _femoral_head, _com,
    LUMBAR, SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R, MIN_VOX,
)

INK = "#1A1C18"
MUTED = "#8d8d86"
ACCENT = "#c8452c"          # the quantity being measured
SECOND = "#2f6fb0"          # the reference it is measured against
THIRD = "#b8860b"           # a second measured quantity in the same picture
PAPER = "#f4f2ec"

# The backdrop has to read as bone at a glance while still losing to the line work. The
# first pair sat within a few counts of the paper and disappeared entirely.
BONE_FAR = np.array([128, 125, 117], np.float32)
BONE_NEAR = np.array([215, 211, 200], np.float32)
FONT = "ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif"


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def surface(vol, view_axis, highlight_ids=None, highlight_mask=None):
    """First-hit surface render along view_axis; returns an RGB image and the kept axes.

    Depth shading only. These are backdrops for line work: a flat silhouette would fight
    the geometry drawn on top, and a fully lit render would bury it.
    """
    occ = vol > 0
    if not occ.any():
        return None, None
    front = occ.argmax(axis=view_axis)
    anyhit = occ.any(axis=view_axis)
    depth = np.where(anyhit, front, np.nan).astype(np.float32)
    d0, d1 = float(np.nanmin(depth)), float(np.nanmax(depth))
    t = np.clip((np.nan_to_num(depth, nan=d1) - d0) / max(1.0, d1 - d0), 0, 1)

    img = np.empty(anyhit.shape + (3,), np.float32)
    img[:] = _rgb(PAPER)
    shade = (BONE_NEAR[None, None, :] * (1 - t)[..., None]
             + BONE_FAR[None, None, :] * t[..., None])
    img[anyhit] = shade[anyhit]

    # A region can be named by label id or given as a mask. The mask form exists because
    # the interesting region is often not a label at all: the vertebral BODY is what
    # body_of() carves out of a vertebra, and the caption is about that carve.
    hany = None
    if highlight_ids:
        hany = np.isin(vol, list(highlight_ids)).any(axis=view_axis)
    if highlight_mask is not None:
        m2 = highlight_mask.any(axis=view_axis)
        hany = m2 if hany is None else (hany | m2)
    if hany is not None:
        img[hany] = img[hany] * 0.55 + _rgb(SECOND) * 0.45

    keep = [a for a in range(3) if a != view_axis]
    return np.clip(img, 0, 255).astype(np.uint8), keep


def png_b64(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Canvas:
    """SVG in patient millimetres. Pass world coordinates; the canvas handles the rest."""

    def __init__(self, img, sp_h, sp_v, origin_h, origin_v, pad=10.0):
        self.rows, self.cols = img.shape[:2]
        self.W = self.cols * sp_h
        self.H = self.rows * sp_v
        self.oh, self.ov = origin_h, origin_v
        self.pad = pad
        self.b64 = png_b64(img)
        self.parts = []

    def pt(self, h_mm, v_mm):
        """World (horizontal, vertical) mm -> SVG units. Vertical is flipped: +v is up."""
        return (h_mm - self.oh), self.H - (v_mm - self.ov)

    def line(self, a, b, colour, width=1.0, dash=None, opacity=1.0):
        x1, y1 = self.pt(*a)
        x2, y2 = self.pt(*b)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round" '
            f'opacity="{opacity}"{d}/>')

    def dot(self, p, colour, r=2.1):
        x, y = self.pt(*p)
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{colour}" '
            f'stroke="{PAPER}" stroke-width="0.9"/>')

    def text(self, p, s, colour=INK, size=6.2, anchor="start", weight="600", dx=0.0, dy=0.0):
        """Halo then glyph, as two elements.

        paint-order="stroke" is the tidy way to do this and a browser honours it, but SVG
        rasterisers do not: they paint fill then stroke, so the paper-coloured halo lands on
        top and the label comes out blank. Since the halo is the only thing keeping type
        legible over bone, and since checking the output means rasterising it, the halo is
        drawn as its own element underneath where every renderer agrees.
        """
        x, y = self.pt(*p)
        x += dx
        y += dy
        common = (f'x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="{FONT}" '
                  f'font-weight="{weight}" text-anchor="{anchor}"')
        self.parts.append(
            f'<text {common} fill="none" stroke="{PAPER}" stroke-width="2.2" '
            f'stroke-opacity="0.92" stroke-linejoin="round">{s}</text>')
        self.parts.append(f'<text {common} fill="{colour}">{s}</text>')

    def arc(self, centre, v1, v2, radius, colour, width=1.2):
        """Sweep from direction v1 to v2 about centre, the short way. Returns a label point."""
        a1 = float(np.arctan2(v1[1], v1[0]))
        a2 = float(np.arctan2(v2[1], v2[0]))
        d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        pts = []
        for i in range(49):
            a = a1 + d * i / 48
            pts.append(self.pt(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)))
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{p}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" opacity="0.95"/>')
        a = a1 + d / 2
        return (centre[0] + radius * 1.42 * np.cos(a), centre[1] + radius * 1.42 * np.sin(a))

    def scalebar(self, mm=50):
        """Bottom-left, in the same units as everything else, so it cannot disagree."""
        x0, y0 = 6.0, 6.0
        a = self.pt(self.oh + x0, self.ov + y0)
        b = self.pt(self.oh + x0 + mm, self.ov + y0)
        self.parts.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
            f'stroke="{INK}" stroke-width="1.1" opacity="0.65"/>')
        self.parts.append(
            f'<text x="{(a[0] + b[0]) / 2:.2f}" y="{a[1] - 2.6:.2f}" fill="{INK}" '
            f'font-size="5.4" opacity="0.65" font-family="{FONT}" text-anchor="middle">'
            f'{mm} mm</text>')

    NOMINAL_W = 190.0

    def svg(self, title, caption):
        pad = self.pad
        # EVERY DIAGRAM IS NORMALISED TO ONE NOMINAL WIDTH. Drawing in patient millimetres
        # is right for the geometry and wrong for the page: a whole pelvis is 190 mm across
        # and a single vertebra 65, so a caption set at a fixed 6 mm was readable on one and
        # swallowed the other, running to eleven lines under a picture three lines tall.
        # Scaling the content group instead keeps type, stroke weight and the anatomy in the
        # same proportion across the whole set, and the scale bar rescales with it, so it
        # stays true.
        s = self.NOMINAL_W / self.W
        draw_w = self.NOMINAL_W
        draw_h = self.H * s
        vb_w = draw_w + 2 * pad
        # THE CAPTION IS WRAPPED, NOT TRUSTED TO FIT. SVG text does not wrap, so a single
        # <text> ran off the right edge and the sentence lost its last third. Width is
        # estimated at half an em per character, which is close for a humanist sans and
        # errs toward breaking early.
        size = 6.0
        per_line = max(20, int((vb_w - 2 * pad) / (size * 0.5)))
        lines, cur = [], ""
        for word in caption.split():
            trial = f"{cur} {word}".strip()
            if len(trial) > per_line and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        cap_h = 6.0 + len(lines) * (size * 1.35)
        vb_h = draw_h + 2 * pad + cap_h
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb_w:.1f} {vb_h:.1f}" '
            f'role="img" aria-label="{title}" preserveAspectRatio="xMidYMid meet" '
            # NO style ATTRIBUTE ON THE ROOT. "width:100%;height:auto" is what a browser
            # wants, and it is also what makes every SVG rasteriser render a blank page --
            # so the one check that matters, looking at the output, becomes impossible.
            # viewBox alone gives the aspect ratio; the page's CSS does the sizing.
            f'width="100%">\n'
            f'<title>{title}</title>\n'
            f'<rect width="100%" height="100%" fill="{PAPER}"/>\n'
            f'<g transform="translate({pad},{pad}) scale({s:.5f})">\n'
            f'<image x="0" y="0" width="{self.W:.2f}" height="{self.H:.2f}" '
            f'href="data:image/png;base64,{self.b64}"/>\n'
            f'{body}\n</g>\n'
            + "\n".join(
                f'<text x="{pad}" y="{draw_h + 2 * pad + 4 + i * size * 1.35:.1f}" '
                f'fill="{INK}" font-size="{size}" opacity="0.78" '
                f'font-family="{FONT}">{ln}</text>'
                for i, ln in enumerate(lines))
            + '\n</svg>\n')


def load(case, labels_dir):
    """Canonicalised to RAS for ANALYSIS AND DRAWING ONLY. Nothing here is written back."""
    img = nib.as_closest_canonical(nib.load(str(Path(labels_dir) / f"{case}_label.nii.gz")))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    sp = np.array(img.header.get_zooms()[:3], float)
    return lab, sp


def crop_to(lab, ids, sp, margin_mm=14.0):
    """Crop to the ids that matter, returning the sub-volume and its origin in world mm."""
    m = np.isin(lab, list(ids))
    if not m.any():
        return None, None
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - np.ceil(margin_mm / sp).astype(int), 0)
    hi = np.minimum(idx.max(0) + np.ceil(margin_mm / sp).astype(int) + 1, np.array(lab.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return lab[sl], lo * sp


# ── views ───────────────────────────────────────────────────────────────────────────────
# RAS after canonicalisation: axis 0 is left->right, axis 1 posterior->anterior,
# axis 2 inferior->superior. A view names which axis it looks ALONG and which two it draws,
# with a sign so the result matches how the study is normally hung -- a lateral lumbar image
# is read with the patient facing left, so anterior runs to the LEFT of the page.
SAGITTAL = dict(view=0, h_axis=1, h_sign=-1, v_axis=2, v_sign=1)
# h_sign is -1 so the patient's LEFT falls on the viewer's RIGHT, which is how an
# anteroposterior study is hung and the opposite of what the raw array order gives.
CORONAL = dict(view=1, h_axis=0, h_sign=-1, v_axis=2, v_sign=1)
AXIAL = dict(view=2, h_axis=0, h_sign=1, v_axis=1, v_sign=-1)


def render_view(vol, sp, origin_mm, view, highlight_ids=None, highlight_mask=None):
    """Render vol in the given view, oriented so rows run top-down and +v is up.

    Returns (image, sp_h, sp_v, origin_h, origin_v) ready to hand to Canvas, plus a
    projection function taking a world 3-vector to canvas (h, v) millimetres.
    """
    img, keep = surface(vol, view["view"], highlight_ids, highlight_mask)
    if img is None:
        return None
    ha, va = view["h_axis"], view["v_axis"]
    # `keep` is ascending and the horizontal axis is always the lower of the two in every
    # view defined here, so the rendered array arrives indexed [h, v] -- rows running along
    # the HORIZONTAL axis. The canvas wants rows = v. Testing for va here instead of ha
    # meant the transpose never fired, the picture was rotated a quarter turn against the
    # geometry drawn on it, and W and H were swapped as well.
    if keep[0] == ha:
        img = np.transpose(img, (1, 0, 2))          # rows = v, cols = h
    if view["v_sign"] > 0:
        img = img[::-1]                             # row 0 must be the LARGEST v
    if view["h_sign"] < 0:
        img = img[:, ::-1]                          # col 0 must be the smallest signed h

    sp_h, sp_v = sp[ha], sp[va]
    lo_h = origin_mm[ha]
    hi_h = lo_h + vol.shape[ha] * sp_h
    lo_v = origin_mm[va]
    sh, sv = view["h_sign"], view["v_sign"]
    origin_h = min(sh * lo_h, sh * hi_h)
    origin_v = sv * lo_v if sv > 0 else -(lo_v + vol.shape[va] * sp_v)

    def project(P):
        return (sh * float(P[ha]), sv * float(P[va]))

    return img, sp_h, sp_v, origin_h, origin_v, project


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


# ── constructions ───────────────────────────────────────────────────────────────────────
def diagram_pelvic_incidence(case, labels_dir):
    """PI, sacral slope and pelvic tilt, which are one picture and one identity.

    Showing them separately would hide the only thing a reader needs to carry away: PI is
    fixed by the shape of the pelvis, SS and PT split it between them according to posture,
    and PI = SS + PT always. Three panels cannot say that; one can.
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

    v = fem - s1c
    vs = np.array([0.0, v[1], v[2]])
    ns = np.array([0.0, s1n[1], s1n[2]])
    ang = lambda a, b: float(np.degrees(np.arccos(                       # noqa: E731
        np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))
    pi_deg = 180.0 - ang(ns, vs)
    ss_deg = ang(ns, np.array([0.0, 0.0, 1.0]))
    pt_deg = pi_deg - ss_deg

    ids = [SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R]
    sub, origin = crop_to(lab, ids, sp, margin_mm=10.0)
    r = render_view(sub, sp, origin, SAGITTAL, highlight_ids=[S1] if S1 in have else [SACRUM])
    if r is None:
        return None, "nothing to render"
    img, sp_h, sp_v, oh, ov, proj = r
    c = Canvas(img, sp_h, sp_v, oh, ov)

    P_s1, P_fem = proj(s1c), proj(fem)
    n2 = _unit(np.array(proj(s1c + s1n * 10.0)) - np.array(P_s1))
    plate = np.array([-n2[1], n2[0]])                 # in-plane, perpendicular to the normal
    to_fem = _unit(np.array(P_fem) - np.array(P_s1))

    # the plate itself, then the perpendicular the angle is actually taken from
    c.line(tuple(np.array(P_s1) - plate * 26), tuple(np.array(P_s1) + plate * 26),
           SECOND, 1.7)
    c.line(P_s1, tuple(np.array(P_s1) + n2 * 46), SECOND, 1.2, dash="3 2.5")
    # horizontal and vertical references
    c.line((P_s1[0] - 34, P_s1[1]), (P_s1[0] + 34, P_s1[1]), MUTED, 0.9, dash="2 3")
    c.line((P_fem[0], P_fem[1] - 8), (P_fem[0], P_fem[1] + 62), MUTED, 0.9, dash="2 3")
    # the line whose angle is the measurement
    c.line(P_s1, P_fem, ACCENT, 1.8)

    # THE THREE ARCS SHARE A VERTEX, so their labels have to be pushed apart deliberately.
    # At equal radii the PI and SS labels landed on top of each other and neither could be
    # read. PI gets the outer radius because it is the quantity the section is named for.
    lp = c.arc(P_s1, n2, to_fem, 26, ACCENT, 1.5)
    c.text(lp, f"PI {pi_deg:.1f}&#176;", ACCENT, 7.6, "middle", dx=-13, dy=-4)
    lp = c.arc(P_s1, np.array([1.0, 0.0]) * np.sign(plate[0] or 1), plate, 12, THIRD, 1.2)
    c.text(lp, f"SS {ss_deg:.1f}&#176;", THIRD, 6.4, "middle", dx=6, dy=9)
    lp = c.arc(P_fem, np.array([0.0, 1.0]), -to_fem, 17, SECOND, 1.2)
    c.text(lp, f"PT {pt_deg:.1f}&#176;", SECOND, 6.4, "middle", dx=-9)

    c.dot(P_s1, SECOND, 2.6)
    c.dot(P_fem, ACCENT, 2.6)
    c.dot(proj(cl), ACCENT, 1.7)
    c.dot(proj(cr), ACCENT, 1.7)
    c.text(P_s1, "S1 endplate midpoint", INK, 5.6, "start", "500", dx=8, dy=-4)
    c.text(P_fem, "femoral head axis", INK, 5.6, "middle", "500", dy=11)
    c.scalebar(50)

    cap = (f"Case {case}. The plate is fitted to the S1 endplate surface; the femoral head "
           f"centres are fitted where each femur meets its acetabulum. "
           f"PI = SS + PT ({pi_deg:.1f} = {ss_deg:.1f} + {pt_deg:.1f}).")
    return c.svg("Pelvic incidence, sacral slope and pelvic tilt", cap), None


def _measure_bar(c, v_mm, h0, h1, colour, label, side="above", tick=3.0):
    """A horizontal span with end ticks and a label -- the width diagrams' only idiom."""
    c.line((h0, v_mm), (h1, v_mm), colour, 1.7)
    for h in (h0, h1):
        c.line((h, v_mm - tick), (h, v_mm + tick), colour, 1.5)
    dy = -4.0 if side == "above" else 9.0
    c.text(((h0 + h1) / 2, v_mm), label, colour, 6.6, "middle", dy=dy)


def diagram_endplate_flare(case, labels_dir):
    """Endplate width against mid-body width, on one lumbar body, in one coronal view.

    These are two numbers the site plots separately -- absolute endplate width by level, and
    their ratio as the osteophyte index -- and they are the same picture. A healthy body is
    an hourglass in the coronal plane; spurs grow at the rim, so the rim widens while the
    waist does not, and the ratio rises. Two spans on one vertebra say that in a way neither
    histogram can.
    """
    from extract_degenerative import body_of                          # noqa: E402

    lab, sp = load(case, labels_dir)
    vid, name = 23, "L4"                       # mid-lumbar: typical, and load-bearing
    m = lab == vid
    if m.sum() < MIN_VOX:
        return None, f"no {name}"
    b = body_of(m)
    if b is None:
        return None, f"no body carved from {name}"

    idx = np.argwhere(b)
    zs = idx[:, 2]

    def span(p0, p1):
        sel = idx[(zs >= np.percentile(zs, p0)) & (zs <= np.percentile(zs, p1))]
        if len(sel) < 60:
            return None
        return (float(np.percentile(sel[:, 0], 1)),
                float(np.percentile(sel[:, 0], 99)),
                float(np.mean(sel[:, 2])))

    rim = span(80, 100)                        # the endplate rim, where spurs grow
    waist = span(38, 62)                       # mid-body, narrowest on a healthy vertebra
    if rim is None or waist is None:
        return None, "band too thin to measure"

    # SHOW ONLY THIS VERTEBRA. The crop box catches the neighbours above and below, and
    # with them in frame the reader cannot tell which bone carries the two spans.
    sub, origin = crop_to(lab, [vid], sp, margin_mm=9.0)
    keep_m = sub == vid
    sub = np.where(keep_m, sub, 0)
    body_sub = np.zeros_like(sub, bool)
    idx_lo = np.argwhere(np.isin(lab, [vid])).min(0)
    off = tuple(int(round(o / z)) for o, z in zip(origin, sp))
    bb = body_of(keep_m)
    if bb is not None:
        body_sub = bb
    r = render_view(sub, sp, origin, CORONAL, highlight_mask=body_sub)
    if r is None:
        return None, "nothing to render"
    img, sp_h, sp_v, oh, ov, proj = r
    c = Canvas(img, sp_h, sp_v, oh, ov)

    out = {}
    for key, (lo, hi, zc), colour in (("rim", rim, ACCENT), ("waist", waist, SECOND)):
        a = proj(np.array([lo * sp[0], 0.0, zc * sp[2]]))
        bb = proj(np.array([hi * sp[0], 0.0, zc * sp[2]]))
        width_mm = abs(hi - lo) * sp[0]
        out[key] = width_mm
        label = (f"superior endplate {width_mm:.1f} mm" if key == "rim"
                 else f"mid-body {width_mm:.1f} mm")
        _measure_bar(c, a[1], min(a[0], bb[0]), max(a[0], bb[0]), colour, label,
                     "above" if key == "rim" else "below")

    ratio = out["rim"] / out["waist"] if out["waist"] > 5 else float("nan")
    c.scalebar(20)
    cap = (f"Case {case}, {name}, anteroposterior view. The body is separated from the "
           f"posterior elements at the anterior wall of the canal, then measured in two "
           f"height bands: the top fifth for the endplate rim, the middle quarter for the "
           f"waist. Endplate flare is their ratio, {ratio:.2f} here; 1.0 is no flare.")
    return c.svg("Endplate width, mid-body width, and the flare between them", cap), None


def diagram_disc_height(case, labels_dir):
    """The interbody gap, measured where a radiologist measures it.

    The subtlety worth drawing is WHY the measurement is confined to a midline column.
    Endplates are concave, so the narrowest part of the space is rim to rim; taking that
    reads 4 to 6 mm against a published 8 to 12. Each column through a 16 mm midline box is
    measured separately and the median reported, which is the mid-sagittal reading.
    """
    from extract_degenerative import body_of                          # noqa: E402

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
    sel_u = iu[(np.abs(iu[:, 0] - cx) <= rx) & (np.abs(iu[:, 1] - cy) <= ry)]
    sel_l = il[(np.abs(il[:, 0] - cx) <= rx) & (np.abs(il[:, 1] - cy) <= ry)]
    if len(sel_u) < 40 or len(sel_l) < 40:
        return None, "midline column too thin"

    cols = {}
    for xx, yy, zz in sel_u:
        k = (xx, yy)
        cols.setdefault(k, [None, None])
        cols[k][0] = zz if cols[k][0] is None else min(cols[k][0], zz)
    for xx, yy, zz in sel_l:
        k = (xx, yy)
        if k in cols:
            cols[k][1] = zz if cols[k][1] is None else max(cols[k][1], zz)
    pairs = [(a, b2) for a, b2 in cols.values() if a is not None and b2 is not None]
    if len(pairs) < 15:
        return None, "too few columns"
    gap_vox = float(np.median([a - b2 for a, b2 in pairs]))
    h_mm = max(0.0, gap_vox * sp[2])
    z_up = float(np.median([a for a, _ in pairs]))
    z_lo = z_up - gap_vox

    sub, origin = crop_to(lab, [upper, lower], sp, margin_mm=8.0)
    r = render_view(sub, sp, origin, SAGITTAL)
    if r is None:
        return None, "nothing to render"
    img, sp_h, sp_v, oh, ov, proj = r
    c = Canvas(img, sp_h, sp_v, oh, ov)

    v_up = proj(np.array([0.0, 0.0, z_up * sp[2]]))[1]
    v_lo = proj(np.array([0.0, 0.0, z_lo * sp[2]]))[1]
    h_a = proj(np.array([0.0, (cy - ry) * sp[1], 0.0]))[0]
    h_b = proj(np.array([0.0, (cy + ry) * sp[1], 0.0]))[0]
    hl, hr = min(h_a, h_b), max(h_a, h_b)

    c.line((hl - 6, v_up), (hr + 6, v_up), SECOND, 1.6)
    c.line((hl - 6, v_lo), (hr + 6, v_lo), SECOND, 1.6)
    for h in (hl, hr):
        c.line((h, v_lo - 9), (h, v_up + 9), MUTED, 0.9, dash="2 2.5")
    mid = (hl + hr) / 2
    c.line((mid, v_lo), (mid, v_up), ACCENT, 2.0)
    c.text((mid, (v_lo + v_up) / 2), f"{h_mm:.1f} mm", ACCENT, 7.0, "start", dx=7, dy=2)
    c.text((mid, v_up), "16 mm midline column", INK, 5.4, "middle", "500", dy=-9)
    c.scalebar(20)

    cap = (f"Case {case}, {name}, lateral view with anterior to the left. Endplates are "
           f"concave, so the space is narrowest rim to rim and measuring there understates "
           f"it. Each column through the midline box is measured separately and the median "
           f"reported.")
    return c.svg("Disc height, measured at the midline", cap), None


BUILDERS = {
    "pelvic_incidence": diagram_pelvic_incidence,
    "endplate_flare": diagram_endplate_flare,
    "disc_height": diagram_disc_height,
}

# Which panels each construction explains. One picture serves every plot that shares a
# construction: PI, SS, PT and the mismatch are all read off the same three points.
PANEL_MAP = {
    "pelvic_incidence": [
        "pelvic_incidence_deg", "sacral_slope_deg", "pelvic_tilt_deg", "pi_ll_mismatch_deg",
        "pi_vs_ll", "pi_by_sex", "aging", "mismatch_age", "agesex_pelvic_tilt_deg",
        "ridge_pelvic_incidence_deg", "ridge_pelvic_tilt_deg",
    ],
    "endplate_flare": [
        "grad_endplate_width", "vertebral_size_sex", "osteophyte", "grad_body_height",
    ],
    "disc_height": [
        "disc_height", "disc_ratio", "disc_by_group", "vacuum",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--out", default="../openspineconsortium.github.io/assets/gallery/measures")
    ap.add_argument("--case", default="0498",
                    help="a case at the cohort median, so the picture is typical")
    ap.add_argument("--only", default=None, help="build one construction by name")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [a.only] if a.only else list(BUILDERS)
    written = {}
    for name in names:
        svg, err = BUILDERS[name](a.case, a.labels)
        if svg is None:
            print(f"  ! {name}: {err}")
            continue
        p = out / f"{name}.svg"
        p.write_text(svg, encoding="utf-8")
        written[name] = PANEL_MAP.get(name, [])
        print(f"  {p}  ({len(svg) / 1024:.0f} kB)")

    if written:
        import json
        idx = out / "index.json"
        # merge, so building one construction cannot delete the others -- the same failure
        # that emptied distributions.json and the mesh index
        prev = {}
        if idx.exists():
            try:
                prev = json.loads(idx.read_text(encoding="utf-8")).get("panels", {})
            except ValueError:
                prev = {}
        for name, panels in written.items():
            for k in panels:
                prev[k] = f"{name}.svg"
        idx.write_text(json.dumps({"panels": prev}, indent=1) + "\n", encoding="utf-8")
        print(f"  {idx}  ({len(prev)} panel(s) mapped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
