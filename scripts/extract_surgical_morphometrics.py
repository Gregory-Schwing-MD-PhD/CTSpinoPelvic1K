"""scripts/extract_surgical_morphometrics.py — the measures that drive operative decisions.

Chosen from what the surgical literature actually decides on, and restricted to what BONE
labels alone can support on a NON-CONTRAST supine CT. Vessels are excluded deliberately:
the iliac confluence position would be the ideal LSTV planning variable, but CT
colonography is unenhanced and the veins have almost no contrast against retroperitoneal
fat.

WHAT SUPINE ACQUISITION PERMITS, AND WHAT IT DOES NOT

  PELVIC INCIDENCE is position-INDEPENDENT -- identical standing, seated and supine,
  because it is a morphological property of the pelvis rather than a posture. So PI from a
  supine CT needs no caveat, and PI is the parameter that dictates how much lordosis a
  given spine requires.

  LUMBAR LORDOSIS is position-dependent, but supine sits only ~4.6 deg below standing
  (50.2 vs 54.8 in published series); it is SEATED that collapses, to ~16. So supine LL is
  reported explicitly AS supine, with the offset stated, rather than dropped or silently
  passed off as standing.

  PELVIC TILT and SACRAL SLOPE are postural and are reported supine for the same reason.

THE MEASURES

  pelvic_incidence_deg   angle at the S1 endplate midpoint between its normal and the line
                         to the bicoxofemoral axis. The decision-maker: PI-LL mismatch
                         beyond ~10 deg predicts residual pain after fusion, including
                         short-segment.
  ll_supine_deg          topmost-lumbar superior to S1 superior endplate angle. Reported
                         with ll_top_vertebra and ll_complete, because these are spine-
                         limited scans: when L1 is clipped the arc starts lower and the
                         angle is smaller than the patient's.
  pi_ll_mismatch_deg     the two together, supine -- computed ONLY where the arc reaches
                         L1, since a truncated lordosis inflates the mismatch.

  crest_height_mm        how far the iliac crest rises ABOVE the L4-5 disc. The lateral
                         corridor: a high crest blocks an L4-5 lateral approach outright,
                         and a published cutoff of 12mm marks rising subsidence risk after
                         oblique lateral fusion at that level.
  rib12_to_crest_mm      the other boundary of the same corridor. This is where the rib
                         work becomes operative: a hypoplastic or absent twelfth rib moves
                         the upper limit of the working window.

  pedicle_width_mm       narrowest transverse width, per level -- it selects screw
                         diameter, and is itself a phenotype: wider at every lumbar level
                         in degenerative stenosis.
  canal_ap_mm            sagittal canal diameter, the decisive measure in degenerative
                         stenosis, with the Torg ratio against body depth.
  wedge_ratio            anterior over posterior body height; compression-fracture
                         screening, opportunistic.

    python scripts/extract_surgical_morphometrics.py --labels data/v5_final --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
# THE PER-LEVEL MEASURES RUN HIGHER THAN THE LUMBAR SPINE, and only they do. Canal depth
# and pedicle width stopped at L1 for no better reason than that this dict did, while
# canal WIDTH and endplate width -- computed by extract_level_gradients.py, which carries
# T11 and T12 -- ran two levels higher in the same figure. Panjabi's thoracic companion
# paper publishes both measures at T11 and T12, so the reference curve was there and our
# own was the half that stopped.
#
# LUMBAR is left alone deliberately. It defines the lordosis arc (which starts at L1 by
# convention, not at T11), the L4/L5 crest lookup, and the level set that pedicle_min_mm
# minimises over -- a number already reported. Only the per-level loop widens.
PER_LEVEL = {18: "T11", 19: "T12", **LUMBAR}
SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R = 26, 29, 30, 31, 32, 33

# RIB BASES ARE NOT CONSTANT ACROSS RELEASES, and getting them wrong is silent.
# Rib n on a side is `base + n`. The left block starts immediately after femur_right, so
# its base is 33 in every scheme here. The RIGHT block's base depends on how many slots
# the left one reserves: 12 (right ribs 46-57) or 13, where a slot is kept for the T13
# rib (right ribs 47-59). Hard-coding 45 is correct for the first and off by one for the
# second, where it reads the left 13th rib as a right 1st and loses the right 12th
# entirely -- a finite, plausible, wrong lowest-rib height, in a measure nothing else
# cross-checks.
#
# This was found by md5-ing two label directories that both look like "the labels": their
# vertebrae and pelvis are voxel-identical and their rib blocks differ by exactly this
# off-by-one. So the bases are read from the `dataset_labels.json` shipped beside the
# labels, which is the source of truth, and never assumed.
RIB_L_DEFAULT, RIB_R_DEFAULT = 33, 45


@lru_cache(maxsize=8)
def _rib_bases(labels_dir: str) -> tuple:
    """(left base, right base) from the scheme beside the labels, or the 12-slot default.

    Cached per directory and resolved inside the worker, not in main(): a value computed
    in main() reaches forked workers and not spawned ones, so it would be right on Linux
    and silently stale on Windows.
    """
    d = Path(labels_dir).resolve()
    for cand in (d / "dataset_labels.json", d.parent / "dataset_labels.json"):
        if not cand.exists():
            continue
        try:
            m = json.loads(cand.read_text()).get("name_to_id") or {}
        except Exception:                                        # noqa: BLE001
            continue
        l1, r1 = m.get("rib_left_1"), m.get("rib_right_1")
        if isinstance(l1, int) and isinstance(r1, int):
            return l1 - 1, r1 - 1
    return RIB_L_DEFAULT, RIB_R_DEFAULT
MIN_VOX = 3000


def _com(mask, sp):
    idx = np.argwhere(mask)
    return (idx.mean(0) * sp) if len(idx) else None


MIN_CANAL_MM2 = 18.0          # a lumbar canal is 200-300 mm^2; a pocket is not
CANAL_CENTROID_TOL_MM = 8.0   # how far a slice canal may sit from the column axis
MIDSAG_HALF_MM = 1.5          # half-width of the mid-sagittal strip the chord is read on


def _canal_column(m, sp):
    """The canal as one continuous column, and the pedicle slab within it.

    WHAT THIS REPLACES, AND WHY. The sagittal canal diameter used to be read off the FIRST
    axial slice in the middle third of the vertebra whose bony ring happened to close, as
    the total anteroposterior extent of the enclosed hole. Three things are wrong with
    that and all three push the same way:

      1. THE SLICE IS WHICHEVER ONE CLOSES FIRST, not an anatomical plane. Every normative
         series fixes the plane at the pedicle -- Maeder et al. (Diagnostics 13:734,
         n=1050) measure "in a plane perpendicular to the longitudinal axis of the spine
         and at the vertebral pedicle levels ... to limit the influence of degenerative
         changes, which typically occur at the intervertebral disc and facet joint
         levels"; Cook & Baker (Int J Spine Surg 15:1072) at the "pediculolaminar level".
         Taking the first closing slice is close to taking a minimum over noisy slices,
         and a minimum manufactures a low tail.
      2. THE RING NEED NOT CLOSE IN AN AXIAL PLANE. A lumbar vertebra in a supine scan of
         a lordotic adult is tilted, and an oblique section through the canal is filled in
         by volume averaging: Eubanks, Cann & Brant-Zawadzki (Radiology 157:243) showed in
         phantom that angled sections "do not always overestimate its diameter ... but can
         make it appear artifactually stenotic". A modern pipeline puts the cost at 22% of
         axis-aligned measurement lines invalid (Nixon et al., BMC Med Imaging 26:134).
      3. THE MAX EXTENT IS NOT THE MIDSAGITTAL DIAMETER. The L4 and especially L5 canal is
         trefoil (Aly & Amin, Orthopedics 36:e229, n=300), so its widest anteroposterior
         chord runs through a lateral recess rather than the midline.

    What it cost: 157 of 736 L3 records read under 11 mm. The smallest midsagittal canal
    anywhere in 1000 level-measurements of living adults is 11.70 mm (Cizmic et al., Acta
    Inform Med 31:200) and the Framingham CT cohort puts congenital absolute stenosis at
    2.62%. Twenty-one per cent is a method failing, not a cohort.

    So: build the canal as a column (Banik et al., J Digit Imaging 23:301 -- close the ring
    before looking for the hole; Diaz-Parra et al., EMBC 2014:5514 -- select it by
    continuity across slices rather than per slice), and return the slab where it is
    closed on all four sides.
    """
    zz = np.nonzero(m.any(axis=(0, 1)))[0]
    if len(zz) < 4:
        return None
    min_px = max(10, int(round(MIN_CANAL_MM2 / (sp[0] * sp[1]))))
    # Banik closes in 3-D with a tubular element 2 mm in radius; the same radius in-plane
    # bridges the interlaminar gap so the ring encloses. A fixed 3x3 would be 2 mm only at
    # 0.7 mm pitch and half that at 0.35.
    rr = max(1, int(round(2.0 / min(sp[0], sp[1]))))
    yy_, xx_ = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    se = (xx_ ** 2 + yy_ ** 2) <= rr ** 2

    cand = {}
    for z in zz:
        s_ = m[:, :, z]
        if s_.sum() < 60:
            continue
        h = ndimage.binary_fill_holes(ndimage.binary_closing(s_, se)) & ~s_
        if not h.any():
            continue
        cc, n = ndimage.label(h)
        if n == 0:
            continue
        sz = ndimage.sum(h, cc, range(1, n + 1))
        big = cc == int(np.argmax(sz)) + 1
        if big.sum() < min_px:
            continue
        xs, ys = np.nonzero(big)
        cand[int(z)] = (big, xs.mean(), ys.mean())
    if len(cand) < 3:
        return None

    # CONTINUITY AS A CENTROID GATE, NOT VOXEL ADJACENCY. Requiring the column to be one
    # 3-D connected component was tried and is too strict: on 3-5 mm slices the canal can
    # step several millimetres between neighbours and the column breaks into pieces too
    # short to measure, which cost 15% of the levels. Gating on distance to the column
    # axis rejects the same wrong regions -- a facet pocket is nowhere near the canal --
    # without demanding that voxels touch.
    cx = np.median([v[1] for v in cand.values()])
    cy = np.median([v[2] for v in cand.values()])
    keep = sorted(z for z, v in cand.items()
                  if np.hypot((v[1] - cx) * sp[0], (v[2] - cy) * sp[1])
                  <= CANAL_CENTROID_TOL_MM)
    if len(keep) < 3:
        return None
    runs, cur = [], [keep[0]]
    for a, b in zip(keep, keep[1:]):
        if b - a <= 2:                      # bridge a single dropped slice
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < 3:
        return None
    # the closed-ring column IS the pediculolaminar extent; its central half is the slab
    lo = run[int(0.25 * (len(run) - 1))]
    hi = run[int(0.75 * (len(run) - 1))]
    return cand, [z for z in run if lo <= z <= hi]


def _canal_ap(cand, slab, sp):
    """Midsagittal canal diameter, as the median over the pedicle slab.

    The chord is read on a strip about the canal own midline and taken as the run
    CONTAINING the canal centroid, so a second enclosed pocket in the same column cannot
    be spliced on. Every normative series measures this quantity and not the maximum
    extent: Verbiest criterion, Cizmic "mediosagittal diameter", Aly & Amin "midsagittal
    diameter", Cook & Baker on the midsagittal reformat, and Panjabi SCD.
    """
    aps = []
    for z in slab:
        if z not in cand:
            continue
        big, xbar, ybar = cand[z]
        xc = int(round(xbar))
        w = max(1, int(round(MIDSAG_HALF_MM / sp[0])))
        col = big[max(0, xc - w): xc + w + 1, :].any(axis=0)
        yy = np.nonzero(col)[0]
        if len(yy) < 2:
            continue
        lbl = ndimage.label(col)[0]
        yc = int(round(ybar))
        k = lbl[yc] if 0 <= yc < len(lbl) and lbl[yc] else int(lbl[yy].max())
        aps.append(float((lbl == k).sum()) * sp[1])
    return float(np.median(aps)) if aps else None


def _canal_front(mask, sp):
    """Anterior wall of the spinal canal, in voxel units along y. None if not found.

    The canal's anterior wall IS the posterior wall of the vertebral body, so it splits
    body from posterior elements without needing a threshold or a fraction.
    """
    zs = np.nonzero(mask.any(axis=(0, 1)))[0]
    if len(zs) < 5:
        return None
    fronts = []
    # sample the middle of the body's height: the canal is open there, whereas near the
    # endplates the ring is incomplete and fill_holes finds nothing
    lo, hi = int(np.percentile(zs, 30)), int(np.percentile(zs, 70))
    for z in range(lo, hi + 1):
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
        # y increases anteriorly in canonical orientation, and the canal sits behind the
        # body, so the canal's largest y is its anterior wall
        fronts.append(int(np.nonzero(big.any(axis=0))[0].max()))
    return float(np.median(fronts)) if fronts else None


def _body_mask(mask, sp):
    """The vertebral BODY, split off from the posterior elements.

    THE LABELS ARE WHOLE VERTEBRAE. Every measure that says "body" -- the endplate
    normal, the Torg denominator, wedging -- is wrong if the spinous process and the
    articular processes are inside it. The spinous process alone roughly triples the
    anteroposterior extent, which is what drove the Torg ratio to 0.2 against a normal
    near 1.0.
    """
    front = _canal_front(mask, sp)
    if front is None:
        # no canal found (a sacral segment, or a body too clipped to close a ring):
        # fall back to the anterior 55% of the y-extent
        ys = np.nonzero(mask.any(axis=(0, 2)))[0]
        if len(ys) < 4:
            return mask
        front = ys.min() + 0.45 * (ys.max() - ys.min())
    out = np.zeros_like(mask)
    out[:, int(np.ceil(front)):, :] = mask[:, int(np.ceil(front)):, :]
    return out if out.sum() >= 50 else mask


def _endplate(mask, sp, superior=True):
    """Centroid and normal of a vertebral endplate, fitted to the ENDPLATE SURFACE.

    WHY NOT A TOP-DECILE SLAB. Selecting the top decile of voxels BY HEIGHT bounds the
    selection with two horizontal planes, so the thinnest direction of what comes back is
    z -- always, whatever the endplate is actually doing. An SVD then returns the vertical
    as the "normal" for every vertebra in every case. That is what put sacral slope at 83
    degrees across the whole release and collapsed lumbar lordosis to 10.

    So fit the SURFACE instead: for each (x, y) column through the body take the topmost
    (or bottommost) voxel, which traces the real plate, and fit a plane to those points.
    The rim rolls off, so the outer margin is dropped and the fit is trimmed once against
    its own residuals -- osteophytes and a clipped corner should not tilt the plate.
    """
    body = _body_mask(mask, sp)
    idx = np.argwhere(body)
    if len(idx) < 80:
        return None, None

    xs, ys = idx[:, 0], idx[:, 1]
    # keep the central core of the plate: the peripheral ring curves away from the plane
    x0, x1 = np.percentile(xs, [18, 82])
    y0, y1 = np.percentile(ys, [18, 82])
    keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    idx = idx[keep]
    if len(idx) < 60:
        return None, None

    # the surface: one z per (x, y) column
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    idx = idx[order]
    col = idx[:, 0] * (body.shape[1] + 1) + idx[:, 1]
    edge = np.nonzero(np.diff(col))[0]
    starts = np.concatenate(([0], edge + 1))
    ends = np.concatenate((edge, [len(idx) - 1]))
    pick = ends if superior else starts          # z is sorted ascending within a column
    pts = idx[pick].astype(float) * sp
    if len(pts) < 40:
        return None, None

    def fit(p):
        c = p.mean(0)
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        return c, (n if n[2] >= 0 else -n)

    c, n = fit(pts)
    resid = np.abs((pts - c) @ n)
    keep = resid <= np.percentile(resid, 80)
    if keep.sum() >= 30:
        c, n = fit(pts[keep])
    return c, n


def _angle(u, v):
    cu = u / (np.linalg.norm(u) + 1e-9)
    cv = v / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cu @ cv, -1, 1))))



def _femoral_head(fem_mask, hip_mask, sp):
    """Centroid of the femoral head: the part of the femur inside the acetabulum.

    Defined by contact rather than by a percentile — the head is what sits against the
    hip bone. Falls back to the whole-femur centroid only when the hip label is absent,
    which is the behaviour this replaces.
    """
    if hip_mask is None or not hip_mask.any():
        return _com(fem_mask, sp)
    # CROP FIRST. A distance transform over the whole volume allocates a 512^3 float64
    # array -- a gigabyte per call, twice per case, across every worker. It OOM-killed
    # the job. The femoral head is where femur and hip meet, so only the box containing
    # both can matter, and the answer is identical.
    both = fem_mask | hip_mask
    idx = np.argwhere(both)
    if not len(idx):
        return _com(fem_mask, sp)
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, np.array(fem_mask.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    fsub, hsub = fem_mask[sl], hip_mask[sl]
    d = ndimage.distance_transform_edt(~hsub, sampling=sp)
    near = fsub & (d <= 6.0)
    if near.sum() < 200:
        near = fsub & (d <= 12.0)
    if near.sum() < 200:
        return _com(fem_mask, sp)
    # centroid of the cropped selection, shifted back into the full volume's frame
    c = _com(near, sp)
    return None if c is None else c + lo * sp


def one(path: str) -> dict:
    stem = Path(path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        img = nib.as_closest_canonical(nib.load(path))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    have = {v: (lab == v) for v in list(PER_LEVEL) + [SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R]
            if (lab == v).sum() >= MIN_VOX}

    # ---- pelvic incidence (position independent) ---------------------------------
    fem = None
    if FEM_L in have and FEM_R in have:
        cl = _femoral_head(have[FEM_L], have.get(HIP_L), sp)
        cr = _femoral_head(have[FEM_R], have.get(HIP_R), sp)
        if cl is not None and cr is not None:
            fem = (cl + cr) / 2
            r["bi_acetabular_mm"] = round(float(np.linalg.norm(cl - cr)), 1)
    elif HIP_L in have and HIP_R in have:
        # fall back to the acetabular centroids; less exact than femoral heads but the
        # bicoxofemoral axis is a midpoint either way
        cl, cr = _com(have[HIP_L], sp), _com(have[HIP_R], sp)
        fem = (cl + cr) / 2
    s1c, s1n = (_endplate(have[S1], sp, True) if S1 in have
                else (None, None))
    if s1c is None and SACRUM in have:
        s1c, s1n = _endplate(have[SACRUM], sp, True)
    # A PLATE WHOSE NORMAL IS NOT ROUGHLY CRANIAL IS NOT A PLATE. The S1 fit sometimes
    # latches onto the near-vertical anterior face of the promontory instead of the
    # superior surface; its normal then points forward rather than up (z-component 0.03 to
    # 0.44, a sacral slope near 88 degrees) and the pelvic incidence that follows is
    # arbitrary. Measured over the release that is about one case in eleven.
    #
    # The gate is on the GEOMETRY, not on the answer. Rejecting a pelvic incidence for
    # being an unlikely number would discard unusual patients along with broken fits;
    # rejecting a sacral endplate whose normal lies more than 60 degrees off the cranial
    # axis discards only fits that cannot be endplates, whatever they imply. The bound is
    # deliberately loose -- a sacral slope of 60 degrees is steep but real, 88 is not.
    # After Tier-1 practice in the automated-CT literature: emit a flagged absence with a
    # reason, never a number nobody can distinguish from a good one.
    MAX_PLATE_TILT_DEG = 60.0
    if s1n is not None:
        tilt = _angle(s1n, np.array([0.0, 0.0, 1.0]))
        r["s1_plate_tilt_deg"] = round(float(tilt), 1)
        r["s1_plate_n_ant"] = round(float(s1n[1]), 3)
        if tilt > MAX_PLATE_TILT_DEG:
            r["s1_plate_rejected"] = 1
            s1c = s1n = None
        # AND THE TILT GATE IS BLIND TO THE DIRECTION OF THE TILT, which is the other half
        # of the failure. `_endplate` orients its normal by the SUPERIOR component alone --
        # `n if n[2] >= 0 else -n` -- so a plane fitted to a surface that slopes the wrong
        # way comes back leaning POSTERIORLY, at the same angle from vertical as a good
        # one, and sails through a gate that measures only that angle.
        #
        # It is not a sign ambiguity; it is a different surface. Rendering the fit shows
        # what it lands on: where the S1 carve produces a compact wedge under L5 the
        # voxels selected lie on the superior end-plate and the normal points up and
        # forward, but where the carve fails -- an irregular blob, or, on some records, an
        # "S1" spanning most of the sacrum -- the plane is fitted to the VENTRAL SURFACE
        # OF THE SACRUM, a long antero-inferior ramp, and pelvic incidence collapses.
        #
        # Measured over the release, 39 of 39 records returning PI <= 17 have a posteriorly
        # leaning normal and 39 of 39 returning PI > 35 have an anteriorly leaning one --
        # a clean separation on a quantity nothing downstream was looking at.
        #
        # So gate on it, in the same spirit as the tilt gate: an S1 superior end-plate in a
        # supine adult has a normal with a positive anterior component. One that does not
        # is not that surface, whatever angle it makes with the vertical. This replaces the
        # figure's ad-hoc "pelvic tilt >= -15 degrees" filter, which was removing the same
        # records by their ANSWER rather than by their geometry.
        elif s1n[1] <= 0.0:
            r["s1_plate_rejected"] = 1
            r["s1_plate_reject_reason"] = "normal leans posteriorly"
            s1c = s1n = None

    if fem is not None and s1c is not None and s1n is not None:
        v = fem - s1c
        # PI is measured in the sagittal plane: drop the left-right component
        vs = np.array([0.0, v[1], v[2]])
        ns = np.array([0.0, s1n[1], s1n[2]])
        if np.linalg.norm(vs) > 1 and np.linalg.norm(ns) > 1e-6:
            # PI is the angle at the endplate midpoint between the plate's PERPENDICULAR
            # and the line to the bicoxofemoral axis. The femoral heads sit below and in
            # front of S1, so that line runs opposite to the superior-pointing normal and
            # the raw angle comes back obtuse; PI is its supplement.
            r["pelvic_incidence_deg"] = round(180.0 - _angle(ns, vs), 1)
            # Sacral slope: the S1 plate against the horizontal, which is the angle its
            # NORMAL makes with the vertical -- not ninety minus that. A horizontal plate
            # has a vertical normal and a slope of zero, and the old form returned 90.
            r["sacral_slope_deg"] = round(_angle(ns, np.array([0.0, 0.0, 1.0])), 1)
            r["pelvic_tilt_deg"] = round(r["pelvic_incidence_deg"]
                                         - r["sacral_slope_deg"], 1)

    # ---- lumbar lordosis, SUPINE -------------------------------------------------
    # WHAT THE TOP OF THE ARC ACTUALLY IS. Lordosis is conventionally L1 superior to S1
    # superior, but lums[0] is the topmost lumbar IN THE FIELD OF VIEW, and these are
    # spine-limited scans. If L1 is clipped the arc starts at L2 and the angle comes back
    # smaller than the patient's -- which then propagates straight into PI-LL mismatch,
    # the one number here a surgeon would act on. So record where the arc really started
    # and how many bodies it spanned, and let the reader gate on it.
    lums = sorted(v for v in have if v in LUMBAR)
    if lums and s1n is not None:
        c1, n1 = _endplate(have[lums[0]], sp, True)
        if n1 is not None:
            r["ll_supine_deg"] = round(_angle(n1, s1n), 1)
            r["ll_top_vertebra"] = LUMBAR.get(lums[0])
            r["ll_levels_spanned"] = len(lums)
            # A six-lumbar spine spans six and still starts at L1, so the test is where
            # the arc STARTS, not how many bodies are under it.
            r["ll_complete"] = int(LUMBAR.get(lums[0]) == "L1")
            if "pelvic_incidence_deg" in r and r["ll_complete"]:
                r["pi_ll_mismatch_deg"] = round(r["pelvic_incidence_deg"]
                                                - r["ll_supine_deg"], 1)

    # ---- pelvic shape: the measures that ARE dimorphic ----------------------------
    # These exist to be checked against a century of anatomy, and to fail loudly if the
    # pelvic labels, the sidedness or the voxel spacing ever go wrong.
    if HIP_L in have and HIP_R in have:
        hl, hr = np.argwhere(have[HIP_L]), np.argwhere(have[HIP_R])
        # widest outer span across both crests, measured along the left-right axis
        allx = np.concatenate([hl[:, 0], hr[:, 0]])
        r["bi_iliac_width_mm"] = round(float(allx.max() - allx.min()) * sp[0], 1)
        # sacral shape: width over height. The most reported pelvic sex difference.
        if SACRUM in have:
            si = np.argwhere(have[SACRUM])
            w = float(si[:, 0].max() - si[:, 0].min()) * sp[0]
            h = float(si[:, 2].max() - si[:, 2].min()) * sp[2]
            if h > 1:
                r["sacral_width_mm"] = round(w, 1)
                r["sacral_height_mm"] = round(h, 1)
                r["sacral_width_ratio"] = round(w / h, 3)

    # pelvic inlet, anteroposterior: sacral promontory forward to the pubic symphysis.
    # The promontory is the most anterior-superior point of S1; the symphysis is the most
    # anterior point of the hips at roughly that height.
    if S1 in have and (HIP_L in have or HIP_R in have):
        si = np.argwhere(have[S1])
        top = si[si[:, 2] >= np.percentile(si[:, 2], 88)]
        if len(top):
            prom = top[np.argmax(top[:, 1])] * sp
            hips = np.zeros_like(lab, bool)
            for h in (HIP_L, HIP_R):
                if h in have:
                    hips |= have[h]
            hi = np.argwhere(hips)
            # the pubic bones meet at the ANTERIOR MIDLINE: that is the symphysis. The
            # first version took the most anterior hip voxel at the promontory's height,
            # which is up on the iliac wing and gave an inlet of 70 mm.
            midx = float(np.median(hi[:, 0]))
            near = hi[np.abs(hi[:, 0] - midx) * sp[0] <= 18.0]
            if len(near) > 50:
                sym = near[np.argmax(near[:, 1])] * sp
                ap = float(np.hypot(sym[1] - prom[1], sym[2] - prom[2]))
                if 60 < ap < 200:
                    r["pelvic_inlet_ap_mm"] = round(ap, 1)
                    if r.get("bi_acetabular_mm"):
                        # rounder inlet -> higher index. The classic shape ratio.
                        r["inlet_index"] = round(ap / r["bi_acetabular_mm"], 3)

    # ---- the lateral corridor ----------------------------------------------------
    crest = None
    if HIP_L in have or HIP_R in have:
        hips = np.zeros_like(lab, bool)
        for h in (HIP_L, HIP_R):
            if h in have:
                hips |= have[h]
        crest = float(np.nonzero(hips.any(axis=(0, 1)))[0].max()) * sp[2]
        r["crest_z_mm"] = round(crest, 1)
    # L4-5 disc level = midpoint between the two bodies
    l4 = next((v for v in have if LUMBAR.get(v) == "L4"), None)
    l5 = next((v for v in have if LUMBAR.get(v) == "L5"), None)
    if crest is not None and l4 and l5:
        # the disc sits between the BODIES. Using whole vertebrae puts L4's inferior
        # articular process and L5's superior one into the span, and those overlap each
        # other by design -- the midpoint of the two would land inside the facet joint
        # rather than at the disc.
        z4 = np.nonzero(_body_mask(have[l4], sp).any(axis=(0, 1)))[0]
        z5 = np.nonzero(_body_mask(have[l5], sp).any(axis=(0, 1)))[0]
        disc = (z4.min() + z5.max()) / 2 * sp[2]
        # POSITIVE means the crest rises above the L4-5 disc and obstructs a lateral
        # approach; the published subsidence cutoff is +12mm
        r["crest_above_l45_mm"] = round(crest - disc, 1)
        r["crest_blocks_l45"] = int((crest - disc) >= 12.0)
    # the other boundary: lowest rib to crest
    ribs = np.zeros_like(lab, bool)
    lowest = None
    for base in _rib_bases(str(Path(path).parent)):
        for n in range(1, 13):
            m = lab == base + n
            if m.sum() > 200:
                ribs |= m
                z = np.nonzero(m.any(axis=(0, 1)))[0].min() * sp[2]
                lowest = z if lowest is None else min(lowest, z)
    if crest is not None and lowest is not None:
        r["rib12_to_crest_mm"] = round(lowest - crest, 1)

    # ---- per-level pedicle width, canal, wedging ---------------------------------
    ped, canal, torg, wedge = {}, {}, {}, {}
    ped_side, ped_mean = {}, {}
    for vid in sorted(have):
        if vid not in PER_LEVEL:
            continue
        m = have[vid]
        idx = np.argwhere(m)
        # SCAN, DO NOT SAMPLE ONE SLICE. This was the single axial slice at the median z
        # of the WHOLE vertebra, spinous process included, so the plane was not even at
        # mid-body. When the bony ring did not happen to close there the level was lost
        # for canal depth AND pedicle width together, which is why both carried about
        # half the n that canal width does. Take the first slice in the middle third
        # whose ring actually closes instead.
        col = _canal_column(m, sp)
        if col is None:
            continue
        cand, slab = col
        zc = int(slab[len(slab) // 2])
        sl = m[:, :, zc]
        ap = _canal_ap(cand, slab, sp)
        if ap is not None:
            canal[PER_LEVEL[vid]] = round(ap, 1)
        big = cand[zc][0]
        if True:
            hy = np.nonzero(big.any(axis=0))[0]
            # TORG DENOMINATOR IS THE BODY, NOT THE VERTEBRA. Measuring the whole slice
            # puts the spinous process in the denominator, roughly tripling it -- which
            # is why the ratio came back near 0.2 against a normal near 1.0. The canal's
            # anterior wall is the body's posterior wall, so the body runs from there
            # forward.
            bsl = sl[:, hy.max():]
            by = np.nonzero(bsl.any(axis=0))[0]
            depth = float(len(by)) * sp[1]
            if depth > 0 and PER_LEVEL[vid] in canal:
                torg[PER_LEVEL[vid]] = round(canal[PER_LEVEL[vid]] / depth, 3)
            # PEDICLE ISTHMUS, PER SIDE, AT THE SLICE WHERE THE PEDICLE EXISTS.
            # Two earlier versions were wrong in opposite directions. The first took the
            # full left-right extent of bone and halved it, canal included, and read
            # ~17mm. The second took the narrowest bone run outboard of the canal over
            # the canal's WHOLE anteroposterior range -- but at the back of the canal
            # that bone is LAMINA, which is thinner than any pedicle, so it read ~5mm
            # where L5 is 15-18.
            #
            # The pedicle is the bridge at the ANTERIOR corner of the canal, joining body
            # to arch, and it exists only over part of the vertebra's height. So: look
            # only at the front third of the canal, and take the widest slice, because a
            # slice below the pedicle has no bridge to measure and would win a minimum
            # for the wrong reason.
            per_side = {"l": [], "r": []}
            zlo, zhi = int(np.percentile(idx[:, 2], 20)), int(np.percentile(idx[:, 2], 80))
            for z in range(zlo, zhi + 1):
                s2 = m[:, :, z]
                if s2.sum() < 60:
                    continue
                h2 = ndimage.binary_fill_holes(s2) & ~s2
                if not h2.any():
                    continue
                c2, n2 = ndimage.label(h2)
                if n2 == 0:
                    continue
                sz = ndimage.sum(h2, c2, range(1, n2 + 1))
                cn = c2 == (int(np.argmax(sz)) + 1)
                if cn.sum() < 20:
                    continue
                cy = np.nonzero(cn.any(axis=0))[0]
                # front third of the canal: where the pedicle is, not the lamina
                y_front = cy[cy >= cy.min() + 0.66 * (cy.max() - cy.min())]
                if len(y_front) < 2:
                    continue
                for side in ("l", "r"):
                    runs_z = []
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
                            runs_z.append(float(seg.max() - seg.min() + 1) * sp[0])
                    if runs_z:
                        per_side[side].append(min(runs_z))
            # MEDIAN ACROSS SLICES, not the widest and not the narrowest. The widest
            # slice is where the transverse process merges into the arch and reads far
            # too broad (L5 came back 22.7mm); the narrowest is a slice with no real
            # bridge at all. The isthmus is a waist, so the typical slice is the honest
            # one.
            got = {k: float(np.median(v)) for k, v in per_side.items() if v}
            if got:
                # KEEP BOTH SIDES. min() is the right number for a screw and the wrong
                # one for morphometry: Panjabi reports PDWr and PDWl separately, and
                # their own left-right gaps run to 1.3 mm at L4, so comparing our min
                # against their mean is biased low before any anatomy is considered.
                # pedicle_mm keeps its clinical meaning; the rest are new.
                ped[PER_LEVEL[vid]] = round(min(got.values()), 1)
                for k, v in got.items():
                    ped_side[f"{PER_LEVEL[vid]}_{k}"] = round(v, 1)
                ped_mean[PER_LEVEL[vid]] = round(sum(got.values()) / len(got), 1)

        # WEDGING IS A PROPERTY OF THE BODY. Taking the posterior fifth of the whole
        # vertebra measures the spinous process, which spans far more height than the
        # posterior body wall does, so every vertebra read as anteriorly collapsed.
        bm = _body_mask(m, sp)
        ys = np.nonzero(bm.any(axis=(0, 2)))[0]
        if len(ys) > 6:
            k = max(2, len(ys) // 5)
            ant = bm[:, ys[-k:], :]
            post = bm[:, ys[:k], :]
            za = np.nonzero(ant.any(axis=(0, 1)))[0]
            zp = np.nonzero(post.any(axis=(0, 1)))[0]
            if len(za) and len(zp):
                wedge[PER_LEVEL[vid]] = round(len(za) / len(zp), 3)

    # pedicle_mean_mm is the like-for-like counterpart to a published PDW, which is
    # reported per side; pedicle_l/_r carry the sides themselves. pedicle_mm stays
    # the min, the pedicle that limits a screw.
    for nm, d in (("pedicle_mm", ped), ("pedicle_mean_mm", ped_mean),
                  ("pedicle_side_mm", ped_side), ("canal_ap_mm", canal),
                  ("torg", torg), ("wedge", wedge)):
        for k, v in d.items():
            r[f"{nm}_{k}"] = v
    # A MINIMUM OVER LEVELS INHERITS EVERY FAILURE AMONG THEM. The narrowest pedicle is
    # the clinically useful number -- it constrains screw diameter -- but taken as a raw
    # minimum across five or six levels it is decided by whichever level was measured
    # worst, not by the anatomy. It returned 1.8 mm on some cases, which is not a pedicle,
    # and 27.3 mm on others, which is not one either.
    #
    # So each level is checked against a physiological envelope before the minimum is
    # taken, and the count of levels that survived is recorded alongside it. A minimum
    # over two levels is a different statistic from a minimum over six and the reader is
    # entitled to know which one they have.
    PLAUSIBLE = {"ped": (3.5, 20.0), "torg": (0.15, 1.2), "wedge": (0.4, 1.8)}

    def _min_of(d, key, name, nd=1):
        lo, hi = PLAUSIBLE[key]
        ok = {k: v for k, v in d.items() if v is not None and lo <= v <= hi}
        if not ok:
            return
        r[name] = round(min(ok.values()), nd)
        r[f"{name}_n_levels"] = len(ok)
        if len(ok) < len(d):
            r[f"{name}_n_dropped"] = len(d) - len(ok)

    # THE MINIMA STAY LUMBAR. pedicle_min_mm is reported as the narrowest LUMBAR pedicle
    # and T11-T12 pedicles are narrower than any of them, so letting the widened loop feed
    # this would silently redefine a published number as a thoracolumbar minimum.
    LUM = {v for v in LUMBAR.values()}
    _min_of({k: v for k, v in ped.items() if k in LUM}, "ped", "pedicle_min_mm", 1)
    _min_of({k: v for k, v in torg.items() if k in LUM}, "torg", "torg_min", 3)
    _min_of({k: v for k, v in wedge.items() if k in LUM}, "wedge", "wedge_min", 3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    labdir = Path(a.labels)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    files = [str(labdir / f"{s}_label.nii.gz") for s in stems
             if (labdir / f"{s}_label.nii.gz").exists()]
    print(f"{len(files)} case(s)\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, files, chunksize=2), 1):
            res.append(x)
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    lut = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
            if s:
                lut[s] = {k: rec.get(k) for k in ("age", "sex", "lstv_label", "config")}
    for x in res:
        x.update(lut.get(x["case"], {}))

    ok = [x for x in res if "error" not in x]
    cols = sorted({k for x in ok for k in x}, key=lambda k: (k != "case", k))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "surgical_morphometrics.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    # WHAT A HUMAN PELVIS AND LUMBAR SPINE ACTUALLY MEASURE. A geometry bug does not
    # announce itself -- an earlier version returned sacral slope 83 deg and a Torg ratio
    # of 0.2 for all 802 cases and reported them in a neat table, and nothing in the run
    # objected. These are generous windows around published adult medians; a median that
    # falls outside one is a defect in the measurement, not a finding about the cohort.
    PLAUSIBLE = {
        "pelvic_incidence_deg": (35, 70),      # adult median ~50-52
        "sacral_slope_deg": (25, 55),          # ~40
        "pelvic_tilt_deg": (2, 25),            # ~13
        "ll_supine_deg": (35, 65),             # ~50 supine, ~55 standing
        "pi_ll_mismatch_deg": (-15, 20),       # centred near zero in the unoperated
        "pedicle_min_mm": (5, 16),             # L1 ~7 rising to L5 ~15
        # NOT the cervical Torg-Pavlov range. That ratio is ~1.0 in the neck, but a
        # lumbar canal is about half the depth of a lumbar body, so ~0.5 is normal here
        # and the cervical figure would condemn a correct measurement.
        "torg_min": (0.35, 0.75),
        # adult pelvis, both sexes pooled
        "bi_iliac_width_mm": (220, 320),
        "bi_acetabular_mm": (130, 210),
        "pelvic_inlet_ap_mm": (95, 145),
        "sacral_width_ratio": (0.9, 2.2),
        "canal_ap_min_mm": (11, 22),           # lumbar midsagittal canal, ~15-18
    }
    suspect = []

    def summarise(key, unit=""):
        v = np.array([x[key] for x in ok if isinstance(x.get(key), (int, float))], float)
        if v.size < 5:
            print(f"  {key:24s} n={v.size} (too few)")
            return
        med = float(np.median(v))
        flag = ""
        if key in PLAUSIBLE:
            lo, hi = PLAUSIBLE[key]
            if not lo <= med <= hi:
                flag = f"   <-- IMPLAUSIBLE (expected {lo}-{hi})"
                suspect.append(key)
        print(f"  {key:24s} n={v.size:4d}  median {med:7.1f}{unit}  "
              f"IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}{flag}")

    print(f"\n  {len(ok)} of {len(res)} cases measured\n")
    for k, u in (("pelvic_incidence_deg", "°"), ("sacral_slope_deg", "°"),
                 ("pelvic_tilt_deg", "°"), ("ll_supine_deg", "°"),
                 ("pi_ll_mismatch_deg", "°"), ("crest_above_l45_mm", "mm"),
                 ("rib12_to_crest_mm", "mm"), ("pedicle_min_mm", "mm"),
                 ("bi_iliac_width_mm", "mm"), ("bi_acetabular_mm", "mm"),
                 ("pelvic_inlet_ap_mm", "mm"), ("sacral_width_ratio", ""),
                 ("inlet_index", ""),
                 ("torg_min", "")):
        summarise(k, u)
    n_ll = sum(1 for x in ok if x.get("ll_supine_deg") is not None)
    n_full = sum(1 for x in ok if x.get("ll_complete"))
    if n_ll:
        print(f"\n  lordosis arc reaches L1 in {n_full}/{n_ll} measured cases; "
              f"PI-LL mismatch is reported for those only")
    blocked = sum(1 for x in ok if x.get("crest_blocks_l45"))
    n_cr = sum(1 for x in ok if x.get("crest_blocks_l45") is not None)
    if n_cr:
        print(f"\n  crest >= 12mm above the L4-5 disc (raised subsidence risk after "
              f"lateral fusion): {blocked}/{n_cr} = {100*blocked/n_cr:.1f}%")
    # BY SEX, on the measures where a difference is expected. This is the sanity check
    # that pelvic incidence could not be: if the pelvic labels, the sidedness or the
    # spacing were wrong, these would move. Reported as a check on the pipeline, not as
    # a finding -- the sex difference in pelvic shape has been known for a century.
    def by_sex(key, unit=""):
        g = {}
        for want, lab_ in (("F", "female"), ("M", "male")):
            v = [x[key] for x in ok
                 if isinstance(x.get(key), (int, float))
                 and str(x.get("sex", "")).strip().upper().startswith(want)]
            if len(v) >= 20:
                g[lab_] = (len(v), float(np.median(v)))
        if len(g) == 2:
            (nf, mf), (nm, mm_) = g["female"], g["male"]
            print(f"  {key:22s} female {mf:7.1f}{unit} (n={nf})   "
                  f"male {mm_:7.1f}{unit} (n={nm})   diff {mf - mm_:+.1f}")

    print("\n  pelvic shape by sex (a check on the pipeline, not a finding):")
    for k, u in (("bi_iliac_width_mm", "mm"), ("bi_acetabular_mm", "mm"),
                 ("pelvic_inlet_ap_mm", "mm"), ("sacral_width_ratio", ""),
                 ("inlet_index", ""), ("pelvic_incidence_deg", "\u00b0")):
        by_sex(k, u)

    print(f"\n  wrote {p}")
    if suspect:
        print(f"\n  *** {len(suspect)} measure(s) outside the plausible range: "
              f"{', '.join(suspect)}")
        print("  *** These are measurement defects, not cohort findings. Do not report "
              "them until the geometry is fixed.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
