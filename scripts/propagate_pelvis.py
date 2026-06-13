"""
propagate_pelvis.py — replace the MODEL pseudolabel pelvis with the patient's
OWN radiologist pelvis GT, carried across acquisitions by registration.

The "separate" cohort (the dominant group: a token whose spine mask and pelvic
mask won DIFFERENT scans) has, for every spine-side scan that needs a pelvis, a
REAL radiologist pelvis GT sitting on that same patient's OTHER (pelvic-side)
scan. Instead of asking a model to guess the pelvis, we warp the real GT from the
pelvic-side scan onto the spine-side scan. It is the identical bone, so this is
strictly higher-fidelity than a population model's completion.

Rigid carry-over (whole-pelvis default; optional per-bone)
----------------------------------------------------------
The bony pelvis moves prone<->supine essentially as ONE rigid unit — it rocks,
tilts and translates but does not deform — so by default we fit a single RIGID
transform (3 rotations + 3 translations), the exact description of "the pelvis as
one solid object." The metric is masked to the moving pelvic mask so the optimizer
aligns the PELVIS and is not pulled off by the differently-articulated spine in the
same scan. On the test cases this seats the sacrum + ilia cleanly on bone.

The pelvis is, strictly, ARTICULATED rigid bone (the SI joints let the sacrum rock
a few degrees relative to the ilia). For cases where that matters, --per_bone ALSO
refines a separate rigid per bone {sacrum, left hip, right hip} from the whole-
pelvis init and composites them, absorbing the articulation. It is opt-in because
the single rigid is already clean on the validated cases.

Every rigid is volume-preserving (det J == 1), so there is no bone-squashing failure
mode and no Jacobian gate. A case falls back to the model only on a real failure:
the bone-HU overlap drops >fail_drop pp vs the native placement, or a bone is
FOV-truncated (vol ratio far below 1 -> no data). Native vs propagated bone-HU
overlap is reported so the carry-over can be shown not to degrade placement quality.

Where this runs
---------------
This is a CONSTRUCTION stage, a sibling of place_fused_masks.py and a step
BETWEEN create_dataset and export — not part of export and not part of the
pseudolabel model machinery. place_fused_masks.py places each mask on its winning
scan; this places the pelvis GT onto the patient's OTHER (spine) scan too, by
registration, in the same placed-space coordinate frame. It only ADDS real GT, so
it is purely additive (the append-only law): v1 (the partial-annotation artifact
that trained the pseudolabeller) is untouched; the propagated pelvis is folded in
only at the DENSE (v2) build, where it REPLACES the model pelvis on accepted
separate cases (the model stays the fallback for rejects + genuinely spine_only).

Output (placed-space, native spine-scan grid — a drop-in placed mask)
---------------------------------------------------------------------
  <out_dir>/<spine_uid>_pelvic_propagated.nii.gz   canonical pelvis
                             (7=sacrum, 8=left_hip, 9=right_hip) on the spine-side
                             scan, ready for export to merge like any placed mask.
  placed_manifest_propagated.json  a placed_manifest.json-STYLE manifest (so
                             export + downstream analysis consume it exactly like a
                             place_fused_masks placed mask): per case a spine+pelvic
                             sub-dict, prov_pelvis='manual_propagated', the native
                             vs propagated bone-HU overlap, and the accept flag.
  propagate_qc.csv           the same metrics, flat, one row per case.

Registration uses SimpleITK (already in the container — no rebuild): a GEOMETRY-
initialized, multi-resolution RIGID registration with the metric masked to the
moving pelvic mask. The QC/gating math is pure numpy and is unit-tested without
SimpleITK.

Usage
-----
  python scripts/propagate_pelvis.py \
      --manifest  data/placed/placed_manifest.json \
      --nifti_dir data/tcia_nifti \
      --pelvic_dir data/placed/pelvic \
      --out_dir   data/placed/pelvic_propagated
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.propagate_pelvis")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Pelvic GT label convention (raw CTPelvic1K "4label") -> canonical, matching
# export_hf.PELVIC_TO_10CLASS exactly. 1=sacrum, 2=left_hip, 3=right_hip.
PELVIC_TO_CANONICAL = {1: 7, 2: 8, 3: 9}
SACRUM, LEFT_HIP, RIGHT_HIP = 7, 8, 9
PELVIS = {SACRUM: "sacrum", LEFT_HIP: "left_hip", RIGHT_HIP: "right_hip"}
BONE_HU = 200
# Lumbar labels in the fixed-scan spine GT: VerSe L1-L6 = 20-25 (the placed
# CTSpine1K convention), or canonical 1-6 as a fallback. Their INFERIOR aspect is
# the L5/S1 junction the radiologist drew — the anatomical anchor for the sacrum.
LUMBAR_VERSE = (20, 21, 22, 23, 24, 25)
LUMBAR_CANON = (1, 2, 3, 4, 5, 6)
# Resolution (mm) the rigid registration runs at. A 6-DOF rigid pelvis fit is fully
# determined at ~2-3mm; registering at the native 512^3 was needless minutes of
# single-threaded smoothing + memory. Overlap is still scored at full res.
REG_MM = 2.5
# Fixed RNG seed for the metric sampler -> the registration is bit-for-bit
# reproducible (NEVER seed from the wall clock here). Determinism is a property
# of the released dataset, not a convenience: same inputs -> same pelvis, always.
SEED = 42


# ===========================================================================
# Pure QC / gating math (no ANTs — unit-tested)
# ===========================================================================

def bone_mask(ct, hu: int = BONE_HU):
    import numpy as np
    return np.asarray(ct) >= hu


def jacobian_stats_in_mask(jac, mask, tol: float = 0.3) -> Tuple[float, float, int]:
    """Inside `mask`, summarize the warp's local volume change. A rigid bone must
    keep det J ~ 1; returns (median det J, fraction with |detJ-1|>tol, n voxels).
    A high bad-fraction means the deformable field is deforming bone -> reject."""
    import numpy as np
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    vals = np.asarray(jac)[m].astype(np.float64)
    med = float(np.median(vals))
    bad = float(np.mean(np.abs(vals - 1.0) > tol))
    return med, bad, n


def bone_fit_fraction(target_ct, warped_bone) -> float:
    """Fraction of warped-mask voxels that land on actual target bone-HU. A good
    registration seats the bone on bone (high); a bad fit floats it in soft
    tissue (low). Uses the SAME rule as place_fused_masks._compute_bone_pct_from
    _files (CT > 200), so it is directly comparable to the native placement."""
    import numpy as np
    m = np.asarray(warped_bone, bool)
    if not m.any():
        return float("nan")
    return float(np.mean(np.asarray(target_ct)[m] > BONE_HU))


def bone_pct(target_ct, mask) -> float:
    """place_fused_masks-identical bone-HU overlap statistic: percent of mask
    voxels with CT > 200, rounded to 1 dp. Lets us verify registration did not
    degrade placement quality (compare to the source mask's native bone_pct)."""
    f = bone_fit_fraction(target_ct, mask)
    return round(f * 100, 1) if f == f else float("nan")


def volume_ratio(warped_bone, source_bone_n: int, *,
                 warped_voxvol: float = 1.0, source_voxvol: float = 1.0) -> float:
    """warped PHYSICAL volume / source physical volume. ~1 for a clean rigid carry-
    over; << 1 means the bone is FOV-truncated on the target. Pass each grid's voxel
    volume (mm^3) since the source mask and the target are on different grids; the
    voxvol defaults (1.0) reduce it to a plain voxel-count ratio."""
    import numpy as np
    if source_bone_n <= 0:
        return float("nan")
    return (float(np.asarray(warped_bone, bool).sum()) * warped_voxvol) / \
           (float(source_bone_n) * source_voxvol)


def gate_case(per_bone: Dict[int, dict], *, min_fit: float, max_jac_bad: float,
              vol_lo: float, vol_hi: float) -> dict:
    """Accept the propagated pelvis only if EVERY present bone seats on bone,
    keeps det J ~ 1, and is not FOV-truncated. Otherwise fall back to the model.
    Returns {accept, reasons[]}."""
    reasons: List[str] = []
    present = [b for b, m in per_bone.items() if m.get("n_warped", 0) > 0
               or m.get("n_source", 0) > 0]
    if not present:
        return {"accept": False, "reasons": ["no_bone_warped"]}
    for b in present:
        m = per_bone[b]
        name = PELVIS.get(b, str(b))
        fit = m.get("bone_fit")
        jb = m.get("jac_bad")
        vr = m.get("vol_ratio")
        if fit is not None and fit == fit and fit < min_fit:
            reasons.append(f"{name}_lowfit={fit:.2f}")
        if jb is not None and jb == jb and jb > max_jac_bad:
            reasons.append(f"{name}_bonewarp={jb:.2f}")
        if vr is not None and vr == vr and not (vol_lo <= vr <= vol_hi):
            reasons.append(f"{name}_volratio={vr:.2f}")
    return {"accept": not reasons, "reasons": reasons}


# ===========================================================================
# SimpleITK registration (guarded import)
# ===========================================================================

def _attach_reg_logging(method, token, tag, every: int):
    """Stream ITK optimizer progress: per-resolution-level changes always, and the
    metric value every `every` iterations (0 = off). Token-prefixed so the lines
    from parallel workers stay attributable."""
    import SimpleITK as sitk
    def _on_level():
        try:
            log.info("  token=%s [%s] -> resolution level %d", token, tag,
                     method.GetCurrentLevel())
        except Exception:                                       # noqa: BLE001
            pass
    method.AddCommand(sitk.sitkMultiResolutionIterationEvent, _on_level)
    if every and every > 0:
        def _on_iter():
            it = method.GetOptimizerIteration()
            if it % every == 0:
                try:
                    log.info("  token=%s [%s] it=%d  metric=%.5f", token, tag, it,
                             method.GetMetricValue())
                except Exception:                               # noqa: BLE001
                    pass
        method.AddCommand(sitk.sitkIterationEvent, _on_iter)


def _aspect_centroid_world(img, labels, aspect: str):
    """World centroid of the inferior ('inf', min superior-Z) or superior ('sup',
    max Z) ~quartile of the labelled voxels. Returns a 3-vector or None."""
    import numpy as np
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(img)                 # z,y,x
    m = np.isin(arr, list(labels))
    if not m.any():
        return None
    kji = np.argwhere(m).astype(float)                # (N, [k,j,i])
    sp = np.asarray(img.GetSpacing(), float)          # (sx,sy,sz) for (i,j,k)
    orig = np.asarray(img.GetOrigin(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    ijk = kji[:, ::-1]                                 # -> (i,j,k)
    world = orig + (D @ (ijk * sp).T).T               # (N,3) physical
    z = world[:, 2]                                    # RAS +Z = superior
    sel = z <= np.percentile(z, 25) if aspect == "inf" else z >= np.percentile(z, 75)
    return world[sel].mean(axis=0)


def landmark_translation(spine_mask_img, sacrum_mask_img):
    """The L5/S1 anchor: a POINT correspondence between the fixed lumbar column's
    INFERIOR aspect (the L5-S1 junction the radiologist drew) and the moving sacrum's
    SUPERIOR aspect (S1 promontory). Returns (translation, center) where translation
    = sac - lum maps the junction onto the sacrum top and `center` (= lum) is the
    pivot to rotate the pelvis about WITHOUT breaking that alignment — so the caller
    can try 180-degree flips (prone<->supine) around the junction. None if missing."""
    lum = _aspect_centroid_world(spine_mask_img, LUMBAR_VERSE, "inf")
    if lum is None:
        lum = _aspect_centroid_world(spine_mask_img, LUMBAR_CANON, "inf")
    sac = _aspect_centroid_world(sacrum_mask_img, (SACRUM,), "sup")
    if lum is None or sac is None:
        return None
    translation = tuple(float(s - l) for s, l in zip(sac, lum))
    return translation, tuple(float(x) for x in lum)


def _endplate(img, labels, aspect):
    """(centroid, normal) of the inferior ('inf') or superior ('sup') endplate of a
    labelled bone, in world mm. The normal is the PCA thin-axis of the endplate slab
    (perpendicular to the disc) ORIENTED to point INFERIOR along the bone, so a fixed
    lumbar-inferior endplate and a moving sacrum-superior endplate get comparably
    oriented axes that can be rotated onto each other. None if too few voxels."""
    import numpy as np
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(img)
    m = np.isin(arr, list(labels))
    if int(m.sum()) < 50:
        return None
    kji = np.argwhere(m).astype(float)
    sp = np.asarray(img.GetSpacing(), float)
    orig = np.asarray(img.GetOrigin(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    world = orig + (D @ (kji[:, ::-1] * sp).T).T              # (N,3) physical
    full_c = world.mean(0)
    z = world[:, 2]
    sel = z <= np.percentile(z, 25) if aspect == "inf" else z >= np.percentile(z, 75)
    slab = world[sel]
    if len(slab) < 20:
        return None
    c = slab.mean(0)
    _, _, vt = np.linalg.svd(slab - c, full_matrices=False)
    n = vt[2]                                                 # endplate normal
    # 'inferior along the bone': inf slab sits below the bulk; sup slab sits above it
    down = (c - full_c) if aspect == "inf" else (full_c - c)
    if float(np.dot(n, down)) < 0:
        n = -n
    return c, n / (np.linalg.norm(n) + 1e-9)


def endplate_alignment(spine_mask_img, sacrum_mask_img):
    """A rigid init that aligns the moving sacrum's SUPERIOR endplate (plane: position
    + orientation) to the fixed last-lumbar INFERIOR endplate — so the sacrum continues
    the lumbar column at the correct TILT, not just the right place. Returns a
    VersorRigid3DTransform or None."""
    import numpy as np
    import SimpleITK as sitk
    f = (_endplate(spine_mask_img, LUMBAR_VERSE, "inf")
         or _endplate(spine_mask_img, LUMBAR_CANON, "inf"))
    m = _endplate(sacrum_mask_img, (SACRUM,), "sup")
    if f is None or m is None:
        return None
    c_f, n_f = f
    c_m, n_m = m
    axis = np.cross(n_f, n_m)
    s = float(np.linalg.norm(axis))
    d = float(np.clip(np.dot(n_f, n_m), -1.0, 1.0))
    e = sitk.VersorRigid3DTransform()
    e.SetCenter(tuple(float(x) for x in c_f))
    if s > 1e-6:                                              # rotate fixed-down -> moving-down
        e.SetRotation(tuple(float(x) for x in axis / s), float(np.arccos(d)))
    e.SetTranslation(tuple(float(x) for x in (c_m - c_f)))    # c_f -> c_m
    return e


def register_and_warp(fixed_ct_path: Path, moving_ct_path: Path,
                      moving_label_img, *, token: str = "", landmark=None,
                      spine_mask=None, rigid_iters: int, dilate_vox: int = 2,
                      per_bone: bool = True, multistart: bool = True,
                      affine: bool = False, log_every: int = 10):
    """Bone-masked RIGID registration, moving CT -> fixed CT; warp the moving label
    (NN) into the fixed grid. Returns (warped_label_sitk, fixed_sitk, moving_sitk);
    the caller does grid-aware QC (source mask and target are on different grids).

    The pelvis is articulated rigid bone: the sacrum + each hemipelvis are rigid,
    but the SI joints let the sacrum rock a few degrees relative to the ilia
    prone<->supine. So (per_bone, default) we first fit ONE whole-pelvis rigid as a
    robust global init, then REFINE a separate rigid per bone {sacrum, L-hip, R-hip}
    masked to that bone, and composite — each bone seats exactly, SI motion absorbed.
    --single_rigid (per_bone=False) keeps the one whole-pelvis transform.

    Every rigid is volume-preserving (det J == 1), so there is no bone-warp gate.
    Registration runs at REG_MM (a 6-DOF fit is determined at ~2-3mm; full 512^3 is
    needless minutes of smoothing/memory); transforms are resolution-independent and
    applied to the FULL-res mask, and bone-HU overlap is scored at full res."""
    import numpy as np
    import SimpleITK as sitk

    fixed = sitk.ReadImage(str(fixed_ct_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_ct_path), sitk.sitkFloat32)

    def _iso(img, interp, default):
        isz, isp = img.GetSize(), img.GetSpacing()
        osz = [max(1, int(round(sz * sp / REG_MM))) for sz, sp in zip(isz, isp)]
        return sitk.Resample(img, osz, sitk.Transform(), interp, img.GetOrigin(),
                             [REG_MM] * 3, img.GetDirection(), float(default),
                             img.GetPixelID())
    fixed_lo = _iso(fixed, sitk.sitkLinear, -1000.0)
    moving_lo = _iso(moving, sitk.sitkLinear, -1000.0)
    # Restrict the metric to BONE on the fixed side: the pelvis sits in a gas-filled
    # abdomen, and bowel gas redistributes completely prone<->supine, so MI on the
    # raw CT locks onto soft-tissue/gas patterns and parks the pelvis off the bone.
    # Sampling only fixed bone (CT>200) forces bone-to-bone alignment.
    fixed_bone_lo = sitk.Cast(sitk.BinaryThreshold(fixed_lo, BONE_HU, 100000, 1, 0),
                              sitk.sitkUInt8)
    # ... but EXCLUDE the lumbar spine from that target. The landmark drops the sacrum
    # at L5, so the moving pelvis overlaps the fixed L-spine, and MI happily climbs the
    # pelvis UP onto the spine (ilia too superior). Subtract the (dilated) spine GT so
    # the pelvis can only register to pelvis/femur bone, never the vertebrae.
    if spine_mask is not None:
        sp_lo = sitk.Resample(spine_mask, fixed_lo, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sp_bin = sitk.BinaryDilate(sitk.BinaryThreshold(sp_lo, 1, 100000, 1, 0),
                                   [3, 3, 3])
        fixed_bone_lo = sitk.And(fixed_bone_lo, sitk.Not(sp_bin))

    def _moving_mask_lo(lo, hi):
        """Downsampled metric mask from the moving label values in [lo, hi]
        (dilated a few voxels) — keeps the optimizer on that structure."""
        m = sitk.BinaryThreshold(moving_label_img, lo, hi, 1, 0)
        if dilate_vox > 0:
            m = sitk.BinaryDilate(m, [int(dilate_vox)] * 3)
        return _iso(sitk.Cast(m, sitk.sitkUInt8), sitk.sitkNearestNeighbor, 0)

    def _run_rigid(mask_lo, init, iters, tag, shrink, smooth):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricFixedMask(fixed_bone_lo)      # sample only fixed BONE
        reg.SetMetricMovingMask(mask_lo)           # ... that maps into the moving pelvis
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2, seed=SEED)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=int(iters))
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrink)
        reg.SetSmoothingSigmasPerLevel(smooth)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init, inPlace=False)
        _attach_reg_logging(reg, token, tag, log_every)
        log.info("  token=%s [%s] starting @%.1fmm (<=%d its)...", token, tag,
                 REG_MM, int(iters))
        tx = reg.Execute(fixed_lo, moving_lo)
        log.info("  token=%s [%s] done: metric=%.5f stop='%s'", token, tag,
                 reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription())
        return tx

    # MULTI-START whole-pelvis rigid. A single init fails bimodally: ~rigid scans
    # carry anatomical DICOM affines so IDENTITY (trust world coords) aligns them,
    # but a big FOV-extent difference between the spine and pelvic scans makes
    # GEOMETRY centering shove the pelvis off by half that difference, beyond the
    # optimizer's reach -> a catastrophic wrong lock. So register from a few inits
    # (identity, geometry, +/- a cranio-caudal slide) and KEEP whichever actually
    # seats the pelvis on bone (highest low-res bone-HU fit of the warped mask).
    mask_whole = _moving_mask_lo(1, 32000)
    fixed_lo_arr = sitk.GetArrayFromImage(fixed_lo)
    fc = fixed_lo.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in fixed_lo.GetSize()])

    def _euler_tz(tz):                       # identity + a world-Z (cranio-caudal) slide
        e = sitk.Euler3DTransform(); e.SetCenter(fc)
        e.SetTranslation((0.0, 0.0, float(tz)))
        return e

    geometry = sitk.CenteredTransformInitializer(
        fixed_lo, moving_lo, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    if multistart:
        cands = [("identity", _euler_tz(0)), ("geometry", geometry)]
        if landmark is not None:
            import math
            T, pivot = landmark
            def _euler_landmark(rx, ry, rz):     # rotate the pelvis ABOUT the junction
                e = sitk.Euler3DTransform(); e.SetCenter(tuple(pivot))
                e.SetRotation(rx, ry, rz); e.SetTranslation(tuple(T))
                return e
            P = math.pi
            # the junction always aligns; the 180-degree flips cover prone<->supine.
            cands += [("L5S1", _euler_landmark(0, 0, 0)),
                      ("L5S1+flipZ", _euler_landmark(0, 0, P)),
                      ("L5S1+flipX", _euler_landmark(P, 0, 0)),
                      ("L5S1+flipY", _euler_landmark(0, P, 0))]
            # endplate alignment: seat the sacrum continuing the lumbar column at the
            # right TILT (orientation), not just translation.
            if spine_mask is not None:
                ep = endplate_alignment(spine_mask, moving_label_img)
                if ep is not None:
                    cands.append(("L5S1plane", ep))
        else:
            # no spine landmark -> fall back to blind cranio-caudal slides.
            cands += [("identity+z", _euler_tz(90)), ("identity-z", _euler_tz(-90))]
    else:
        cands = [("geometry", geometry)]

    def _lo_fit(tx):                         # fraction of warped pelvis on target bone
        wa = sitk.GetArrayFromImage(sitk.Resample(
            moving_label_img, fixed_lo, tx, sitk.sitkNearestNeighbor, 0,
            sitk.sitkInt16)) > 0
        return float((fixed_lo_arr[wa] > BONE_HU).mean()) if wa.any() else -1.0

    best_tx = best_fit = best_name = None
    for name, init in cands:
        # An init with no overlap makes ITK throw "all samples map outside moving
        # image buffer" — skip that candidate, do NOT kill the case: another init
        # (often the L5/S1 landmark) is exactly what recovers these.
        try:
            fit_raw = _lo_fit(init)                  # the init pose, BEFORE optimizing
            tx = _run_rigid(mask_whole, init, rigid_iters, f"rigid:{name}",
                            [4, 2, 1], [4, 2, 0])
            fit_opt = _lo_fit(tx)
        except Exception as exc:                                # noqa: BLE001
            log.warning("  token=%s start=%-10s FAILED (%s) — skipped", token, name,
                        str(exc).splitlines()[-1][:80])
            continue
        # The registration OBJECTIVE (MI) can disagree with bone overlap on these
        # gas/contrast-heavy abdomens and drift AWAY from a correct anatomical init.
        # If the optimizer made the bone seating WORSE, keep the (anatomical) init.
        if fit_raw > fit_opt:
            tx, fit = init, fit_raw
            log.info("  token=%s start=%-10s lo-fit=%.3f  (raw init KEPT; optimizer "
                     "drifted to %.3f)", token, name, fit_raw, fit_opt)
        else:
            fit = fit_opt
            log.info("  token=%s start=%-10s lo-fit=%.3f  (raw init %.3f)", token,
                     name, fit_opt, fit_raw)
        if best_fit is None or fit > best_fit:
            best_tx, best_fit, best_name = tx, fit, name
    if best_tx is None:
        raise RuntimeError("all inits failed to register (no overlap)")
    full = best_tx
    if multistart:
        log.info("  token=%s multi-start winner: %s (lo-fit=%.3f)", token,
                 best_name, best_fit)

    if affine:
        # EXPERIMENT: allow scale + shear on top of the best rigid. If this fixes the
        # failures, the two scans' affines disagree on physical scale; if not, scale
        # is ruled out and the problem is purely pose (gas/soft-tissue local minima).
        ra = sitk.ImageRegistrationMethod()
        ra.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        ra.SetMetricFixedMask(fixed_bone_lo)
        ra.SetMetricMovingMask(mask_whole)
        ra.SetMetricSamplingStrategy(ra.RANDOM)
        ra.SetMetricSamplingPercentage(0.2, seed=SEED)
        ra.SetInterpolator(sitk.sitkLinear)
        ra.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=int(rigid_iters))
        ra.SetOptimizerScalesFromPhysicalShift()
        ra.SetShrinkFactorsPerLevel([2, 1])
        ra.SetSmoothingSigmasPerLevel([1, 0])
        ra.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        ra.SetMovingInitialTransform(full)          # refine on top of the rigid
        aff0 = sitk.AffineTransform(3); aff0.SetCenter(fc)
        ra.SetInitialTransform(aff0, inPlace=False)
        _attach_reg_logging(ra, token, "affine", log_every)
        log.info("  token=%s [affine] refining (scale+shear) from rigid...", token)
        try:
            aff = ra.Execute(fixed_lo, moving_lo)
            cand = sitk.CompositeTransform([full, aff])
            f2 = _lo_fit(cand)
            log.info("  token=%s [affine] done lo-fit=%.3f (rigid was %.3f)",
                     token, f2, best_fit)
            if f2 > best_fit:
                full = cand
        except Exception as exc:                                # noqa: BLE001
            log.warning("  token=%s [affine] failed (%s) — keeping rigid", token,
                        str(exc).splitlines()[-1][:80])

    def _warp(label_img, tx):
        return sitk.Resample(label_img, fixed, tx, sitk.sitkNearestNeighbor,
                             0.0, sitk.sitkInt16)

    if not per_bone:
        return _warp(moving_label_img, full), fixed, moving

    # Per-bone refinement from the global pose: each bone is truly rigid, so a
    # separate rigid seats it exactly and the SI-joint articulation is absorbed.
    fsz = fixed.GetSize()
    comp = np.zeros((fsz[2], fsz[1], fsz[0]), dtype="int16")           # z,y,x
    for b in (SACRUM, LEFT_HIP, RIGHT_HIP):
        mask_b = _moving_mask_lo(b, b)
        if float(sitk.GetArrayFromImage(mask_b).sum()) < 10:
            continue                                                   # bone absent
        tx_b = _run_rigid(mask_b, sitk.Euler3DTransform(full),
                          max(50, int(rigid_iters) // 2),
                          f"bone:{PELVIS[b]}", [2, 1], [1, 0])
        bone_img = sitk.BinaryThreshold(moving_label_img, b, b, b, 0)  # value-b
        wb = sitk.GetArrayFromImage(_warp(bone_img, tx_b))
        comp[wb == b] = b
    warped_img = sitk.GetImageFromArray(comp)
    warped_img.CopyInformation(fixed)
    return warped_img, fixed, moving


# ===========================================================================
# Per-patient orchestration
# ===========================================================================

def _to_canonical_pelvis(lab):
    """Remap raw CTPelvic1K {1,2,3} -> canonical {7,8,9}; pass through if already
    canonical (values already in {7,8,9})."""
    import numpy as np
    lab = np.asarray(lab)
    if set(np.unique(lab)) <= ({0} | set(PELVIS)):
        return lab.astype("int16")
    out = np.zeros_like(lab, dtype="int16")
    for src, dst in PELVIC_TO_CANONICAL.items():
        out[lab == src] = dst
    return out


def process_patient(case: dict, *, nifti_dir: Path, pelvic_dir: Path,
                    out_dir: Path, reg_kw: dict, gate_kw: dict,
                    spine_dir: Path = None) -> Optional[dict]:
    import numpy as np

    tok = str(case.get("patient_token", "?"))
    pv = case.get("pelvic", {}) or {}
    sp = case.get("spine", {}) or {}
    pelvic_uid = pv.get("series_uid")
    spine_uid = sp.get("series_uid")
    # The native placement quality is read straight from placed_manifest.json —
    # the SAME bone-HU overlap value place_fused_masks._compute_bone_pct_from_files
    # wrote for this pelvis (not a recomputation), so src vs prop is apples-to-apples.
    src_bone_pct = pv.get("bone_pct")
    if src_bone_pct is None:
        src_bone_pct = pv.get("pelvic_bone_pct", case.get("pelvic_bone_pct"))

    # target grid = the patient's NATIVE spine-side scan (placed-space), so the
    # propagated pelvis is just another placed mask export merges normally.
    fixed_ct = nifti_dir / f"{spine_uid}.nii.gz" if spine_uid else None
    moving_ct = nifti_dir / f"{pelvic_uid}.nii.gz" if pelvic_uid else None
    placed = pv.get("placed")
    pelvic_mask = Path(placed) if placed else None
    if pelvic_mask and not pelvic_mask.exists():
        cand = pelvic_dir / Path(placed).name
        pelvic_mask = cand if cand.exists() else None

    miss = [n for n, p in (("fixed_ct", fixed_ct), ("moving_ct", moving_ct),
                           ("pelvic_mask", pelvic_mask))
            if p is None or not Path(p).exists()]
    if miss:
        return {"token": tok, "spine_uid": spine_uid or "",
                "status": "missing:" + ",".join(miss), "accept": 0}

    import SimpleITK as sitk
    # one sitk thread per worker process: avoids oversubscription under the
    # process pool AND removes the last source of run-to-run nondeterminism.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    # Guard against a degenerate/scout series being selected for either side: a
    # scout/topogram is a thin projection (a tiny dimension) that cannot overlap a
    # full CT. Read headers only (cheap) and skip with a NAMED status so a wrong
    # series surfaces instead of a cryptic "samples map outside" registration error.
    def _geom(path):
        r = sitk.ImageFileReader(); r.SetFileName(str(path)); r.ReadImageInformation()
        size, spc = r.GetSize(), r.GetSpacing()
        return size, spc, [s * p for s, p in zip(size, spc)]
    f_size, _f_sp, f_ext = _geom(fixed_ct)
    m_size, _m_sp, m_ext = _geom(moving_ct)
    log.info("token=%s  fixed(spine) uid=%s size=%s extent=%dx%dx%dmm  "
             "moving(pelvic) uid=%s size=%s extent=%dx%dx%dmm", tok, spine_uid,
             f_size, *[int(x) for x in f_ext], pelvic_uid, m_size,
             *[int(x) for x in m_ext])
    SCOUT_MIN_SLICES, SCOUT_MIN_EXT_MM = 16, 40.0
    scout = []
    if min(f_size) < SCOUT_MIN_SLICES or min(f_ext) < SCOUT_MIN_EXT_MM:
        scout.append(f"fixed/spine({spine_uid})")
    if min(m_size) < SCOUT_MIN_SLICES or min(m_ext) < SCOUT_MIN_EXT_MM:
        scout.append(f"moving/pelvic({pelvic_uid})")
    if scout:
        return {"token": tok, "spine_uid": spine_uid or "",
                "pelvic_uid": pelvic_uid or "", "accept": 0,
                "status": "scout_or_degenerate:" + ",".join(scout)}

    lab_img = sitk.ReadImage(str(pelvic_mask), sitk.sitkInt16)
    src_arr = _to_canonical_pelvis(sitk.GetArrayFromImage(lab_img))   # z,y,x
    canon_img = sitk.GetImageFromArray(src_arr)
    canon_img.CopyInformation(lab_img)

    # The fixed scan's spine GT drives the L5/S1 landmark init + the endplate-tilt
    # init, AND is subtracted from the bone target so the pelvis can't climb the
    # spine. Resolve + load the placed spine mask once.
    landmark = None
    spine_img = None
    spine_placed = sp.get("placed")
    spine_mask_p = Path(spine_placed) if spine_placed else None
    if spine_mask_p and not spine_mask_p.exists() and spine_dir and spine_uid:
        cand = spine_dir / f"{spine_uid}_seg_placed.nii.gz"
        spine_mask_p = cand if cand.exists() else None
    if spine_mask_p and spine_mask_p.exists():
        try:
            spine_img = sitk.ReadImage(str(spine_mask_p), sitk.sitkInt16)
            landmark = landmark_translation(spine_img, canon_img)
        except Exception as exc:                                # noqa: BLE001
            log.warning("token=%s spine mask failed (%s) — using blind inits", tok, exc)
    if landmark is not None:
        log.info("token=%s L5/S1 landmark translation = (%.0f, %.0f, %.0f) mm",
                 tok, *landmark[0])

    warped_img, fixed_img, moving_img = register_and_warp(
        fixed_ct, moving_ct, canon_img, token=tok, landmark=landmark,
        spine_mask=spine_img, **reg_kw)
    warped = sitk.GetArrayFromImage(warped_img).astype("int16")    # fixed grid
    # QC only needs the HU threshold (CT > 200), and HU fits in int16 — half the
    # memory of float32 per worker (matters at high worker counts).
    fixed_np = sitk.GetArrayFromImage(fixed_img).astype("int16")
    moving_np = sitk.GetArrayFromImage(moving_img).astype("int16")
    # rigid is exactly volume-preserving -> det J == 1 everywhere (no bone-warp).
    jac = np.ones(warped.shape, dtype="float32")

    # The placed mask and the raw pelvic CT are on DIFFERENT grids, so resample the
    # source mask into the moving-CT grid (physical-space) for grid-matched QC.
    src_on_moving = sitk.Resample(canon_img, moving_img, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0.0, sitk.sitkInt16)
    src_moving = sitk.GetArrayFromImage(src_on_moving).astype("int16")
    fixed_vox = float(np.prod(fixed_img.GetSpacing()))     # mm^3 / voxel
    moving_vox = float(np.prod(moving_img.GetSpacing()))

    per_bone: Dict[int, dict] = {}
    for b in PELVIS:
        wb = warped == b                                    # fixed grid
        n_src = int((src_moving == b).sum())                # moving grid
        per_bone[b] = {
            "n_source": n_src,
            "n_warped": int(wb.sum()),
            "bone_fit": bone_fit_fraction(fixed_np, wb),
            "vol_ratio": volume_ratio(wb, n_src, warped_voxvol=fixed_vox,
                                      source_voxvol=moving_vox),
        }
        med, bad, _ = jacobian_stats_in_mask(jac, wb, tol=gate_kw["jac_tol"])
        per_bone[b]["jac_med"] = med
        per_bone[b]["jac_bad"] = bad

    decision = gate_case(
        per_bone, min_fit=gate_kw["min_fit"], max_jac_bad=gate_kw["max_jac_bad"],
        vol_lo=gate_kw["vol_lo"], vol_hi=gate_kw["vol_hi"])

    # Bone-HU overlap of the WHOLE pelvis on BOTH scans, recomputed with the SAME
    # bone_pct() (CT > 200) so src vs prop is rigorously apples-to-apples — this is
    # what proves registration did not degrade placement quality. The placed_manifest
    # value place_fused wrote is kept only as a consistency cross-check.
    src_pct = bone_pct(moving_np, src_moving > 0)         # native pelvic-scan overlap
    prop_pct = bone_pct(fixed_np, warped > 0)             # propagated spine-scan overlap
    prop_bone_pct = prop_pct
    bone_pct_drop = (src_pct - prop_pct) if (src_pct == src_pct
                    and prop_pct == prop_pct) else float("nan")
    try:
        manifest_pct = float(src_bone_pct) if src_bone_pct is not None else float("nan")
    except (TypeError, ValueError):
        manifest_pct = float("nan")

    # Acceptance keys on the RELATIVE drop, not an absolute bone-HU floor: filled
    # (marrow-inclusive) masks are only ~45-55% >200HU even natively, so the same
    # threshold on both sides cancels and the drop is calibration-free. Reject to the
    # model only on a genuine failure (drop > fail_drop); a small drop is still REAL
    # GT and beats a model guess. drop_target (1pp) is the reported ideal, not a gate.
    reasons = list(decision["reasons"])
    reg_failed = (bone_pct_drop == bone_pct_drop
                  and bone_pct_drop > gate_kw["fail_drop"])
    if reg_failed:
        reasons.append(f"reg_failed_drop={bone_pct_drop:.1f}pp")
    accept = int(bool(decision["accept"]) and not reg_failed)

    # save in the native spine-CT grid (sitk carries the fixed geometry exactly).
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spine_uid}_pelvic_propagated.nii.gz"
    sitk.WriteImage(warped_img, str(out_path))

    row = {"token": tok, "spine_uid": spine_uid or "",
           "pelvic_uid": pelvic_uid or "", "status": "ok",
           "out_file": str(out_path), "accept": accept,
           "reasons": ";".join(reasons),
           "src_bone_pct": _r(src_pct, 1), "prop_bone_pct": _r(prop_bone_pct, 1),
           "bone_pct_drop": _r(bone_pct_drop, 1),
           "native_bone_pct_manifest": _r(manifest_pct, 1)}
    for b, name in PELVIS.items():
        m = per_bone[b]
        row[f"{name}_fit"] = _r(m["bone_fit"])
        row[f"{name}_bonepct"] = _r(m["bone_fit"] * 100 if m["bone_fit"] ==
                                    m["bone_fit"] else float("nan"), 1)
        row[f"{name}_jacbad"] = _r(m["jac_bad"])
        row[f"{name}_jacmed"] = _r(m["jac_med"])
        row[f"{name}_volratio"] = _r(m["vol_ratio"])
    return row


def _r(v, n=4):
    return round(v, n) if isinstance(v, float) and v == v else ""


# ===========================================================================
# Main
# ===========================================================================

PROPAGATED_MANIFEST_SCHEMA = "propagated_pelvis/1"

# place_fused_masks case fields that are pure patient/acquisition pass-through.
_CARRY_FIELDS = ("position", "age", "sex", "patient_weight", "patient_size",
                 "convolution_kernel", "manufacturer", "manufacturer_model",
                 "slice_thickness", "kvp", "lstv_pelvic", "lstv_vertebral",
                 "lstv_agreement", "lstv_confusion_zone", "lstv_class")


def _build_manifest_case(case: dict, row: dict) -> dict:
    """A placed_manifest.json-style case for the propagated pelvis, so export and
    downstream analysis consume it exactly like a place_fused_masks placed mask.

    The propagated pelvis now lives on the SPINE series, so spine and pelvic share
    one series_uid (a fused-like case). prov_pelvis='manual_propagated' marks it as
    REAL radiologist GT moved by deterministic registration — never a model guess.
    The native vs propagated bone-HU overlap is carried for downstream analysis."""
    sp = dict(case.get("spine", {}) or {})
    pv_src = case.get("pelvic", {}) or {}
    spine_uid = row.get("spine_uid", "")
    out = {
        "patient_token": row.get("token"),
        "match_type": "propagated",
        "prov_spine": "manual",
        "prov_pelvis": "manual_propagated",
        "spine": sp,
        # propagated pelvis, on the spine scan -> a placed mask in the same shape
        # place_fused writes (series_uid + placed + bone_pct + position).
        "pelvic": {
            "series_uid": spine_uid,
            "placed": row.get("out_file", ""),
            "bone_pct": row.get("prop_bone_pct", ""),
            "position": sp.get("position"),
            "source_series_uid": (row.get("pelvic_uid", "")
                                  or pv_src.get("series_uid", "")),  # where GT lived
            "bone_pct_before": row.get("src_bone_pct", ""),
            "bone_pct_after": row.get("prop_bone_pct", ""),
            "bone_pct_drop": row.get("bone_pct_drop", ""),
            "native_bone_pct_manifest": row.get("native_bone_pct_manifest", ""),
        },
        # the registration QC, persisted for downstream analysis / visualization.
        "propagation": {
            "accept": int(row.get("accept", 0) or 0),
            "reasons": row.get("reasons", ""),
            "seed": SEED,
            "per_bone": {name: {
                "bone_pct": row.get(f"{name}_bonepct", ""),
                "jac_med": row.get(f"{name}_jacmed", ""),
                "jac_bad": row.get(f"{name}_jacbad", ""),
                "vol_ratio": row.get(f"{name}_volratio", ""),
            } for name in PELVIS.values()},
        },
    }
    for f in _CARRY_FIELDS:
        if f in case:
            out[f] = case[f]
    return out


def _worker(task: dict) -> dict:
    """Picklable entry point for the process pool: unpack and run one patient."""
    tok = str(task["case"].get("patient_token", "?"))
    try:
        return process_patient(
            task["case"], nifti_dir=task["nifti_dir"], pelvic_dir=task["pelvic_dir"],
            out_dir=task["out_dir"], reg_kw=task["reg_kw"], gate_kw=task["gate_kw"],
            spine_dir=task.get("spine_dir"))
    except Exception as exc:                                     # noqa: BLE001
        return {"token": tok, "status": f"fail:{exc}", "accept": 0}


# test = fast end-to-end smoke (few cases, low iters); production = full quality.
MODE_PRESETS = {
    "test":       dict(limit=5, rigid_iters=200),
    # multi-start runs ~4 registrations/case; the optimizer hits its gradient
    # tolerance well before 500, so 300 is plenty and ~40% cheaper.
    "production": dict(limit=0, rigid_iters=300),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--nifti_dir", required=True, type=Path)
    ap.add_argument("--pelvic_dir", required=True, type=Path)
    ap.add_argument("--spine_dir", type=Path, default=None,
                    help="placed spine GT masks (<spine_uid>_seg_placed.nii.gz), used "
                         "for the L5/S1 landmark init. Optional; absent -> blind inits.")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--mode", choices=("test", "production"), default="production",
                    help="test = fast smoke on a few cases; production = full run.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2),
                    help="parallel registration processes (each sitk single-thread).")
    # explicit overrides (default None -> take the mode preset)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rigid_iters", type=int, default=None,
                    help="cap the rigid optimizer iterations (per level).")
    ap.add_argument("--dilate_vox", type=int, default=2,
                    help="dilate the moving pelvic metric mask by this many voxels "
                         "(small margin; radius 5 on a 512^3 volume is ~10x slower "
                         "for no benefit).")
    ap.add_argument("--per_bone", dest="per_bone", action="store_true", default=False,
                    help="(opt-in) ALSO refine a separate rigid per bone {sacrum, "
                         "L-hip, R-hip} from the whole-pelvis init, to absorb the "
                         "few-degree SI-joint articulation. Default is ONE whole-"
                         "pelvis rigid, which seats the masks cleanly on the test "
                         "cases; enable only if a case looks misplaced.")
    ap.add_argument("--no_multistart", dest="multistart", action="store_false",
                    default=True,
                    help="disable multi-start init (identity / geometry / +-z slide). "
                         "Multi-start fixes the bimodal catastrophic failures where a "
                         "single GEOMETRY init locks onto the wrong place; off = one "
                         "GEOMETRY init (faster, but ~28%% of cases mis-register).")
    ap.add_argument("--affine", action="store_true",
                    help="EXPERIMENT: after the best rigid, allow scale+shear (affine) "
                         "refinement. Tests whether the scans' affines disagree on "
                         "physical scale; kept only if it improves bone-fit.")
    ap.add_argument("--reg_log_every", type=int, default=10,
                    help="log the ITK optimizer metric every N iterations "
                         "(0 = only level changes + stage start/done).")
    # gate thresholds
    ap.add_argument("--min_fit", type=float, default=0.0,
                    help="OFF by default (0.0). An ABSOLUTE bone-HU floor is wrong "
                         "for CTPelvic1K masks: they are FILLED (marrow-inclusive), "
                         "so only ~45-55%% of voxels are >200 HU even natively. The "
                         "calibration-free signal is the drop vs native (--fail_drop).")
    ap.add_argument("--max_jac_bad", type=float, default=0.10,
                    help="reject if >this fraction of bone voxels have |detJ-1|>tol "
                         "(rigid -> always 0; kept for the optional deformable path).")
    ap.add_argument("--jac_tol", type=float, default=0.30)
    ap.add_argument("--vol_lo", type=float, default=0.70)
    ap.add_argument("--vol_hi", type=float, default=1.30)
    ap.add_argument("--drop_target", type=float, default=1.0,
                    help="REPORT-ONLY reference: the run reports what %% of placed "
                         "pelves stay within this many bone-HU overlap pp of the "
                         "native placement (default 1.0 — the <=1%% ideal).")
    ap.add_argument("--fail_drop", type=float, default=8.0,
                    help="THE acceptance gate: fall back to the model only when the "
                         "bone-HU overlap drops >this many pp vs the native placement "
                         "(a genuine registration failure). Lenient on purpose — a "
                         "slightly-degraded REAL pelvis still beats a model guess.")
    ap.add_argument("--tokens", default="",
                    help="run ONLY these patient tokens (debug); separate by comma, "
                         "colon, or space (e.g. --tokens 74:54:154 or --tokens 74).")
    ap.add_argument("--resume", action="store_true",
                    help="idempotent: skip cases already ACCEPTED in a prior "
                         "propagate_qc.csv (whose mask still exists) and carry their "
                         "result forward; re-register only the rejected/missing ones "
                         "(e.g. to recover failures after enabling multi-start).")
    args = ap.parse_args()

    preset = MODE_PRESETS[args.mode]
    limit = args.limit if args.limit is not None else preset["limit"]
    rigid_iters = (args.rigid_iters if args.rigid_iters is not None
                   else preset["rigid_iters"])
    log.info("mode=%s  limit=%s  rigid_iters=%d  dilate_vox=%d  per_bone=%s  "
             "multistart=%s  seed=%d (deterministic, %s rigid @%.1fmm)", args.mode,
             limit or "all", rigid_iters, args.dilate_vox, args.per_bone,
             args.multistart, SEED,
             "per-bone" if args.per_bone else "whole-pelvis", REG_MM)

    data = json.loads(args.manifest.read_text())
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if isinstance(cases, dict):
        cases = list(cases.values())
    separate = [c for c in cases
                if (c.get("match_type") or c.get("config")) == "separate"]
    want = {t for t in re.split(r"[,:;\s]+", args.tokens.strip()) if t}
    if want:
        separate = [c for c in separate
                    if str(c.get("patient_token", "?")) in want]
        log.info("--tokens: restricted to %d case(s): %s", len(separate),
                 sorted(want))
    if limit:
        separate = separate[:limit]
    log.info("separate-cohort patients to propagate: %d", len(separate))
    if not separate:
        log.warning("no 'separate' cases in %s — nothing to propagate.",
                    args.manifest)
        return 0

    reg_kw = dict(rigid_iters=rigid_iters, dilate_vox=args.dilate_vox,
                  per_bone=args.per_bone, multistart=args.multistart,
                  affine=args.affine, log_every=args.reg_log_every)
    gate_kw = dict(min_fit=args.min_fit, max_jac_bad=args.max_jac_bad,
                   jac_tol=args.jac_tol, vol_lo=args.vol_lo, vol_hi=args.vol_hi,
                   fail_drop=args.fail_drop)

    case_by_token = {str(c.get("patient_token", "?")): c for c in separate}
    out_csv = args.out_csv or (args.out_dir / "propagate_qc.csv")

    # Resume: carry forward cases already ACCEPTED in a prior run (mask present) and
    # re-register only the rest. Read the prior CSV BEFORE it is overwritten below.
    done_rows: Dict[str, dict] = {}
    if args.resume and out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            tok = str(r.get("token", ""))
            if r.get("status") != "ok" or str(r.get("accept")) != "1" or not tok:
                continue
            of = r.get("out_file", "")
            mask = (Path(of) if of else
                    args.out_dir / f"{r.get('spine_uid', '')}_pelvic_propagated.nii.gz")
            if mask.exists():
                done_rows[tok] = r
        before = len(separate)
        separate = [c for c in separate
                    if str(c.get("patient_token", "?")) not in done_rows]
        log.info("resume: %d already accepted (skipped) | %d to (re)register",
                 len(done_rows), len(separate))

    tasks = [dict(case=c, nifti_dir=args.nifti_dir, pelvic_dir=args.pelvic_dir,
                  out_dir=args.out_dir, reg_kw=reg_kw, gate_kw=gate_kw,
                  spine_dir=args.spine_dir)
             for c in separate]
    workers = max(1, min(args.workers, len(tasks)))
    log.info("registering %d pelves across %d workers ...", len(tasks), workers)

    rows: List[dict] = []
    n_acc = 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, t): str(t["case"].get("patient_token", "?"))
                for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            tok = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:                            # noqa: BLE001
                row = {"token": tok, "status": f"fail:{exc}", "accept": 0}
            rows.append(row)
            n_acc += int(row.get("accept") == 1)
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(tasks) - i) / rate if rate > 0 else 0.0
            log.info("[%d/%d] token=%s  %s  accept=%s  drop=%s pp  | %d ok-acc  "
                     "%.2f/s  elapsed %dm%02ds  ETA %dm%02ds  %s",
                     i, len(tasks), tok, row.get("status"), row.get("accept"),
                     row.get("bone_pct_drop", "?"), n_acc, rate,
                     int(elapsed) // 60, int(elapsed) % 60,
                     int(eta) // 60, int(eta) % 60, row.get("reasons", ""))

    # carry forward the resumed (already-accepted) cases so the CSV/manifest/summary
    # describe the FULL cohort, not just this run's re-registered subset.
    if done_rows:
        rows = list(done_rows.values()) + rows
        log.info("resume: carried forward %d previously-accepted cases", len(done_rows))

    cols = ["token", "spine_uid", "status", "out_file", "accept", "reasons",
            "src_bone_pct", "prop_bone_pct", "bone_pct_drop",
            "native_bone_pct_manifest"]
    for name in PELVIS.values():
        cols += [f"{name}_fit", f"{name}_bonepct", f"{name}_jacbad",
                 f"{name}_jacmed", f"{name}_volratio"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    # placed_manifest.json-style manifest so export + downstream analysis consume
    # the propagated pelvis exactly like a place_fused_masks placed mask.
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    manifest_cases = [_build_manifest_case(case_by_token.get(r["token"], {}), r)
                      for r in ok_rows if r.get("token") in case_by_token]
    manifest_out = args.out_dir / "placed_manifest_propagated.json"
    manifest_out.write_text(json.dumps({
        "schema_version": PROPAGATED_MANIFEST_SCHEMA,
        "mode": args.mode, "seed": SEED, "drop_target": args.drop_target,
        "fail_drop": args.fail_drop,
        "n_cases": len(manifest_cases),
        "n_accepted": sum(c["propagation"]["accept"] for c in manifest_cases),
        "n_fallback": sum(1 - c["propagation"]["accept"] for c in manifest_cases),
        "cases": manifest_cases,
    }, indent=2, default=str))

    ok = [r for r in rows if r.get("status") == "ok"]
    acc = [r for r in ok if r.get("accept") == 1]
    log.info("=" * 72)
    log.info("PROPAGATION: %d processed | %d accepted (real-GT pelvis) | "
             "%d -> model fallback", len(ok), len(acc), len(ok) - len(acc))

    def _nums(records, key):
        return sorted(float(r[key]) for r in records
                      if isinstance(r.get(key), (int, float)))

    def _mean(v):
        return sum(v) / len(v) if v else float("nan")

    def _med(v):
        return v[len(v) // 2] if v else float("nan")

    # ---- RESULTS: bone-HU overlap BEFORE (native) vs AFTER (propagated) -------
    for label, recs in (("ALL placed", ok), ("ACCEPTED (-> dataset)", acc)):
        before = _nums(recs, "src_bone_pct")
        after = _nums(recs, "prop_bone_pct")
        drops = _nums(recs, "bone_pct_drop")
        if not before:
            continue
        log.info("-" * 72)
        log.info("RESULTS — bone-HU overlap, registration-placed pelves [%s, n=%d]",
                 label, len(before))
        log.info("  BEFORE (native placement)  : mean %5.1f  median %5.1f %%",
                 _mean(before), _med(before))
        log.info("  AFTER  (propagated)        : mean %5.1f  median %5.1f %%",
                 _mean(after), _med(after))
        if drops:
            within = sum(1 for d in drops if d <= args.drop_target)
            log.info("  degradation (before-after) : median %+.2f  max %+.2f pp"
                     "   |  %d/%d (%.0f%%) within %.1f pp",
                     _med(drops), drops[-1], within, len(drops),
                     100 * within / len(drops), args.drop_target)
    # per-bone AFTER overlap (sacrum / left_hip / right_hip), accepted set
    for name in PELVIS.values():
        pb = _nums(acc, f"{name}_bonepct")
        if pb:
            log.info("  AFTER per-bone %-10s: mean %5.1f  median %5.1f %%",
                     name, _mean(pb), _med(pb))
    log.info("=" * 72)
    log.info("Deterministic (seed=%d), pelvis-masked RIGID carry-over of the "
             "patient's OWN radiologist pelvis onto the spine scan — the highest-"
             "fidelity pelvis the data allows, no model guess. Cases drop to the "
             "model only on a genuine registration failure (bone-HU overlap drop "
             ">%.1f pp vs native, or FOV-truncation). wrote -> %s  (+%s)",
             SEED, args.fail_drop, out_csv, manifest_out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
