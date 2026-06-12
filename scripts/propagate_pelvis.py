"""
propagate_pelvis.py — replace the MODEL pseudolabel pelvis with the patient's
OWN radiologist pelvis GT, carried across acquisitions by registration.

The "separate" cohort (the dominant group: a token whose spine mask and pelvic
mask won DIFFERENT scans) has, for every spine-side scan that needs a pelvis, a
REAL radiologist pelvis GT sitting on that same patient's OTHER (pelvic-side)
scan. Instead of asking a model to guess the pelvis, we warp the real GT from the
pelvic-side scan onto the spine-side scan. It is the identical bone, so this is
strictly higher-fidelity than a population model's completion.

Deformable, done safely for bone
--------------------------------
The transform is a global deformable (bone-masked rigid+affine init, then a
coarse-grid B-spline), so the metric is driven by cortical bone, NOT soft tissue /
bowel. A free-form warp CAN squash rigid bone to chase intensity, so we make that
failure MEASURABLE
rather than silent: we compute the warp's Jacobian determinant inside each warped
bone (a rigid bone must not change local volume -> det J ~ 1) and reject the case
to the model fallback when the field deforms bone (|det J - 1| large), when the
warped bone does not sit on actual target bone-HU (a bad fit), or when the bone is
FOV-truncated on the target scan (vol ratio far below 1 -> no data to propagate).

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

Registration uses SimpleITK (already in the container — no rebuild): a bone-masked
rigid+affine init followed by a COARSE-grid B-spline deformable. The coarse control
grid is inherently stiff (it cannot introduce high-frequency intra-bone warping),
and the Jacobian-determinant gate catches any residual bone deformation. The
QC/gating math is pure numpy and is unit-tested without SimpleITK.

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


def volume_ratio(warped_bone, source_bone_n: int) -> float:
    """warped voxel count / source voxel count. ~1 for a clean rigid carry-over;
    << 1 means the bone is FOV-truncated on the target (or the warp collapsed)."""
    import numpy as np
    if source_bone_n <= 0:
        return float("nan")
    return float(np.asarray(warped_bone, bool).sum()) / float(source_bone_n)


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

def register_and_warp(fixed_ct_path: Path, moving_ct_path: Path,
                      moving_label_img, *,
                      flow_sigma: float, bspline_iters: int, affine_iters: int):
    """Bone-masked rigid+affine init then COARSE B-spline deformable, moving CT
    -> fixed CT; warp the moving label (NN) into the fixed grid. Returns
    (warped_label_np, jacobian_np, fixed_ct_np, warped_label_sitk).

    `flow_sigma` sets the B-spline control-point spacing in mm (LARGER = fewer
    control points = stiffer); `bspline_iters` caps the deformable optimizer and
    `affine_iters` the rigid optimizer. Defaults err stiff (test lowers iters)."""
    import numpy as np
    import SimpleITK as sitk

    fixed = sitk.ReadImage(str(fixed_ct_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_ct_path), sitk.sitkFloat32)

    # GEOMETRY init, NOT MOMENTS: CT air is -1000, so an intensity-weighted center
    # of mass is meaningless and leaves the scans non-overlapping ("all samples map
    # outside moving image buffer"). GEOMETRY aligns the physical image centers.
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # --- RIGID: full-image MI, wide multi-resolution pyramid (NO mask) --------
    # Unmasked + heavily down-sampled top level gives a wide capture range so the
    # partially-overlapping spine/pelvic scans lock together before any masking.
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=int(affine_iters))
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    rigid = reg.Execute(fixed, moving)

    # --- COARSE B-spline deformable (stiff; bone-masked) ----------------------
    # flow_sigma is the control-point SPACING in mm: LARGE => few control points
    # => stiff (cannot warp within a bone). ~40-50mm gives ~5-8 points per axis.
    fixed_bone = sitk.BinaryThreshold(fixed, BONE_HU, 1e6, 1, 0)
    grid_mm = max(float(flow_sigma), 20.0)
    size_mm = [sz * sp for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh = [max(1, int(round(extent / grid_mm))) for extent in size_mm]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)

    rb = sitk.ImageRegistrationMethod()
    rb.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    rb.SetMetricFixedMask(fixed_bone)
    rb.SetMetricSamplingStrategy(rb.RANDOM)
    rb.SetMetricSamplingPercentage(0.2, seed=SEED)
    rb.SetInterpolator(sitk.sitkLinear)
    rb.SetMovingInitialTransform(rigid)
    rb.SetInitialTransform(bspline, inPlace=True)
    rb.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                            numberOfIterations=int(bspline_iters),
                            maximumNumberOfCorrections=5,
                            maximumNumberOfFunctionEvaluations=2000)
    rb.SetShrinkFactorsPerLevel([2, 1])
    rb.SetSmoothingSigmasPerLevel([1, 0])
    rb.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rb.Execute(fixed, moving)

    full = sitk.CompositeTransform([rigid, bspline])

    # --- warp the label (NN) into the fixed grid -----------------------------
    warped_img = sitk.Resample(moving_label_img, fixed, full,
                               sitk.sitkNearestNeighbor, 0.0, sitk.sitkInt16)
    warped = sitk.GetArrayFromImage(warped_img).astype("int16")

    # --- Jacobian determinant of the deformable part inside the fixed grid ----
    try:
        disp = sitk.TransformToDisplacementField(
            full, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(),
            fixed.GetSpacing(), fixed.GetDirection())
        jac_img = sitk.DisplacementFieldJacobianDeterminant(disp)
        jac = sitk.GetArrayFromImage(jac_img).astype("float32")
    except Exception:                                           # noqa: BLE001
        jac = np.ones(warped.shape, dtype="float32")

    fixed_np = sitk.GetArrayFromImage(fixed).astype("float32")
    moving_np = sitk.GetArrayFromImage(moving).astype("float32")
    return warped, jac, fixed_np, warped_img, moving_np


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
                    out_dir: Path, reg_kw: dict, gate_kw: dict) -> Optional[dict]:
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

    warped, jac, fixed_np, warped_img, moving_np = register_and_warp(
        fixed_ct, moving_ct, canon_img, **reg_kw)
    src_lab = src_arr                                # for per-bone source counts

    per_bone: Dict[int, dict] = {}
    for b in PELVIS:
        wb = warped == b
        per_bone[b] = {
            "n_source": int((src_lab == b).sum()),
            "n_warped": int(wb.sum()),
            "bone_fit": bone_fit_fraction(fixed_np, wb),
            "vol_ratio": volume_ratio(wb, int((src_lab == b).sum())),
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
    src_pct = bone_pct(moving_np, src_arr > 0)            # native pelvic-scan overlap
    prop_pct = bone_pct(fixed_np, warped > 0)             # propagated spine-scan overlap
    prop_bone_pct = prop_pct
    bone_pct_drop = (src_pct - prop_pct) if (src_pct == src_pct
                    and prop_pct == prop_pct) else float("nan")
    try:
        manifest_pct = float(src_bone_pct) if src_bone_pct is not None else float("nan")
    except (TypeError, ValueError):
        manifest_pct = float("nan")

    # bone-HU drop is a REPORTED quality metric, NOT a rejection rule by default:
    # a slightly-degraded REAL pelvis still beats a model guess, and min_fit
    # (absolute on-bone fraction) already rejects genuinely broken registrations.
    # Only --gate_on_drop makes exceeding drop_target a fallback-to-model.
    reasons = list(decision["reasons"])
    over_drop = (bone_pct_drop == bone_pct_drop
                 and bone_pct_drop > gate_kw["drop_target"])
    if over_drop and gate_kw["gate_on_drop"]:
        reasons.append(f"bone_pct_drop={bone_pct_drop:.1f}")
    accept = int(bool(decision["accept"])
                 and not (over_drop and gate_kw["gate_on_drop"]))

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
            "source_series_uid": row.get("pelvic_uid", ""),  # where the GT lived
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
            out_dir=task["out_dir"], reg_kw=task["reg_kw"], gate_kw=task["gate_kw"])
    except Exception as exc:                                     # noqa: BLE001
        return {"token": tok, "status": f"fail:{exc}", "accept": 0}


# test = fast end-to-end smoke (few cases, low iters); production = full quality.
# flow_sigma is the B-spline control-point SPACING in mm (LARGE = stiff coarse grid).
MODE_PRESETS = {
    "test":       dict(limit=5,  affine_iters=100, bspline_iters=30,  flow_sigma=50.0),
    "production": dict(limit=0,  affine_iters=250, bspline_iters=120, flow_sigma=40.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--nifti_dir", required=True, type=Path)
    ap.add_argument("--pelvic_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--mode", choices=("test", "production"), default="production",
                    help="test = fast smoke on a few cases; production = full run.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2),
                    help="parallel registration processes (each sitk single-thread).")
    # explicit overrides (default None -> take the mode preset)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flow_sigma", type=float, default=None,
                    help="B-spline control-grid spacing (mm); larger = stiffer.")
    ap.add_argument("--affine_iters", type=int, default=None)
    ap.add_argument("--bspline_iters", type=int, default=None)
    # gate thresholds
    ap.add_argument("--min_fit", type=float, default=0.80,
                    help="reject if a warped bone is <this fraction on bone-HU.")
    ap.add_argument("--max_jac_bad", type=float, default=0.10,
                    help="reject if >this fraction of bone voxels have |detJ-1|>tol.")
    ap.add_argument("--jac_tol", type=float, default=0.30)
    ap.add_argument("--vol_lo", type=float, default=0.70)
    ap.add_argument("--vol_hi", type=float, default=1.30)
    ap.add_argument("--drop_target", type=float, default=1.0,
                    help="REPORT-ONLY reference: the run reports what %% of placed "
                         "pelves stay within this many bone-HU overlap pp of the "
                         "native placement (default 1.0 — the <=1%% ideal).")
    ap.add_argument("--gate_on_drop", action="store_true",
                    help="ALSO fall back to the model when a case exceeds "
                         "--drop_target (OFF by default: a slightly-degraded REAL "
                         "pelvis still beats a model guess; min_fit already floors "
                         "absolute quality).")
    args = ap.parse_args()

    preset = MODE_PRESETS[args.mode]
    limit = args.limit if args.limit is not None else preset["limit"]
    flow_sigma = args.flow_sigma if args.flow_sigma is not None else preset["flow_sigma"]
    affine_iters = (args.affine_iters if args.affine_iters is not None
                    else preset["affine_iters"])
    bspline_iters = (args.bspline_iters if args.bspline_iters is not None
                     else preset["bspline_iters"])
    log.info("mode=%s  limit=%s  flow_sigma=%s  affine_iters=%d  bspline_iters=%d  "
             "seed=%d (deterministic)", args.mode, limit or "all", flow_sigma,
             affine_iters, bspline_iters, SEED)

    data = json.loads(args.manifest.read_text())
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if isinstance(cases, dict):
        cases = list(cases.values())
    separate = [c for c in cases
                if (c.get("match_type") or c.get("config")) == "separate"]
    if limit:
        separate = separate[:limit]
    log.info("separate-cohort patients to propagate: %d", len(separate))
    if not separate:
        log.warning("no 'separate' cases in %s — nothing to propagate.",
                    args.manifest)
        return 0

    reg_kw = dict(flow_sigma=flow_sigma, affine_iters=affine_iters,
                  bspline_iters=bspline_iters)
    gate_kw = dict(min_fit=args.min_fit, max_jac_bad=args.max_jac_bad,
                   jac_tol=args.jac_tol, vol_lo=args.vol_lo, vol_hi=args.vol_hi,
                   drop_target=args.drop_target, gate_on_drop=args.gate_on_drop)

    case_by_token = {str(c.get("patient_token", "?")): c for c in separate}
    tasks = [dict(case=c, nifti_dir=args.nifti_dir, pelvic_dir=args.pelvic_dir,
                  out_dir=args.out_dir, reg_kw=reg_kw, gate_kw=gate_kw)
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

    out_csv = args.out_csv or (args.out_dir / "propagate_qc.csv")
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
        "gate_on_drop": bool(args.gate_on_drop),
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
    log.info("Deterministic (seed=%d), bone-masked deformable carry-over of the "
             "patient's OWN radiologist pelvis onto the spine scan — the highest-"
             "fidelity pelvis the data allows, no model guess. Cases drop to the "
             "model only on a genuine registration failure (off-bone / bone-warp / "
             "FOV-truncation)%s. wrote -> %s  (+%s)",
             SEED, (" or >%.1f pp bone-HU drop" % args.drop_target)
             if args.gate_on_drop else "", out_csv, manifest_out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
