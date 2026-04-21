"""
orientation_fix.py — Second-pass pelvis-AP orientation consistency check (v3).

Runs AFTER place_fused_masks.py. Reads placed_manifest.json, detects AP-
inverted scans via body-anchored geometry, emits parallel _orientation_fixed
NIfTIs next to the originals, and writes a new manifest where every case
explicitly lists the CT + mask paths that should be loaded.

KEY CONTRACT
============
The output manifest (placed_manifest_orientation_fixed.json) is fully self-
describing. For every case, both sides carry absolute paths:

  spine.ct_nifti    → CT to load for the spine panel
  spine.placed      → spine mask to load
  pelvic.ct_nifti   → CT to load for the pelvic panel
  pelvic.placed     → pelvic mask to load

For inverted cases these paths resolve to the *_orientation_fixed files;
for non-inverted cases they resolve to the originals. No external routing,
no staging dirs, no UID-lookup logic downstream. visualize_qc.py becomes
one-line: load what the manifest says.

DETECTION (body-center)
=======================
A correctly-oriented pelvis has its bone mass sitting in the POSTERIOR half
of the body. Body cross-section is roughly elliptical → bounding-box center
is a stable mid-AP reference. Pelvic bone mass (sacrum + iliac crest +
ischium) is unambiguously posterior.

  delta_posterior_mm = body_center_Y − pelvic_centroid_Y     (world RAS+)

  delta > +threshold  →  pelvic mass posterior of body middle  (CORRECT)
  delta < −threshold  →  pelvic mass anterior of body middle   (INVERTED)
  |delta| ≤ threshold →  ambiguous — don't flip

FLIP
====
Full 3D resampling that reflects each volume across the CT's world-Y
bounding-box center. Using a COMMON external mirror plane (instead of
np.flip's per-volume self-center) preserves mask-on-CT alignment even
when CT and placed masks were produced by different code paths with
different Y translations in their affines.

FLIP SET BY MATCH TYPE
======================
  fused        → CT (shared), spine mask, pelvic mask
  pelvic_only  → CT, pelvic mask
  separate     → pelvic CT, pelvic mask  (spine CT is a different series;
                 pelvic-based detector can't judge it → untouched)
  spine_only   → skipped (no pelvic anatomy)

USAGE
=====
    python scripts/orientation_fix.py \\
        --manifest   data/placed/placed_manifest.json \\
        --nifti_dir  data/tcia_nifti \\
        --placed_dir data/placed \\
        --workers    16 \\
        [--threshold_mm 10.0] [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orientation_fix")

PELV_SACRUM    = 1
PELV_LEFT_HIP  = 2
PELV_RIGHT_HIP = 3

SUFFIX            = "_orientation_fixed"
BODY_HU_THRESHOLD = -500.0
DEFAULT_THRESHOLD = 10.0   # mm


# ── Affine sanity ────────────────────────────────────────────────────────────

def _valid_affine(aff) -> bool:
    if aff is None:
        return False
    a = np.asarray(aff, dtype=np.float64)
    if a.shape != (4, 4):
        return False
    if not np.all(np.isfinite(a)):
        return False
    col_norms = np.linalg.norm(a[:3, :3], axis=0)
    if (col_norms < 1e-6).any():
        return False
    if abs(float(np.linalg.det(a[:3, :3]))) < 1e-9:
        return False
    return True


# ── Detection ────────────────────────────────────────────────────────────────

def _detect_ap_inversion_body_center(
    placed_pelvic_path: Path,
    ct_path: Path,
    threshold_mm: float,
) -> Tuple[Optional[bool], float, float, str]:
    """
    Returns (is_inverted, delta_posterior_mm, delta_sacrum_hip_mm, status_str).
    """
    import nibabel as nib

    try:
        pelv_img  = nib.load(str(placed_pelvic_path))
        pelv_data = np.asarray(pelv_img.dataobj)
        pelv_aff  = pelv_img.affine
    except Exception as e:
        return None, 0.0, 0.0, f"pelvic_load_failed:{type(e).__name__}"

    if not _valid_affine(pelv_aff):
        return None, 0.0, 0.0, "pelvic_affine_invalid"

    pelv_vox = np.argwhere(pelv_data > 0)
    if len(pelv_vox) < 1000:
        return None, 0.0, 0.0, f"insufficient_pelvic({len(pelv_vox)})"

    pelv_cent_vox   = pelv_vox.mean(axis=0)
    pelv_cent_world = pelv_aff @ np.append(pelv_cent_vox, 1.0)
    pelv_y = float(pelv_cent_world[1])

    # Secondary signal (logged for debugging borderline cases)
    sacrum_vox = np.argwhere(pelv_data == PELV_SACRUM)
    hip_vox    = np.argwhere(
        (pelv_data == PELV_LEFT_HIP) | (pelv_data == PELV_RIGHT_HIP)
    )
    if len(sacrum_vox) >= 500 and len(hip_vox) >= 1000:
        axis0_mm_per_vox = float(np.linalg.norm(pelv_aff[:3, 0]))
        delta_sacrum_hip_mm = (
            (hip_vox[:, 0].mean() - sacrum_vox[:, 0].mean()) * axis0_mm_per_vox
        )
    else:
        delta_sacrum_hip_mm = 0.0

    if not ct_path.exists():
        return None, 0.0, delta_sacrum_hip_mm, f"ct_missing:{ct_path.name}"
    try:
        ct_img = nib.load(str(ct_path))
        ct_aff = ct_img.affine
    except Exception as e:
        return None, 0.0, delta_sacrum_hip_mm, f"ct_load_failed:{type(e).__name__}"

    if not _valid_affine(ct_aff):
        return None, 0.0, delta_sacrum_hip_mm, "ct_affine_invalid"

    try:
        ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)
    except Exception as e:
        return None, 0.0, delta_sacrum_hip_mm, f"ct_read_failed:{type(e).__name__}"

    body_mask = ct_data > BODY_HU_THRESHOLD
    if not body_mask.any():
        return None, 0.0, delta_sacrum_hip_mm, "no_body_found"

    body_vox = np.argwhere(body_mask)
    if len(body_vox) < 1000:
        return None, 0.0, delta_sacrum_hip_mm, f"insufficient_body({len(body_vox)})"

    body_bbox_center_vox = (body_vox.min(axis=0) + body_vox.max(axis=0)) / 2.0
    body_center_world    = ct_aff @ np.append(body_bbox_center_vox, 1.0)
    body_y = float(body_center_world[1])

    delta_posterior_mm = body_y - pelv_y

    if delta_posterior_mm > threshold_mm:
        return False, delta_posterior_mm, delta_sacrum_hip_mm, "ok"
    if delta_posterior_mm < -threshold_mm:
        return True,  delta_posterior_mm, delta_sacrum_hip_mm, "inverted"
    return False, delta_posterior_mm, delta_sacrum_hip_mm, "indeterminate"


# ── Flip ─────────────────────────────────────────────────────────────────────

def _bbox_y_extent(img) -> Tuple[float, float]:
    """Return (y_min, y_max) world-Y extent of the volume's voxel bounding box."""
    shape = img.shape[:3]
    aff = img.affine.astype(np.float64)
    corners = np.array([
        [0, 0, 0], [shape[0]-1, 0, 0], [0, shape[1]-1, 0], [0, 0, shape[2]-1],
        [shape[0]-1, shape[1]-1, 0], [shape[0]-1, 0, shape[2]-1],
        [0, shape[1]-1, shape[2]-1],
        [shape[0]-1, shape[1]-1, shape[2]-1],
    ], dtype=np.float64)
    h = np.hstack([corners, np.ones((8, 1))])
    world_y = (aff @ h.T).T[:, 1]
    return float(world_y.min()), float(world_y.max())


def _flip_world_y_resample(
    src: Path,
    dst: Path,
    mirror_plane_y: float,
    is_label_map: bool,
    label: str = "",
) -> bool:
    """
    Reflect the volume across the world plane  Y = mirror_plane_y  via full
    3D resampling. Output shape & affine == input; only the data content
    is reshuffled to reflect the world-space mirror.

    Why resampling, not np.flip
    ---------------------------
    np.flip(data, axis=k) mirrors the anatomy about the VOLUME'S OWN
    voxel-center along axis k, which in world space is
        Y_self_center = aff[1, k] * (N_k - 1) / 2  +  aff[1, 3]
    i.e., it depends on the volume's shape AND its Y translation.

    For a fused case, the TCIA CT (LPS, dcm2niix-produced) and the placed
    masks (PIR, placement-produced) are aligned in world space but have
    DIFFERENT Y translations in their affines — so their self-centers
    differ, and independent np.flip calls displace CT and masks relative
    to each other by (aff_ct[1,3] - aff_mask[1,3]) in world Y.

    Reflecting both about an EXTERNAL common plane (here: the CT's bbox-Y
    center) applies the identical world-space transformation to every
    file, preserving relative alignment. That's "doing the same thing to
    both" in the way that actually matters for mask-on-CT geometry.
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    try:
        img = nib.load(str(src))
        data = np.asarray(img.dataobj)
        aff = img.affine.astype(np.float64)
        inv_aff = np.linalg.inv(aff)

        shape = data.shape
        if len(shape) != 3:
            log.error("  flip FAILED %s: expected 3D, got shape %s", src.name, shape)
            return False

        # Process slice-by-slice along axis 2 to keep peak memory bounded
        # (per-slice coord scratch = ~3 * shape[0] * shape[1] * 4 bytes).
        order = 0 if is_label_map else 1
        cval  = 0 if is_label_map else -1024.0
        src_data = data if is_label_map else data.astype(np.float32)

        output = np.empty_like(data)

        for k in range(shape[2]):
            ii, jj = np.meshgrid(
                np.arange(shape[0], dtype=np.float32),
                np.arange(shape[1], dtype=np.float32),
                indexing="ij",
            )
            # Output voxel (i, j, k) → output world (x, y, z)
            wx = aff[0, 0]*ii + aff[0, 1]*jj + aff[0, 2]*k + aff[0, 3]
            wy = aff[1, 0]*ii + aff[1, 1]*jj + aff[1, 2]*k + aff[1, 3]
            wz = aff[2, 0]*ii + aff[2, 1]*jj + aff[2, 2]*k + aff[2, 3]

            # Mirror the Y coordinate about the external common plane
            wy_src = (2.0 * mirror_plane_y) - wy

            # Mirrored world → source voxel in the SAME file's voxel grid
            si = inv_aff[0, 0]*wx + inv_aff[0, 1]*wy_src + inv_aff[0, 2]*wz + inv_aff[0, 3]
            sj = inv_aff[1, 0]*wx + inv_aff[1, 1]*wy_src + inv_aff[1, 2]*wz + inv_aff[1, 3]
            sk = inv_aff[2, 0]*wx + inv_aff[2, 1]*wy_src + inv_aff[2, 2]*wz + inv_aff[2, 3]

            coords = np.stack([si, sj, sk], axis=0)
            slice_out = map_coordinates(
                src_data, coords, order=order, mode="constant", cval=cval
            )
            output[..., k] = slice_out.astype(data.dtype)

        new_img = nib.Nifti1Image(output, aff, header=img.header)
        new_img.set_data_dtype(img.get_data_dtype())

        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(dst))

        y_min, y_max = _bbox_y_extent(img)
        log.info(
            "  flipped %s: mirror_y=%.2f  shape=%s  own_y_range=[%.1f, %.1f]  own_y_center=%.2f  is_label=%s",
            label or src.name, mirror_plane_y, tuple(shape),
            y_min, y_max, (y_min + y_max) / 2, is_label_map,
        )
        return True
    except Exception as e:
        log.error("  flip FAILED %s → %s: %s", src.name, dst.name, e)
        return False


def _verify_bone_alignment(ct_path: Path, mask_path: Path) -> Optional[dict]:
    """
    Post-flip sanity: sample the CT at every pelvic-mask voxel's world
    position and compute bone% (HU > 200) + mean HU. If the flip worked,
    bone% should come out comparable to the pre-flip value from the
    manifest (~70% for a well-placed pelvic mask). bone% under ~30% means
    mask and CT are not aligned in world space.
    """
    try:
        import nibabel as nib
        from scipy.ndimage import map_coordinates

        ct   = nib.load(str(ct_path))
        mask = nib.load(str(mask_path))
        mask_data = np.asarray(mask.dataobj)

        vox = np.argwhere(mask_data > 0)
        if len(vox) < 1000:
            return None

        h      = np.hstack([vox.astype(np.float64), np.ones((len(vox), 1))])
        world  = (mask.affine.astype(np.float64) @ h.T).T[:, :3]
        inv_ct = np.linalg.inv(ct.affine.astype(np.float64))
        hw     = np.hstack([world, np.ones((len(world), 1))])
        ct_vox = (inv_ct @ hw.T).T[:, :3]

        ct_data = np.asarray(ct.dataobj, dtype=np.float32)
        sampled = map_coordinates(
            ct_data, ct_vox.T, order=0, mode="constant", cval=-1000.0
        )

        return {
            "bone_pct":   float((sampled > 200).sum()) / len(sampled) * 100.0,
            "mean_hu":    float(sampled.mean()),
            "n_mask_vox": int(len(vox)),
        }
    except Exception as e:
        log.error("  verify_bone_alignment failed: %s", e)
        return None


def _with_suffix(p: Path, suffix: str) -> Path:
    """foo.nii.gz → foo{suffix}.nii.gz"""
    name = p.name
    if name.endswith(".nii.gz"):
        return p.with_name(name[:-7] + suffix + ".nii.gz")
    if name.endswith(".nii"):
        return p.with_name(name[:-4] + suffix + ".nii")
    return p.with_name(p.stem + suffix + p.suffix)


# ── Per-case worker ──────────────────────────────────────────────────────────

def _process_case(args) -> dict:
    case, threshold_mm, nifti_dir_str, dry_run = args
    nifti_dir = Path(nifti_dir_str)

    token      = str(case.get("patient_token", "?"))
    match_type = case.get("match_type", "unknown")

    out: dict = {
        "patient_token":        token,
        "match_type":           match_type,
        "orientation_check":    {
            "status":              None,
            "delta_posterior_mm": None,
            "delta_sacrum_hip_mm": None,
            "threshold_mm":       threshold_mm,
            "detector":           "body_center",
        },
        "orientation_fixed":    None,
        "orientation_original": None,
        "error":                None,
    }

    pv            = case.get("pelvic") or {}
    pelvic_placed = pv.get("placed")
    pelvic_uid    = pv.get("series_uid")

    if not pelvic_placed or not Path(pelvic_placed).exists():
        out["orientation_check"]["status"] = "skipped_no_pelvic"
        return out

    if not pelvic_uid:
        out["orientation_check"]["status"] = "skipped_no_uid"
        return out

    ct_path = nifti_dir / f"{pelvic_uid}.nii.gz"
    is_inv, delta_post, delta_sh, status = _detect_ap_inversion_body_center(
        Path(pelvic_placed), ct_path, threshold_mm
    )

    out["orientation_check"]["status"]              = status
    out["orientation_check"]["delta_posterior_mm"]  = (
        round(delta_post, 1) if is_inv is not None else None
    )
    out["orientation_check"]["delta_sacrum_hip_mm"] = (
        round(delta_sh, 1)   if is_inv is not None else None
    )

    if is_inv is None or not is_inv:
        return out

    # ── Flagged as inverted — flip files about CT's world-Y center ──────────
    # Key: all three files mirror about the SAME external world-Y plane
    # so their relative alignment is preserved after the flip.
    import nibabel as nib
    try:
        ct_img = nib.load(str(ct_path))
        ct_y_min, ct_y_max = _bbox_y_extent(ct_img)
        mirror_plane_y = (ct_y_min + ct_y_max) / 2.0
    except Exception as e:
        out["error"] = f"mirror_plane_compute_failed:{type(e).__name__}"
        return out

    log.info(
        "  [token=%s] common mirror plane (world Y): %.2f  "
        "(from CT y_range=[%.1f, %.1f])",
        token, mirror_plane_y, ct_y_min, ct_y_max,
    )

    flipped: Dict[str, str] = {}
    originals: Dict[str, Optional[str]] = {
        "ct":             str(ct_path),
        "ct_uid":         pelvic_uid,
        "pelvic_placed":  str(pelvic_placed),
        "spine_placed":   None,
    }

    ct_src = ct_path
    if ct_src.exists():
        ct_dst = _with_suffix(ct_src, SUFFIX)
        ok = dry_run or _flip_world_y_resample(
            ct_src, ct_dst, mirror_plane_y,
            is_label_map=False, label=f"CT[token={token}]",
        )
        if dry_run or ok:
            flipped["ct"] = str(ct_dst)

    pm_src = Path(pelvic_placed)
    pm_dst = _with_suffix(pm_src, SUFFIX)
    ok = dry_run or _flip_world_y_resample(
        pm_src, pm_dst, mirror_plane_y,
        is_label_map=True, label=f"pelvic_mask[token={token}]",
    )
    if dry_run or ok:
        flipped["pelvic_placed"] = str(pm_dst)

    sp = case.get("spine") or {}
    if match_type == "fused" and sp.get("placed"):
        sp_src = Path(sp["placed"])
        if sp_src.exists():
            sp_dst = _with_suffix(sp_src, SUFFIX)
            originals["spine_placed"] = str(sp_src)
            ok = dry_run or _flip_world_y_resample(
                sp_src, sp_dst, mirror_plane_y,
                is_label_map=True, label=f"spine_mask[token={token}]",
            )
            if dry_run or ok:
                flipped["spine_placed"] = str(sp_dst)

    # Post-flip sanity check: mask-on-flipped-CT bone%. If this comes out
    # comparable to the pre-flip manifest bone_pct, alignment is preserved.
    if not dry_run and "ct" in flipped and "pelvic_placed" in flipped:
        chk = _verify_bone_alignment(Path(flipped["ct"]), Path(flipped["pelvic_placed"]))
        if chk is not None:
            log.info(
                "  [token=%s] post-flip alignment check: bone%%=%.1f  mean_HU=%.1f  "
                "(pre-flip manifest pelvic bone_pct=%.1f)",
                token, chk["bone_pct"], chk["mean_hu"],
                (case.get("pelvic") or {}).get("bone_pct", -1),
            )
            out["orientation_check"]["post_flip_bone_pct"] = round(chk["bone_pct"], 1)
            out["orientation_check"]["post_flip_mean_hu"]  = round(chk["mean_hu"],  1)

    out["orientation_fixed"]    = flipped if flipped else None
    out["orientation_original"] = originals if flipped else None
    return out


# ── Manifest merge ───────────────────────────────────────────────────────────

def _merge_case(case: dict,
                ori_by_token: Dict[str, dict],
                nifti_dir: Path) -> dict:
    """
    Build the output case entry. Key invariants for every case:

      1. spine.ct_nifti / pelvic.ct_nifti carry explicit absolute paths
         (flipped for inverted cases, original otherwise).
      2. spine.series_uid / pelvic.series_uid are PATCHED for inverted
         cases to the _orientation_fixed stem, so any downstream consumer
         that resolves CT via `{nifti_dir}/{series_uid}.nii.gz` — e.g.
         visualize_qc.py — lands on the flipped CT without code changes.
         Original UIDs are preserved in series_uid_original.
      3. spine.placed / pelvic.placed point at the flipped mask files for
         inverted cases; originals kept in placed_original.
    """
    merged = dict(case)
    tok    = str(case.get("patient_token", "?"))
    match_type = merged.get("match_type", "unknown")

    ori = ori_by_token.get(tok) or {}
    ofx = ori.get("orientation_fixed") or {}
    flipped_ct_path = ofx.get("ct")

    # Derive the flipped UID from the filename. The flipped CT lives in
    # nifti_dir with a _orientation_fixed-suffixed filename, so its "UID"
    # (the filename stem) is just the original UID + SUFFIX.
    flipped_ct_uid: Optional[str] = None
    if flipped_ct_path:
        stem = Path(flipped_ct_path).name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        flipped_ct_uid = stem

    if ori:
        merged["orientation_check"] = ori["orientation_check"]
    if ori.get("orientation_fixed"):
        merged["orientation_fixed"]    = ori["orientation_fixed"]
        merged["orientation_original"] = ori["orientation_original"]

    # Which side(s) flip their CT?
    spine_ct_flips  = (match_type == "fused")
    pelvic_ct_flips = (match_type in ("fused", "separate", "pelvic_only"))

    def _resolve_ct(uid: Optional[str], flips: bool) -> Optional[str]:
        if not uid:
            return None
        if flipped_ct_path and flips:
            return flipped_ct_path
        return str((nifti_dir / f"{uid}.nii.gz").resolve())

    sp = merged.get("spine")
    if isinstance(sp, dict):
        new_sp = dict(sp)
        orig_spine_uid = sp.get("series_uid")

        # Patch series_uid → flipped stem so UID-based CT resolution hits
        # the flipped file. Keep the original UID for traceability.
        if spine_ct_flips and flipped_ct_uid:
            new_sp["series_uid_original"] = orig_spine_uid
            new_sp["series_uid"]          = flipped_ct_uid

        new_sp["ct_nifti"] = _resolve_ct(new_sp["series_uid"], spine_ct_flips)
        if spine_ct_flips and flipped_ct_path:
            new_sp["ct_nifti_original"] = str(
                (nifti_dir / f"{orig_spine_uid}.nii.gz").resolve()
            )
        if ofx.get("spine_placed"):
            new_sp["placed"]          = ofx["spine_placed"]
            new_sp["placed_original"] = (
                (ori.get("orientation_original") or {}).get("spine_placed")
            )
        merged["spine"] = new_sp

    pv = merged.get("pelvic")
    if isinstance(pv, dict):
        new_pv = dict(pv)
        orig_pelvic_uid = pv.get("series_uid")

        if pelvic_ct_flips and flipped_ct_uid:
            new_pv["series_uid_original"] = orig_pelvic_uid
            new_pv["series_uid"]          = flipped_ct_uid

        new_pv["ct_nifti"] = _resolve_ct(new_pv["series_uid"], pelvic_ct_flips)
        if pelvic_ct_flips and flipped_ct_path:
            new_pv["ct_nifti_original"] = str(
                (nifti_dir / f"{orig_pelvic_uid}.nii.gz").resolve()
            )
        if ofx.get("pelvic_placed"):
            new_pv["placed"]          = ofx["pelvic_placed"]
            new_pv["placed_original"] = (
                (ori.get("orientation_original") or {}).get("pelvic_placed")
            )
        merged["pelvic"] = new_pv

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest",     required=True, type=Path)
    ap.add_argument("--nifti_dir",    required=True, type=Path)
    ap.add_argument("--placed_dir",   required=True, type=Path)
    ap.add_argument("--threshold_mm", default=DEFAULT_THRESHOLD, type=float)
    ap.add_argument("--workers",      default=16, type=int)
    ap.add_argument("--dry_run",      action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest); return

    manifest = json.loads(args.manifest.read_text())
    cases    = manifest.get("cases", [])
    log.info("Loaded %d cases from %s", len(cases), args.manifest)
    log.info("Detector: body_center  |  threshold: |delta_posterior_mm| > %.1f mm",
             args.threshold_mm)
    if args.dry_run:
        log.warning("DRY RUN — no files will be written")

    work = [(c, args.threshold_mm, str(args.nifti_dir), args.dry_run)
            for c in cases]

    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_case, w) for w in work]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:
                log.error("Worker crash: %s", e)
            if i % 50 == 0 or i == len(work):
                log.info("  processed %d / %d", i, len(work))

    # ── Summary ─────────────────────────────────────────────────────────────
    by_status: Dict[str, int] = {}
    deltas_post: List[float]  = []
    deltas_sh:   List[float]  = []
    for r in results:
        s = r["orientation_check"]["status"] or "unknown"
        by_status[s] = by_status.get(s, 0) + 1
        dp = r["orientation_check"]["delta_posterior_mm"]
        ds = r["orientation_check"]["delta_sacrum_hip_mm"]
        if dp is not None: deltas_post.append(dp)
        if ds is not None: deltas_sh.append(ds)

    log.info("=" * 68)
    log.info("Detection summary:")
    for s in sorted(by_status):
        log.info("  %-30s : %d", s, by_status[s])
    if deltas_post:
        arr = np.asarray(deltas_post)
        log.info("  delta_posterior_mm (PRIMARY)  n=%d  min=%+6.1f  p10=%+6.1f  "
                 "p50=%+6.1f  p90=%+6.1f  max=%+6.1f",
                 len(arr), arr.min(),
                 float(np.percentile(arr, 10)),
                 float(np.percentile(arr, 50)),
                 float(np.percentile(arr, 90)),
                 arr.max())
    if deltas_sh:
        arr = np.asarray(deltas_sh)
        log.info("  delta_sacrum_hip_mm (info)    n=%d  min=%+6.1f  p10=%+6.1f  "
                 "p50=%+6.1f  p90=%+6.1f  max=%+6.1f",
                 len(arr), arr.min(),
                 float(np.percentile(arr, 10)),
                 float(np.percentile(arr, 50)),
                 float(np.percentile(arr, 90)),
                 arr.max())
    log.info("=" * 68)

    inverted = [r for r in results
                if r["orientation_check"]["status"] == "inverted"]
    if inverted:
        log.info("Inverted cases (most negative delta_posterior_mm first):")
        inverted.sort(
            key=lambda r: (r["orientation_check"]["delta_posterior_mm"] or 0)
        )
        for r in inverted:
            n_files = len(r["orientation_fixed"] or {})
            log.info("  token=%-12s  delta_post=%+7.1f  delta_sh=%+7.1f  "
                     "match=%-10s  files=%d",
                     r["patient_token"],
                     r["orientation_check"]["delta_posterior_mm"],
                     r["orientation_check"]["delta_sacrum_hip_mm"],
                     r["match_type"],
                     n_files)

    # ── Merge and write manifest ────────────────────────────────────────────
    ori_by_token = {r["patient_token"]: r for r in results}
    new_cases = [_merge_case(c, ori_by_token, args.nifti_dir) for c in cases]

    out_path = args.placed_dir / "placed_manifest_orientation_fixed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "n_cases":            manifest.get("n_cases", len(new_cases)),
        "n_fused":            manifest.get("n_fused"),
        "n_separate":         manifest.get("n_separate"),
        "n_spine_only":       manifest.get("n_spine_only"),
        "n_pelvic_only":      manifest.get("n_pelvic_only"),
        "n_ap_inverted":      len(inverted),
        "n_ap_ok":            by_status.get("ok", 0),
        "n_ap_indeterminate": by_status.get("indeterminate", 0),
        "n_ap_skipped":       (
            by_status.get("skipped_no_pelvic", 0)
            + by_status.get("skipped_no_uid", 0)
        ),
        "threshold_mm":       args.threshold_mm,
        "detector":           "body_center",
        "dry_run":            args.dry_run,
        "schema_version":     "v3_explicit_ct_paths",
        "cases":              new_cases,
    }
    out_path.write_text(json.dumps(out_doc, indent=2, default=str))
    log.info("Manifest → %s", out_path)
    log.info("Schema: every case carries explicit spine.ct_nifti and "
             "pelvic.ct_nifti. visualize_qc.py should read these paths "
             "directly; no UID-based resolution.")


if __name__ == "__main__":
    main()
