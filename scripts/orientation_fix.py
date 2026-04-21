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
np.flip(data, axis=0) with the affine UNCHANGED. Preserves the PIR
orientation claim while making the data match the anatomy.

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

def _find_ap_axis(aff: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Return (voxel_axis_index, y_components) for the voxel axis most aligned
    with world Y (the anterior-posterior axis in nibabel's RAS+ convention).

    Each column of aff[:3, :3] is the world-space direction vector of the
    corresponding voxel axis. Row 1 (world Y component) tells us which
    voxel axis carries the AP direction.

    For a PIR volume (placed masks) this returns 0.
    For an LPS volume (typical TCIA CT from dcm2niix) this returns 1.
    For an LAS volume this also returns 1 (sign differs but abs-max is 1).
    """
    y_components = np.abs(aff[:3, :3][1, :].astype(np.float64))
    ap_axis = int(np.argmax(y_components))
    return ap_axis, y_components


def _flip_along_world_y(src: Path, dst: Path, label: str = "") -> Optional[int]:
    """
    Flip the volume along the voxel axis most aligned with world Y (AP).
    Affine is preserved — flipping data (not the affine) is what makes the
    PIR/LPS claim become anatomically accurate for an inverted scan.

    Returns the voxel axis that was flipped, or None on failure.

    Previously this function was hard-coded to axis=0, which worked for PIR
    masks but silently mirrored LPS CTs left-right instead of flipping them
    anterior-posterior. The affine-driven axis choice is the fix.
    """
    import nibabel as nib
    try:
        img  = nib.load(str(src))
        data = np.asarray(img.dataobj)
        aff  = img.affine

        ap_axis, y_components = _find_ap_axis(aff)

        # Sanity: warn if no voxel axis is clearly AP-aligned (oblique volume)
        total = float(y_components.sum()) + 1e-9
        dominance = float(y_components[ap_axis]) / total
        if dominance < 0.7:
            log.warning(
                "  flip %s: AP axis unclear (dominance=%.2f, y_components=%s)"
                " — using axis %d anyway",
                label or src.name, dominance,
                [f"{v:.2f}" for v in y_components.tolist()],
                ap_axis,
            )

        flipped = np.ascontiguousarray(np.flip(data, axis=ap_axis))
        new_img = nib.Nifti1Image(flipped, aff, header=img.header)
        new_img.set_data_dtype(img.get_data_dtype())

        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(dst))

        log.info(
            "  flipped %s: ap_axis=%d  shape=%s  y_components=%s",
            label or src.name, ap_axis, tuple(data.shape),
            [f"{v:.2f}" for v in y_components.tolist()],
        )
        return ap_axis
    except Exception as e:
        log.error("  flip FAILED %s → %s: %s", src.name, dst.name, e)
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

    # ── Flagged as inverted — flip files ─────────────────────────────────────
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
        axis = None if dry_run else _flip_along_world_y(
            ct_src, ct_dst, label=f"CT[token={token}]"
        )
        if dry_run or axis is not None:
            flipped["ct"] = str(ct_dst)
            flipped["ct_ap_axis"] = axis if not dry_run else None

    pm_src = Path(pelvic_placed)
    pm_dst = _with_suffix(pm_src, SUFFIX)
    axis = None if dry_run else _flip_along_world_y(
        pm_src, pm_dst, label=f"pelvic_mask[token={token}]"
    )
    if dry_run or axis is not None:
        flipped["pelvic_placed"] = str(pm_dst)
        flipped["pelvic_ap_axis"] = axis if not dry_run else None

    sp = case.get("spine") or {}
    if match_type == "fused" and sp.get("placed"):
        sp_src = Path(sp["placed"])
        if sp_src.exists():
            sp_dst = _with_suffix(sp_src, SUFFIX)
            originals["spine_placed"] = str(sp_src)
            axis = None if dry_run else _flip_along_world_y(
                sp_src, sp_dst, label=f"spine_mask[token={token}]"
            )
            if dry_run or axis is not None:
                flipped["spine_placed"] = str(sp_dst)
                flipped["spine_ap_axis"] = axis if not dry_run else None

    out["orientation_fixed"]    = flipped if flipped else None
    out["orientation_original"] = originals if flipped else None
    return out


# ── Manifest merge ───────────────────────────────────────────────────────────

def _merge_case(case: dict,
                ori_by_token: Dict[str, dict],
                nifti_dir: Path) -> dict:
    """
    Build the output case entry. Key invariant: every case has explicit
    spine.ct_nifti and pelvic.ct_nifti pointing at the file to load — no
    UID-based resolution needed downstream.
    """
    merged = dict(case)
    tok    = str(case.get("patient_token", "?"))
    match_type = merged.get("match_type", "unknown")

    ori = ori_by_token.get(tok) or {}
    ofx = ori.get("orientation_fixed") or {}
    flipped_ct_path = ofx.get("ct")

    if ori:
        merged["orientation_check"] = ori["orientation_check"]
    if ori.get("orientation_fixed"):
        merged["orientation_fixed"]    = ori["orientation_fixed"]
        merged["orientation_original"] = ori["orientation_original"]

    # Which side(s) flip their CT?
    #   fused        → spine and pelvic share the CT, both sides flip
    #   separate     → only pelvic CT flips; spine CT is a different series
    #   pelvic_only  → pelvic CT flips; no spine side
    #   spine_only   → nothing flips
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
        new_sp["ct_nifti"] = _resolve_ct(sp.get("series_uid"), spine_ct_flips)
        if spine_ct_flips and flipped_ct_path:
            new_sp["ct_nifti_original"] = str(
                (nifti_dir / f"{sp.get('series_uid')}.nii.gz").resolve()
            )
        # Also patch the placed mask path if it was flipped
        if ofx.get("spine_placed"):
            new_sp["placed"]          = ofx["spine_placed"]
            new_sp["placed_original"] = (
                (ori.get("orientation_original") or {}).get("spine_placed")
            )
        merged["spine"] = new_sp

    pv = merged.get("pelvic")
    if isinstance(pv, dict):
        new_pv = dict(pv)
        new_pv["ct_nifti"] = _resolve_ct(pv.get("series_uid"), pelvic_ct_flips)
        if pelvic_ct_flips and flipped_ct_path:
            new_pv["ct_nifti_original"] = str(
                (nifti_dir / f"{pv.get('series_uid')}.nii.gz").resolve()
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
