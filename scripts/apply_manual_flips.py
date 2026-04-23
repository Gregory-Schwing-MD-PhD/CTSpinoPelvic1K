"""
apply_manual_flips.py — Apply reviewer-curated AP flips.

Reads:
  --manifest   data/placed/placed_manifest.json   (source of truth from Step B)
  --flip_list  configs/flip_list.json             (reviewer-curated token list)

For each token listed in flip_list.json, reflects CT + spine + pelvic NIfTI
across the CT's world-Y bounding-box center using full 3D resampling. All
three files are mirrored about the SAME external world plane so their
relative mask-on-CT alignment is preserved after the flip.

No heuristic detection. Tokens not in flip_list.json are left untouched.

AFFECTED FILES BY MATCH TYPE
============================
  fused        → CT (shared), spine mask, pelvic mask
  pelvic_only  → CT, pelvic mask
  separate     → pelvic CT, pelvic mask  (spine CT is a different series;
                 its flip status cannot be inferred from the pelvic token,
                 so it is left alone unless a second review pass lists it
                 separately)
  spine_only   → CT, spine mask

Writes:
  data/placed/placed_manifest_orientation_fixed.json

  For every case (flipped or not), the output manifest carries explicit
  spine.ct_nifti and pelvic.ct_nifti paths. For flipped cases:
    - series_uid is patched to the _orientation_fixed stem so any
      UID-based CT resolution lands on the flipped file
    - placed paths point at the flipped mask files
    - series_uid_original, ct_nifti_original, placed_original preserve the
      pre-flip references for traceability

USAGE
=====
    python scripts/apply_manual_flips.py \\
        --manifest   data/placed/placed_manifest.json \\
        --flip_list  configs/flip_list.json \\
        --nifti_dir  data/tcia_nifti \\
        --placed_dir data/placed \\
        --workers    16
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
log = logging.getLogger("apply_manual_flips")

SUFFIX = "_orientation_fixed"


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


# ── Geometry + flip ──────────────────────────────────────────────────────────

def _bbox_y_extent(img) -> Tuple[float, float]:
    """World-Y min/max of a volume's voxel bounding box (8 corners)."""
    shape = img.shape[:3]
    aff   = img.affine.astype(np.float64)
    corners = np.array([
        [0, 0, 0],
        [shape[0]-1, 0, 0], [0, shape[1]-1, 0], [0, 0, shape[2]-1],
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
    Reflect src across the world plane Y = mirror_plane_y via full 3D
    resampling. Output shape + affine = input; only data content changes.

    Using a COMMON external mirror plane (not np.flip's per-volume
    self-center) preserves mask-on-CT alignment when CT and masks have
    different Y translations in their affines.

    PERF NOTE: the XY-plane contribution to world coordinates is identical
    for every Z slice — only the aff[*,2]*k + aff[*,3] offset changes
    across slices. We hoist the meshgrid and the XY contributions out of
    the per-slice loop, roughly halving the per-volume work.
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    try:
        img = nib.load(str(src))
        if not _valid_affine(img.affine):
            log.error("  flip FAILED %s: invalid affine", src.name)
            return False

        data    = np.asarray(img.dataobj)
        aff     = img.affine.astype(np.float64)
        inv_aff = np.linalg.inv(aff)
        shape   = data.shape
        if len(shape) != 3:
            log.error("  flip FAILED %s: expected 3D, got shape %s", src.name, shape)
            return False

        order = 0 if is_label_map else 1
        cval  = 0 if is_label_map else -1024.0
        src_data = data if is_label_map else data.astype(np.float32)

        # ── Hoisted (slice-invariant) terms ────────────────────────────
        # Per-voxel in-plane world contributions: depend only on (i, j),
        # not on k. We compute them once here instead of shape[2] times
        # in the old per-slice loop.
        ii, jj = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            indexing="ij",
        )
        wx_xy = aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 3]  # add aff[0,2]*k + aff[0,3] per-k
        wy_xy = aff[1, 0] * ii + aff[1, 1] * jj + aff[1, 3]
        wz_xy = aff[2, 0] * ii + aff[2, 1] * jj + aff[2, 3]

        # Note: the two "+ aff[.,3]" terms above fold in the translation
        # once; the per-slice update below adds only "aff[.,2]*k" (the
        # z-column contribution, sans translation), so we subtract the
        # translation back out to keep arithmetic identical to the
        # original formulation.
        wx_xy -= aff[0, 3]
        wy_xy -= aff[1, 3]
        wz_xy -= aff[2, 3]

        # Inverse-affine coefficients are also slice-invariant.
        ia = inv_aff  # shorthand

        output = np.empty_like(data)
        for k in range(shape[2]):
            # World coordinates for this slice: XY contribution + Z column + translation.
            wx = wx_xy + aff[0, 2] * k + aff[0, 3]
            wy = wy_xy + aff[1, 2] * k + aff[1, 3]
            wz = wz_xy + aff[2, 2] * k + aff[2, 3]

            # Reflect across the mirror plane in world-Y.
            wy_src = (2.0 * mirror_plane_y) - wy

            # Map back to voxel coordinates via the inverse affine.
            si = ia[0, 0]*wx + ia[0, 1]*wy_src + ia[0, 2]*wz + ia[0, 3]
            sj = ia[1, 0]*wx + ia[1, 1]*wy_src + ia[1, 2]*wz + ia[1, 3]
            sk = ia[2, 0]*wx + ia[2, 1]*wy_src + ia[2, 2]*wz + ia[2, 3]

            coords = np.stack([si, sj, sk], axis=0)
            slice_out = map_coordinates(
                src_data, coords, order=order, mode="constant", cval=cval
            )
            output[..., k] = slice_out.astype(data.dtype)

        new_img = nib.Nifti1Image(output, aff, header=img.header)
        new_img.set_data_dtype(img.get_data_dtype())

        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(dst))

        log.info(
            "  flipped %s: mirror_y=%.2f  shape=%s  is_label=%s",
            label or src.name, mirror_plane_y, tuple(shape), is_label_map,
        )
        return True
    except Exception as e:
        log.error("  flip FAILED %s → %s: %s", src.name, dst.name, e)
        return False


def _verify_bone_alignment(ct_path: Path, mask_path: Path) -> Optional[dict]:
    """
    Sample CT at mask voxels' world positions; return bone% + mean HU.
    bone% close to pre-flip value means the flip preserved alignment.
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
    name = p.name
    if name.endswith(".nii.gz"):
        return p.with_name(name[:-7] + suffix + ".nii.gz")
    if name.endswith(".nii"):
        return p.with_name(name[:-4] + suffix + ".nii")
    return p.with_name(p.stem + suffix + p.suffix)


def _resolve_sides_to_flip(case: dict, entry: dict) -> List[str]:
    """
    Determine which sides get flipped for a case, honoring an optional
    'sides' override in the flip entry.

    Returns a subset of ["spine", "pelvic"]:
      "spine"  → flip the spine CT + spine mask  (if both exist)
      "pelvic" → flip the pelvic CT + pelvic mask (if both exist)

    For fused cases, "spine" and "pelvic" refer to the SAME shared CT; the
    caller should de-dup on CT UID to avoid flipping it twice.

    Defaults when 'sides' is omitted:
      fused        → ["spine", "pelvic"]   (shared CT flips once; both masks flip)
      pelvic_only  → ["pelvic"]
      spine_only   → ["spine"]
      separate     → ["pelvic"]            (conservative: pelvic-based body-center
                                            cue drove the original detector; spine
                                            side left alone unless explicitly listed)
    """
    match_type = case.get("match_type", "unknown")
    sides = entry.get("sides")
    if sides:
        return [s for s in sides if s in ("spine", "pelvic")]

    if match_type == "fused":
        return ["spine", "pelvic"]
    if match_type == "pelvic_only":
        return ["pelvic"]
    if match_type == "spine_only":
        return ["spine"]
    if match_type == "separate":
        return ["pelvic"]
    return []


# ── Per-case worker ──────────────────────────────────────────────────────────

def _process_flip(args) -> dict:
    """Flip one case's files. Called in a subprocess."""
    case, nifti_dir_str, entry = args
    nifti_dir = Path(nifti_dir_str)

    token      = str(case.get("patient_token", "?"))
    match_type = case.get("match_type", "unknown")
    sides      = _resolve_sides_to_flip(case, entry)

    out: dict = {
        "patient_token":        token,
        "match_type":           match_type,
        "flip_review":          {
            "reviewer": entry.get("reviewer", ""),
            "date":     entry.get("date", ""),
            "notes":    entry.get("notes", ""),
            "sides":    sides,
        },
        "orientation_fixed":    None,
        "orientation_original": None,
        "post_flip_bone_pct":   None,
        "post_flip_mean_hu":    None,
        "error":                None,
    }

    pv            = case.get("pelvic") or {}
    sp            = case.get("spine")  or {}
    pelvic_placed = pv.get("placed")
    pelvic_uid    = pv.get("series_uid")
    spine_placed  = sp.get("placed")
    spine_uid     = sp.get("series_uid")

    if not sides:
        out["error"] = f"no_sides_to_flip (match_type={match_type})"
        return out

    import nibabel as nib

    # Plan the CT flip(s). For fused, spine_uid == pelvic_uid → one CT.
    # For separate with sides=["spine","pelvic"], two different CTs.
    planned_cts: List[Tuple[str, str, str]] = []  # (side_label, uid, placed_path_for_geom)
    seen_uids: set = set()

    if "spine" in sides and spine_uid:
        if spine_uid not in seen_uids:
            planned_cts.append(("spine", spine_uid, spine_placed or pelvic_placed or ""))
            seen_uids.add(spine_uid)
    if "pelvic" in sides and pelvic_uid:
        if pelvic_uid not in seen_uids:
            planned_cts.append(("pelvic", pelvic_uid, pelvic_placed or spine_placed or ""))
            seen_uids.add(pelvic_uid)

    if not planned_cts:
        out["error"] = "no_resolvable_ct_for_requested_sides"
        return out

    # For each planned CT, compute its own mirror plane (its own Y bbox
    # center) and flip the CT plus any masks that live on it. Masks that
    # share the CT with the same side flip on that CT's plane.
    flipped:   Dict[str, str]           = {}
    originals: Dict[str, Optional[str]] = {
        "pelvic_placed": str(pelvic_placed) if pelvic_placed else None,
        "spine_placed":  str(spine_placed)  if spine_placed  else None,
        "pelvic_ct":     None,
        "spine_ct":      None,
        "pelvic_ct_uid": pelvic_uid,
        "spine_ct_uid":  spine_uid,
    }

    # CT paths we flipped (by UID) and their mirror planes — used to route
    # mask flips onto the right plane.
    ct_flipped_by_uid: Dict[str, Tuple[str, float]] = {}  # uid → (dst_path, mirror_y)

    for side_label, uid, _anchor_mask in planned_cts:
        ct_src = nifti_dir / f"{uid}.nii.gz"
        if not ct_src.exists():
            log.warning("[token=%s] side=%s CT missing: %s — skipping this side",
                        token, side_label, ct_src.name)
            continue
        try:
            ct_img = nib.load(str(ct_src))
            y_min, y_max = _bbox_y_extent(ct_img)
            mirror_y = (y_min + y_max) / 2.0
        except Exception as e:
            log.error("[token=%s] side=%s: mirror_plane compute failed: %s",
                      token, side_label, e)
            continue

        log.info(
            "[token=%s  match=%s] side=%s  anchor_uid=%s  mirror plane Y=%.2f  "
            "(CT y_range=[%.1f, %.1f])",
            token, match_type, side_label, uid, mirror_y, y_min, y_max,
        )

        ct_dst = _with_suffix(ct_src, SUFFIX)
        if _flip_world_y_resample(
            ct_src, ct_dst, mirror_y,
            is_label_map=False, label=f"CT[token={token},side={side_label}]",
        ):
            ct_flipped_by_uid[uid] = (str(ct_dst), mirror_y)
            if side_label == "spine":
                originals["spine_ct"] = str(ct_src)
            else:
                originals["pelvic_ct"] = str(ct_src)

    # Now flip masks, using each mask's associated CT's mirror plane.
    #   spine mask  → spine CT's plane (or pelvic CT's plane if fused)
    #   pelvic mask → pelvic CT's plane (or spine CT's plane if fused)
    def _plane_for_uid(uid: Optional[str]) -> Optional[float]:
        if not uid: return None
        entry = ct_flipped_by_uid.get(uid)
        return entry[1] if entry else None

    if "spine" in sides and spine_placed and Path(spine_placed).exists():
        plane = _plane_for_uid(spine_uid)
        if plane is None:
            log.warning("[token=%s] side=spine: no plane available (CT not flipped) "
                        "— skipping spine mask flip", token)
        else:
            sp_src = Path(spine_placed)
            sp_dst = _with_suffix(sp_src, SUFFIX)
            if _flip_world_y_resample(
                sp_src, sp_dst, plane,
                is_label_map=True, label=f"spine_mask[token={token}]",
            ):
                flipped["spine_placed"] = str(sp_dst)

    if "pelvic" in sides and pelvic_placed and Path(pelvic_placed).exists():
        plane = _plane_for_uid(pelvic_uid)
        if plane is None:
            log.warning("[token=%s] side=pelvic: no plane available (CT not flipped) "
                        "— skipping pelvic mask flip", token)
        else:
            pm_src = Path(pelvic_placed)
            pm_dst = _with_suffix(pm_src, SUFFIX)
            if _flip_world_y_resample(
                pm_src, pm_dst, plane,
                is_label_map=True, label=f"pelvic_mask[token={token}]",
            ):
                flipped["pelvic_placed"] = str(pm_dst)

    # Record which CTs were flipped (by side-label, for the manifest merge).
    # fused shares one CT, so spine_ct_flipped == pelvic_ct_flipped in that case.
    if "spine" in sides and spine_uid in ct_flipped_by_uid:
        flipped["spine_ct"] = ct_flipped_by_uid[spine_uid][0]
    if "pelvic" in sides and pelvic_uid in ct_flipped_by_uid:
        flipped["pelvic_ct"] = ct_flipped_by_uid[pelvic_uid][0]
    # Back-compat: the merge code reads flipped["ct"] for the "primary"
    # flipped CT. Prefer pelvic for non-spine-only cases.
    if match_type == "spine_only" and "spine_ct" in flipped:
        flipped["ct"] = flipped["spine_ct"]
    elif "pelvic_ct" in flipped:
        flipped["ct"] = flipped["pelvic_ct"]
    elif "spine_ct" in flipped:
        flipped["ct"] = flipped["spine_ct"]

    # ── Post-flip alignment check ───────────────────────────────────────────
    # Pick the most informative mask/CT pair on the same coordinate system.
    check_ct   = None
    check_mask = None
    check_side = None
    if "pelvic_ct" in flipped and "pelvic_placed" in flipped:
        check_ct, check_mask, check_side = flipped["pelvic_ct"], flipped["pelvic_placed"], "pelvic"
    elif "spine_ct" in flipped and "spine_placed" in flipped:
        check_ct, check_mask, check_side = flipped["spine_ct"],  flipped["spine_placed"],  "spine"

    if check_ct and check_mask:
        chk = _verify_bone_alignment(Path(check_ct), Path(check_mask))
        if chk is not None:
            pre_bone = (case.get(check_side) or {}).get("bone_pct", -1)
            log.info(
                "[token=%s] post-flip alignment (%s): bone%%=%.1f  mean_HU=%.1f  "
                "(pre-flip manifest %s bone_pct=%.1f)",
                token, check_side, chk["bone_pct"], chk["mean_hu"],
                check_side, pre_bone,
            )
            out["post_flip_bone_pct"] = round(chk["bone_pct"], 1)
            out["post_flip_mean_hu"]  = round(chk["mean_hu"],  1)

    out["orientation_fixed"]    = flipped if flipped else None
    out["orientation_original"] = originals if flipped else None
    return out


# ── Manifest merge ───────────────────────────────────────────────────────────

def _merge_case(case: dict,
                flips_by_token: Dict[str, dict],
                flip_review_by_token: Dict[str, dict],
                nifti_dir: Path) -> dict:
    """
    Build the output case entry. Every case carries explicit
    spine.ct_nifti and pelvic.ct_nifti paths. For flipped cases:
      - each side's series_uid is patched independently so UID-based
        lookups land on the flipped CT for that side
      - placed paths point at flipped mask files (per side)
      - *_original fields preserve the pre-flip references

    For separate cases with sides=["spine"], only the spine side's UID
    + ct_nifti + placed get patched; pelvic side passes through.
    """
    merged     = dict(case)
    tok        = str(case.get("patient_token", "?"))
    match_type = merged.get("match_type", "unknown")

    fr        = flip_review_by_token.get(tok) or {}
    flip_rec  = flips_by_token.get(tok) or {}
    ofx       = flip_rec.get("orientation_fixed") or {}
    post_bone = flip_rec.get("post_flip_bone_pct")
    post_hu   = flip_rec.get("post_flip_mean_hu")

    # Per-side flipped CT paths. flipped["spine_ct"] / flipped["pelvic_ct"]
    # are explicitly set by _process_flip.
    spine_ct_flipped_path  = ofx.get("spine_ct")
    pelvic_ct_flipped_path = ofx.get("pelvic_ct")

    def _uid_from_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        stem = Path(p).name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        return stem

    spine_flipped_uid  = _uid_from_path(spine_ct_flipped_path)
    pelvic_flipped_uid = _uid_from_path(pelvic_ct_flipped_path)

    if ofx:
        merged["orientation_fixed"]    = ofx
        merged["orientation_original"] = flip_rec.get("orientation_original")
    if fr or post_bone is not None:
        merged["orientation_check"] = {
            "status":             "flipped" if ofx else "ok",
            "source":             "manual_flip_list",
            "reviewer":           fr.get("reviewer", ""),
            "date":               fr.get("date", ""),
            "notes":              fr.get("notes", ""),
            "sides":              fr.get("sides") or (flip_rec.get("flip_review") or {}).get("sides", []),
            "post_flip_bone_pct": post_bone,
            "post_flip_mean_hu":  post_hu,
        }
    else:
        merged["orientation_check"] = {"status": "ok", "source": "unreviewed"}

    # Spine side
    sp = merged.get("spine")
    if isinstance(sp, dict):
        new_sp = dict(sp)
        orig_spine_uid = sp.get("series_uid")
        if spine_flipped_uid:
            new_sp["series_uid_original"] = orig_spine_uid
            new_sp["series_uid"]          = spine_flipped_uid
            new_sp["ct_nifti"]            = spine_ct_flipped_path
            new_sp["ct_nifti_original"]   = str(
                (nifti_dir / f"{orig_spine_uid}.nii.gz").resolve()
            ) if orig_spine_uid else None
        else:
            new_sp["ct_nifti"] = (
                str((nifti_dir / f"{orig_spine_uid}.nii.gz").resolve())
                if orig_spine_uid else None
            )
        if ofx.get("spine_placed"):
            new_sp["placed"]          = ofx["spine_placed"]
            new_sp["placed_original"] = (flip_rec.get("orientation_original") or {}).get("spine_placed")
        merged["spine"] = new_sp

    # Pelvic side
    pv = merged.get("pelvic")
    if isinstance(pv, dict):
        new_pv = dict(pv)
        orig_pelvic_uid = pv.get("series_uid")
        if pelvic_flipped_uid:
            new_pv["series_uid_original"] = orig_pelvic_uid
            new_pv["series_uid"]          = pelvic_flipped_uid
            new_pv["ct_nifti"]            = pelvic_ct_flipped_path
            new_pv["ct_nifti_original"]   = str(
                (nifti_dir / f"{orig_pelvic_uid}.nii.gz").resolve()
            ) if orig_pelvic_uid else None
        else:
            new_pv["ct_nifti"] = (
                str((nifti_dir / f"{orig_pelvic_uid}.nii.gz").resolve())
                if orig_pelvic_uid else None
            )
        if ofx.get("pelvic_placed"):
            new_pv["placed"]          = ofx["pelvic_placed"]
            new_pv["placed_original"] = (flip_rec.get("orientation_original") or {}).get("pelvic_placed")
        merged["pelvic"] = new_pv

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest",   required=True, type=Path,
                    help="Stage 2 placed_manifest.json")
    ap.add_argument("--flip_list",  required=True, type=Path,
                    help="configs/flip_list.json")
    ap.add_argument("--nifti_dir",  required=True, type=Path)
    ap.add_argument("--placed_dir", required=True, type=Path)
    ap.add_argument("--workers",    default=16, type=int)
    args = ap.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest); return
    if not args.flip_list.exists():
        log.error("Flip list not found: %s", args.flip_list); return

    manifest = json.loads(args.manifest.read_text())
    cases    = manifest.get("cases", [])

    flip_doc = json.loads(args.flip_list.read_text())
    flip_entries = flip_doc.get("flips", [])
    flip_review_by_token: Dict[str, dict] = {
        str(e["token"]): e for e in flip_entries if "token" in e
    }

    # Exclusions — tokens to drop entirely from the output manifest because
    # the mask/series pairing is wrong (not just an AP issue).
    exclusion_entries = flip_doc.get("exclusions", [])
    exclusions_by_token: Dict[str, dict] = {
        str(e["token"]): e for e in exclusion_entries if "token" in e
    }

    log.info("Loaded %d cases from %s", len(cases), args.manifest)
    log.info("Loaded %d flip entries from %s",
             len(flip_review_by_token), args.flip_list)
    log.info("Loaded %d exclusion entries", len(exclusions_by_token))
    if flip_review_by_token:
        log.info("Tokens to flip:    %s",
                 sorted(flip_review_by_token.keys(), key=lambda s: (len(s), s)))
    if exclusions_by_token:
        log.info("Tokens to exclude: %s",
                 sorted(exclusions_by_token.keys(), key=lambda s: (len(s), s)))

    # Cross-references: which listed tokens actually exist in the manifest?
    manifest_tokens = {str(c.get("patient_token", "?")) for c in cases}
    flip_missing = sorted(set(flip_review_by_token) - manifest_tokens)
    excl_missing = sorted(set(exclusions_by_token) - manifest_tokens)
    if flip_missing:
        log.warning("flip_list tokens not in manifest (skipped): %s", flip_missing)
    if excl_missing:
        log.warning("exclusion tokens not in manifest (skipped): %s", excl_missing)

    # Sanity: overlap between flips and exclusions is a reviewer bug
    both = sorted(set(flip_review_by_token) & set(exclusions_by_token))
    if both:
        log.error(
            "TOKENS LISTED IN BOTH flips AND exclusions — aborting: %s", both
        )
        log.error("A token cannot be both flipped and excluded. Remove from one list.")
        return

    # Build work list — only tokens in flips AND present in manifest AND not excluded
    work = []
    for c in cases:
        tok = str(c.get("patient_token", "?"))
        if tok in exclusions_by_token:
            continue
        if tok in flip_review_by_token:
            work.append((c, str(args.nifti_dir), flip_review_by_token[tok]))

    results: List[dict] = []
    if work:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_flip, w) for w in work]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    results.append(f.result())
                except Exception as e:
                    log.error("Worker crash: %s", e)
                if i % 10 == 0 or i == len(work):
                    log.info("  processed %d / %d", i, len(work))
    else:
        log.info("No tokens in flip_list match the manifest — nothing to flip.")

    flips_by_token = {r["patient_token"]: r for r in results}

    # ── Summary ─────────────────────────────────────────────────────────────
    n_flipped = sum(1 for r in results if r.get("orientation_fixed"))
    n_failed  = sum(1 for r in results if r.get("error"))
    n_excluded_applied = len(set(exclusions_by_token) & manifest_tokens)
    log.info("=" * 68)
    log.info("Summary:")
    log.info("  Flip requested     : %d", len(flip_review_by_token))
    log.info("  Flip not in manif. : %d", len(flip_missing))
    log.info("  Flipped OK         : %d", n_flipped)
    log.info("  Flip failed        : %d", n_failed)
    log.info("  Exclude requested  : %d", len(exclusions_by_token))
    log.info("  Excluded (applied) : %d", n_excluded_applied)
    log.info("  Exclude missing    : %d", len(excl_missing))
    if n_failed:
        for r in results:
            if r.get("error"):
                log.warning("  token=%-12s  error=%s",
                            r["patient_token"], r["error"])
    if exclusions_by_token:
        log.info("Excluded tokens (dropped from output manifest):")
        for tok in sorted(exclusions_by_token, key=lambda s: (len(s), s)):
            e = exclusions_by_token[tok]
            in_manifest = tok in manifest_tokens
            log.info(
                "  token=%-12s  reviewer=%-20s  reason=%-20s  in_manifest=%s",
                tok, e.get("reviewer", "?"), e.get("reason", "?"),
                "yes" if in_manifest else "NO (no-op)",
            )
    log.info("=" * 68)

    # ── Merge + write manifest (excluding tokens in exclusions_by_token) ────
    new_cases = [
        _merge_case(c, flips_by_token, flip_review_by_token, args.nifti_dir)
        for c in cases
        if str(c.get("patient_token", "?")) not in exclusions_by_token
    ]

    out_path = args.placed_dir / "placed_manifest_orientation_fixed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "n_cases":              len(new_cases),
        "n_cases_input":        manifest.get("n_cases", len(cases)),
        "n_fused":              manifest.get("n_fused"),
        "n_separate":           manifest.get("n_separate"),
        "n_spine_only":         manifest.get("n_spine_only"),
        "n_pelvic_only":        manifest.get("n_pelvic_only"),
        "n_manually_flipped":   n_flipped,
        "n_flip_requested":     len(flip_review_by_token),
        "n_flip_missing":       len(flip_missing),
        "n_flip_failed":        n_failed,
        "n_excluded":           n_excluded_applied,
        "n_exclude_requested":  len(exclusions_by_token),
        "n_exclude_missing":    len(excl_missing),
        "flip_list_path":       str(args.flip_list),
        "excluded_tokens":      sorted(
            set(exclusions_by_token) & manifest_tokens,
            key=lambda s: (len(s), s),
        ),
        "detector":             "manual_review",
        "schema_version":       "v6_manual_flips_with_exclusions",
        "cases":                new_cases,
    }
    out_path.write_text(json.dumps(out_doc, indent=2, default=str))
    log.info("Manifest → %s", out_path)
    log.info("Schema: v6_manual_flips_with_exclusions (no heuristic detector). "
             "spine.ct_nifti and pelvic.ct_nifti carry explicit paths per case. "
             "Excluded tokens are dropped from the output manifest.")


if __name__ == "__main__":
    main()
