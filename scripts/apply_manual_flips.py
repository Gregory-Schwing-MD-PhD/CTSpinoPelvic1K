"""
apply_manual_flips.py — Apply reviewer-curated AP flips and label swaps.

Reads:
  --manifest   data/placed/placed_manifest.json   (source of truth from Step B)
  --flip_list  configs/flip_list.json             (reviewer-curated overrides)

Three independent override mechanisms in flip_list.json:

  1. flips         — tokens whose CT + masks need spatial AP-flipping for
                     visual consistency (e.g., normalizing all cases to a
                     prone-looking display orientation). Reflects across
                     the CT's world-Y bbox center via 3D resampling.
                     Output suffix: _orientation_fixed.

  2. label_swaps   — tokens whose mask file has anatomically inverted label
                     VALUES (most often hip 8 ↔ 9 for upstream CTPelvic1K
                     files where the source labels were placed on the wrong
                     anatomical sides — verified via TS-vs-GT hip Dice
                     audit + axial visual QC). Voxels stay where they are;
                     only label values change. Output suffix:
                     _orientation_fixed (same as flips, so a token in BOTH
                     lists ends with one corrected file containing both
                     fixes).

  3. exclusions    — tokens to drop entirely from the output manifest
                     because the mask/series pairing is fundamentally
                     wrong (mis-assigned series, etc.).

Composition: a single token can appear in both `flips` AND `label_swaps`.
The spatial flip is applied first, then the label remap is applied to the
flipped output (so the final file has both corrections). Tokens in
`exclusions` are dropped before either operation.

AFFECTED FILES BY MATCH TYPE (for flips)
========================================
  fused        → CT (shared), spine mask, pelvic mask
  pelvic_only  → CT, pelvic mask
  separate     → pelvic CT, pelvic mask  (spine side conservative default)
  spine_only   → CT, spine mask

AFFECTED FILES (for label_swaps)
================================
  Default target is `pelvic` — the typical use case is hip-label
  inversion in CTPelvic1K source files. The optional `target` field on
  each label_swap entry can override:
    target: "pelvic"  (default; remaps the placed pelvic mask)
    target: "spine"   (remaps the placed spine mask)
    target: "both"    (remaps both)

Writes:
  data/placed/placed_manifest_orientation_fixed.json

  For every case, the output manifest carries explicit spine.ct_nifti and
  pelvic.ct_nifti paths. For modified cases (flipped, swapped, or both):
    - placed paths point at the modified mask files (_orientation_fixed)
    - For flipped cases, series_uid is patched to the flipped CT stem
    - *_original fields preserve the pre-modification references
    - orientation_check carries reviewer + provenance metadata, including
      a label_swap subfield when label_swaps applied to this token

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
from typing import Any, Dict, List, Optional, Tuple

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


# ── Geometry + spatial flip (unchanged from prior version) ───────────────────

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

        ii, jj = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            indexing="ij",
        )
        wx_xy = aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 3]
        wy_xy = aff[1, 0] * ii + aff[1, 1] * jj + aff[1, 3]
        wz_xy = aff[2, 0] * ii + aff[2, 1] * jj + aff[2, 3]

        wx_xy -= aff[0, 3]
        wy_xy -= aff[1, 3]
        wz_xy -= aff[2, 3]

        ia = inv_aff

        output = np.empty_like(data)
        for k in range(shape[2]):
            wx = wx_xy + aff[0, 2] * k + aff[0, 3]
            wy = wy_xy + aff[1, 2] * k + aff[1, 3]
            wz = wz_xy + aff[2, 2] * k + aff[2, 3]

            wy_src = (2.0 * mirror_plane_y) - wy

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


# ── Label-swap operation ─────────────────────────────────────────────────────

def _apply_label_swap(
    src: Path,
    dst: Path,
    swap_pairs: List[Tuple[int, int]],
    label: str = "",
) -> Optional[Dict[str, int]]:
    """
    Read `src`, swap label values according to `swap_pairs`, write to `dst`.
    Affine and header are preserved exactly — only voxel VALUES change.

    Each pair (a, b) in swap_pairs is bidirectional: voxels with value a
    become b, and voxels with value b become a. The swap is performed in
    parallel (all reads, then all writes) so multiple pairs cannot
    accidentally chain into each other.

    If src == dst, the file is overwritten in place. The typical caller
    pattern is:
      - flip_list.json says token T needs both a spatial flip and a label
        swap. The spatial flip writes flipped data to dst (the
        _orientation_fixed file). _apply_label_swap is then called with
        src=dst=that same _orientation_fixed file, modifying it in place.
      - flip_list.json says token T needs only a label swap. Caller passes
        src=original placed mask, dst=_orientation_fixed file; the swap
        runs on a fresh copy.

    Returns a dict of {label_value: voxel_count} for each label that was
    affected (pre-swap counts), or None on failure. Useful for logging
    and for verifying the swap was non-trivial.
    """
    import nibabel as nib

    try:
        img = nib.load(str(src))
        if not _valid_affine(img.affine):
            log.error("  label_swap FAILED %s: invalid affine", src.name)
            return None

        data = np.asarray(img.dataobj)
        if data.ndim != 3:
            log.error("  label_swap FAILED %s: expected 3D, got shape %s",
                      src.name, data.shape)
            return None

        # Build the swap as a remap dictionary, then apply via np.where chain.
        # Doing it in two passes (read-only mask, then write) avoids the
        # chaining bug where swapping a→b followed by b→a undoes itself.
        new_data = data.copy()
        affected: Dict[int, int] = {}
        for a, b in swap_pairs:
            mask_a = (data == a)
            mask_b = (data == b)
            n_a = int(mask_a.sum())
            n_b = int(mask_b.sum())
            new_data[mask_a] = b
            new_data[mask_b] = a
            affected[a] = n_a
            affected[b] = n_b

        new_img = nib.Nifti1Image(
            new_data.astype(img.get_data_dtype()),
            img.affine,
            header=img.header,
        )
        new_img.set_data_dtype(img.get_data_dtype())

        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(dst))

        log.info(
            "  label_swap %s: pairs=%s  affected_voxels=%s",
            label or src.name, swap_pairs, affected,
        )
        return affected
    except Exception as e:
        log.error("  label_swap FAILED %s → %s: %s", src.name, dst.name, e)
        return None


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


def _normalize_swap_pairs(raw: Any) -> List[Tuple[int, int]]:
    """
    Accept either ``[[a,b], [c,d]]`` or ``[a,b]`` shorthand and normalize
    to ``[(a,b), (c,d)]``. Validates that each pair is two ints and the
    values within a pair differ.
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(
        isinstance(x, int) for x in raw
    ):
        # shorthand: a single pair as [a, b]
        raw = [raw]
    out: List[Tuple[int, int]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"label swap pair must be [a,b], got {pair!r}")
        a, b = pair
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError(f"label swap pair must be two ints, got {pair!r}")
        if a == b:
            raise ValueError(f"label swap pair has equal values: {pair!r}")
        out.append((a, b))
    return out


def _resolve_label_swap_target(entry: dict) -> str:
    """Returns one of 'pelvic', 'spine', 'both'. Defaults to 'pelvic'."""
    t = (entry.get("target") or "pelvic").lower()
    if t not in ("pelvic", "spine", "both"):
        raise ValueError(f"unknown label_swap target {t!r}")
    return t


# ── Per-case worker ──────────────────────────────────────────────────────────

def _process_case(args) -> dict:
    """
    Apply spatial flip and/or label swap to a single case. A case may have:
      - a flip entry only        → spatial flip
      - a label_swap entry only  → label remap on a fresh copy of placed mask
      - BOTH                     → spatial flip first, then label remap on
                                   the flipped output (so the final file
                                   carries both corrections)
    """
    case, nifti_dir_str, flip_entry, swap_entry = args
    nifti_dir = Path(nifti_dir_str)

    token      = str(case.get("patient_token", "?"))
    match_type = case.get("match_type", "unknown")

    out: dict = {
        "patient_token":        token,
        "match_type":           match_type,
        "flip_review":          None,
        "label_swap_review":    None,
        "orientation_fixed":    None,
        "orientation_original": None,
        "post_flip_bone_pct":   None,
        "post_flip_mean_hu":    None,
        "label_swap_applied":   None,
        "error":                None,
    }

    pv            = case.get("pelvic") or {}
    sp            = case.get("spine")  or {}
    pelvic_placed = pv.get("placed")
    pelvic_uid    = pv.get("series_uid")
    spine_placed  = sp.get("placed")
    spine_uid     = sp.get("series_uid")

    flipped:   Dict[str, str]                     = {}
    originals: Dict[str, Optional[str]] = {
        "pelvic_placed": str(pelvic_placed) if pelvic_placed else None,
        "spine_placed":  str(spine_placed)  if spine_placed  else None,
        "pelvic_ct":     None,
        "spine_ct":      None,
        "pelvic_ct_uid": pelvic_uid,
        "spine_ct_uid":  spine_uid,
    }
    ct_flipped_by_uid: Dict[str, Tuple[str, float]] = {}

    # ── Phase 1: spatial flip (if applicable) ───────────────────────────────
    if flip_entry:
        sides = _resolve_sides_to_flip(case, flip_entry)
        out["flip_review"] = {
            "reviewer": flip_entry.get("reviewer", ""),
            "date":     flip_entry.get("date", ""),
            "notes":    flip_entry.get("notes", ""),
            "sides":    sides,
        }

        if not sides:
            out["error"] = f"no_sides_to_flip (match_type={match_type})"
            return out

        import nibabel as nib

        planned_cts: List[Tuple[str, str, str]] = []
        seen_uids: set = set()
        if "spine" in sides and spine_uid and spine_uid not in seen_uids:
            planned_cts.append(("spine", spine_uid, spine_placed or pelvic_placed or ""))
            seen_uids.add(spine_uid)
        if "pelvic" in sides and pelvic_uid and pelvic_uid not in seen_uids:
            planned_cts.append(("pelvic", pelvic_uid, pelvic_placed or spine_placed or ""))
            seen_uids.add(pelvic_uid)

        if not planned_cts:
            out["error"] = "no_resolvable_ct_for_requested_sides"
            return out

        for side_label, uid, _anchor_mask in planned_cts:
            ct_src = nifti_dir / f"{uid}.nii.gz"
            if not ct_src.exists():
                log.warning("[token=%s] side=%s CT missing: %s — skipping side",
                            token, side_label, ct_src.name)
                continue
            try:
                ct_img = nib.load(str(ct_src))
                y_min, y_max = _bbox_y_extent(ct_img)
                mirror_y = (y_min + y_max) / 2.0
            except Exception as e:
                log.error("[token=%s] side=%s mirror_plane fail: %s",
                          token, side_label, e)
                continue

            log.info(
                "[token=%s match=%s] side=%s uid=%s mirror_y=%.2f",
                token, match_type, side_label, uid, mirror_y,
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

        def _plane_for_uid(uid: Optional[str]) -> Optional[float]:
            if not uid: return None
            entry = ct_flipped_by_uid.get(uid)
            return entry[1] if entry else None

        if "spine" in sides and spine_placed and Path(spine_placed).exists():
            plane = _plane_for_uid(spine_uid)
            if plane is None:
                log.warning("[token=%s] spine: no plane available — skipping spine mask flip", token)
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
                log.warning("[token=%s] pelvic: no plane available — skipping pelvic mask flip", token)
            else:
                pm_src = Path(pelvic_placed)
                pm_dst = _with_suffix(pm_src, SUFFIX)
                if _flip_world_y_resample(
                    pm_src, pm_dst, plane,
                    is_label_map=True, label=f"pelvic_mask[token={token}]",
                ):
                    flipped["pelvic_placed"] = str(pm_dst)

        if "spine" in sides and spine_uid in ct_flipped_by_uid:
            flipped["spine_ct"] = ct_flipped_by_uid[spine_uid][0]
        if "pelvic" in sides and pelvic_uid in ct_flipped_by_uid:
            flipped["pelvic_ct"] = ct_flipped_by_uid[pelvic_uid][0]
        if match_type == "spine_only" and "spine_ct" in flipped:
            flipped["ct"] = flipped["spine_ct"]
        elif "pelvic_ct" in flipped:
            flipped["ct"] = flipped["pelvic_ct"]
        elif "spine_ct" in flipped:
            flipped["ct"] = flipped["spine_ct"]

    # ── Phase 2: label swap (if applicable) ─────────────────────────────────
    if swap_entry:
        try:
            swap_pairs = _normalize_swap_pairs(swap_entry.get("swap"))
            target     = _resolve_label_swap_target(swap_entry)
        except ValueError as e:
            out["error"] = f"invalid_label_swap: {e}"
            return out

        if not swap_pairs:
            log.warning("[token=%s] label_swap entry has no 'swap' pairs — skipping", token)
        else:
            out["label_swap_review"] = {
                "reviewer": swap_entry.get("reviewer", ""),
                "date":     swap_entry.get("date", ""),
                "notes":    swap_entry.get("notes", ""),
                "swap":     [list(p) for p in swap_pairs],
                "target":   target,
            }

            applied_summary: Dict[str, Any] = {}

            # For each target side, decide source/destination paths:
            #   - If a spatial flip already wrote a _orientation_fixed file,
            #     run the swap in place on that file.
            #   - Otherwise, run the swap from the original placed mask
            #     into a fresh _orientation_fixed file.
            for side in ("pelvic", "spine"):
                if target not in (side, "both"):
                    continue

                placed_orig = pelvic_placed if side == "pelvic" else spine_placed
                placed_key  = "pelvic_placed" if side == "pelvic" else "spine_placed"
                if not placed_orig or not Path(placed_orig).exists():
                    log.warning("[token=%s] label_swap target=%s but no placed mask — skipping",
                                token, side)
                    continue

                if placed_key in flipped:
                    # Modify the flipped file in place
                    src_path = Path(flipped[placed_key])
                    dst_path = src_path
                else:
                    src_path = Path(placed_orig)
                    dst_path = _with_suffix(src_path, SUFFIX)
                    # Track that the mask is being modified even without a flip
                    if side == "pelvic":
                        originals.setdefault("pelvic_placed",
                                             str(placed_orig))
                    else:
                        originals.setdefault("spine_placed",
                                             str(placed_orig))

                affected = _apply_label_swap(
                    src_path, dst_path, swap_pairs,
                    label=f"{side}_mask[token={token}]",
                )
                if affected is not None:
                    flipped[placed_key] = str(dst_path)
                    applied_summary[side] = {
                        "src": str(src_path),
                        "dst": str(dst_path),
                        "swap_pairs": [list(p) for p in swap_pairs],
                        "affected_voxels": affected,
                    }

            if applied_summary:
                out["label_swap_applied"] = applied_summary

    # ── Post-modification alignment check (only for flipped outputs) ────────
    check_ct = check_mask = check_side = None
    if "pelvic_ct" in flipped and "pelvic_placed" in flipped:
        check_ct, check_mask, check_side = flipped["pelvic_ct"], flipped["pelvic_placed"], "pelvic"
    elif "spine_ct" in flipped and "spine_placed" in flipped:
        check_ct, check_mask, check_side = flipped["spine_ct"], flipped["spine_placed"], "spine"

    if check_ct and check_mask:
        chk = _verify_bone_alignment(Path(check_ct), Path(check_mask))
        if chk is not None:
            pre_bone = (case.get(check_side) or {}).get("bone_pct", -1)
            log.info(
                "[token=%s] post-modify alignment (%s): bone%%=%.1f mean_HU=%.1f (pre=%.1f)",
                token, check_side, chk["bone_pct"], chk["mean_hu"], pre_bone,
            )
            out["post_flip_bone_pct"] = round(chk["bone_pct"], 1)
            out["post_flip_mean_hu"]  = round(chk["mean_hu"],  1)

    out["orientation_fixed"]    = flipped if flipped else None
    out["orientation_original"] = originals if flipped else None
    return out


# ── Manifest merge ───────────────────────────────────────────────────────────

def _merge_case(case: dict,
                results_by_token: Dict[str, dict],
                flip_review_by_token: Dict[str, dict],
                swap_review_by_token: Dict[str, dict],
                nifti_dir: Path) -> dict:
    """
    Build the output case entry. Every case carries explicit
    spine.ct_nifti and pelvic.ct_nifti paths. For modified cases:
      - each side's series_uid is patched if its CT was flipped
      - placed paths point at modified mask files (per side)
      - *_original fields preserve the pre-modification references
      - orientation_check carries both flip and label_swap provenance
    """
    merged     = dict(case)
    tok        = str(case.get("patient_token", "?"))
    match_type = merged.get("match_type", "unknown")

    fr        = flip_review_by_token.get(tok) or {}
    sw        = swap_review_by_token.get(tok) or {}
    rec       = results_by_token.get(tok) or {}
    ofx       = rec.get("orientation_fixed") or {}
    post_bone = rec.get("post_flip_bone_pct")
    post_hu   = rec.get("post_flip_mean_hu")
    swap_appl = rec.get("label_swap_applied")

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
        merged["orientation_original"] = rec.get("orientation_original")

    # orientation_check covers both flip and label_swap provenance
    if fr or sw or post_bone is not None:
        status_parts = []
        if fr and ofx:
            status_parts.append("flipped")
        if sw and swap_appl:
            status_parts.append("label_swapped")
        status = "+".join(status_parts) if status_parts else "ok"

        oc: Dict[str, Any] = {
            "status":             status,
            "source":             "manual_flip_list",
            "post_flip_bone_pct": post_bone,
            "post_flip_mean_hu":  post_hu,
        }
        if fr:
            oc["flip"] = {
                "reviewer": fr.get("reviewer", ""),
                "date":     fr.get("date", ""),
                "notes":    fr.get("notes", ""),
                "sides":    fr.get("sides") or (rec.get("flip_review") or {}).get("sides", []),
            }
        if sw:
            oc["label_swap"] = {
                "reviewer":         sw.get("reviewer", ""),
                "date":             sw.get("date", ""),
                "notes":            sw.get("notes", ""),
                "swap":             sw.get("swap", []),
                "target":           sw.get("target", "pelvic"),
                "affected_voxels":  swap_appl,
            }
        merged["orientation_check"] = oc
    else:
        merged["orientation_check"] = {"status": "ok", "source": "unreviewed"}

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
            new_sp["placed_original"] = (rec.get("orientation_original") or {}).get("spine_placed")
        merged["spine"] = new_sp

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
            new_pv["placed_original"] = (rec.get("orientation_original") or {}).get("pelvic_placed")
        merged["pelvic"] = new_pv

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest",   required=True, type=Path)
    ap.add_argument("--flip_list",  required=True, type=Path)
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
    flip_entries  = flip_doc.get("flips", [])
    swap_entries  = flip_doc.get("label_swaps", [])
    excl_entries  = flip_doc.get("exclusions", [])

    flip_review_by_token: Dict[str, dict] = {
        str(e["token"]): e for e in flip_entries if "token" in e
    }
    swap_review_by_token: Dict[str, dict] = {
        str(e["token"]): e for e in swap_entries if "token" in e
    }
    exclusions_by_token:  Dict[str, dict] = {
        str(e["token"]): e for e in excl_entries if "token" in e
    }

    log.info("Loaded %d cases from %s", len(cases), args.manifest)
    log.info("Loaded %d flip entries", len(flip_review_by_token))
    log.info("Loaded %d label_swap entries", len(swap_review_by_token))
    log.info("Loaded %d exclusion entries", len(exclusions_by_token))
    if flip_review_by_token:
        log.info("Tokens to flip:        %s",
                 sorted(flip_review_by_token.keys(), key=lambda s: (len(s), s)))
    if swap_review_by_token:
        log.info("Tokens to label_swap:  %s",
                 sorted(swap_review_by_token.keys(), key=lambda s: (len(s), s)))
    if exclusions_by_token:
        log.info("Tokens to exclude:     %s",
                 sorted(exclusions_by_token.keys(), key=lambda s: (len(s), s)))

    manifest_tokens = {str(c.get("patient_token", "?")) for c in cases}
    flip_missing = sorted(set(flip_review_by_token) - manifest_tokens)
    swap_missing = sorted(set(swap_review_by_token) - manifest_tokens)
    excl_missing = sorted(set(exclusions_by_token) - manifest_tokens)
    if flip_missing:
        log.warning("flip_list flips not in manifest (skipped): %s", flip_missing)
    if swap_missing:
        log.warning("flip_list label_swaps not in manifest (skipped): %s", swap_missing)
    if excl_missing:
        log.warning("flip_list exclusions not in manifest (skipped): %s", excl_missing)

    # Sanity: tokens cannot be both modified and excluded
    bad = (set(flip_review_by_token) | set(swap_review_by_token)) & set(exclusions_by_token)
    if bad:
        log.error("TOKENS LISTED IN exclusions AND (flips or label_swaps) — aborting: %s",
                  sorted(bad))
        return

    # Build work list — tokens with ANY override that aren't excluded
    needs_modification = set(flip_review_by_token) | set(swap_review_by_token)
    work = []
    for c in cases:
        tok = str(c.get("patient_token", "?"))
        if tok in exclusions_by_token:
            continue
        if tok in needs_modification:
            work.append((
                c,
                str(args.nifti_dir),
                flip_review_by_token.get(tok),
                swap_review_by_token.get(tok),
            ))

    results: List[dict] = []
    if work:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_case, w) for w in work]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    results.append(f.result())
                except Exception as e:
                    log.error("Worker crash: %s", e)
                if i % 10 == 0 or i == len(work):
                    log.info("  processed %d / %d", i, len(work))
    else:
        log.info("No tokens in flip_list match the manifest — nothing to do.")

    results_by_token = {r["patient_token"]: r for r in results}

    # ── Summary ─────────────────────────────────────────────────────────────
    n_flipped       = sum(1 for r in results
                          if r.get("orientation_fixed")
                          and r.get("flip_review"))
    n_swapped       = sum(1 for r in results if r.get("label_swap_applied"))
    n_both          = sum(1 for r in results
                          if r.get("flip_review")
                          and r.get("label_swap_applied")
                          and r.get("orientation_fixed"))
    n_failed        = sum(1 for r in results if r.get("error"))
    n_excluded_app  = len(set(exclusions_by_token) & manifest_tokens)
    log.info("=" * 68)
    log.info("Summary:")
    log.info("  Flip requested        : %d", len(flip_review_by_token))
    log.info("  Flip not in manifest  : %d", len(flip_missing))
    log.info("  Flipped OK            : %d", n_flipped)
    log.info("  Swap requested        : %d", len(swap_review_by_token))
    log.info("  Swap not in manifest  : %d", len(swap_missing))
    log.info("  Label-swapped OK      : %d", n_swapped)
    log.info("  Both flip & swap      : %d", n_both)
    log.info("  Modify failed         : %d", n_failed)
    log.info("  Exclude requested     : %d", len(exclusions_by_token))
    log.info("  Excluded (applied)    : %d", n_excluded_app)
    log.info("  Exclude missing       : %d", len(excl_missing))
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

    new_cases = [
        _merge_case(c, results_by_token, flip_review_by_token,
                    swap_review_by_token, args.nifti_dir)
        for c in cases
        if str(c.get("patient_token", "?")) not in exclusions_by_token
    ]

    out_path = args.placed_dir / "placed_manifest_orientation_fixed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "n_cases":               len(new_cases),
        "n_cases_input":         manifest.get("n_cases", len(cases)),
        "n_fused":               manifest.get("n_fused"),
        "n_separate":            manifest.get("n_separate"),
        "n_spine_only":          manifest.get("n_spine_only"),
        "n_pelvic_only":         manifest.get("n_pelvic_only"),
        "n_manually_flipped":    n_flipped,
        "n_label_swapped":       n_swapped,
        "n_both":                n_both,
        "n_flip_requested":      len(flip_review_by_token),
        "n_flip_missing":        len(flip_missing),
        "n_swap_requested":      len(swap_review_by_token),
        "n_swap_missing":        len(swap_missing),
        "n_modify_failed":       n_failed,
        "n_excluded":            n_excluded_app,
        "n_exclude_requested":   len(exclusions_by_token),
        "n_exclude_missing":     len(excl_missing),
        "flip_list_path":        str(args.flip_list),
        "excluded_tokens":       sorted(
            set(exclusions_by_token) & manifest_tokens,
            key=lambda s: (len(s), s),
        ),
        "detector":              "manual_review",
        "schema_version":        "v7_flips_with_label_swaps",
        "cases":                 new_cases,
    }
    out_path.write_text(json.dumps(out_doc, indent=2, default=str))
    log.info("Manifest → %s", out_path)
    log.info("Schema: v7_flips_with_label_swaps (adds label_swaps section "
             "for upstream label-value corrections; composes with flips).")


if __name__ == "__main__":
    main()
