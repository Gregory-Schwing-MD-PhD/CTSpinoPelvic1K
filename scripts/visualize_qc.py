"""
visualize_qc.py — QC figures for CTSpinoPelvic1K placed masks.

Reads placed_manifest.json (from place_fused_masks.py) and generates
per-case PNG figures:

  Standard (--per_case):     3 rows × 3 cols  (spine / pelvic / overlay)
  Debug    (--tokens / --debug): 3 rows × 4 cols
    Col 0  Raw CT (spine series)
    Col 1  CT + spine seg
    Col 2  Pelvic CT + pelvic mask  (own CT for separate cases)
    Col 3  Compatibility metrics panel (computed + manifest values)

Data orientation: stored volumes are PIR (axis 0 = Posterior, axis 1 = Inferior,
axis 2 = patient-Right as index increases). This script reorients every input
to PIR via `_load()`, then slices and displays so that:
  Row 1 "Coronal"   dim=0  -- head at top, feet at bottom
  Row 2 "Axial"     dim=1  -- anterior at top, posterior (spine) at bottom
  Row 3 "Sagittal"  dim=2  -- head at top, spine runs vertically (requires transpose)

IS_fail cases (vertebral centroid ordering check failed) are routed to
  per_case/is_fail/
with a red IS_ORDER_FAIL banner. All other cases go to per_case/{match_type}/.

Usage
-----
  python scripts/visualize_qc.py \
      --manifest  data/placed/placed_manifest.json \
      --nifti_dir data/tcia_nifti \
      --placed_dir data/placed \
      --out_dir   data/qc_figures
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.visualize_qc")

_HU_MIN, _HU_MAX = -200, 800

_SPINE_LABELS  = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
_PELVIC_LABELS = {1: "sacrum", 2: "left_hip", 3: "right_hip"}

_SPINE_COLORS = {
    20: (0.20, 0.40, 0.90, 0.55),
    21: (0.15, 0.55, 0.85, 0.55),
    22: (0.10, 0.70, 0.75, 0.55),
    23: (0.10, 0.75, 0.50, 0.55),
    24: (0.15, 0.80, 0.30, 0.55),
    25: (0.60, 0.90, 0.10, 0.55),
}
_PELVIC_COLORS = {
    1: (0.95, 0.20, 0.15, 0.60),
    2: (0.95, 0.55, 0.10, 0.60),
    3: (0.95, 0.85, 0.10, 0.60),
}

_COMPAT_COLORS = {"OK": "#44ff88", "WARNING": "#ffaa00", "FAIL": "#ff4444"}

_BONE_HU          = 200.0
_PELVIC_BONE_WARN = 20.0
_PELVIC_BONE_FAIL =  5.0
_SPINE_BONE_WARN  = 30.0
_SPINE_BONE_FAIL  = 10.0
_FOV_OVERLAP_FAIL = 80.0


# ===========================================================================
# Image helpers
# ===========================================================================

def _load(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
    img      = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)
    data     = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff  = img.affine @ nib.orientations.inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _window(ct: np.ndarray) -> np.ndarray:
    return np.clip((ct - _HU_MIN) / (_HU_MAX - _HU_MIN), 0.0, 1.0)


def _overlay(base, lbl_sl, cmap):
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    else:
        rgb = base.astype(np.float32).copy()
    for v, (r, g, b, a) in cmap.items():
        m = lbl_sl == v
        if not m.any():
            continue
        rgb[m, 0] = rgb[m, 0] * (1 - a) + r * a
        rgb[m, 1] = rgb[m, 1] * (1 - a) + g * a
        rgb[m, 2] = rgb[m, 2] * (1 - a) + b * a
    return np.clip(rgb, 0.0, 1.0)


def _choose_slices(ct, seg, pelv):
    sp = np.argwhere(seg  > 0) if seg  is not None and (seg  > 0).any() else None
    pv = np.argwhere(pelv > 0) if pelv is not None and (pelv > 0).any() else None

    i = (int(sp[:, 0].mean()) if sp is not None
         else (int(pv[:, 0].mean()) if pv is not None else ct.shape[0] // 2))
    j = (int(pv[:, 1].mean()) if pv is not None
         else (int(sp[:, 1].mean()) if sp is not None else ct.shape[1] // 2))

    if sp is not None and pv is not None:
        seg_zs  = np.where(seg.any(axis=(0, 1)))[0]
        pelv_zs = np.where(pelv.any(axis=(0, 1)))[0]
        both    = sorted(set(seg_zs.tolist()) & set(pelv_zs.tolist()))
        k = (both[len(both) // 4] if both
             else ((int(seg_zs.min()) + int(pelv_zs.max())) // 2
                   if len(seg_zs) and len(pelv_zs) else ct.shape[2] // 2))
    elif sp is not None:
        k = int(np.where(seg.any(axis=(0, 1)))[0].mean())
    elif pv is not None:
        k = int(np.where(pelv.any(axis=(0, 1)))[0].mean())
    else:
        k = ct.shape[2] // 2

    return (int(np.clip(i, 0, ct.shape[0] - 1)),
            int(np.clip(j, 0, ct.shape[1] - 1)),
            int(np.clip(k, 0, ct.shape[2] - 1)))


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """Orient a 2D slice for radiological display assuming PIR source.

    PIR: axis 0 = P (A->P as idx++), axis 1 = I (S->I as idx++), axis 2 = R (L->R as idx++).

    dim=0 (coronal):  slice shape (I, R). imshow default: row 0 top = I=0 = superior. OK as-is.
    dim=1 (axial):    slice shape (P, R). imshow default: row 0 top = P=0 = anterior. OK as-is.
    dim=2 (sagittal): slice shape (P, I). We want row=I (head up), col=P. Transpose.
    """
    return arr2d.T if dim == 2 else arr2d


def _display_shape(shape_3d: Tuple[int, ...], dim: int) -> Tuple[int, int]:
    raw = tuple(s for d, s in enumerate(shape_3d) if d != dim)
    return raw[::-1] if dim == 2 else raw


def _grey_panel(shape_2d):
    return np.full((*shape_2d, 3), 0.12, dtype=np.float32)


def _safe_slice(arr: np.ndarray, dim: int, idx: int) -> np.ndarray:
    clamped = int(np.clip(idx, 0, arr.shape[dim] - 1))
    s = [slice(None)] * arr.ndim
    s[dim] = clamped
    return arr[tuple(s)]


# ===========================================================================
# Placement compatibility metrics
# ===========================================================================

def _compute_placement_metrics(ct, mask, mask_type="pelvic"):
    mn   = tuple(min(a, b) for a, b in zip(ct.shape, mask.shape))
    ct_c = ct  [:mn[0], :mn[1], :mn[2]]
    mk_c = mask[:mn[0], :mn[1], :mn[2]]

    total_vox   = int((mask  > 0).sum())
    clipped_vox = int((mk_c > 0).sum())
    fov_pct     = clipped_vox / max(1, total_vox) * 100.0

    mask_bool = (mk_c > 0)
    if mask_bool.any():
        hu_vals  = ct_c[mask_bool].astype(np.float32)
        mean_hu  = float(hu_vals.mean())
        bone_pct = float((hu_vals > _BONE_HU).sum()) / max(1, int(mask_bool.sum())) * 100.0
    else:
        mean_hu = bone_pct = 0.0

    z_pct = (float(np.where(mask_bool)[2].mean()) / max(1, ct_c.shape[2] - 1) * 100.0
             if mask_bool.any() else 50.0)

    bone_warn = _PELVIC_BONE_WARN if mask_type == "pelvic" else _SPINE_BONE_WARN
    bone_fail = _PELVIC_BONE_FAIL if mask_type == "pelvic" else _SPINE_BONE_FAIL

    issues = []
    if total_vox < 100:
        issues.append(f"TOO_FEW_VOX({total_vox})")
    if fov_pct < _FOV_OVERLAP_FAIL:
        issues.append(f"FOV_OVERLAP_LOW({fov_pct:.0f}%)")
    if bone_pct < bone_fail:
        issues.append(f"BONE_FAIL({bone_pct:.0f}%)")
    elif bone_pct < bone_warn:
        issues.append(f"BONE_LOW({bone_pct:.0f}%)")

    if any("FAIL" in i or "TOO_FEW" in i for i in issues):
        overall = "FAIL"
    elif issues:
        overall = "WARNING"
    else:
        overall = "OK"

    return {
        "mask_vox":          total_vox,
        "fov_overlap_pct":   round(fov_pct,  1),
        "bone_coverage_pct": round(bone_pct, 1),
        "mean_hu_in_mask":   round(mean_hu,  0),
        "z_position_pct":    round(z_pct,    1),
        "issues":            issues,
        "overall":           overall,
    }


def _render_metrics_panel(ax_row, spine_m, pelvic_m, token, match_type,
                           image_source="", seg_source="",
                           manifest_spine=None, manifest_pelvic=None):
    def _text_block(ax, lines, bg="#111111"):
        ax.set_facecolor(bg)
        ax.axis("off")
        y = 0.97
        for text, size, color in lines:
            ax.text(0.06, y, text, transform=ax.transAxes,
                    fontsize=size, color=color, va="top",
                    fontfamily="monospace", clip_on=True)
            y -= max(0.11, 0.95 / max(len(lines), 1))

    isrc_short = (image_source or "?")[:14]
    lines0 = [
        (f"TOKEN:  {token}", 8, "#ffffff"),
        (f"TYPE:   {match_type}", 7, "#aaaaaa"),
        (f"IMG:    {isrc_short}", 6, "#888888"),
    ]

    if manifest_spine:
        method_short = (manifest_spine.get("method","") or "")[:20]
        uid_short    = (manifest_spine.get("series_uid","") or "")[-12:]
        lines0 += [
            ("-- SPINE (manifest) ---", 6, "#555555"),
            (f"bone%:  {manifest_spine.get('bone_pct','?'):>5}  "
             f"IS_ok={manifest_spine.get('IS_ok','?')}", 7, "#bbbbbb"),
            (f"method: {method_short}", 6, "#888888"),
            (f"uid:    ...{uid_short}", 6, "#666666"),
        ]
    elif spine_m:
        oc = _COMPAT_COLORS[spine_m["overall"]]
        lines0 += [
            ("-- SPINE SEG ----------", 6, "#555555"),
            (f"Voxels:  {spine_m['mask_vox']:>8,}",        7, "#cccccc"),
            (f"FOV in:  {spine_m['fov_overlap_pct']:>6.0f} %", 7, "#cccccc"),
            (f"Bone:    {spine_m['bone_coverage_pct']:>6.0f} %", 7, "#cccccc"),
            (f"Mean HU: {spine_m['mean_hu_in_mask']:>6.0f}",   7, "#cccccc"),
            (f"Status:  {spine_m['overall']}", 8, oc),
        ]
    else:
        lines0.append(("-- SPINE  (none) ------", 6, "#444444"))

    if manifest_spine and spine_m:
        oc = _COMPAT_COLORS[spine_m["overall"]]
        lines0 += [
            ("-- SPINE (computed) ---", 6, "#444444"),
            (f"Bone:    {spine_m['bone_coverage_pct']:>6.0f} %  {spine_m['overall']}",
             6, oc),
        ]
    _text_block(ax_row[0], lines0)

    if manifest_pelvic:
        uid_short = (manifest_pelvic.get("series_uid","") or "")[-12:]
        lines1 = [
            ("-- PELVIC (manifest) --", 6, "#555555"),
            (f"bone%:  {manifest_pelvic.get('bone_pct','?'):>5}", 7, "#bbbbbb"),
            (f"uid:    ...{uid_short}", 6, "#666666"),
        ]
        if pelvic_m:
            oc = _COMPAT_COLORS[pelvic_m["overall"]]
            lines1 += [
                ("-- PELVIC (computed) --", 6, "#444444"),
                (f"Voxels:  {pelvic_m['mask_vox']:>8,}", 7, "#cccccc"),
                (f"FOV in:  {pelvic_m['fov_overlap_pct']:>6.0f} %", 7, "#cccccc"),
                (f"Bone:    {pelvic_m['bone_coverage_pct']:>6.0f} %", 7, "#cccccc"),
                (f"Mean HU: {pelvic_m['mean_hu_in_mask']:>6.0f}", 7, "#cccccc"),
                (f"Z-pos:   {pelvic_m['z_position_pct']:>6.0f} %", 7, "#cccccc"),
                (f"Status:  {pelvic_m['overall']}", 8, oc),
            ]
            for issue in pelvic_m.get("issues", []):
                lines1.append((f"  ! {issue}", 6, "#ff8800"))
    elif pelvic_m:
        oc = _COMPAT_COLORS[pelvic_m["overall"]]
        lines1 = [
            ("-- PELVIC MASK --------", 6, "#555555"),
            (f"Voxels:  {pelvic_m['mask_vox']:>8,}",        7, "#cccccc"),
            (f"FOV in:  {pelvic_m['fov_overlap_pct']:>6.0f} %", 7, "#cccccc"),
            (f"Bone:    {pelvic_m['bone_coverage_pct']:>6.0f} %", 7, "#cccccc"),
            (f"Mean HU: {pelvic_m['mean_hu_in_mask']:>6.0f}",   7, "#cccccc"),
            (f"Z-pos:   {pelvic_m['z_position_pct']:>6.0f} %",  7, "#cccccc"),
            (f"Status:  {pelvic_m['overall']}", 8, oc),
        ]
        for issue in pelvic_m.get("issues", []):
            lines1.append((f"  ! {issue}", 6, "#ff8800"))
    else:
        lines1 = [("-- PELVIC MASK --------", 6, "#555555"),
                  ("  (none)", 7, "#444444")]
    _text_block(ax_row[1], lines1)

    both = [m for m in [spine_m, pelvic_m] if m]
    if both:
        worst = "OK"
        for m in both:
            if m["overall"] == "FAIL": worst = "FAIL"
            elif m["overall"] == "WARNING" and worst != "FAIL": worst = "WARNING"
        vc   = _COMPAT_COLORS[worst]
        icon = "OK" if worst == "OK" else "WARN" if worst == "WARNING" else "FAIL"
        ax_row[2].set_facecolor(vc + "18")
        ax_row[2].axis("off")
        ax_row[2].text(0.5, 0.62, f"{icon}",
                       transform=ax_row[2].transAxes,
                       fontsize=13, color=vc, va="center", ha="center",
                       fontfamily="monospace", fontweight="bold")
        all_issues = []
        for m in both:
            all_issues.extend(m.get("issues", []))
        if all_issues:
            ax_row[2].text(0.5, 0.30, "\n".join(all_issues[:3]),
                           transform=ax_row[2].transAxes,
                           fontsize=6, color="#ff9900", va="center", ha="center",
                           fontfamily="monospace")
    else:
        ax_row[2].axis("off")


def _add_metrics_overlay(ax, metrics, row):
    if row != 1:
        return
    bone    = metrics.get("bone_coverage_pct", 0)
    overall = metrics.get("overall", "OK")
    color   = _COMPAT_COLORS[overall]
    icon    = "OK" if overall == "OK" else "WARN" if overall == "WARNING" else "FAIL"
    ax.text(0.02, 0.04, f"{icon} bone={bone:.0f}%",
            transform=ax.transAxes, fontsize=7, color=color,
            va="bottom", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111111", alpha=0.75,
                      ec=color, lw=0.8))


# ===========================================================================
# Section filler: 3 rows x 3 cols
# ===========================================================================

def _fill_section(axes, col_offset, spine_ct_path, spine_seg_path,
                  pelvic_ct_path, pelvic_mask_path, same_space, token,
                  show_metrics=False):
    has_spine = has_pelvic = False
    ct_sp = seg_arr = ct_pv = pelv_arr = None
    spine_metrics = pelvic_metrics = None

    if spine_ct_path and spine_ct_path.exists() and spine_seg_path and spine_seg_path.exists():
        try:
            ct_sp, _  = _load(spine_ct_path)
            seg_arr, _ = _load(spine_seg_path)
            seg_arr    = seg_arr.astype(np.int32)
            mn = tuple(min(a, b) for a, b in zip(ct_sp.shape, seg_arr.shape))
            ct_sp = ct_sp[:mn[0],:mn[1],:mn[2]]
            seg_arr = seg_arr[:mn[0],:mn[1],:mn[2]]
            has_spine = True
            if show_metrics:
                spine_metrics = _compute_placement_metrics(ct_sp, seg_arr, "spine")
        except Exception as e:
            log.warning("token=%s spine load: %s", token, e)

    if pelvic_ct_path and pelvic_ct_path.exists() and pelvic_mask_path and pelvic_mask_path.exists():
        try:
            ct_pv, _   = _load(pelvic_ct_path)
            pelv_arr, _ = _load(pelvic_mask_path)
            pelv_arr = pelv_arr.astype(np.int32)
            mn = tuple(min(a, b) for a, b in zip(ct_pv.shape, pelv_arr.shape))
            ct_pv    = ct_pv[:mn[0],:mn[1],:mn[2]]
            pelv_arr = pelv_arr[:mn[0],:mn[1],:mn[2]]
            has_pelvic = True
            if show_metrics:
                pelvic_metrics = _compute_placement_metrics(ct_pv, pelv_arr, "pelvic")
        except Exception as e:
            log.warning("token=%s pelvic load: %s", token, e)

    if not has_spine and not has_pelvic:
        log.warning("token=%s: nothing to display", token)
        return

    ref_ct = ct_sp if has_spine else ct_pv

    i_cor, j_ax, k_sag = _choose_slices(
        ref_ct,
        seg_arr  if has_spine  else None,
        pelv_arr if (has_pelvic and same_space) else None,
    )
    plane_names = [f"Coronal i={i_cor}", f"Axial j={j_ax}", f"Sagittal k={k_sag}"]

    if has_pelvic and not same_space:
        pi_cor, pj_ax, pk_sag = _choose_slices(ct_pv, None, pelv_arr)
    else:
        pi_cor, pj_ax, pk_sag = i_cor, j_ax, k_sag
    pelvic_plane_idx = {0: pi_cor, 1: pj_ax, 2: pk_sag}

    for row, (dim, idx) in enumerate([(0, i_cor), (1, j_ax), (2, k_sag)]):
        pv_idx = pelvic_plane_idx[row]

        ref_shape_2d = _display_shape(ref_ct.shape, dim)

        ax = axes[row, col_offset]
        if has_spine:
            ct_2d  = _display_slice(_safe_slice(ct_sp, dim, idx), dim)
            seg_2d = _display_slice(_safe_slice(seg_arr, dim, idx), dim)
            rgb    = _overlay(_window(ct_2d), seg_2d, _SPINE_COLORS)
        else:
            rgb = _grey_panel(ref_shape_2d)
            if row == 1:
                ax.text(0.5, 0.5, "No Spine Scan", transform=ax.transAxes,
                        color="#888888", fontsize=9, fontweight="bold",
                        ha="center", va="center")
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.axis("off")

        ax = axes[row, col_offset + 1]
        pelvic_slice_empty = has_pelvic and not (_safe_slice(pelv_arr, dim, pv_idx) > 0).any()
        if has_pelvic:
            ct_2d   = _display_slice(_safe_slice(ct_pv, dim, pv_idx), dim)
            pelv_2d = _display_slice(_safe_slice(pelv_arr, dim, pv_idx), dim)
            rgb     = _overlay(_window(ct_2d), pelv_2d, _PELVIC_COLORS)
        else:
            rgb = _grey_panel(ref_shape_2d)
            if row == 1:
                ax.text(0.5, 0.5, "No Pelvic Scan", transform=ax.transAxes,
                        color="#888888", fontsize=9, fontweight="bold",
                        ha="center", va="center")
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        if pelvic_slice_empty and row == 1:
            ax.text(0.5, 0.5, "MASK OUTSIDE SLICE\n(pelvis below FOV)",
                    transform=ax.transAxes, color="#ff6600", fontsize=7,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="#111111", alpha=0.7))
        ax.axis("off")

        ax = axes[row, col_offset + 2]
        if has_spine:
            ct_2d  = _display_slice(_safe_slice(ct_sp, dim, idx), dim)
            seg_2d = _display_slice(_safe_slice(seg_arr, dim, idx), dim)
            rgb    = _overlay(_window(ct_2d), seg_2d, _SPINE_COLORS)
            if has_pelvic and same_space:
                pelv_2d = _display_slice(_safe_slice(pelv_arr, dim, pv_idx), dim)
                rgb     = _overlay(rgb, pelv_2d, _PELVIC_COLORS)
        elif has_pelvic:
            ct_2d   = _display_slice(_safe_slice(ct_pv, dim, pv_idx), dim)
            pelv_2d = _display_slice(_safe_slice(pelv_arr, dim, pv_idx), dim)
            rgb     = _overlay(_window(ct_2d), pelv_2d, _PELVIC_COLORS)
        else:
            rgb = _grey_panel(ref_shape_2d)
        ax.imshow(rgb, aspect="auto", interpolation="nearest")

        if show_metrics and pelvic_metrics and same_space:
            _add_metrics_overlay(ax, pelvic_metrics, row)

        if (has_pelvic and same_space
                and not (_safe_slice(pelv_arr, dim, pv_idx) > 0).any()
                and row == 1):
            ax.text(0.5, 0.97, "! pelvic mask outside crop window",
                    transform=ax.transAxes, color="#ff6600", fontsize=6,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round", fc="#111111", alpha=0.6))
        ax.axis("off")

        if col_offset == 0:
            axes[row, 0].text(-0.05, 0.5, plane_names[row],
                              transform=axes[row, 0].transAxes,
                              fontsize=7, color="#aaaaaa",
                              rotation=90, va="center", ha="right")


# ===========================================================================
# Debug section filler: 3 rows x 4 cols
# ===========================================================================

def _fill_debug_section(axes, col_offset, ct, spine_seg, pelvic_mask,
                        spine_metrics, pelvic_metrics, token, match_type,
                        image_source, seg_source,
                        manifest_spine=None, manifest_pelvic=None,
                        pelvic_ct=None):
    is_separate = pelvic_ct is not None
    pv_ct = pelvic_ct if is_separate else ct

    i_cor, j_ax, k_sag = _choose_slices(ct, spine_seg,
                                         pelvic_mask if not is_separate else None)
    if is_separate and pelvic_mask is not None:
        _, pj_ax, pk_sag = _choose_slices(pv_ct, None, pelvic_mask)
    else:
        pj_ax, pk_sag = j_ax, k_sag

    plane_names = [f"Coronal i={i_cor}", f"Axial j={j_ax}", f"Sagittal k={k_sag}"]

    for row, (dim, idx) in enumerate([(0, i_cor), (1, j_ax), (2, k_sag)]):
        pv_idx = {0: i_cor, 1: pj_ax, 2: pk_sag}[row]

        ct_win = _window(_display_slice(_safe_slice(ct, dim, idx), dim))

        ax = axes[row, col_offset]
        ax.imshow(np.stack([ct_win, ct_win, ct_win], axis=-1),
                  aspect="auto", interpolation="nearest")
        ax.axis("off")
        if row == 0:
            ax.set_title("Raw CT (spine)", fontsize=8, color="#cccccc", pad=2)
        if col_offset == 0:
            ax.text(-0.05, 0.5, plane_names[row], transform=ax.transAxes,
                    fontsize=7, color="#aaaaaa", rotation=90, va="center", ha="right")

        ax = axes[row, col_offset + 1]
        if spine_seg is not None:
            mn = tuple(min(a, b) for a, b in zip(ct.shape, spine_seg.shape))
            ct_c  = ct[:mn[0],:mn[1],:mn[2]]
            seg_c = spine_seg[:mn[0],:mn[1],:mn[2]]
            ct_2d  = _display_slice(_safe_slice(ct_c, dim, idx), dim)
            seg_2d = _display_slice(_safe_slice(seg_c, dim, idx), dim)
            rgb    = _overlay(_window(ct_2d), seg_2d, _SPINE_COLORS)
        else:
            rgb = np.stack([ct_win, ct_win, ct_win], axis=-1)
            if row == 1:
                ax.text(0.5, 0.5, "No Spine Seg", transform=ax.transAxes,
                        color="#666666", fontsize=8, ha="center", va="center")
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.axis("off")
        if row == 0:
            ax.set_title("CT + Spine Seg", fontsize=8, color="#cccccc", pad=2)

        ax = axes[row, col_offset + 2]
        if pelvic_mask is not None:
            mn   = tuple(min(a, b) for a, b in zip(pv_ct.shape, pelvic_mask.shape))
            pvc  = pv_ct[:mn[0],:mn[1],:mn[2]]
            pkc  = pelvic_mask[:mn[0],:mn[1],:mn[2]]
            ct_2d   = _display_slice(_safe_slice(pvc, dim, pv_idx), dim)
            pelv_2d = _display_slice(_safe_slice(pkc, dim, pv_idx), dim)
            rgb     = _overlay(_window(ct_2d), pelv_2d, _PELVIC_COLORS)
        else:
            pv_win = _window(_display_slice(_safe_slice(pv_ct, dim, pv_idx), dim))
            rgb    = np.stack([pv_win, pv_win, pv_win], axis=-1)
            if row == 1:
                ax.text(0.5, 0.5, "No Pelvic Mask", transform=ax.transAxes,
                        color="#666666", fontsize=8, ha="center", va="center")
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.axis("off")
        if row == 0:
            series_label = "separate series" if is_separate else "same CT -> conflict check"
            ax.set_title(f"Pelvic CT + Mask\n({series_label})",
                         fontsize=7, color="#cccccc", pad=2)
        if is_separate and row == 1:
            ax.text(0.5, 0.97, "different series (separate case)",
                    transform=ax.transAxes, color="#88aaff", fontsize=6,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round", fc="#111111", alpha=0.7))

        if pelvic_metrics and row == 1:
            _add_metrics_overlay(ax, pelvic_metrics, row)

        if (not is_separate and pelvic_mask is not None and pelvic_metrics
                and pelvic_metrics["overall"] in ("WARNING","FAIL") and row == 1):
            bc    = pelvic_metrics["bone_coverage_pct"]
            sc    = _COMPAT_COLORS[pelvic_metrics["overall"]]
            label = ("SERIES CONFLICT?" if pelvic_metrics["overall"] == "FAIL"
                     else "CHECK PLACEMENT")
            ax.text(0.5, 0.97, f"{label}  bone={bc:.0f}%",
                    transform=ax.transAxes, color=sc, fontsize=6,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round", fc="#111111", alpha=0.8, ec=sc, lw=0.8))

        ax = axes[row, col_offset + 3]
        if row == 0:
            ax.set_title("Compatibility", fontsize=8, color="#cccccc", pad=2)

    _render_metrics_panel(
        ax_row          = [axes[r, col_offset + 3] for r in range(3)],
        spine_m         = spine_metrics,
        pelvic_m        = pelvic_metrics,
        token           = token,
        match_type      = match_type,
        image_source    = image_source,
        seg_source      = seg_source,
        manifest_spine  = manifest_spine,
        manifest_pelvic = manifest_pelvic,
    )


# ===========================================================================
# Per-case figure
# ===========================================================================

def make_case_figure(case, source_type, out_path, debug_layout=False):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        log.error("matplotlib required"); return False

    token       = case.get("token", "?")
    source_type = case.get("match_type", source_type)
    same_space  = case.get("same_space", source_type == "fused")

    if not debug_layout:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9),
                                  gridspec_kw={"hspace": 0.04, "wspace": 0.04})
        fig.patch.set_facecolor("#111111")
        for ax in axes.flat: ax.set_facecolor("#111111"); ax.axis("off")
        for ci, title in enumerate(["Spine CT + labels", "Pelvic CT + mask", "Fused overlay"]):
            axes[0, ci].set_title(title, fontsize=9, color="#cccccc", pad=3)

        _fill_section(axes=axes, col_offset=0,
                      spine_ct_path=case.get("spine_ct"),
                      spine_seg_path=case.get("spine_seg"),
                      pelvic_ct_path=case.get("pelvic_ct"),
                      pelvic_mask_path=case.get("pelvic_mask"),
                      same_space=same_space, token=token)

        patches = []
        for v, name in _SPINE_LABELS.items():
            r, g, b, _ = _SPINE_COLORS[v]
            patches.append(mpatches.Patch(facecolor=(r,g,b), label=name))
        for v, name in _PELVIC_LABELS.items():
            r, g, b, _ = _PELVIC_COLORS[v]
            patches.append(mpatches.Patch(facecolor=(r,g,b), label=name))
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   fontsize=7, bbox_to_anchor=(0.5, 0.0),
                   facecolor="#222222", labelcolor="#dddddd", edgecolor="#444444")

        mask_vx = case.get("mask_voxels", -1)
        if source_type in ("fused", "separate"):
            if mask_vx == 0:
                vx_str, ttl_color = "  ! PLACED MASK HAS 0 VOXELS", "#ff6600"
            elif mask_vx == -1 and case.get("pelvic_mask") is None:
                vx_str, ttl_color = "  ! PLACED MASK FILE MISSING", "#ff3333"
            else:
                vx_str    = f"  ({mask_vx:,} pelvic voxels placed)" if mask_vx > 0 else ""
                ttl_color = "#ffffff"
        else:
            vx_str, ttl_color = "", "#ffffff"

        type_display = source_type
        if source_type == "separate":
            type_display = "separate (different series)"

        if case.get("is_fail"):
            is_fail_lbl = case.get("is_fail_labels", [])
            lbl_str     = f"  labels={is_fail_lbl}" if is_fail_lbl else ""
            fig.suptitle(
                f"IS_ORDER_FAIL -- Token {token}  [{type_display}]{lbl_str}{vx_str}",
                fontsize=11, y=1.002, color="#ff4444")
            for ax in axes.flat:
                for spine_side in ax.spines.values():
                    spine_side.set_edgecolor("#ff4444")
                    spine_side.set_linewidth(1.5)
                    spine_side.set_visible(True)
        else:
            fig.suptitle(f"Token {token}  [{type_display}]{vx_str}",
                         fontsize=11, y=1.002, color=ttl_color)

    else:
        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor("#111111")
        gs   = GridSpec(3, 4, figure=fig, hspace=0.06, wspace=0.05,
                        width_ratios=[1, 1, 1, 1.15])
        axes = np.empty((3, 4), dtype=object)
        for r in range(3):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gs[r, c])
                axes[r, c].set_facecolor("#111111")
                axes[r, c].axis("off")

        ct = spine_seg = pelvic_mask_arr = None
        spine_m = pelvic_m = None
        image_source = case.get("image_source", "dcm2niix")
        seg_source   = case.get("seg_source",   "")

        ct_path = case.get("spine_ct") or case.get("pelvic_ct")
        if ct_path and Path(ct_path).exists():
            try:
                ct, _ = _load(Path(ct_path))
            except Exception as e:
                log.warning("token=%s CT load: %s", token, e)

        if ct is None:
            for ax in axes.flat: ax.axis("off")
            axes[1,1].text(0.5, 0.5, f"CT LOAD FAILED\ntoken={token}",
                           transform=axes[1,1].transAxes,
                           color="#ff4444", fontsize=10, ha="center", va="center")
            fig.suptitle(f"Token {token}  [{source_type}]  CT LOAD FAILED",
                         fontsize=11, y=1.002, color="#ff4444")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            return False

        seg_path = case.get("spine_seg")
        if seg_path and Path(seg_path).exists():
            try:
                spine_seg_raw, _ = _load(Path(seg_path))
                spine_seg = spine_seg_raw.astype(np.int32)
                mn = tuple(min(a,b) for a,b in zip(ct.shape, spine_seg.shape))
                ct        = ct[:mn[0],:mn[1],:mn[2]]
                spine_seg = spine_seg[:mn[0],:mn[1],:mn[2]]
                spine_m   = _compute_placement_metrics(ct, spine_seg, "spine")
            except Exception as e:
                log.warning("token=%s spine seg: %s", token, e)

        pelvic_ct_arr = None
        pm_path = case.get("pelvic_mask")
        if pm_path and Path(pm_path).exists():
            try:
                pm_raw, _ = _load(Path(pm_path))
                pelvic_mask_arr = pm_raw.astype(np.int32)

                if (not same_space
                        and case.get("pelvic_ct")
                        and case.get("pelvic_ct") != ct_path):
                    pv_ct_path = case.get("pelvic_ct")
                    if Path(pv_ct_path).exists():
                        try:
                            pelvic_ct_arr, _ = _load(Path(pv_ct_path))
                        except Exception as e:
                            log.warning("token=%s pelvic CT load: %s", token, e)

                ref_for_clip = pelvic_ct_arr if pelvic_ct_arr is not None else ct
                mn = tuple(min(a,b) for a,b in zip(ref_for_clip.shape,
                                                     pelvic_mask_arr.shape))
                pelvic_mask_arr = pelvic_mask_arr[:mn[0],:mn[1],:mn[2]]
                if pelvic_ct_arr is not None:
                    pelvic_ct_arr = pelvic_ct_arr[:mn[0],:mn[1],:mn[2]]
                else:
                    ct = ct[:mn[0],:mn[1],:mn[2]]
                    if spine_seg is not None:
                        spine_seg = spine_seg[:mn[0],:mn[1],:mn[2]]

                ref_for_metrics = pelvic_ct_arr if pelvic_ct_arr is not None else ct
                pelvic_m = _compute_placement_metrics(ref_for_metrics,
                                                       pelvic_mask_arr, "pelvic")
            except Exception as e:
                log.warning("token=%s pelvic mask: %s", token, e)

        if pelvic_m and pelvic_m["overall"] == "FAIL":
            log.error("token=%s  SERIES_CONFLICT  bone=%.0f%%  fov=%.0f%%",
                      token, pelvic_m["bone_coverage_pct"], pelvic_m["fov_overlap_pct"])
        elif pelvic_m and pelvic_m["overall"] == "WARNING":
            log.warning("token=%s  LOW_BONE  bone=%.0f%%  issues=%s",
                        token, pelvic_m["bone_coverage_pct"], pelvic_m["issues"])

        _fill_debug_section(
            axes=axes, col_offset=0, ct=ct,
            spine_seg=spine_seg, pelvic_mask=pelvic_mask_arr,
            spine_metrics=spine_m, pelvic_metrics=pelvic_m,
            token=token, match_type=source_type,
            image_source=image_source, seg_source=seg_source,
            manifest_spine   = case.get("manifest_spine"),
            manifest_pelvic  = case.get("manifest_pelvic"),
            pelvic_ct        = pelvic_ct_arr,
        )

        worst = "OK"
        for m in [spine_m, pelvic_m]:
            if m and m["overall"] == "FAIL": worst = "FAIL"
            elif m and m["overall"] == "WARNING" and worst != "FAIL": worst = "WARNING"

        if case.get("is_fail"):
            is_fail_lbl = case.get("is_fail_labels", [])
            lbl_str     = f"  labels={is_fail_lbl}" if is_fail_lbl else ""
            fig.suptitle(
                f"IS_ORDER_FAIL -- Token {token}  [{source_type}]{lbl_str}",
                fontsize=10, y=1.002, color="#ff4444")
        else:
            ttl_color = _COMPAT_COLORS[worst]
            icon      = "OK" if worst == "OK" else "WARN" if worst == "WARNING" else "FAIL"
            fig.suptitle(f"Token {token}  [{source_type}]  {icon}",
                         fontsize=10, y=1.002, color=ttl_color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ===========================================================================
# Case loading from placed_manifest.json
# ===========================================================================

def load_cases_from_manifest(
    manifest_json: Path,
    nifti_dir:     Path,
    n_each:        int,
    tokens_filter: Optional[Set[str]] = None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    manifest  = json.loads(Path(manifest_json).read_text())
    nifti_dir = Path(nifti_dir)

    fused_cases:    List[dict] = []
    separate_cases: List[dict] = []
    pelvic_cases:   List[dict] = []
    spine_cases:    List[dict] = []

    for case in manifest.get("cases", []):
        token      = str(case.get("patient_token", "?"))
        match_type = case.get("match_type", "spine_only")

        if tokens_filter and token not in tokens_filter:
            continue

        sp = case.get("spine")
        pv = case.get("pelvic")

        spine_ct    = (nifti_dir / f"{sp['series_uid']}.nii.gz") if sp else None
        pelvic_ct   = (nifti_dir / f"{pv['series_uid']}.nii.gz") if pv else None
        spine_seg   = Path(sp["placed"]) if sp and sp.get("placed") else None
        pelvic_mask = Path(pv["placed"]) if pv and pv.get("placed") else None

        same_space = bool(
            match_type == "fused" and sp and pv
            and sp.get("series_uid") == pv.get("series_uid")
        )

        mask_voxels = -1
        if pelvic_mask and pelvic_mask.exists():
            try:
                import nibabel as _nib
                mask_voxels = int(
                    (_nib.load(str(pelvic_mask)).get_fdata(dtype=float) > 0).sum()
                )
            except Exception:
                pass

        is_fail        = bool(sp and sp.get("IS_ok") is False)
        is_fail_labels = (sp.get("labels") or []) if sp else []

        case_dict = {
            "token":           token,
            "match_type":      match_type,
            "same_space":      same_space,
            "spine_ct":        spine_ct    if spine_ct    and spine_ct.exists()    else None,
            "spine_seg":       spine_seg   if spine_seg   and spine_seg.exists()   else None,
            "pelvic_ct":       pelvic_ct   if pelvic_ct   and pelvic_ct.exists()   else None,
            "pelvic_mask":     pelvic_mask if pelvic_mask and pelvic_mask.exists() else None,
            "mask_voxels":     mask_voxels,
            "image_source":    "dcm2niix",
            "seg_source":      sp.get("method","") if sp else "",
            "manifest_spine":  sp,
            "manifest_pelvic": pv,
            "is_fail":         is_fail,
            "is_fail_labels":  is_fail_labels,
        }

        if match_type == "fused":
            if n_each > 0 and len(fused_cases) >= n_each: continue
            fused_cases.append(case_dict)
        elif match_type == "separate":
            if n_each > 0 and len(separate_cases) >= n_each: continue
            separate_cases.append(case_dict)
        elif match_type == "spine_only":
            if n_each > 0 and len(spine_cases) >= n_each: continue
            spine_cases.append(case_dict)
        elif match_type == "pelvic_only":
            if n_each > 0 and len(pelvic_cases) >= n_each: continue
            pelvic_cases.append(case_dict)

    n_is_fail = sum(1 for c in fused_cases+separate_cases+spine_cases+pelvic_cases
                    if c.get("is_fail"))
    log.info(
        "Cases loaded: fused=%d  separate=%d  spine_only=%d  pelvic_only=%d  "
        "is_fail=%d%s",
        len(fused_cases), len(separate_cases), len(spine_cases), len(pelvic_cases),
        n_is_fail,
        f"  [token_filter={sorted(tokens_filter)}]" if tokens_filter else "",
    )
    return fused_cases, separate_cases, pelvic_cases, spine_cases


def _render_case_worker(args):
    case, source_type, out_path_str, debug_layout = args
    out_path = Path(out_path_str)
    if out_path.exists():
        return case["token"], True, "skip", ""
    try:
        ok = make_case_figure(case, source_type, out_path, debug_layout=debug_layout)
        return case["token"], ok, "ok" if ok else "fail", ""
    except Exception:
        import traceback
        return case["token"], False, "fail", traceback.format_exc()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--manifest",   required=True, type=Path)
    p.add_argument("--nifti_dir",  required=True, type=Path)
    p.add_argument("--placed_dir", required=True, type=Path)
    p.add_argument("--out_dir",    required=True, type=Path)
    p.add_argument("--n_each",     default=3, type=int)
    p.add_argument("--per_case",   action="store_true")
    p.add_argument("--only",       default=None,
                   choices=["fused","pelvic_only","spine_only","separate"])
    p.add_argument("--workers",    default=16, type=int)
    p.add_argument("--tokens",     default="", type=str,
                   help="Comma-separated patient tokens. Implies --per_case + debug layout.")
    p.add_argument("--debug",      action="store_true",
                   help="Force 4-col debug layout for all per-case figures.")
    args = p.parse_args()

    token_filter: Optional[Set[str]] = None
    if args.tokens.strip():
        token_filter = {t.strip() for t in args.tokens.split(",") if t.strip()}
        args.per_case = True
        log.info("Token filter: %d tokens -> %s", len(token_filter), sorted(token_filter))

    debug_layout = bool(args.debug or token_filter)
    n_load = 0 if args.per_case else args.n_each

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        return

    fused_cases, separate_cases, pelvic_cases, spine_cases = load_cases_from_manifest(
        args.manifest, args.nifti_dir, n_load, tokens_filter=token_filter,
    )

    if not fused_cases and not separate_cases and not pelvic_cases and not spine_cases:
        log.error("No cases found -- verify place_fused_masks.py has run.")
        return

    n_ok = n_skip = n_fail = 0

    if args.per_case:
        all_sources = [
            ("fused",       fused_cases,    "fused"),
            ("separate",    separate_cases, "separate"),
            ("spine_only",  spine_cases,    "spine_only"),
            ("pelvic_only", pelvic_cases,   "pelvic_only"),
        ]
        if args.only:
            all_sources = [(st, c, sd) for st, c, sd in all_sources if st == args.only]

        out_base = args.out_dir / ("debug" if debug_layout else "per_case")
        work: List[Tuple] = []

        for source_type, cases, subdir in all_sources:
            if not cases: continue
            for c in cases:
                suffix = "_debug" if debug_layout else ""
                fname  = f"token_{c['token']}_qc{suffix}.png"
                dest = (out_base / "is_fail") if c.get("is_fail") else (out_base / subdir)
                dest.mkdir(parents=True, exist_ok=True)
                work.append((c, source_type, str(dest / fname), debug_layout))

        n_is_fail_work = sum(1 for w in work if w[0].get("is_fail"))
        log.info("Per-case QC: %d figures  debug_layout=%s  workers=%d",
                 len(work), debug_layout, args.workers)
        log.info("  is_fail cases: %d -> %s/is_fail/", n_is_fail_work, out_base)

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_render_case_worker, w): w[0]["token"] for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    token, ok, msg, err = fut.result()
                except Exception as e:
                    n_fail += 1
                    log.warning("WORKER CRASH: %s", e)
                    continue
                if msg == "skip": n_skip += 1
                elif ok:          n_ok   += 1
                else:
                    n_fail += 1
                    log.warning("  [%d/%d]  FAIL  token=%s: %s",
                                i, len(work), token,
                                err.strip().splitlines()[-1] if err else "unknown")
                if i % 25 == 0 or i == len(work):
                    elapsed = time.time() - t0
                    eta     = elapsed / i * (len(work) - i) if i < len(work) else 0
                    log.info("  -- [%d/%d]  ok=%d  skip=%d  fail=%d  "
                             "elapsed=%.0fs  ETA=%.0fs --",
                             i, len(work), n_ok, n_skip, n_fail, elapsed, eta)

        log.info("%d rendered  %d skipped  %d failed -> %s",
                 n_ok, n_skip, n_fail, out_base)

    log.info("QC checklist:")
    log.info("  fused/:        sacrum below L5, hips bilateral, overlap <5%%")
    log.info("  pelvic_only/:  sacrum + hips present, no floating voxels")
    log.info("  spine_only/:   L1-L5/L6 labelled, no gaps or mislabels")
    log.info("  separate/:     spine and pelvic on different CTs -- check both panels")
    log.info("  is_fail/:      IS_ORDER_FAIL -- inspect for misplacement or prone case")


if __name__ == "__main__":
    main()
