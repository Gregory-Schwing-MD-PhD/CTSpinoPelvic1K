#!/usr/bin/env python3
"""
viz_pelvic_dimensions.py — publication-quality figure of pelvic ring axes.

Anatomical orientation
----------------------
This script follows the same PIR convention as scripts/visualize_qc.py:
all volumes are reoriented to PIR voxel space on load (axis 0 = P, A->P
as index increases; axis 1 = I, S->I as index increases; axis 2 = R,
L->R as index increases). All slicing, PCA, and display logic assumes
this canonical layout, so the figures come out in the standard
radiological orientation regardless of the source NIfTI's orientation
code.

Display orientation per slice axis (matches visualize_qc.py):
  - Coronal (slice axis 0):  shape (I, R). imshow as-is. Head up.
  - Axial   (slice axis 1):  shape (P, R). imshow as-is. Anterior up.
  - Sagittal (slice axis 2): shape (P, I). Transpose so I is vertical (head up).

PCA is computed in WORLD-MM space using the post-reorientation affine,
so eigenvector directions correspond cleanly to anatomical SI/AP/ML.

The script measures principal-axis dimensions of three pelvic-ring
structures — sacrum (label 7), left hip (8), right hip (9) — plus the
INTER-HIP mediolateral extent (the most-lateral-to-most-lateral pelvic
ring width, which is the binding constraint on in-plane patch size).

Two figures, one script. Both publication-ready (300 DPI):

  1. Single-case figure: three orthogonal views (axial / coronal /
     sagittal) through the sacrum centroid, with structure overlays
     and PCA axes.

  2. Cohort figure: 4-panel histogram (sacrum SI/AP/ML + inter-hip ML)
     with median/p90/p99 + patch-coverage % per --plans entry.

USAGE
-----
  python viz_pelvic_dimensions.py \\
      --token  77 --config fused --cohort \\
      --dataset_dir data/hf_export \\
      --plans nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_100G.json \\
      --out figures/pelvic_dimensions

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pelvic_dim")

# Unified label scheme (matches the rest of CTSpinoPelvic1K)
SACRUM_LABEL    = 7
LEFT_HIP_LABEL  = 8
RIGHT_HIP_LABEL = 9

PELVIC_LABELS: List[Tuple[int, str]] = [
    (SACRUM_LABEL,    "sacrum"),
    (LEFT_HIP_LABEL,  "left_hip"),
    (RIGHT_HIP_LABEL, "right_hip"),
]

AXIS_NAMES = ("craniocaudal", "anteroposterior", "mediolateral")
AXIS_COLORS = {
    "craniocaudal":    "#d62728",
    "anteroposterior": "#2ca02c",
    "mediolateral":    "#1f77b4",
}
AXIS_SHORT = {
    "craniocaudal":    "SI",
    "anteroposterior": "AP",
    "mediolateral":    "ML",
}
STRUCTURE_OVERLAY_COLOR = {
    "sacrum":    (1.0, 0.0, 1.0),
    "left_hip":  (0.0, 0.7, 1.0),
    "right_hip": (1.0, 0.7, 0.0),
}

# HU windowing (matches scripts/visualize_qc.py)
_HU_MIN, _HU_MAX = -200, 800


# =============================================================================
# Orientation: PIR canonicalization (mirrors visualize_qc.py)
# =============================================================================

def _load_pir(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI and reorient to PIR voxel space.

    Returns (data, post_reorientation_affine). The affine is updated
    so voxel-to-world coordinates remain correct after reorientation,
    which is critical for PCA-in-world-space.
    """
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff,
    )
    img      = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)
    data     = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff  = img.affine @ inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _load_pir_int(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data, aff = _load_pir(path)
    return data.astype(np.int16), aff


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """
    Orient a 2D slice for radiological display assuming PIR source.

      dim=0 (coronal):  slice (I, R). imshow row 0 top = I=0 = superior. OK.
      dim=1 (axial):    slice (P, R). imshow row 0 top = P=0 = anterior. OK.
      dim=2 (sagittal): slice (P, I). Transpose so I is vertical (head up).
    """
    return arr2d.T if dim == 2 else arr2d


def _hu_window(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr - _HU_MIN) / (_HU_MAX - _HU_MIN), 0.0, 1.0)


# =============================================================================
# PCA in world (mm) space
# =============================================================================

@dataclass
class StructurePCA:
    label_id:        int
    name:            str
    centroid_mm:     np.ndarray
    centroid_vox:    np.ndarray
    axes_world:      np.ndarray
    extents_mm:      Dict[str, float]
    n_voxels:        int
    voxel_volume_mm3: float

    def volume_mm3(self) -> float:
        return self.n_voxels * self.voxel_volume_mm3


def _voxel_to_world(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homog = np.concatenate([vox, np.ones((len(vox), 1))], axis=1)
    return (affine @ homog.T).T[:, :3]


def compute_structure_pca(label_vol: np.ndarray, affine: np.ndarray,
                           label_id: int, name: str,
                           min_voxels: int = 100) -> Optional[StructurePCA]:
    """PCA on a labeled structure's voxels in physical (mm) space."""
    sv = np.argwhere(label_vol == label_id)
    if len(sv) < min_voxels:
        return None

    world_pts = _voxel_to_world(sv, affine)
    centroid_mm  = world_pts.mean(axis=0)
    centroid_vox = sv.mean(axis=0)
    centered = world_pts - centroid_mm

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    eigvecs = vt.T

    projections = centered @ eigvecs
    extents = projections.max(axis=0) - projections.min(axis=0)

    order = np.argsort(extents)[::-1]
    eigvecs = eigvecs[:, order]
    extents = extents[order]
    extents_mm = {AXIS_NAMES[i]: float(extents[i]) for i in range(3)}

    vox_vol = float(np.abs(np.linalg.det(affine[:3, :3])))

    return StructurePCA(
        label_id=label_id, name=name,
        centroid_mm=centroid_mm, centroid_vox=centroid_vox,
        axes_world=eigvecs, extents_mm=extents_mm,
        n_voxels=len(sv), voxel_volume_mm3=vox_vol,
    )


def compute_inter_hip_extent(label_vol: np.ndarray,
                              affine: np.ndarray) -> Optional[float]:
    """Pelvic-ring ML width, lateral-most to lateral-most."""
    lh = np.argwhere(label_vol == LEFT_HIP_LABEL)
    rh = np.argwhere(label_vol == RIGHT_HIP_LABEL)
    if len(lh) == 0 or len(rh) == 0:
        return None
    lh_world = _voxel_to_world(lh, affine)
    rh_world = _voxel_to_world(rh, affine)
    # World X axis is left-right
    lh_xs = lh_world[:, 0]
    rh_xs = rh_world[:, 0]
    return float(max(lh_xs.max(), rh_xs.max()) -
                 min(lh_xs.min(), rh_xs.min()))


# =============================================================================
# PCA-axis projection into the displayed plane
# =============================================================================

def _world_eigvec_to_pir_voxel_dir(eigvec_world: np.ndarray,
                                    affine: np.ndarray) -> np.ndarray:
    """
    Project a world-space unit vector into PIR voxel-direction space by
    solving aff[:3,:3] @ vox_dir = eigvec_world. Uses the post-reorient
    affine so this corresponds correctly to the PIR axes in use.
    """
    vox_dir = np.linalg.solve(affine[:3, :3], eigvec_world)
    return vox_dir


def _draw_axes_overlay(ax, pca: StructurePCA, view_axis: int,
                        plane_axes_disp: Tuple[int, int],
                        spacing: np.ndarray,
                        affine: np.ndarray,
                        label_prefix: str = "") -> None:
    """
    Draw three PCA eigenvectors as line segments through the structure
    centroid in the current displayed plane. Segments labeled with axis
    name + length in mm.

    plane_axes_disp = (h_axis, v_axis) describing which PIR voxel axes
    map to the display's horizontal and vertical. For axial/coronal the
    natural slice axes are used directly; for sagittal a swap is applied
    (because we transpose the slice for head-up display).
    """
    h_ax, v_ax = plane_axes_disp

    # Centroid in mm along each PIR voxel axis (used for both display
    # axes regardless of transpose, since spacing is per-voxel-axis).
    cx_mm = pca.centroid_vox[h_ax] * spacing[h_ax]
    cy_mm = pca.centroid_vox[v_ax] * spacing[v_ax]

    for i, axis_name in enumerate(AXIS_NAMES):
        eigvec_world = pca.axes_world[:, i]
        length_mm    = pca.extents_mm[axis_name]
        # Convert world-space eigvec to PIR voxel-direction; then take
        # the components along the two PIR axes that span this slice.
        vox_dir = _world_eigvec_to_pir_voxel_dir(eigvec_world, affine)

        # Multiply by spacing to convert voxel-direction components into
        # mm-direction components in the displayed plane
        dx = vox_dir[h_ax] * spacing[h_ax] * (length_mm / 2.0) / max(
            np.linalg.norm(vox_dir * spacing), 1e-9)
        dy = vox_dir[v_ax] * spacing[v_ax] * (length_mm / 2.0) / max(
            np.linalg.norm(vox_dir * spacing), 1e-9)

        in_plane_len = np.hypot(dx, dy)
        out_of_plane_frac = 1.0 - (in_plane_len / max(length_mm / 2.0, 1e-6))

        color = AXIS_COLORS[axis_name]
        prefix = f"{label_prefix} " if label_prefix else ""
        if out_of_plane_frac > 0.85:
            ax.plot([cx_mm], [cy_mm], "o", color=color, markersize=9,
                    markerfacecolor="none", markeredgewidth=2.0,
                    label=f"{prefix}{AXIS_SHORT[axis_name]}={length_mm:.0f} mm (⊥)")
        else:
            ax.plot([cx_mm - dx, cx_mm + dx], [cy_mm - dy, cy_mm + dy],
                    color=color, linewidth=2.0,
                    label=f"{prefix}{AXIS_SHORT[axis_name]}={length_mm:.0f} mm")
            ax.plot([cx_mm - dx, cx_mm + dx], [cy_mm - dy, cy_mm + dy],
                    color=color, marker="|", markersize=7, linestyle="none")


def _draw_patch_box(ax, patch_mm: Tuple[float, float, float],
                     plane_axes_disp: Tuple[int, int],
                     centroid_xy: Tuple[float, float],
                     color: str, label: str) -> None:
    from matplotlib.patches import Rectangle
    h_ax, v_ax = plane_axes_disp
    w = patch_mm[h_ax]
    h = patch_mm[v_ax]
    cx, cy = centroid_xy
    rect = Rectangle((cx - w / 2, cy - h / 2), w, h,
                     linewidth=1.6, edgecolor=color,
                     facecolor="none", linestyle="--",
                     label=label)
    ax.add_patch(rect)


def _slice_extent_mm(vol_shape: Tuple[int, ...], view_axis: int,
                      spacing: np.ndarray) -> Tuple[Tuple[float, float, float, float], Tuple[int, int]]:
    """
    Compute imshow's `extent` in millimeters for a slice perpendicular
    to `view_axis`, plus the (h_ax, v_ax) tuple describing which PIR
    voxel axes correspond to the displayed horizontal and vertical
    after any required transpose.
    """
    h_ax, v_ax = [a for a in range(3) if a != view_axis]
    if view_axis == 2:
        # Sagittal: transpose puts I (axis 1) on rows, P (axis 0) on cols
        # so display h = P (h_ax), display v = I (v_ax) -- already (h_ax, v_ax)
        # since we picked the remaining two in order. Just return as-is.
        h_ax, v_ax = 0, 1   # P horizontal, I vertical
    x_mm = vol_shape[h_ax] * spacing[h_ax]
    y_mm = vol_shape[v_ax] * spacing[v_ax]
    return (0.0, x_mm, 0.0, y_mm), (h_ax, v_ax)


# =============================================================================
# Single-case figure
# =============================================================================

def render_single_case(ct: np.ndarray, label: np.ndarray, affine: np.ndarray,
                        pcas: Dict[str, StructurePCA],
                        inter_hip_mm: Optional[float],
                        out_path: Path,
                        case_id: str = "",
                        patch_specs: Optional[List[Dict]] = None) -> None:
    """
    Three orthogonal panels in PIR voxel space:
      Axial    = slice axis 1 (I); display (P, R), anterior up.
      Coronal  = slice axis 0 (P); display (I, R), head up.
      Sagittal = slice axis 2 (R); display (P, I) -> transposed to head up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    spacing = np.linalg.norm(affine[:3, :3], axis=0)

    # Slicing centroid: prefer sacrum
    if "sacrum" in pcas:
        slice_centroid = pcas["sacrum"].centroid_vox
    elif "left_hip" in pcas:
        slice_centroid = pcas["left_hip"].centroid_vox
    else:
        slice_centroid = np.array([s // 2 for s in ct.shape])
    p_idx, i_idx, r_idx = (int(round(c)) for c in slice_centroid)

    views = [
        ("Axial",    1, i_idx),
        ("Coronal",  0, p_idx),
        ("Sagittal", 2, r_idx),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), constrained_layout=False)

    for col, (view_name, view_axis, idx) in enumerate(views):
        ax = axes[col]

        # Pull and orient the CT and label slices
        idx = int(np.clip(idx, 0, ct.shape[view_axis] - 1))
        s = [slice(None)] * 3
        s[view_axis] = idx
        ct_raw  = ct[tuple(s)]
        lbl_raw = label[tuple(s)]
        ct_disp  = _display_slice(ct_raw, view_axis)
        lbl_disp = _display_slice(lbl_raw, view_axis)

        extent, plane_axes_disp = _slice_extent_mm(ct.shape, view_axis, spacing)

        ax.imshow(_hu_window(ct_disp), cmap="gray",
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")

        # Mask overlays
        for name, pca in pcas.items():
            mask = (lbl_disp == pca.label_id).astype(np.float32)
            if not mask.any():
                continue
            r, g, b = STRUCTURE_OVERLAY_COLOR[name]
            rgba = np.zeros((*mask.shape, 4))
            rgba[..., 0] = r
            rgba[..., 1] = g
            rgba[..., 2] = b
            rgba[..., 3] = mask * 0.30
            ax.imshow(rgba, extent=extent, origin="upper", aspect="equal",
                      interpolation="nearest")

        # PCA axes through each structure's centroid
        for name, pca in pcas.items():
            prefix = "" if name == "sacrum" else ("LH" if name == "left_hip" else "RH")
            _draw_axes_overlay(ax, pca, view_axis, plane_axes_disp,
                               spacing, affine, label_prefix=prefix)

        # Candidate patch boxes (centered on sacrum centroid)
        if patch_specs and "sacrum" in pcas:
            h_ax, v_ax = plane_axes_disp
            cx_mm = pcas["sacrum"].centroid_vox[h_ax] * spacing[h_ax]
            cy_mm = pcas["sacrum"].centroid_vox[v_ax] * spacing[v_ax]
            patch_palette = ["#ff7f00", "#984ea3", "#a65628", "#f781bf"]
            for i, ps in enumerate(patch_specs):
                _draw_patch_box(ax, ps["patch_mm"], plane_axes_disp,
                                (cx_mm, cy_mm),
                                color=patch_palette[i % len(patch_palette)],
                                label=f"{ps['label']} patch")

        # Anatomic orientation labels in the corners (same convention as
        # radiology workstations)
        def _orient_label(x, y, text, ha, va):
            ax.text(x, y, text, transform=ax.transAxes, color="white",
                    fontsize=10, fontweight="bold", va=va, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
        if view_name == "Axial":
            _orient_label(0.50, 0.98, "A", "center", "top")
            _orient_label(0.50, 0.02, "P", "center", "bottom")
            _orient_label(0.98, 0.50, "R", "right",  "center")
            _orient_label(0.02, 0.50, "L", "left",   "center")
        elif view_name == "Coronal":
            _orient_label(0.50, 0.98, "S", "center", "top")
            _orient_label(0.50, 0.02, "I", "center", "bottom")
            _orient_label(0.98, 0.50, "R", "right",  "center")
            _orient_label(0.02, 0.50, "L", "left",   "center")
        else:  # Sagittal — after transpose: rows=I (head up), cols=P
            _orient_label(0.50, 0.98, "S", "center", "top")
            _orient_label(0.50, 0.02, "I", "center", "bottom")
            _orient_label(0.98, 0.50, "P", "right",  "center")
            _orient_label(0.02, 0.50, "A", "left",   "center")

        ax.set_title(view_name, fontsize=12)
        ax.set_xlabel("mm", fontsize=9)
        if col == 0:
            ax.set_ylabel("mm", fontsize=9)
        ax.tick_params(labelsize=8)

    structure_handles = [
        Patch(facecolor=STRUCTURE_OVERLAY_COLOR[name], alpha=0.4,
              edgecolor="black", linewidth=0.5,
              label=name.replace("_", " "))
        for name in ("sacrum", "left_hip", "right_hip") if name in pcas
    ]
    axis_handles = [
        Line2D([0], [0], color=AXIS_COLORS[a], linewidth=2.2,
               label=AXIS_SHORT[a])
        for a in AXIS_NAMES
    ]
    patch_handles = []
    if patch_specs:
        patch_palette = ["#ff7f00", "#984ea3", "#a65628", "#f781bf"]
        for i, ps in enumerate(patch_specs):
            patch_handles.append(Line2D(
                [0], [0],
                color=patch_palette[i % len(patch_palette)],
                linestyle="--", linewidth=1.6,
                label=f"{ps['label']} "
                      f"({ps['patch_mm'][0]:.0f}×{ps['patch_mm'][1]:.0f}×{ps['patch_mm'][2]:.0f} mm)"))

    handles = structure_handles + axis_handles + patch_handles
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.02))

    title_lines = ["Pelvic principal axes"]
    if case_id:
        title_lines[0] += f"  ({case_id})"
    if "sacrum" in pcas:
        s = pcas["sacrum"].extents_mm
        title_lines.append(
            f"sacrum: SI={s['craniocaudal']:.0f}  "
            f"AP={s['anteroposterior']:.0f}  "
            f"ML={s['mediolateral']:.0f} mm")
    if inter_hip_mm is not None:
        title_lines.append(f"inter-hip ML extent = {inter_hip_mm:.0f} mm")
    fig.suptitle("    ".join(title_lines), fontsize=12, y=0.97)

    fig.subplots_adjust(top=0.91, bottom=0.18, left=0.05, right=0.98, wspace=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# =============================================================================
# Cohort figure (unchanged from previous version)
# =============================================================================

def render_cohort_figure(records: List[Dict], out_path: Path,
                          patch_specs: Optional[List[Dict]] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        log.error("No cohort records to plot.")
        return

    panels = []
    for axis_name in AXIS_NAMES:
        vals = np.array([
            r["sacrum"]["extents_mm"][axis_name]
            for r in records if r.get("sacrum")
        ])
        panels.append({
            "title":          f"sacrum {AXIS_SHORT[axis_name]}",
            "subtitle":       axis_name,
            "data":           vals,
            "color":          AXIS_COLORS[axis_name],
            "patch_axis_idx": {"craniocaudal": 0,
                               "anteroposterior": 1,
                               "mediolateral":    2}[axis_name],
        })
    inter_vals = np.array([
        r["inter_hip_ml_mm"] for r in records
        if r.get("inter_hip_ml_mm") is not None
    ])
    panels.append({
        "title":          "inter-hip ML",
        "subtitle":       "pelvic-ring width",
        "data":           inter_vals,
        "color":          "#9467bd",
        "patch_axis_idx": 2,
    })

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=False)

    for col, panel in enumerate(panels):
        ax = axes[col]
        data = panel["data"]
        if len(data) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(panel["title"])
            continue

        ax.hist(data, bins=30, color=panel["color"], alpha=0.55,
                edgecolor="black", linewidth=0.6)

        med = np.median(data)
        p90 = np.percentile(data, 90)
        p99 = np.percentile(data, 99)
        ax.axvline(med, color="black", linestyle="-",  linewidth=1.4,
                   label=f"median = {med:.0f} mm")
        ax.axvline(p90, color="black", linestyle="--", linewidth=1.2,
                   label=f"p90 = {p90:.0f} mm")
        ax.axvline(p99, color="black", linestyle=":",  linewidth=1.0,
                   label=f"p99 = {p99:.0f} mm")

        if patch_specs:
            for ps in patch_specs:
                p_mm = ps["patch_mm"][panel["patch_axis_idx"]]
                frac = float((data <= p_mm).mean())
                ax.axvline(p_mm, color=ps.get("color", "#ff7f00"),
                           linestyle="-.", linewidth=1.6,
                           label=f"{ps['label']}: {p_mm:.0f} mm "
                                 f"({frac*100:.0f}% cov.)")

        ax.set_xlabel("Extent (mm)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Cases", fontsize=10)
        ax.set_title(f"{panel['title']}\n({panel['subtitle']})", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    n = len(records)
    fig.suptitle(
        f"Pelvic principal-axis extents across the cohort  (N = {n})",
        fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.05, right=0.99, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# =============================================================================
# Plans loading / cohort iteration / main
# =============================================================================

def load_patch_specs(plans_paths: List[Path], config: str = "3d_fullres") -> List[Dict]:
    specs = []
    palette = ["#ff7f00", "#984ea3", "#a65628", "#f781bf", "#377eb8"]
    for i, p in enumerate(plans_paths):
        data = json.loads(p.read_text())
        cfg = data.get("configurations", {}).get(config, {})
        patch = cfg.get("patch_size")
        spacing = cfg.get("spacing") or data.get("original_median_spacing_after_transp")
        if patch is None or spacing is None:
            log.warning("Skipping %s: no patch_size or spacing", p.name)
            continue
        patch_mm = tuple(float(patch[k]) * float(spacing[k]) for k in range(3))
        stem = p.stem
        label = stem
        for marker in ("Plans_", "plans_"):
            if marker in stem:
                label = stem.split(marker, 1)[1]
                break
        specs.append({
            "label":    label,
            "patch_mm": patch_mm,
            "color":    palette[i % len(palette)],
        })
    return specs


def iter_label_paths(dataset_dir: Path) -> List[Tuple[Path, str]]:
    lbl_dir = dataset_dir / "labels"
    out = []
    for p in sorted(lbl_dir.glob("*_label.nii.gz")):
        stem = p.name.replace("_label.nii.gz", "")
        out.append((p, stem))
    return out


def compute_cohort_records(dataset_dir: Path,
                            limit: Optional[int] = None) -> List[Dict]:
    paths = iter_label_paths(dataset_dir)
    if limit:
        paths = paths[:limit]
    log.info("Computing pelvic PCA across %d label files (PIR-canonicalized)...",
             len(paths))
    records = []
    n_with_sacrum = n_with_both_hips = n_with_inter_hip = 0
    skipped = 0
    for i, (p, stem) in enumerate(paths, 1):
        try:
            arr, aff = _load_pir_int(p)
            rec: Dict = {"case_id": stem}
            for label_id, name in PELVIC_LABELS:
                pca = compute_structure_pca(arr, aff, label_id, name)
                rec[name] = None if pca is None else {
                    "extents_mm":  pca.extents_mm,
                    "volume_mm3":  pca.volume_mm3(),
                    "n_voxels":    pca.n_voxels,
                }
            inter = compute_inter_hip_extent(arr, aff)
            rec["inter_hip_ml_mm"] = inter
            if rec.get("sacrum"):                                     n_with_sacrum += 1
            if rec.get("left_hip") and rec.get("right_hip"):          n_with_both_hips += 1
            if inter is not None:                                     n_with_inter_hip += 1
            if not (rec.get("sacrum") or rec.get("left_hip") or rec.get("right_hip")):
                skipped += 1; continue
            records.append(rec)
        except Exception as e:
            log.warning("  [%d/%d] %s failed: %s", i, len(paths), stem, e)
            skipped += 1; continue
        if i % 50 == 0:
            log.info("  [%d/%d] computed", i, len(paths))
    log.info(
        "Cohort done: %d records  (sacrum=%d  both_hips=%d  inter_hip=%d  skipped=%d)",
        len(records), n_with_sacrum, n_with_both_hips, n_with_inter_hip, skipped)
    return records


def summarize_cohort(records: List[Dict]) -> None:
    log.info("Cohort summary:")
    for axis_name in AXIS_NAMES:
        vals = np.array([r["sacrum"]["extents_mm"][axis_name]
                         for r in records if r.get("sacrum")])
        if len(vals) == 0:
            continue
        log.info("  sacrum %-16s n=%4d  median=%5.1f  IQR=[%5.1f, %5.1f]  "
                 "p90=%5.1f  p99=%5.1f  max=%5.1f mm",
                 axis_name, len(vals),
                 float(np.median(vals)),
                 float(np.percentile(vals, 25)),
                 float(np.percentile(vals, 75)),
                 float(np.percentile(vals, 90)),
                 float(np.percentile(vals, 99)),
                 float(vals.max()))
    inter = np.array([r["inter_hip_ml_mm"] for r in records
                      if r.get("inter_hip_ml_mm") is not None])
    if len(inter) > 0:
        log.info("  inter-hip ML        n=%4d  median=%5.1f  IQR=[%5.1f, %5.1f]  "
                 "p90=%5.1f  p99=%5.1f  max=%5.1f mm",
                 len(inter),
                 float(np.median(inter)),
                 float(np.percentile(inter, 25)),
                 float(np.percentile(inter, 75)),
                 float(np.percentile(inter, 90)),
                 float(np.percentile(inter, 99)),
                 float(inter.max()))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pelvic principal-axis dimension figure (publication quality).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset_dir", required=True, type=Path)
    ap.add_argument("--token", default=None, type=str)
    ap.add_argument("--config", default="fused", type=str,
                    choices=["fused", "spine_only", "pelvic_native"])
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--cohort_limit", default=None, type=int)
    ap.add_argument("--cohort_cache", default=None, type=Path)
    ap.add_argument("--plans", action="append", default=[], type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out_path = args.out
    if out_path.suffix == "":
        single_out = out_path.with_suffix(".png")
        cohort_out = out_path.parent / (out_path.name + "_cohort.png")
    else:
        single_out = out_path
        cohort_out = out_path.with_name(out_path.stem + "_cohort.png")

    patch_specs: List[Dict] = []
    if args.plans:
        patch_specs = load_patch_specs(args.plans)
        log.info("Loaded %d patch specs from --plans", len(patch_specs))

    if args.token is not None:
        try:
            stem = f"{int(args.token):04d}_{args.config}"
        except ValueError:
            stem = f"{args.token}_{args.config}"
        ct_path  = args.dataset_dir / "ct"     / f"{stem}_ct.nii.gz"
        lbl_path = args.dataset_dir / "labels" / f"{stem}_label.nii.gz"
        if not ct_path.exists() or not lbl_path.exists():
            log.error("Case files not found: %s / %s", ct_path, lbl_path)
            return 1
        log.info("Loading case %s (PIR-canonicalized)", stem)
        ct,  ct_aff  = _load_pir(ct_path)
        lbl, lbl_aff = _load_pir_int(lbl_path)
        if not np.allclose(ct_aff, lbl_aff, atol=1e-3):
            log.warning("CT and label affines differ after PIR canonicalization. "
                        "Using label affine for geometry.")
        affine = lbl_aff

        # Trim to common shape (CT and label might differ by 1-2 voxels
        # at the edges due to source data). Keep the smaller bounding
        # box so all PIR voxel coordinates remain valid for both.
        mn = tuple(min(a, b) for a, b in zip(ct.shape, lbl.shape))
        ct  = ct [:mn[0], :mn[1], :mn[2]]
        lbl = lbl[:mn[0], :mn[1], :mn[2]]

        pcas: Dict[str, StructurePCA] = {}
        for label_id, name in PELVIC_LABELS:
            pca = compute_structure_pca(lbl, affine, label_id, name)
            if pca is not None:
                pcas[name] = pca
                e = pca.extents_mm
                log.info("  %-9s: SI=%5.1f  AP=%5.1f  ML=%5.1f  vol=%6.0f mm³",
                         name, e["craniocaudal"], e["anteroposterior"],
                         e["mediolateral"], pca.volume_mm3())
        if not pcas:
            log.error("No pelvic structures found in %s", stem)
            return 1
        inter = compute_inter_hip_extent(lbl, affine)
        if inter is not None:
            log.info("  inter-hip ML extent: %.1f mm", inter)
        render_single_case(ct, lbl, affine, pcas, inter, single_out,
                           case_id=stem, patch_specs=patch_specs)

    if args.cohort:
        cache = args.cohort_cache
        if cache is None:
            cache = cohort_out.with_suffix(".json")
        if cache.exists():
            log.info("Loading cached cohort records from %s", cache)
            records = json.loads(cache.read_text())
        else:
            records = compute_cohort_records(args.dataset_dir,
                                              limit=args.cohort_limit)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(records, indent=2))
            log.info("Cached %d cohort records to %s", len(records), cache)
        render_cohort_figure(records, cohort_out, patch_specs=patch_specs)
        summarize_cohort(records)

    if not args.token and not args.cohort:
        log.error("Provide --token (single-case) and/or --cohort.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
