#!/usr/bin/env python3
"""
viz_ts_case.py — Visualize TotalSegmentator vs GT for a single case.

Anatomical orientation
----------------------
All inputs (CT, GT, TS prediction) are reoriented to PIR voxel space on
load (axis 0 = P, A->P; axis 1 = I, S->I; axis 2 = R, L->R), matching
scripts/visualize_qc.py and scripts/viz_pelvic_dimensions.py.

After PIR canonicalization:
  - Axial slice    : index along axis 1 (I); displayed (P, R), anterior up
  - Coronal slice  : index along axis 0 (P); displayed (I, R), head up
  - Sagittal slice : index along axis 2 (R); transposed to (I, P), head up
This way figures come out in the standard radiologic orientation
regardless of whether the source NIfTIs are RAS / LPS / PIR / etc.

Outputs (under <out_dir>/<token>_<config>/):
  ct.nii.gz             — CT in PIR space
  gt_unified.nii.gz     — GT in PIR space, unified 10-class scheme
  pred_unified.nii.gz   — TS prediction in PIR space, remapped to unified scheme
  pred_diff.nii.gz      — non-zero where pred != gt; encoded pred*100+gt
  axial_mosaic.png      — 12 axial slices: CT / CT+GT / CT+pred (3 rows)
  orthogonal.png        — axial + coronal + sagittal through GT L5 centroid
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viz_ts")

TS_TO_UNIFIED: Dict[int, int] = {
    31: 1, 30: 2, 29: 3, 28: 4, 27: 5,
    25: 7, 77: 8, 78: 9,
}
CLASS_NAMES = {
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6", 7: "sacrum", 8: "hip_left", 9: "hip_right",
}
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0), 2: (1.0, 0.6, 0.0), 3: (1.0, 1.0, 0.0),
    4: (0.0, 1.0, 0.0), 5: (0.0, 1.0, 1.0), 6: (0.0, 0.0, 1.0),
    7: (1.0, 0.0, 1.0), 8: (0.6, 0.0, 0.8), 9: (1.0, 0.4, 0.7),
}

# HU windowing (matches visualize_qc.py and the other viz scripts)
_HU_LO, _HU_HI = -200, 800


# =============================================================================
# Orientation helpers (mirror visualize_qc.py)
# =============================================================================

def _load_pir(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI and reorient to PIR voxel space.
    Returns (data, post-reorient affine). The affine is updated so
    voxel-to-world stays correct after reorientation.
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


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """
    Orient a 2D slice for display assuming PIR source.
      dim=0 (coronal):  shape (I, R). imshow row 0 top = I=0 = superior. OK.
      dim=1 (axial):    shape (P, R). imshow row 0 top = P=0 = anterior. OK.
      dim=2 (sagittal): shape (P, I). Transpose so I is vertical (head up).
    """
    return arr2d.T if dim == 2 else arr2d


def _hu_window(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr - _HU_LO) / (_HU_HI - _HU_LO), 0.0, 1.0)


def _orient_label(ax, view_name: str) -> None:
    """Drop S/I/A/P/L/R labels in the panel corners for radiologic reading."""
    def _t(x, y, text, ha, va):
        ax.text(x, y, text, transform=ax.transAxes, color="white",
                fontsize=9, fontweight="bold", va=va, ha=ha,
                bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.6))
    if view_name == "Axial":
        _t(0.50, 0.98, "A", "center", "top"); _t(0.50, 0.02, "P", "center", "bottom")
        _t(0.98, 0.50, "R", "right",  "center"); _t(0.02, 0.50, "L", "left",   "center")
    elif view_name == "Coronal":
        _t(0.50, 0.98, "S", "center", "top"); _t(0.50, 0.02, "I", "center", "bottom")
        _t(0.98, 0.50, "R", "right",  "center"); _t(0.02, 0.50, "L", "left",   "center")
    elif view_name == "Sagittal":
        _t(0.50, 0.98, "S", "center", "top"); _t(0.50, 0.02, "I", "center", "bottom")
        _t(0.98, 0.50, "P", "right",  "center"); _t(0.02, 0.50, "A", "left",   "center")


# =============================================================================
# TS resampling
# =============================================================================

def resample_ts_to_gt_pir(ts_path: Path, gt_path: Path) -> np.ndarray:
    """
    Resample TS prediction onto GT geometry, then PIR-canonicalize the
    result. We do the resample in the original (pre-reorient) space
    because SimpleITK's resampler reads the on-disk affine; afterward
    we round-trip through nibabel + _load_pir to get the PIR layout.

    Returns a 3D int16 array with the same shape and orientation as the
    PIR-loaded GT, with TS labels remapped to the unified 10-class scheme.
    """
    import nibabel as nib
    import SimpleITK as sitk

    moving = sitk.ReadImage(str(ts_path), sitk.sitkInt32)
    fixed  = sitk.ReadImage(str(gt_path), sitk.sitkInt32)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(fixed)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    resampled = rs.Execute(moving)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        tmp_path = tf.name
    try:
        sitk.WriteImage(resampled, tmp_path)
        # Round-trip through PIR canonicalization so the array layout
        # matches our PIR-loaded GT and CT.
        arr_pir, _ = _load_pir(Path(tmp_path))
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    arr_int = arr_pir.astype(np.int32)
    unified = np.zeros_like(arr_int, dtype=np.int16)
    for ts_id, cls in TS_TO_UNIFIED.items():
        unified[arr_int == ts_id] = cls
    return unified


# =============================================================================
# Color overlay
# =============================================================================

def label_to_rgba(label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = label.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask = label == cls
        if not mask.any():
            continue
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = alpha
    return rgba


# =============================================================================
# Mosaic: 3 rows (CT / CT+GT / CT+pred) x N axial slices
# =============================================================================

def make_mosaic_png(ct, gt, pred, centre_axial_idx, out_path,
                     n_slices=12, gap=8):
    """
    Axial mosaic. In PIR voxel space, axial = perpendicular to axis 1 (I),
    so a "z" slice means slicing axis 1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    AX = 1   # axial slicing axis in PIR
    half = n_slices // 2
    indices = list(range(
        max(0, centre_axial_idx - half * gap),
        min(ct.shape[AX], centre_axial_idx + half * gap),
        gap))[:n_slices]
    if not indices:
        indices = [centre_axial_idx]

    rows, cols = 3, len(indices)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.4))
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for col, idx in enumerate(indices):
        # Axial slice: take ct[:, idx, :] -> shape (P, R)
        s = [slice(None)] * 3
        s[AX] = idx
        ct_2d   = _display_slice(ct[tuple(s)], AX)
        gt_2d   = _display_slice(gt[tuple(s)], AX)
        pred_2d = _display_slice(pred[tuple(s)], AX)
        ct_win  = _hu_window(ct_2d)

        axes[0, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[0, col].set_title(f"i={idx}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[1, col].imshow(label_to_rgba(gt_2d, 0.5), origin="upper", aspect="equal")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].text(-0.10, 0.5, "GT", transform=axes[1, col].transAxes,
                              fontsize=10, color="#cccccc",
                              rotation=90, va="center", ha="right")

        axes[2, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[2, col].imshow(label_to_rgba(pred_2d, 0.5), origin="upper", aspect="equal")
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].text(-0.10, 0.5, "TS pred", transform=axes[2, col].transAxes,
                              fontsize=10, color="#cccccc",
                              rotation=90, va="center", ha="right")

        # Anatomic labels only on the first column to reduce clutter
        if col == 0:
            for r in range(3):
                _orient_label(axes[r, col], "Axial")

    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(1, 10)
               if (gt == c).any() or (pred == c).any()]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Axial mosaic (PIR) — centre i={centre_axial_idx}", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Orthogonal: axial + coronal + sagittal x (CT / CT+GT / CT+pred)
# =============================================================================

def make_orth_view(ct, gt, pred, centre_pir, out_path):
    """
    Three orthogonal views, three rows each.

    PIR voxel-axis convention:
      Axial    = slice axis 1 (I); display (P, R), anterior up
      Coronal  = slice axis 0 (P); display (I, R), head up
      Sagittal = slice axis 2 (R); display (P, I) -> transposed, head up

    centre_pir is (P_idx, I_idx, R_idx), the PIR voxel coordinates of
    the slicing centroid (typically the L5 or S1 centroid).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p_idx, i_idx, r_idx = centre_pir

    views = [
        ("Axial",    1, i_idx),
        ("Coronal",  0, p_idx),
        ("Sagittal", 2, r_idx),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))

    for col, (name, view_axis, idx) in enumerate(views):
        idx = int(np.clip(idx, 0, ct.shape[view_axis] - 1))
        s = [slice(None)] * 3
        s[view_axis] = idx
        ct_2d   = _display_slice(ct[tuple(s)],   view_axis)
        gt_2d   = _display_slice(gt[tuple(s)],   view_axis)
        pred_2d = _display_slice(pred[tuple(s)], view_axis)
        ct_win  = _hu_window(ct_2d)

        # Row 0: CT only
        axes[0, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[0, col].set_title(f"CT — {name}", fontsize=10)
        axes[0, col].axis("off")
        _orient_label(axes[0, col], name)

        # Row 1: CT + GT overlay
        axes[1, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[1, col].imshow(label_to_rgba(gt_2d, 0.55), origin="upper", aspect="equal")
        axes[1, col].set_title(f"GT — {name}", fontsize=10)
        axes[1, col].axis("off")
        _orient_label(axes[1, col], name)

        # Row 2: CT + TS prediction overlay
        axes[2, col].imshow(ct_win, cmap="gray", origin="upper", aspect="equal")
        axes[2, col].imshow(label_to_rgba(pred_2d, 0.55), origin="upper", aspect="equal")
        axes[2, col].set_title(f"TS pred — {name}", fontsize=10)
        axes[2, col].axis("off")
        _orient_label(axes[2, col], name)

    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(1, 10)
               if (gt == c).any() or (pred == c).any()]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Per-case driver
# =============================================================================

def viz_one_case(token, config, dataset_dir, pred_dir, out_dir) -> bool:
    import nibabel as nib

    case_dir = out_dir / f"{token}_{config}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ct_dir  = dataset_dir / "ct"
    lbl_dir = dataset_dir / "labels"

    try:
        stem = f"{int(token):04d}_{config}"
    except ValueError:
        stem = f"{token}_{config}"
    ct_path = ct_dir  / f"{stem}_ct.nii.gz"
    gt_path = lbl_dir / f"{stem}_label.nii.gz"

    if not ct_path.exists() or not gt_path.exists():
        log.error("[%s/%s] CT or label not found. tried: %s  %s",
                  token, config, ct_path, gt_path)
        return False
    log.info("[%s/%s] CT=%s  GT=%s", token, config, ct_path.name, gt_path.name)

    ts_candidates = [
        pred_dir / f"{token}_{config}" / "segmentation.nii.gz",
        pred_dir / token / "segmentation.nii.gz",
    ]
    ts_path = next((p for p in ts_candidates if p.exists()), None)
    if ts_path is None:
        log.error("[%s/%s] No TS prediction found.", token, config)
        return False

    # PIR-canonicalize CT, GT, and TS prediction
    ct_pir, ct_aff   = _load_pir(ct_path)
    gt_pir, gt_aff   = _load_pir(gt_path)
    gt_pir = gt_pir.astype(np.int16)
    pred_pir = resample_ts_to_gt_pir(ts_path, gt_path)

    if pred_pir.shape != gt_pir.shape:
        log.error("[%s/%s] PIR shape mismatch after resample: pred=%s gt=%s",
                  token, config, pred_pir.shape, gt_pir.shape)
        return False
    # Trim CT to common shape if needed
    mn = tuple(min(a, b) for a, b in zip(ct_pir.shape, gt_pir.shape))
    ct_pir   = ct_pir  [:mn[0], :mn[1], :mn[2]]
    gt_pir   = gt_pir  [:mn[0], :mn[1], :mn[2]]
    pred_pir = pred_pir[:mn[0], :mn[1], :mn[2]]

    # Save canonicalized NIfTIs (with the PIR affine) so they open in
    # Slicer / ITK-SNAP in the same orientation as the figures.
    nib.save(nib.Nifti1Image(ct_pir.astype(np.float32), gt_aff),
             case_dir / "ct.nii.gz")
    nib.save(nib.Nifti1Image(gt_pir.astype(np.int16), gt_aff),
             case_dir / "gt_unified.nii.gz")
    nib.save(nib.Nifti1Image(pred_pir.astype(np.int16), gt_aff),
             case_dir / "pred_unified.nii.gz")
    diff = np.where(pred_pir == gt_pir, 0,
                    pred_pir.astype(np.int32) * 100 + gt_pir.astype(np.int32))
    nib.save(nib.Nifti1Image(diff.astype(np.int32), gt_aff),
             case_dir / "pred_diff.nii.gz")

    # L5 centroid in PIR voxel coordinates -> (P_idx, I_idx, R_idx)
    l5_vox = np.argwhere(gt_pir == 5)
    s1_vox = np.argwhere(gt_pir == 7)
    if len(l5_vox) > 0:
        centre_pir = tuple(int(round(c)) for c in l5_vox.mean(axis=0))
    elif len(s1_vox) > 0:
        centre_pir = tuple(int(round(c)) for c in s1_vox.mean(axis=0))
    else:
        centre_pir = (gt_pir.shape[0] // 2, gt_pir.shape[1] // 2, gt_pir.shape[2] // 2)
    log.info("[%s/%s] PIR centroid (P, I, R) = %s", token, config, centre_pir)

    # Mosaic uses the I-axis (axial) center
    make_mosaic_png(ct_pir, gt_pir, pred_pir,
                     centre_axial_idx=centre_pir[1],
                     out_path=case_dir / "axial_mosaic.png")
    make_orth_view(ct_pir, gt_pir, pred_pir, centre_pir,
                    out_path=case_dir / "orthogonal.png")
    log.info("[%s/%s] Wrote: %s/", token, config, case_dir)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, type=Path)
    ap.add_argument("--pred_dir",    required=True, type=Path)
    ap.add_argument("--out_dir",     required=True, type=Path)
    ap.add_argument("--token",   default=None, type=str)
    ap.add_argument("--config",  default=None, type=str,
                    choices=[None, "fused", "spine_only", "pelvic_native"])
    ap.add_argument("--tokens",  default="", type=str)
    ap.add_argument("--configs", default="", type=str)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.tokens and args.configs:
        toks = [t.strip() for t in args.tokens.split(",") if t.strip()]
        cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
        if len(toks) != len(cfgs):
            log.error("--tokens and --configs must have same count")
            sys.exit(1)
        cases = list(zip(toks, cfgs))
    elif args.token and args.config:
        cases = [(args.token, args.config)]
    else:
        log.error("Provide (--token + --config) or (--tokens + --configs)")
        sys.exit(1)

    n_ok = 0
    for token, config in cases:
        if viz_one_case(token, config, args.dataset_dir, args.pred_dir, args.out_dir):
            n_ok += 1
    log.info("DONE  ok=%d/%d  output=%s", n_ok, len(cases), args.out_dir)


if __name__ == "__main__":
    main()
