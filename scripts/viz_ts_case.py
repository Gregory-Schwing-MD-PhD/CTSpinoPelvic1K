#!/usr/bin/env python3
"""
viz_ts_case.py — Visualize TotalSegmentator vs GT for a single case.

Outputs (under <out_dir>/<token>_<config>/):
  ct.nii.gz             — original CT (copy)
  gt_unified.nii.gz     — GT in unified 10-class scheme
  pred_unified.nii.gz   — TS prediction RESAMPLED + REMAPPED to GT space
  pred_diff.nii.gz      — non-zero where pred != gt; encoded pred*100+gt
  axial_mosaic.png      — 3 rows × 12 axial slices: CT / CT+GT / CT+pred
  orthogonal.png        — axial+coronal+sagittal through GT L5 centroid
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


def resample_ts_to_gt(ts_path: Path, gt_path: Path) -> np.ndarray:
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
        arr = np.asarray(nib.load(tmp_path).dataobj, dtype=np.int32)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
    unified = np.zeros_like(arr, dtype=np.int16)
    for ts_id, cls in TS_TO_UNIFIED.items():
        unified[arr == ts_id] = cls
    return unified


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


def normalize_ct(ct, lo=-200, hi=1500):
    out = np.clip(ct, lo, hi)
    return (out - lo) / (hi - lo)


def make_mosaic_png(ct, gt, pred, centre_z, out_path, n_slices=12, gap=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    half = n_slices // 2
    z_indices = list(range(
        max(0, centre_z - half * gap),
        min(ct.shape[2], centre_z + half * gap),
        gap))[:n_slices]
    if not z_indices:
        z_indices = [centre_z]
    rows, cols = 3, len(z_indices)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.4))
    if cols == 1:
        axes = axes.reshape(rows, 1)
    for col, z in enumerate(z_indices):
        ct_slice = normalize_ct(ct[:, :, z])
        gt_slice = gt[:, :, z]
        pred_slice = pred[:, :, z]
        axes[0, col].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"z={z}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[1, col].imshow(label_to_rgba(gt_slice, 0.5).transpose(1, 0, 2), origin="lower")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("GT", fontsize=10)
        axes[2, col].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[2, col].imshow(label_to_rgba(pred_slice, 0.5).transpose(1, 0, 2), origin="lower")
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel("TS pred", fontsize=10)
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(1, 10)
               if (gt == c).any() or (pred == c).any()]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Axial mosaic — centre z={centre_z}", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_orth_view(ct, gt, pred, centre_xyz, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cx, cy, cz = centre_xyz
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    views = [
        ("Axial",    lambda v: v[:, :, cz].T),
        ("Coronal",  lambda v: v[:, cy, :].T),
        ("Sagittal", lambda v: v[cx, :, :].T),
    ]
    for col, (name, slicer) in enumerate(views):
        ct_s = normalize_ct(slicer(ct))
        gt_s = slicer(gt)
        pred_s = slicer(pred)
        axes[0, col].imshow(ct_s, cmap="gray", origin="lower")
        axes[0, col].set_title(f"CT — {name}", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(ct_s, cmap="gray", origin="lower")
        axes[1, col].imshow(label_to_rgba(gt_s, 0.55), origin="lower")
        axes[1, col].set_title(f"GT — {name}", fontsize=10)
        axes[1, col].axis("off")
        axes[2, col].imshow(ct_s, cmap="gray", origin="lower")
        axes[2, col].imshow(label_to_rgba(pred_s, 0.55), origin="lower")
        axes[2, col].set_title(f"TS pred — {name}", fontsize=10)
        axes[2, col].axis("off")
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(1, 10)
               if (gt == c).any() or (pred == c).any()]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def viz_one_case(token, config, dataset_dir, pred_dir, out_dir) -> bool:
    import nibabel as nib
    case_dir = out_dir / f"{token}_{config}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ct_dir  = dataset_dir / "ct"
    lbl_dir = dataset_dir / "labels"

    # Match export_hf.py naming: {token:04d}_{config}_ct.nii.gz
    try:
        stem = f"{int(token):04d}_{config}"
    except ValueError:
        stem = f"{token}_{config}"
    ct_path = ct_dir / f"{stem}_ct.nii.gz"
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

    ct = np.asarray(nib.load(str(ct_path)).dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(str(gt_path)).dataobj, dtype=np.int16)
    pred = resample_ts_to_gt(ts_path, gt_path)
    if pred.shape != gt.shape:
        log.error("[%s/%s] Shape mismatch: pred=%s gt=%s",
                  token, config, pred.shape, gt.shape)
        return False

    aff = nib.load(str(gt_path)).affine
    nib.save(nib.Nifti1Image(ct, aff), case_dir / "ct.nii.gz")
    nib.save(nib.Nifti1Image(gt.astype(np.int16), aff), case_dir / "gt_unified.nii.gz")
    nib.save(nib.Nifti1Image(pred.astype(np.int16), aff), case_dir / "pred_unified.nii.gz")
    diff = np.where(pred == gt, 0, pred.astype(np.int32) * 100 + gt.astype(np.int32))
    nib.save(nib.Nifti1Image(diff.astype(np.int32), aff), case_dir / "pred_diff.nii.gz")

    l5_vox = np.argwhere(gt == 5)
    s1_vox = np.argwhere(gt == 7)
    if len(l5_vox) > 0:
        centre = tuple(int(round(c)) for c in l5_vox.mean(axis=0))
    elif len(s1_vox) > 0:
        centre = tuple(int(round(c)) for c in s1_vox.mean(axis=0))
    else:
        centre = (gt.shape[0] // 2, gt.shape[1] // 2, gt.shape[2] // 2)

    make_mosaic_png(ct, gt, pred, centre[2], case_dir / "axial_mosaic.png")
    make_orth_view(ct, gt, pred, centre, case_dir / "orthogonal.png")
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
