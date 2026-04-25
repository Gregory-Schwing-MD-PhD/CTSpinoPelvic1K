#!/usr/bin/env python3
"""
benchmark_totalseg.py — TotalSegmentator zero-shot benchmark on CTSpinoPelvic1K.

Zero-shot inference cares about aggregation, not splits — by default this
benchmarks the ENTIRE dataset (all three configs) and stratifies the results
at aggregation time by config, LSTV class, and match_type.

Features:
1. Orientation-safe resample via SimpleITK round-trip
2. Per-vertebra columns in Table 5 (L1, L2, L3, L4, L5)
3. Config-aware Dice scoring (only score classes the GT can have)
4. Optional --config filter (all | fused | spine_only | pelvic_native)
5. Optional --tokens subset for sharding / debugging
6. TS prediction cache key includes config
7. kd-tree HD95 (50-100x faster than brute-force pairwise)
8. ASSD + MSD surface metrics for surgical-planning relevance
9. Streaming per-case JSONL log (resumable; no metric data loss on job timeout)

Surface metrics (added Apr 2026)
--------------------------------
Three boundary-distance numbers, all computed from the same per-class
surface-voxel point clouds via scipy cKDTree. Adding ASSD + MSD on top of
HD95 is essentially free since they reuse the same trees + queries.

  HD95  - 95th percentile of the union of nearest-neighbor distances.
          Worst-case boundary error with 5% outlier tolerance. Sensitive
          to spurious blobs of mislabeled voxels far from the true
          surface. Reported in mm.

  ASSD  - Average Symmetric Surface Distance. Mean of nearest-neighbor
          distances symmetrically: from each pred-surface voxel to its
          nearest GT-surface voxel, AND vice versa, then averaged
          together. The headline metric for surgical-planning relevance:
          ASSD < 1mm typically considered safe for screw-trajectory
          planning of SI / iliosacral hardware, 1-2mm marginal,
          >2mm unsafe.

  MSD   - Mean Surface Distance, directional. Two values reported:
            msd_pred_to_gt: average distance from predicted boundary to
                            nearest GT boundary. High = pred has voxels
                            far from any truth surface (over-segmentation
                            into nearby tissue).
            msd_gt_to_pred: average distance from GT boundary to nearest
                            pred boundary. High = pred misses parts of
                            the true surface (under-segmentation).
          Diagnostic for asymmetric errors that ASSD averages away.

NOTE: --fast mode is available but NOT the default. For publication-quality
TotalSegmentator numbers, leave it off.

Resumability
------------
Two layers of resumability:

1. Predictions cached on disk under <pred_dir>/<token>_<config>/segmentation.nii.gz.
   If a prediction file exists, TS inference is skipped on the next run.

2. Per-case metric results are streamed to <out_dir>/per_case_partial.jsonl
   immediately after each case completes (with fsync), so a wall-clock kill
   only loses the in-flight case. On startup, this file is loaded and any
   (token, config) pairs already present are skipped entirely (no re-Dice
   computation).

After all cases complete, the final aggregation pass produces
benchmark_results.json + benchmark_summary.json + paper_tables.txt + CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.benchmark_ts")

CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6",
    7: "sacrum",
    8: "hip_left", 9: "hip_right",
}

TS_TO_UNIFIED: Dict[int, int] = {
    31: 1, 30: 2, 29: 3, 28: 4, 27: 5,
    25: 7, 77: 8, 78: 9,
}

TS_ROI_SUBSET = [
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
    "vertebrae_L4", "vertebrae_L5",
    "sacrum", "hip_left", "hip_right",
]

FOREGROUND_CLASSES     = list(range(1, 10))
TS_CAPABLE_CLASSES     = [1, 2, 3, 4, 5, 7, 8, 9]
TS_INCAPABLE_CLASSES   = [6]
JUNCTION_CLASSES       = [5, 6, 7]
JUNCTION_WINDOW_MM     = 40.0

SCOREABLE_BY_CONFIG: Dict[str, List[int]] = {
    "fused":         [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "spine_only":    [1, 2, 3, 4, 5, 6],
    "pelvic_native": [7, 8, 9],
}

# Surface-distance fields persisted per case. Each is a {class_id: float|None}
# dict matching the layout of the existing `dice` field.
SURFACE_METRIC_FIELDS = ("hd95", "assd", "msd_pred_to_gt", "msd_gt_to_pred")


# =============================================================================
# Metrics
# =============================================================================

def dice(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    p = pred == cls
    g = gt   == cls
    if not p.any() and not g.any():
        return float("nan")
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (p & g).sum() / denom)


def _surface_points_mm(mask: np.ndarray,
                        spacing: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """Return Nx3 surface-voxel coordinates in physical millimeters, or
    None if the mask has no extractable surface."""
    from scipy.ndimage import binary_erosion
    if not mask.any():
        return None
    surf = mask ^ binary_erosion(mask)
    if not surf.any():
        return None
    sp = np.asarray(spacing, dtype=np.float32)
    return np.argwhere(surf).astype(np.float32) * sp


def surface_metrics(pred: np.ndarray, gt: np.ndarray, cls: int,
                     vox_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                     ) -> Dict[str, float]:
    """
    Compute HD95, ASSD, and directional MSD between (pred==cls) and
    (gt==cls), all in physical millimeter space.

    Returns a dict with keys: hd95, assd, msd_pred_to_gt, msd_gt_to_pred.
    Values are floats in mm, or NaN if either mask is empty / has no
    extractable surface.

    All four metrics share one pair of cKDTrees built from the per-mask
    surface point clouds, so the cost is dominated by the tree
    construction plus two batched nearest-neighbor queries, regardless
    of how many metrics we ask for. Adding ASSD + MSD on top of HD95 is
    essentially free.

    Symmetric formulations:
      HD95 = 95th percentile of the UNION of both directed distance sets.
      ASSD = mean of the UNION of both directed distance sets.
      MSD_X_to_Y = mean of distances from X-surface points to nearest
                   Y-surface point (directional).
    """
    from scipy.spatial import cKDTree

    nan_dict = {k: float("nan") for k in SURFACE_METRIC_FIELDS}

    p = (pred == cls).astype(bool)
    g = (gt   == cls).astype(bool)
    p_pts = _surface_points_mm(p, vox_spacing)
    g_pts = _surface_points_mm(g, vox_spacing)
    if p_pts is None or g_pts is None:
        return nan_dict

    p_tree = cKDTree(p_pts)
    g_tree = cKDTree(g_pts)
    p_to_g, _ = g_tree.query(p_pts, k=1, workers=-1)
    g_to_p, _ = p_tree.query(g_pts, k=1, workers=-1)

    union = np.concatenate([p_to_g, g_to_p])
    return {
        "hd95":            float(np.percentile(union, 95)),
        "assd":            float(union.mean()),
        "msd_pred_to_gt":  float(p_to_g.mean()),
        "msd_gt_to_pred":  float(g_to_p.mean()),
    }


# Back-compat shim. Anything that imported hausdorff95 directly still works,
# but internally now goes through the unified surface_metrics() path.
def hausdorff95(pred: np.ndarray, gt: np.ndarray, cls: int,
                 vox_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    return surface_metrics(pred, gt, cls, vox_spacing)["hd95"]


def junction_analysis(pred, gt, affine, window_mm=JUNCTION_WINDOW_MM) -> Dict:
    vox_sz_z = float(np.abs(np.linalg.norm(affine[:3, 2])))
    half     = int(np.ceil(window_mm / 2.0 / max(vox_sz_z, 0.1)))
    l5_vox = np.argwhere(gt == 5)
    if len(l5_vox) > 0:
        centre_z = int(round(float(l5_vox[:, 2].mean())))
    else:
        s1_vox = np.argwhere(gt == 7)
        if len(s1_vox) == 0:
            return {"error": "no_L5_or_sacrum_in_GT", "window_mm": window_mm}
        centre_z = int(s1_vox[:, 2].max())
    lo = max(0, centre_z - half)
    hi = min(gt.shape[2], centre_z + half)
    gt_w   = gt  [..., lo:hi]
    pred_w = pred[..., lo:hi]
    jxn_dice = {cls: dice(pred_w, gt_w, cls) for cls in JUNCTION_CLASSES}
    jxn_vals = [v for _, v in jxn_dice.items() if v == v and v is not None]
    mean_jxn_dsc = float(np.mean(jxn_vals)) if jxn_vals else float("nan")

    confusion   = defaultdict(lambda: defaultdict(int))
    total_jv    = 0
    for gt_cls in JUNCTION_CLASSES:
        mask = gt_w == gt_cls
        if not mask.any():
            continue
        total_jv += int(mask.sum())
        pred_vals, counts = np.unique(pred_w[mask], return_counts=True)
        for pv, cnt in zip(pred_vals.tolist(), counts.tolist()):
            confusion[gt_cls][int(pv)] += cnt
    error_vox = sum(cnt for gt_c, pd in confusion.items()
                    for pc, cnt in pd.items() if pc != gt_c)
    error_rate = float(error_vox / total_jv) if total_jv > 0 else float("nan")
    l5_total = sum(confusion.get(5, {}).values())
    l5_as_s1 = int(confusion.get(5, {}).get(7, 0))
    l5_sacrum_rate = float(l5_as_s1 / l5_total) if l5_total > 0 else float("nan")
    s1_total = sum(confusion.get(7, {}).values())
    s1_as_l5 = int(confusion.get(7, {}).get(5, 0))
    s1_l5_rate = float(s1_as_l5 / s1_total) if s1_total > 0 else float("nan")
    ts_has_l6 = bool((pred_w == 6).any())

    return {
        "centre_z":             centre_z,
        "window_lo_hi":         [lo, hi],
        "window_mm":            window_mm,
        "n_junction_voxels":    total_jv,
        "mean_junction_dsc":    round(mean_jxn_dsc, 4),
        "junction_dice":        {
            CLASS_NAMES[c]: (round(v, 4) if v == v else None)
            for c, v in jxn_dice.items()
        },
        "labelling_error_rate":  round(error_rate, 4) if error_rate == error_rate else None,
        "l5_called_sacrum_rate": round(l5_sacrum_rate, 4) if l5_sacrum_rate == l5_sacrum_rate else None,
        "s1_called_l5_rate":     round(s1_l5_rate, 4) if s1_l5_rate == s1_l5_rate else None,
        "ts_assigned_l6":        ts_has_l6,
        "confusion_matrix": {
            CLASS_NAMES.get(gt_c, str(gt_c)): {
                CLASS_NAMES.get(pc, str(pc)): cnt for pc, cnt in pd.items()
            } for gt_c, pd in confusion.items()
        },
    }


# =============================================================================
# TotalSegmentator inference + label remapping
# =============================================================================

def run_totalseg(ct_path: Path, out_dir: Path, device="gpu",
                 fast=False, force=False) -> Optional[Path]:
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    ml_path = out_dir / "segmentation.nii.gz"
    if ml_path.exists() and not force:
        return ml_path
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("    TS: running on %s  (device=%s fast=%s)",
             ct_path.name, device, fast)
    t0 = time.time()
    try:
        pred = totalsegmentator(
            input      = nib.load(str(ct_path)),
            output     = None,
            task       = "total",
            ml         = True,
            device     = device,
            fast       = fast,
            roi_subset = TS_ROI_SUBSET,
            verbose    = False,
        )
        nib.save(pred, str(ml_path))
        log.info("    TS: done %.0fs", time.time() - t0)
        return ml_path
    except Exception as e:
        log.error("    TS FAILED: %s", e)
        return None


def resample_and_remap(ts_path: Path, ref_path: Path) -> np.ndarray:
    import nibabel as nib
    import SimpleITK as sitk
    moving = sitk.ReadImage(str(ts_path), sitk.sitkInt32)
    fixed  = sitk.ReadImage(str(ref_path), sitk.sitkInt32)
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
        pred_nib = nib.load(tmp_path)
        arr = np.asarray(pred_nib.dataobj, dtype=np.int32)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
    ref_nib = nib.load(str(ref_path))
    if arr.shape != ref_nib.shape:
        raise ValueError(f"Post-resample shape mismatch: pred={arr.shape} ref={ref_nib.shape}")
    unified = np.zeros_like(arr, dtype=np.int16)
    for ts_id, cls in TS_TO_UNIFIED.items():
        unified[arr == ts_id] = cls
    return unified


# =============================================================================
# Per-case driver
# =============================================================================

def benchmark_one(token, ct_path, label_path, pred_dir, case_meta,
                   device="gpu", fast=False, skip_surface=False,
                   force_ts=False, junction_window=JUNCTION_WINDOW_MM) -> Dict:
    """
    Run TS on this case, compute Dice + (optionally) HD95/ASSD/MSD
    surface metrics + junction analysis, and return a single result
    dict.

    `skip_surface=True` skips ALL surface metrics (HD95, ASSD, MSD).
    """
    import nibabel as nib
    config    = case_meta.get("config", "fused")
    scoreable = SCOREABLE_BY_CONFIG.get(config, FOREGROUND_CLASSES)
    result: Dict = {
        "token": token, "config": config,
        "match_type":    case_meta.get("match_type", ""),
        "lstv_label":    case_meta.get("lstv_label", "normal"),
        "position":      case_meta.get("position", "unknown"),
        "has_l6":        case_meta.get("has_l6", False),
        "lstv_agreement": case_meta.get("lstv_agreement"),
        "lstv_confusion_zone": case_meta.get("lstv_confusion_zone"),
        "scoreable_classes": scoreable,
        "ok": False, "error": None,
        "dice": {},
        "hd95": {}, "assd": {}, "msd_pred_to_gt": {}, "msd_gt_to_pred": {},
        "junction": {},
        "n_gt_classes": 0, "l6_gt_present": False, "l6_dice_meaningful": None,
    }
    try:
        cache_subdir = pred_dir / f"{token}_{config}"
        ts_pred = run_totalseg(ct_path, cache_subdir,
                                device=device, fast=fast, force=force_ts)
        if ts_pred is None:
            raise RuntimeError("TotalSegmentator inference failed")
        pred = resample_and_remap(ts_pred, label_path)
        gt_img = nib.load(str(label_path))
        gt     = np.asarray(gt_img.dataobj, dtype=np.int16)
        affine = gt_img.affine
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch pred={pred.shape} gt={gt.shape}")
        gt_classes = sorted({int(v) for v in np.unique(gt)} - {0})
        result["n_gt_classes"]  = len(gt_classes)
        result["l6_gt_present"] = bool(6 in gt_classes)

        # Dice (incl. hips, classes 8 and 9 -- already in scoreable for
        # fused + pelvic_native)
        for cls in FOREGROUND_CLASSES:
            if cls not in scoreable:
                result["dice"][cls] = None
                continue
            d = dice(pred, gt, cls)
            result["dice"][cls] = round(d, 4) if d == d else None
        if result["l6_gt_present"]:
            result["l6_dice_meaningful"] = 0.0

        # Surface metrics: compute HD95 + ASSD + MSD all in one shot per
        # class so we only build kd-trees once per (class, case). Fields
        # `hd95`, `assd`, `msd_pred_to_gt`, `msd_gt_to_pred` are filled
        # per scoreable TS-capable class.
        if not skip_surface:
            for cls in TS_CAPABLE_CLASSES:
                if cls not in scoreable:
                    for f in SURFACE_METRIC_FIELDS:
                        result[f][cls] = None
                    continue
                m = surface_metrics(pred, gt, cls, vox_spacing=spacing)
                for f in SURFACE_METRIC_FIELDS:
                    v = m[f]
                    result[f][cls] = round(v, 4) if v == v else None

        if config == "fused":
            result["junction"] = junction_analysis(pred, gt, affine,
                                                    window_mm=junction_window)
        else:
            result["junction"] = {"error": f"junction_analysis_skipped_config_{config}"}
        result["ok"] = True

        # Compact log line: Dice for spine + sacrum + both hips, plus
        # the surgical-relevance number (sacrum ASSD).
        d_str = "  ".join(
            f"{CLASS_NAMES[c]}={(result['dice'].get(c) if result['dice'].get(c) is not None else float('nan')):.3f}"
            for c in [1, 2, 3, 4, 5, 7, 8, 9])
        sacrum_assd = result["assd"].get(7) if not skip_surface else None
        sac_str = f"sac_ASSD={sacrum_assd:.2f}mm" if sacrum_assd is not None else ""
        log.info("  %-6s  cfg=%s  %s  %s  lstv=%s",
                 token, config, d_str, sac_str, result["lstv_label"])
    except Exception:
        result["error"] = traceback.format_exc()
        log.error("  FAIL token=%s cfg=%s: %s", token, config,
                  result["error"].splitlines()[-1])
    return result


# =============================================================================
# Streaming JSONL log: load + append helpers
# =============================================================================

def load_completed_results(per_case_log: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Read the streaming JSONL file from any prior run(s) and return:
      - the list of result dicts (deduplicated; keeps the LATEST entry per
        (token, config) so a re-run that recomputed metrics overwrites the
        prior entry)
      - a set of "{token}__{config}" keys we've already seen and don't need
        to redo

    Returns ([], set()) if the file doesn't exist.
    """
    if not per_case_log.exists():
        return [], set()

    by_key: Dict[str, Dict] = {}     # latest record per (token, config)
    n_lines = n_bad = 0
    with open(per_case_log, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            tok = rec.get("token")
            cfg = rec.get("config")
            if not tok or not cfg:
                n_bad += 1
                continue
            by_key[f"{tok}__{cfg}"] = rec   # later occurrences overwrite

    results = list(by_key.values())
    seen = set(by_key.keys())
    log.info("Resumed from %s: %d lines, %d unique (token, config) entries (bad=%d)",
             per_case_log, n_lines, len(seen), n_bad)
    return results, seen


def append_result_jsonl(per_case_log: Path, result: Dict) -> None:
    """
    Append one case's result as a JSON line, then fsync so it survives a
    SIGKILL. The default=str fallback handles things like Path objects in
    error tracebacks.
    """
    per_case_log.parent.mkdir(parents=True, exist_ok=True)
    with open(per_case_log, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# =============================================================================
# Aggregation
# =============================================================================

def _nanmean(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.mean(v)), 4) if v else None

def _nanstd(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.std(v)), 4) if len(v) > 1 else None


def _normalize_for_aggregation(r: Dict) -> Dict:
    """
    JSONL roundtrip turns int dict keys into strings. Convert them back
    so class_stats() can index by int cls IDs. Idempotent — safe to call
    on dicts that were never round-tripped. Handles Dice + all four
    surface metrics.
    """
    for field in ("dice",) + SURFACE_METRIC_FIELDS:
        d = r.get(field) or {}
        new_d = {}
        for k, v in d.items():
            try:
                new_d[int(k)] = v
            except (ValueError, TypeError):
                new_d[k] = v
        r[field] = new_d
    return r


def class_stats(cases):
    """Per-class summary: Dice + each surface metric, all with mean and std."""
    stats = {}
    for cls in FOREGROUND_CLASSES:
        entry = {"n_cases": len(cases)}
        # Dice
        dvals = [r["dice"].get(cls) for r in cases]
        entry["dice_mean"] = _nanmean(dvals)
        entry["dice_std"]  = _nanstd(dvals)
        # Surface metrics — only aggregated if at least one case carries them
        for f in SURFACE_METRIC_FIELDS:
            vals = [r.get(f, {}).get(cls) for r in cases if r.get(f)]
            entry[f"{f}_mean"] = _nanmean(vals)
            entry[f"{f}_std"]  = _nanstd(vals)
        stats[CLASS_NAMES[cls]] = entry
    return stats


def junction_stats(cases):
    jxns = [r["junction"] for r in cases
            if r.get("junction") and "error" not in r["junction"]]
    return {
        "n": len(jxns),
        "mean_junction_dsc": _nanmean([j.get("mean_junction_dsc") for j in jxns]),
        "labelling_error_rate_mean": _nanmean([j.get("labelling_error_rate") for j in jxns]),
        "l5_called_sacrum_rate_mean": _nanmean([j.get("l5_called_sacrum_rate") for j in jxns]),
        "s1_called_l5_rate_mean": _nanmean([j.get("s1_called_l5_rate") for j in jxns]),
        "ts_assigned_l6_count": sum(1 for j in jxns if j.get("ts_assigned_l6")),
    }


def aggregate(results):
    results = [_normalize_for_aggregation(r) for r in results]
    ok = [r for r in results if r.get("ok")]
    def _sub(cases, label):
        if not cases:
            return {"n": 0, "label": label}
        return {"n": len(cases), "label": label,
                "classes": class_stats(cases),
                "junction": junction_stats(cases)}
    summary = {
        "n_total": len(results), "n_ok": len(ok),
        "n_fail": len(results) - len(ok),
        "subgroups": {
            "all":           _sub(ok, "All cases"),
            "fused_only":    _sub([r for r in ok if r["config"] == "fused"], "Fused only"),
            "spine_only":    _sub([r for r in ok if r["config"] == "spine_only"], "Spine only"),
            "pelvic_native": _sub([r for r in ok if r["config"] == "pelvic_native"], "Pelvic native"),
            "normal":        _sub([r for r in ok if r["lstv_label"].lower() == "normal"], "Normal"),
            "any_lstv":      _sub([r for r in ok if r["lstv_label"].lower() not in ("normal","unknown","")], "Any LSTV"),
            "sacralization": _sub([r for r in ok if r["lstv_label"].lower() == "sacralization"], "Sacralization"),
            "lumbarization": _sub([r for r in ok if r["lstv_label"].lower() == "lumbarization"], "Lumbarization"),
        },
    }
    return summary


def format_table5(summary) -> str:
    """Publication-ready per-vertebra Dice table."""
    def _f(v):
        return f"{v:>5.3f}" if v is not None and v == v else f"{'—':>5}"

    lines = [
        "",
        "=" * 104,
        "  TABLE 5  —  TotalSegmentator Zero-Shot Benchmark (per-vertebra Dice)",
        "=" * 104,
    ]
    header = (
        f"\n  {'Subgroup':<26}  {'n':>3}  "
        f"{'L1':>5}  {'L2':>5}  {'L3':>5}  {'L4':>5}  {'L5':>5}  "
        f"{'L6':>5}  {'Sac':>5}  {'HipL':>5}  {'HipR':>5}  {'JxnDSC':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * 100)

    for key in ["all", "fused_only", "spine_only", "pelvic_native",
                "normal", "any_lstv", "sacralization", "lumbarization"]:
        sg = summary["subgroups"].get(key)
        if not sg or sg.get("n", 0) == 0:
            continue
        cls = sg.get("classes", {})
        jxn = sg.get("junction", {})

        lumbar_cells = "  ".join(
            _f(cls.get(n, {}).get("dice_mean"))
            for n in ("L1", "L2", "L3", "L4", "L5")
        )
        pelvic_cells = "  ".join(
            _f(cls.get(n, {}).get("dice_mean"))
            for n in ("sacrum", "hip_left", "hip_right")
        )
        l6_cell  = f"{'—':>5}"
        jxn_cell = f"{_f(jxn.get('mean_junction_dsc')):>7}"

        lines.append(
            f"  {sg['label']:<26}  {sg['n']:>3}  "
            f"{lumbar_cells}  {l6_cell}  {pelvic_cells}  {jxn_cell}"
        )
    lines.append("")
    lines.append("  † L6 column = '—': TS has no L6 label.")
    return "\n".join(lines)


def format_table_surface(summary) -> str:
    """
    Per-class surface-distance table — surgical-planning headline.

    Rows: subgroup. Columns: ASSD and HD95 for sacrum + both hips
    (the classes that matter for SI screw / iliosacral fixation
    planning), plus a sacrum MSD asymmetry indicator
    (msd_pred_to_gt - msd_gt_to_pred) — positive means TS over-segments
    the sacrum on average, negative means it under-segments.
    """
    def _fmm(v):
        return f"{v:>5.2f}" if v is not None and v == v else f"{'—':>5}"

    lines = [
        "",
        "=" * 110,
        "  TABLE 6  —  Surface-distance metrics (mm) for surgical-planning relevance",
        "=" * 110,
        "",
        "  ASSD = Average Symmetric Surface Distance (lower better; <1mm typical safe threshold)",
        "  HD95 = 95th-percentile Hausdorff distance (lower better; outlier-sensitive)",
        "  MSD asymm = msd_pred_to_gt - msd_gt_to_pred for sacrum",
        "              (positive = TS over-segments, negative = TS under-segments)",
    ]
    header = (
        f"\n  {'Subgroup':<24}  {'n':>3}  "
        f"{'Sac ASSD':>9}  {'Sac HD95':>9}  {'HipL ASSD':>10}  {'HipL HD95':>10}  "
        f"{'HipR ASSD':>10}  {'HipR HD95':>10}  {'Sac MSD asymm':>14}"
    )
    lines.append(header)
    lines.append("  " + "-" * 106)

    for key in ["all", "fused_only", "spine_only", "pelvic_native",
                "normal", "any_lstv", "sacralization", "lumbarization"]:
        sg = summary["subgroups"].get(key)
        if not sg or sg.get("n", 0) == 0:
            continue
        cls = sg.get("classes", {})
        sac = cls.get("sacrum", {})
        lh  = cls.get("hip_left", {})
        rh  = cls.get("hip_right", {})

        msd_p2g = sac.get("msd_pred_to_gt_mean")
        msd_g2p = sac.get("msd_gt_to_pred_mean")
        if msd_p2g is not None and msd_g2p is not None:
            asymm_str = f"{(msd_p2g - msd_g2p):+5.2f}"
        else:
            asymm_str = "    —"

        lines.append(
            f"  {sg['label']:<24}  {sg['n']:>3}  "
            f"{_fmm(sac.get('assd_mean')):>9}  {_fmm(sac.get('hd95_mean')):>9}  "
            f"{_fmm(lh.get('assd_mean')):>10}  {_fmm(lh.get('hd95_mean')):>10}  "
            f"{_fmm(rh.get('assd_mean')):>10}  {_fmm(rh.get('hd95_mean')):>10}  "
            f"{asymm_str:>14}"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(results: List[Dict], path: Path) -> None:
    """Per-case CSV. Includes Dice + all four surface metrics per class."""
    import csv
    rows = []
    for r in results:
        if not r.get("ok"):
            continue
        r = _normalize_for_aggregation(r)
        row = {"token": r["token"], "config": r["config"],
               "match_type": r["match_type"], "lstv_label": r["lstv_label"],
               "position": r["position"], "has_l6": r["has_l6"],
               "l6_gt_present": r["l6_gt_present"]}
        for cls in FOREGROUND_CLASSES:
            name = CLASS_NAMES[cls]
            row[f"dice_{name}"] = r["dice"].get(cls, "")
            for f in SURFACE_METRIC_FIELDS:
                row[f"{f}_{name}"] = (r.get(f, {}) or {}).get(cls, "")
        jxn = r.get("junction", {}) if isinstance(r.get("junction"), dict) else {}
        row["junction_dsc"] = jxn.get("mean_junction_dsc", "")
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Per-case CSV → %s  (%d rows)", path, len(rows))


# =============================================================================
# Case selection
# =============================================================================

def select_cases(ds, args) -> List:
    """
    Zero-shot benchmarking — no train/val/test filter. Run TS on every
    available case and stratify subgroups (config, LSTV class, match_type)
    only at aggregation time.
    """
    if args.config == "all":
        universe = ds.filter(present_only=True)
    else:
        universe = ds.filter(config=args.config, present_only=True)
    if args.tokens.strip():
        toks = {t.strip() for t in args.tokens.split(",") if t.strip()}
        universe = [c for c in universe if c.token in toks]
    return universe


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default=None, type=Path)
    ap.add_argument("--hf_repo_id",  default="anonymous-mlhc/CTSpinoPelvic1K")
    ap.add_argument("--hf_token",    default=None)
    ap.add_argument("--config",     default="all",
                    choices=["all", "fused", "spine_only", "pelvic_native"],
                    help="Which dataset config to benchmark.")
    ap.add_argument("--tokens",      default="", type=str,
                    help="Optional comma-separated token subset (debugging / sharding).")
    ap.add_argument("--device",    default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--fast",      action="store_true",
                    help="TS --fast mode (NOT recommended for publication)")
    ap.add_argument("--force_ts",  action="store_true")
    ap.add_argument("--force_recompute_metrics", action="store_true",
                    help="Ignore per_case_partial.jsonl and recompute Dice + "
                         "surface metrics for every case (predictions still cached).")
    # Canonical flag (covers HD95, ASSD, MSD).
    ap.add_argument("--skip_surface", action="store_true",
                    help="Skip ALL surface metrics (HD95, ASSD, MSD).")
    # Legacy alias for old SLURM scripts that still pass --skip_hd95.
    ap.add_argument("--skip_hd95", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--window_mm", default=JUNCTION_WINDOW_MM, type=float)
    ap.add_argument("--out_dir",  required=True, type=Path)
    ap.add_argument("--pred_dir", default=None,  type=Path)
    args = ap.parse_args()

    skip_surface = bool(args.skip_surface or args.skip_hd95)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.pred_dir or (args.out_dir / "ts_preds")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Streaming per-case JSONL log lives under out_dir, not pred_dir, so
    # different shards / different runs each get their own metrics log
    # while sharing the prediction cache.
    per_case_log = args.out_dir / "per_case_partial.jsonl"

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_interface import CTSpinoPelvic1K

    if args.dataset_dir:
        ds = CTSpinoPelvic1K(args.dataset_dir)
    else:
        ds = CTSpinoPelvic1K.from_hub(
            repo_id=args.hf_repo_id,
            token=args.hf_token or os.environ.get("HF_TOKEN"),
        )
    log.info(ds.stats())

    cases = select_cases(ds, args)
    if not cases:
        log.error("No cases found for config='%s' tokens='%s'.",
                  args.config, args.tokens)
        sys.exit(1)

    from collections import Counter
    cfg_counts = Counter(c.config for c in cases)
    log.info("Benchmarking %d cases  config=%s  per-config=%s  device=%s  fast=%s  "
             "skip_surface=%s",
             len(cases), args.config, dict(cfg_counts), args.device, args.fast,
             skip_surface)

    # ── Resume: read prior streamed results, build skip set ─────────────────
    if args.force_recompute_metrics:
        log.info("--force_recompute_metrics: ignoring %s", per_case_log)
        results: List[Dict] = []
        already_done: Set[str] = set()
    else:
        results, already_done = load_completed_results(per_case_log)

    cases_to_run = [c for c in cases
                    if f"{c.token}__{c.config}" not in already_done]
    n_skip = len(cases) - len(cases_to_run)
    if n_skip:
        log.info("Skipping %d already-completed (token, config) pairs from %s",
                 n_skip, per_case_log.name)

    # ── HF download (if needed) for cases we still have to do ──────────────
    if not args.dataset_dir and cases_to_run:
        try:
            from huggingface_hub import hf_hub_download
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            for case in cases_to_run:
                for rel in (f"ct/{case.ct_path.name}",
                            f"labels/{case.label_path.name}"):
                    target = ds.root / rel
                    if not target.exists():
                        hf_hub_download(
                            repo_id=args.hf_repo_id, repo_type="dataset",
                            filename=rel, token=hf_token,
                            local_dir=str(ds.root),
                        )
        except Exception as e:
            log.warning("HF download: %s", e)

    # ── Main loop with streaming write ─────────────────────────────────────
    t0 = time.time()
    for i, case in enumerate(cases_to_run, 1):
        log.info("[%d/%d]  token=%-6s  config=%-14s  lstv=%s",
                 i, len(cases_to_run), case.token, case.config, case.lstv_label)
        case_meta = {
            "config":              case.config,
            "match_type":          case.match_type,
            "lstv_label":          case.lstv_label,
            "position":            case.position,
            "has_l6":              case.has_l6,
            "lstv_agreement":      case.lstv_agreement,
            "lstv_confusion_zone": case.lstv_confusion_zone,
        }
        r = benchmark_one(
            token           = case.token,
            ct_path         = case.ct_path,
            label_path      = case.label_path,
            pred_dir        = pred_dir,
            case_meta       = case_meta,
            device          = args.device,
            fast            = args.fast,
            skip_surface    = skip_surface,
            force_ts        = args.force_ts,
            junction_window = args.window_mm,
        )
        results.append(r)
        # Stream this case's result to disk before moving to the next case.
        # If the job is killed at this point, we lose at most one in-flight
        # case's metrics (the prediction itself is still cached).
        append_result_jsonl(per_case_log, r)

        elapsed = time.time() - t0
        log.info("  progress %d/%d  elapsed=%.0fs", i, len(cases_to_run), elapsed)

    # ── Final aggregation pass ─────────────────────────────────────────────
    summary = aggregate(results)
    t5 = format_table5(summary)
    t6 = format_table_surface(summary)
    print(t5)
    print(t6)
    (args.out_dir / "paper_tables.txt").write_text(t5 + "\n" + t6)
    (args.out_dir / "benchmark_results.json").write_text(
        json.dumps({"config": vars(args), "summary": summary, "per_case": results},
                   indent=2, default=str))
    (args.out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    write_csv(results, args.out_dir / "benchmark_per_case.csv")

    log.info("DONE  ok=%d  fail=%d  total_time=%.0fs  per_case_log=%s",
             summary["n_ok"], summary["n_fail"], time.time() - t0, per_case_log)


if __name__ == "__main__":
    main()
