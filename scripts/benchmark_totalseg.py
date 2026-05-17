#!/usr/bin/env python3
"""
benchmark_totalseg.py — TotalSegmentator zero-shot benchmark on CTSpinoPelvic1K.

Apr 2026 v6 update — 6-way LSTV subgroups via splits v6 patient_subtypes
========================================================================
Tables 5 and 6 now report 6-way LSTV subgroups (was 4-way). The new
taxonomy mirrors splits_5fold.json schema v6:

  normal | lumb | sacr_count | semisacralization | sacralization | ambiguous

The previous 4-way scheme silently merged semisacralization into
sacralization (parser bug in mask_index.py). With the parser fix, the 2
semi patients are now their own subgroup.

Subgroup binning is splits-aware:
  1. If --splits_file points at v6+ splits, use patient_subtypes[token]
     (6-way: normal/lumb/sacr_count/semisacralization/sacralization/ambiguous).
  2. If v5 splits, use token_info[token].lstv_subtype (legacy 4-way).
  3. Otherwise fall back to per-record lstv_label string (3-way).

Patient-level aggregation is unchanged from v8/v9: separate-mode patients
(spine_only + pelvic_native records) are merged into one patient row.

Re-aggregation without re-running TS:
    python3 scripts/benchmark_totalseg.py \\
        --reaggregate_only \\
        --out_dir     results/totalseg_bench_<jobid>/_merged \\
        --splits_file data/hf_export/splits_5fold.json

Pass --record_level to disable patient-level dedup and report records
(legacy behavior; useful for debugging).

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
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
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6", 7: "sacrum", 8: "hip_left", 9: "hip_right",
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

SURFACE_METRIC_FIELDS = ("hd95", "assd", "msd_pred_to_gt", "msd_gt_to_pred")
EXIT_CIRCUIT_BREAKER = 2


# =============================================================================
# 6-way LSTV subgroup taxonomy
# =============================================================================

# Canonical 6-way subgroups (mirrors splits_5fold.json schema v6 SUBTYPES)
SUBGROUP_NORMAL = "normal"
SUBGROUP_LUMB   = "lumb"
SUBGROUP_SACR_COUNT = "sacr_count"
SUBGROUP_SEMI   = "semisacralization"
SUBGROUP_SACR   = "sacralization"
SUBGROUP_AMBIG  = "ambiguous"
SUBGROUPS_6WAY = (
    SUBGROUP_NORMAL,
    SUBGROUP_LUMB,
    SUBGROUP_SACR_COUNT,
    SUBGROUP_SEMI,
    SUBGROUP_SACR,
    SUBGROUP_AMBIG,
)


def _canonicalize_lstv_subtype(s) -> str:
    """Coerce arbitrary LSTV string to a 6-way subgroup label.

    Recognized inputs (case-insensitive substrings):
      - 'ambig*'  -> ambiguous
      - 'lumb*'   -> lumb
      - 'semi*'   -> semisacralization
      - 'sacr_count' / 'sacrcount' -> sacr_count
      - 'sacr*'   -> sacralization (default sacr bucket)
      - everything else -> normal
    """
    if s is None:
        return SUBGROUP_NORMAL
    v = str(s).strip().lower()
    if not v or v in ("unknown", "none", "n/a", "na"):
        return SUBGROUP_NORMAL
    if v.startswith("ambig"):
        return SUBGROUP_AMBIG
    if "lumb" in v:
        return SUBGROUP_LUMB
    # Order matters: check 'semi' before 'sacr' since "semisacralization"
    # contains both substrings.
    if "semi" in v:
        return SUBGROUP_SEMI
    if "sacr_count" in v or "sacrcount" in v:
        return SUBGROUP_SACR_COUNT
    if "sacr" in v:
        return SUBGROUP_SACR
    return SUBGROUP_NORMAL


def load_splits_subtype_map(splits_path: Optional[Path]) -> Tuple[Dict[str, str], int]:
    """Load token -> subtype map from splits_5fold.json.

    Schema-aware:
      v6+: data['patient_subtypes'] is a flat dict {token: subtype}
      v5:  data['token_info'] is a dict {token: {'lstv_subtype': ...}}
      <v5: returns {} (no subtype info)

    Returns (token -> subtype dict, schema_version).
    """
    if splits_path is None or not splits_path.exists():
        return {}, 0
    try:
        data = json.loads(splits_path.read_text())
    except Exception as e:
        log.warning("Could not parse splits %s: %s", splits_path, e)
        return {}, 0

    schema = int(data.get("schema_version", 0))
    out: Dict[str, str] = {}

    if schema >= 6:
        ps = data.get("patient_subtypes") or {}
        if isinstance(ps, dict):
            out = {str(k): str(v) for k, v in ps.items()}
        log.info("Loaded splits_5fold.json schema v%d (%d tokens) — 6-way subtype binning",
                 schema, len(out))
    elif schema >= 5:
        info = data.get("token_info") or {}
        if isinstance(info, dict):
            for k, rec in info.items():
                if not isinstance(rec, dict):
                    continue
                sub = rec.get("lstv_subtype")
                if sub:
                    out[str(k)] = str(sub)
        log.warning("Splits schema v%d < 6; semisacralization unavailable. "
                    "Using legacy 4-way binning. Re-run generate_5fold_splits.py "
                    "to upgrade to v6.", schema)
    else:
        log.warning("Splits schema v%d < 5; no subtype info. Falling back to "
                    "lstv_label strings (3-way).", schema)

    return out, schema


def resolve_subgroup(record: Dict, splits_subtypes: Dict[str, str]) -> str:
    """Resolve LSTV 6-way subgroup for a per-record dict.

    Priority:
      1. splits_subtypes[token] if available (canonical 6-way)
      2. record['lstv_label'] string (fallback)
    """
    token = str(record.get("token", ""))
    if token and token in splits_subtypes:
        return _canonicalize_lstv_subtype(splits_subtypes[token])
    return _canonicalize_lstv_subtype(record.get("lstv_label"))


# Back-compat alias for code that called the old 4-way name.
load_splits_token_info = load_splits_subtype_map


# =============================================================================
# Metrics (unchanged)
# =============================================================================

def dice(pred, gt, cls):
    p = pred == cls; g = gt == cls
    if not p.any() and not g.any(): return float("nan")
    denom = p.sum() + g.sum()
    return float(2 * (p & g).sum() / denom) if denom > 0 else float("nan")


def _surface_points_mm(mask, spacing):
    from scipy.ndimage import binary_erosion
    if not mask.any(): return None
    surf = mask ^ binary_erosion(mask)
    if not surf.any(): return None
    return np.argwhere(surf).astype(np.float32) * np.asarray(spacing, dtype=np.float32)


def surface_metrics(pred, gt, cls, vox_spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    from scipy.spatial import cKDTree
    nan_dict = {k: float("nan") for k in SURFACE_METRIC_FIELDS}
    p_pts = _surface_points_mm((pred == cls).astype(bool), vox_spacing)
    g_pts = _surface_points_mm((gt == cls).astype(bool), vox_spacing)
    if p_pts is None or g_pts is None: return nan_dict
    p_tree, g_tree = cKDTree(p_pts), cKDTree(g_pts)
    p_to_g, _ = g_tree.query(p_pts, k=1, workers=-1)
    g_to_p, _ = p_tree.query(g_pts, k=1, workers=-1)
    union = np.concatenate([p_to_g, g_to_p])
    return {
        "hd95": float(np.percentile(union, 95)),
        "assd": float(union.mean()),
        "msd_pred_to_gt": float(p_to_g.mean()),
        "msd_gt_to_pred": float(g_to_p.mean()),
    }


def hausdorff95(pred, gt, cls, vox_spacing=(1.0, 1.0, 1.0)):
    return surface_metrics(pred, gt, cls, vox_spacing)["hd95"]


def junction_analysis(pred, gt, affine, window_mm=JUNCTION_WINDOW_MM) -> Dict:
    vox_sz_z = float(np.abs(np.linalg.norm(affine[:3, 2])))
    half = int(np.ceil(window_mm / 2.0 / max(vox_sz_z, 0.1)))
    l5_vox = np.argwhere(gt == 5)
    if len(l5_vox) > 0:
        centre_z = int(round(float(l5_vox[:, 2].mean())))
    else:
        s1_vox = np.argwhere(gt == 7)
        if len(s1_vox) == 0:
            return {"error": "no_L5_or_sacrum_in_GT", "window_mm": window_mm}
        centre_z = int(s1_vox[:, 2].max())
    lo, hi = max(0, centre_z - half), min(gt.shape[2], centre_z + half)
    gt_w, pred_w = gt[..., lo:hi], pred[..., lo:hi]
    jxn_dice = {cls: dice(pred_w, gt_w, cls) for cls in JUNCTION_CLASSES}
    jxn_vals = [v for v in jxn_dice.values() if v == v and v is not None]
    mean_jxn_dsc = float(np.mean(jxn_vals)) if jxn_vals else float("nan")
    confusion = defaultdict(lambda: defaultdict(int))
    total_jv = 0
    for gt_cls in JUNCTION_CLASSES:
        mask = gt_w == gt_cls
        if not mask.any(): continue
        total_jv += int(mask.sum())
        pv, cnt = np.unique(pred_w[mask], return_counts=True)
        for p, c in zip(pv.tolist(), cnt.tolist()):
            confusion[gt_cls][int(p)] += c
    error_vox = sum(c for gc, pd in confusion.items() for pc, c in pd.items() if pc != gc)
    error_rate = float(error_vox / total_jv) if total_jv > 0 else float("nan")
    l5_total = sum(confusion.get(5, {}).values())
    l5_as_s1 = int(confusion.get(5, {}).get(7, 0))
    l5_sacrum_rate = float(l5_as_s1 / l5_total) if l5_total > 0 else float("nan")
    s1_total = sum(confusion.get(7, {}).values())
    s1_as_l5 = int(confusion.get(7, {}).get(5, 0))
    s1_l5_rate = float(s1_as_l5 / s1_total) if s1_total > 0 else float("nan")
    return {
        "centre_z": centre_z, "window_lo_hi": [lo, hi], "window_mm": window_mm,
        "n_junction_voxels": total_jv,
        "mean_junction_dsc": round(mean_jxn_dsc, 4),
        "junction_dice": {CLASS_NAMES[c]: (round(v, 4) if v == v else None)
                            for c, v in jxn_dice.items()},
        "labelling_error_rate": round(error_rate, 4) if error_rate == error_rate else None,
        "l5_called_sacrum_rate": round(l5_sacrum_rate, 4) if l5_sacrum_rate == l5_sacrum_rate else None,
        "s1_called_l5_rate": round(s1_l5_rate, 4) if s1_l5_rate == s1_l5_rate else None,
        "ts_assigned_l6": bool((pred_w == 6).any()),
        "confusion_matrix": {
            CLASS_NAMES.get(gc, str(gc)): {
                CLASS_NAMES.get(pc, str(pc)): cnt for pc, cnt in pd.items()
            } for gc, pd in confusion.items()
        },
    }


# =============================================================================
# TS inference + remap (unchanged)
# =============================================================================

def run_totalseg(ct_path, out_dir, device="gpu", fast=False, force=False) -> Optional[Path]:
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    ml_path = out_dir / "segmentation.nii.gz"
    if ml_path.exists() and not force: return ml_path
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("    TS: running on %s (device=%s fast=%s)", ct_path.name, device, fast)
    t0 = time.time()
    try:
        pred = totalsegmentator(
            input=nib.load(str(ct_path)), output=None, task="total",
            ml=True, device=device, fast=fast,
            roi_subset=TS_ROI_SUBSET, verbose=False)
        nib.save(pred, str(ml_path))
        log.info("    TS: done %.0fs", time.time() - t0)
        return ml_path
    except Exception as e:
        log.error("    TS FAILED: %s", e)
        return None


def resample_and_remap(ts_path, ref_path):
    import nibabel as nib
    import SimpleITK as sitk
    moving = sitk.ReadImage(str(ts_path), sitk.sitkInt32)
    fixed = sitk.ReadImage(str(ref_path), sitk.sitkInt32)
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
    ref_nib = nib.load(str(ref_path))
    if arr.shape != ref_nib.shape:
        raise ValueError(f"shape mismatch: pred={arr.shape} ref={ref_nib.shape}")
    unified = np.zeros_like(arr, dtype=np.int16)
    for ts_id, cls in TS_TO_UNIFIED.items():
        unified[arr == ts_id] = cls
    return unified


# =============================================================================
# Per-case driver (unchanged)
# =============================================================================

def benchmark_one(token, ct_path, label_path, pred_dir, case_meta,
                   device="gpu", fast=False, skip_surface=False,
                   force_ts=False, junction_window=JUNCTION_WINDOW_MM) -> Dict:
    import nibabel as nib
    config = case_meta.get("config", "fused")
    scoreable = SCOREABLE_BY_CONFIG.get(config, FOREGROUND_CLASSES)
    result = {
        "token": token, "config": config,
        "match_type": case_meta.get("match_type", ""),
        "lstv_label": case_meta.get("lstv_label", "normal"),
        "position": case_meta.get("position", "unknown"),
        "has_l6": case_meta.get("has_l6", False),
        "lstv_agreement": case_meta.get("lstv_agreement"),
        "lstv_confusion_zone": case_meta.get("lstv_confusion_zone"),
        "scoreable_classes": scoreable,
        "ok": False, "error": None,
        "dice": {}, "hd95": {}, "assd": {},
        "msd_pred_to_gt": {}, "msd_gt_to_pred": {},
        "junction": {},
        "n_gt_classes": 0, "l6_gt_present": False, "l6_dice_meaningful": None,
    }
    try:
        cache_subdir = pred_dir / f"{token}_{config}"
        ts_pred = run_totalseg(ct_path, cache_subdir, device=device, fast=fast, force=force_ts)
        if ts_pred is None: raise RuntimeError("TotalSegmentator inference failed")
        pred = resample_and_remap(ts_pred, label_path)
        gt_img = nib.load(str(label_path))
        gt = np.asarray(gt_img.dataobj, dtype=np.int16)
        affine = gt_img.affine
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch")
        gt_classes = sorted({int(v) for v in np.unique(gt)} - {0})
        result["n_gt_classes"] = len(gt_classes)
        result["l6_gt_present"] = bool(6 in gt_classes)
        for cls in FOREGROUND_CLASSES:
            if cls not in scoreable:
                result["dice"][cls] = None; continue
            d = dice(pred, gt, cls)
            result["dice"][cls] = round(d, 4) if d == d else None
        if result["l6_gt_present"]: result["l6_dice_meaningful"] = 0.0
        if not skip_surface:
            for cls in TS_CAPABLE_CLASSES:
                if cls not in scoreable:
                    for f in SURFACE_METRIC_FIELDS: result[f][cls] = None
                    continue
                m = surface_metrics(pred, gt, cls, vox_spacing=spacing)
                for f in SURFACE_METRIC_FIELDS:
                    v = m[f]
                    result[f][cls] = round(v, 4) if v == v else None
        if config == "fused":
            result["junction"] = junction_analysis(pred, gt, affine, window_mm=junction_window)
        else:
            result["junction"] = {"error": f"junction_skipped_config_{config}"}
        result["ok"] = True
        d_str = "  ".join(
            f"{CLASS_NAMES[c]}={(result['dice'].get(c) if result['dice'].get(c) is not None else float('nan')):.3f}"
            for c in [1,2,3,4,5,7,8,9])
        log.info("  %-6s cfg=%s %s lstv=%s",
                 token, config, d_str, result["lstv_label"])
    except Exception:
        result["error"] = traceback.format_exc()
        log.error("  FAIL %s/%s: %s", token, config, result["error"].splitlines()[-1])
    return result


# =============================================================================
# JSONL load/save (unchanged)
# =============================================================================

def load_completed_results(per_case_log: Path) -> Tuple[List[Dict], Set[str], int, int]:
    if not per_case_log.exists(): return [], set(), 0, 0
    by_key, n_lines, n_bad = {}, 0, 0
    with open(per_case_log) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_lines += 1
            try: rec = json.loads(line)
            except json.JSONDecodeError: n_bad += 1; continue
            tok, cfg = rec.get("token"), rec.get("config")
            if not tok or not cfg: n_bad += 1; continue
            by_key[f"{tok}__{cfg}"] = rec
    seeded, done, n_ok, n_fail = [], set(), 0, 0
    for k, r in by_key.items():
        if r.get("ok"):
            seeded.append(r); done.add(k); n_ok += 1
        else: n_fail += 1
    log.info("Resumed from %s: %d lines, %d unique (bad=%d, ok=%d, fail=%d)",
             per_case_log, n_lines, len(by_key), n_bad, n_ok, n_fail)
    return seeded, done, n_ok, n_fail


def append_result_jsonl(per_case_log: Path, result: Dict):
    per_case_log.parent.mkdir(parents=True, exist_ok=True)
    with open(per_case_log, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
        f.flush()
        try: os.fsync(f.fileno())
        except OSError: pass


# =============================================================================
# Patient-level deduplication (unchanged from v8/v9)
# =============================================================================

def _normalize_for_aggregation(r: Dict) -> Dict:
    """Coerce JSONL-roundtripped str keys back to int."""
    for field in ("dice",) + SURFACE_METRIC_FIELDS:
        d = r.get(field) or {}
        new_d = {}
        for k, v in d.items():
            try: new_d[int(k)] = v
            except (ValueError, TypeError): new_d[k] = v
        r[field] = new_d
    return r


def merge_patient_records(records: List[Dict]) -> Dict:
    """Merge multiple per-config records for the same patient into one."""
    if not records:
        return {}
    if len(records) == 1:
        return records[0]

    records = [_normalize_for_aggregation(r) for r in records]
    out = dict(records[0])

    for field in ("dice",) + SURFACE_METRIC_FIELDS:
        merged: Dict[int, Optional[float]] = {}
        for cls in FOREGROUND_CLASSES:
            vals = []
            for r in records:
                v = (r.get(field) or {}).get(cls)
                if v is not None and v == v:
                    vals.append(float(v))
            merged[cls] = float(np.mean(vals)) if vals else None
        out[field] = merged

    out["junction"] = {"error": "no_fused_record_for_patient"}
    for r in records:
        if r.get("config") == "fused" and r.get("junction") and "error" not in r["junction"]:
            out["junction"] = r["junction"]
            break

    out["ok"] = any(r.get("ok") for r in records)
    out["l6_gt_present"] = any(r.get("l6_gt_present") for r in records)
    out["n_gt_classes"] = max(r.get("n_gt_classes", 0) for r in records)

    configs = sorted({r.get("config", "?") for r in records})
    out["config"] = "+".join(configs) if len(configs) > 1 else configs[0]
    out["_n_records_merged"] = len(records)

    for r in records:
        lab = str(r.get("lstv_label", "")).strip().lower()
        if lab and lab != "normal":
            out["lstv_label"] = r["lstv_label"]
            break

    return out


def deduplicate_to_patient_level(records: List[Dict]) -> List[Dict]:
    """Group records by token and merge each group into one patient record."""
    by_token: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_token[str(r.get("token", ""))].append(r)
    return [merge_patient_records(recs) for recs in by_token.values()]


# =============================================================================
# Aggregation — splits-aware (6-way) + patient-level
# =============================================================================

def _nanmean(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.mean(v)), 4) if v else None

def _nanstd(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.std(v)), 4) if len(v) > 1 else None


def class_stats(cases):
    stats = {}
    for cls in FOREGROUND_CLASSES:
        entry = {"n_cases": len(cases)}
        dvals = [r["dice"].get(cls) for r in cases]
        entry["dice_mean"] = _nanmean(dvals)
        entry["dice_std"] = _nanstd(dvals)
        for f in SURFACE_METRIC_FIELDS:
            vals = [r.get(f, {}).get(cls) for r in cases if r.get(f)]
            entry[f"{f}_mean"] = _nanmean(vals)
            entry[f"{f}_std"] = _nanstd(vals)
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


def aggregate(results: List[Dict],
              splits_subtypes: Optional[Dict[str, str]] = None,
              patient_level: bool = True,
              splits_schema: int = 0) -> Dict:
    """Aggregate per-case results into 6-way subgroup tables.

    Args:
        results: list of per-record dicts from the JSONL log.
        splits_subtypes: dict {token: subtype} from splits_5fold.json.
            v6+ -> 6-way; v5 -> 4-way; empty -> 3-way fallback via lstv_label.
        patient_level: if True, records deduplicated by token and
            multi-config records merged before aggregation.
        splits_schema: source schema version (for reporting).
    """
    if splits_subtypes is None: splits_subtypes = {}
    results = [_normalize_for_aggregation(r) for r in results]
    ok_records = [r for r in results if r.get("ok")]

    if patient_level:
        ok = deduplicate_to_patient_level(ok_records)
        log.info("Aggregation: deduped %d records -> %d patients",
                 len(ok_records), len(ok))
    else:
        ok = ok_records
        log.info("Aggregation: %d records (record-level mode)", len(ok))

    for r in ok:
        r["_subgroup"] = resolve_subgroup(r, splits_subtypes)

    def _sub(cases, label):
        if not cases: return {"n": 0, "label": label}
        return {"n": len(cases), "label": label,
                "classes": class_stats(cases),
                "junction": junction_stats(cases)}

    # 6-way subgroup buckets
    sub_normal     = [r for r in ok if r["_subgroup"] == SUBGROUP_NORMAL]
    sub_lumb       = [r for r in ok if r["_subgroup"] == SUBGROUP_LUMB]
    sub_sacr_count = [r for r in ok if r["_subgroup"] == SUBGROUP_SACR_COUNT]
    sub_semi       = [r for r in ok if r["_subgroup"] == SUBGROUP_SEMI]
    sub_sacr       = [r for r in ok if r["_subgroup"] == SUBGROUP_SACR]
    sub_ambig      = [r for r in ok if r["_subgroup"] == SUBGROUP_AMBIG]

    # Composite: any LSTV (everything except normal)
    sub_any_lstv   = sub_lumb + sub_sacr_count + sub_semi + sub_sacr + sub_ambig
    # Composite: any sacralization-flavored (sacr_count + semi + sacr)
    sub_any_sacr   = sub_sacr_count + sub_semi + sub_sacr

    # Determine reporting source
    if splits_schema >= 6:
        source_str = "splits_v6_patient_subtypes"
    elif splits_schema >= 5:
        source_str = "splits_v5_token_info"
    elif splits_subtypes:
        source_str = "splits_legacy"
    else:
        source_str = "lstv_label_string"

    def _is_config(r, cfg):
        c = str(r.get("config", ""))
        return cfg in c.split("+")

    summary = {
        "n_total_records": len(results),
        "n_ok_records": len(ok_records),
        "n_fail_records": len(results) - len(ok_records),
        "patient_level": patient_level,
        "subgroup_source": source_str,
        "subgroup_taxonomy": "6-way" if splits_schema >= 6 else (
            "4-way (legacy)" if splits_schema >= 5 else "3-way (lstv_label fallback)"),
        "subgroup_counts": {
            SUBGROUP_NORMAL:     len(sub_normal),
            SUBGROUP_LUMB:       len(sub_lumb),
            SUBGROUP_SACR_COUNT: len(sub_sacr_count),
            SUBGROUP_SEMI:       len(sub_semi),
            SUBGROUP_SACR:       len(sub_sacr),
            SUBGROUP_AMBIG:      len(sub_ambig),
        },
        "subgroups": {
            "all":           _sub(ok, "All cases"),
            "fused_only":    _sub([r for r in ok if _is_config(r, "fused")], "Fused only"),
            "spine_only":    _sub([r for r in ok if _is_config(r, "spine_only")], "Spine only"),
            "pelvic_native": _sub([r for r in ok if _is_config(r, "pelvic_native")], "Pelvic native"),
            "normal":             _sub(sub_normal, "Normal"),
            "any_lstv":           _sub(sub_any_lstv, "Any LSTV"),
            "any_sacralization":  _sub(sub_any_sacr, "Any sacralization (semi+full+count)"),
            "lumb":               _sub(sub_lumb, "Lumbarization"),
            "sacr_count":         _sub(sub_sacr_count, "Sacralization (count-style)"),
            "semisacralization":  _sub(sub_semi, "Semi-sacralization"),
            "sacralization":      _sub(sub_sacr, "Sacralization (full)"),
            "ambiguous":          _sub(sub_ambig, "Ambiguous"),
        },
    }
    return summary


# =============================================================================
# Tables (6-way row order)
# =============================================================================

TABLE_ROW_ORDER = [
    "all", "fused_only", "spine_only", "pelvic_native",
    "normal", "any_lstv", "any_sacralization",
    "lumb", "sacr_count", "semisacralization", "sacralization", "ambiguous",
]


def _level_str(summary) -> str:
    return "patients" if summary.get("patient_level", True) else "records"


def format_table5(summary) -> str:
    def _f(v):
        return f"{v:>5.3f}" if v is not None and v == v else f"{'—':>5}"
    level = _level_str(summary)
    lines = [
        "",
        "=" * 110,
        f"  TABLE 5  —  TotalSegmentator Zero-Shot Benchmark (per-vertebra Dice; n = {level})",
        "=" * 110,
    ]
    src_map = {
        "splits_v6_patient_subtypes": "splits_5fold.json:patient_subtypes (6-way LSTV)",
        "splits_v5_token_info":       "splits_5fold.json:token_info (4-way LSTV, legacy)",
        "splits_legacy":              "splits file (unknown schema)",
        "lstv_label_string":          "per-record lstv_label string (3-way fallback)",
    }
    src = src_map.get(summary.get("subgroup_source", ""), "unknown")
    lines.append(f"  Subgroup source: {src}")
    lines.append(f"  Taxonomy:        {summary.get('subgroup_taxonomy', '?')}")
    if summary.get("patient_level", True):
        lines.append(f"  Aggregation:     patient-level (separate-mode patients merged)")
    else:
        lines.append(f"  Aggregation:     record-level (each spine/pelvic record counted separately)")
    header = (
        f"\n  {'Subgroup':<40}  {'n':>3}  "
        f"{'L1':>5}  {'L2':>5}  {'L3':>5}  {'L4':>5}  {'L5':>5}  "
        f"{'L6':>5}  {'Sac':>5}  {'HipL':>5}  {'HipR':>5}  {'JxnDSC':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * 114)
    for key in TABLE_ROW_ORDER:
        sg = summary["subgroups"].get(key)
        if not sg or sg.get("n", 0) == 0: continue
        cls = sg.get("classes", {})
        jxn = sg.get("junction", {})
        lumbar = "  ".join(_f(cls.get(n, {}).get("dice_mean"))
                            for n in ("L1","L2","L3","L4","L5"))
        pelvic = "  ".join(_f(cls.get(n, {}).get("dice_mean"))
                            for n in ("sacrum","hip_left","hip_right"))
        lines.append(
            f"  {sg['label']:<40}  {sg['n']:>3}  {lumbar}  {'—':>5}  {pelvic}  "
            f"{_f(jxn.get('mean_junction_dsc')):>7}"
        )
    lines.append("")
    lines.append("  † L6 column = '—': TS has no L6 label.")
    lines.append("  ‡ Subgroup definitions (filename qualifier on pelvic mask):")
    lines.append("    - Lumbarization        : 6 lumbar vertebrae or has_l6")
    lines.append("    - Sacralization (count): vert=SACR + n_lumb=4 (no qualifier)")
    lines.append("    - Semi-sacralization   : '_semisacralization_' in pelvic mask")
    lines.append("    - Sacralization (full) : '_sacralization_' in pelvic mask + 0/5 lumbars")
    lines.append("    - Ambiguous            : cross-anatomy disagreement (vert vs pelvic)")
    return "\n".join(lines)


def format_table_surface(summary) -> str:
    def _fmm(v):
        return f"{v:>5.2f}" if v is not None and v == v else f"{'—':>5}"
    level = _level_str(summary)
    lines = [
        "",
        "=" * 124,
        f"  TABLE 6  —  Surface-distance metrics (mm) for surgical-planning relevance (n = {level})",
        "=" * 124,
        "",
        "  ASSD = Average Symmetric Surface Distance (lower = better)",
        "  HD95 = 95th-percentile Hausdorff distance (lower = better; outlier-sensitive)",
        "  MSD asymm = msd_pred_to_gt - msd_gt_to_pred for sacrum",
        "              (positive = TS over-segments, negative = TS under-segments)",
    ]
    header = (
        f"\n  {'Subgroup':<40}  {'n':>3}  "
        f"{'Sac ASSD':>9}  {'Sac HD95':>9}  {'HipL ASSD':>10}  {'HipL HD95':>10}  "
        f"{'HipR ASSD':>10}  {'HipR HD95':>10}  {'Sac MSD asym':>13}"
    )
    lines.append(header)
    lines.append("  " + "-" * 122)
    for key in TABLE_ROW_ORDER:
        sg = summary["subgroups"].get(key)
        if not sg or sg.get("n", 0) == 0: continue
        cls = sg.get("classes", {})
        sac, lh, rh = cls.get("sacrum", {}), cls.get("hip_left", {}), cls.get("hip_right", {})
        msd_p2g, msd_g2p = sac.get("msd_pred_to_gt_mean"), sac.get("msd_gt_to_pred_mean")
        asymm = f"{(msd_p2g - msd_g2p):+5.2f}" if (msd_p2g is not None and msd_g2p is not None) else "    —"
        lines.append(
            f"  {sg['label']:<40}  {sg['n']:>3}  "
            f"{_fmm(sac.get('assd_mean')):>9}  {_fmm(sac.get('hd95_mean')):>9}  "
            f"{_fmm(lh.get('assd_mean')):>10}  {_fmm(lh.get('hd95_mean')):>10}  "
            f"{_fmm(rh.get('assd_mean')):>10}  {_fmm(rh.get('hd95_mean')):>10}  "
            f"{asymm:>13}"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(results: List[Dict], path: Path,
              splits_subtypes: Optional[Dict[str, str]] = None,
              patient_level: bool = True) -> None:
    """Per-case CSV with 6-way subgroup column."""
    import csv
    if splits_subtypes is None: splits_subtypes = {}
    results = [_normalize_for_aggregation(r) for r in results]
    ok_records = [r for r in results if r.get("ok")]
    cases = deduplicate_to_patient_level(ok_records) if patient_level else ok_records

    rows = []
    for r in cases:
        sub = resolve_subgroup(r, splits_subtypes)
        row = {
            "token": r["token"], "config": r["config"],
            "match_type": r["match_type"],
            "lstv_label": r["lstv_label"],
            "lstv_subgroup": sub,
            "position": r["position"], "has_l6": r["has_l6"],
            "l6_gt_present": r["l6_gt_present"],
            "n_records_merged": r.get("_n_records_merged", 1),
        }
        for cls in FOREGROUND_CLASSES:
            name = CLASS_NAMES[cls]
            row[f"dice_{name}"] = r["dice"].get(cls, "")
            for f in SURFACE_METRIC_FIELDS:
                row[f"{f}_{name}"] = (r.get(f, {}) or {}).get(cls, "")
        jxn = r.get("junction", {}) if isinstance(r.get("junction"), dict) else {}
        row["junction_dsc"] = jxn.get("mean_junction_dsc", "")
        rows.append(row)
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Per-case CSV → %s (%d rows, %s)",
             path, len(rows), "patients" if patient_level else "records")


# =============================================================================
# Case selection (unchanged)
# =============================================================================

def select_cases(ds, args) -> List:
    if args.config == "all":
        universe = ds.filter(present_only=True)
    else:
        universe = ds.filter(config=args.config, present_only=True)
    if args.tokens.strip():
        toks = {t.strip() for t in args.tokens.split(",") if t.strip()}
        universe = [c for c in universe if c.token in toks]
    return universe


# =============================================================================
# Output writers
# =============================================================================

def write_final_outputs(results, args, out_dir, per_case_log, total_t0,
                         splits_subtypes=None, splits_schema=0,
                         partial=False, patient_level=True,
                         also_write_record_level=True):
    summary = aggregate(results, splits_subtypes=splits_subtypes,
                         splits_schema=splits_schema, patient_level=patient_level)
    t5 = format_table5(summary)
    t6 = format_table_surface(summary)
    if not partial:
        print(t5); print(t6)
    suffix = "_partial" if partial else ""
    (out_dir / f"paper_tables{suffix}.txt").write_text(t5 + "\n" + t6)
    (out_dir / f"benchmark_results{suffix}.json").write_text(
        json.dumps({"config": vars(args), "summary": summary, "per_case": results},
                   indent=2, default=str))
    (out_dir / f"benchmark_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    write_csv(results, out_dir / f"benchmark_per_case{suffix}.csv",
              splits_subtypes=splits_subtypes, patient_level=patient_level)

    if also_write_record_level and patient_level and not partial:
        rec_summary = aggregate(results, splits_subtypes=splits_subtypes,
                                 splits_schema=splits_schema, patient_level=False)
        rec_t5 = format_table5(rec_summary)
        rec_t6 = format_table_surface(rec_summary)
        (out_dir / "paper_tables_record_level.txt").write_text(rec_t5 + "\n" + rec_t6)
        (out_dir / "benchmark_summary_record_level.json").write_text(
            json.dumps(rec_summary, indent=2, default=str))
        write_csv(results, out_dir / "benchmark_per_case_record_level.csv",
                  splits_subtypes=splits_subtypes, patient_level=False)
        log.info("Also wrote record-level supplementary outputs.")

    log.info("%s ok_records=%d fail_records=%d total=%.0fs",
             "PARTIAL" if partial else "DONE",
             summary["n_ok_records"], summary["n_fail_records"],
             time.time() - total_t0)
    return summary


def reaggregate_only(args, splits_subtypes: Dict[str, str], splits_schema: int) -> int:
    per_case_log = args.out_dir / "per_case_partial.jsonl"
    if not per_case_log.exists():
        log.error("No %s found.", per_case_log)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, _, n_ok, n_fail = load_completed_results(per_case_log)
    if not results:
        log.error("No successful records in %s", per_case_log)
        return 1
    log.info("Re-aggregating %d records (schema v%d, level=%s)",
             len(results), splits_schema,
             "patient" if not args.record_level else "record")
    write_final_outputs(results, args, args.out_dir, per_case_log, time.time(),
                         splits_subtypes=splits_subtypes,
                         splits_schema=splits_schema,
                         partial=False, patient_level=not args.record_level)
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default=None, type=Path)
    ap.add_argument("--hf_repo_id", default="anonymous-neurips-ED/CTSpinoPelvic1K")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--config", default="all",
                     choices=["all", "fused", "spine_only", "pelvic_native"])
    ap.add_argument("--tokens", default="", type=str)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force_ts", action="store_true")
    ap.add_argument("--force_recompute_metrics", action="store_true")
    ap.add_argument("--retry_failed", action="store_true", default=True)
    ap.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    ap.add_argument("--skip_surface", action="store_true")
    ap.add_argument("--skip_hd95", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--window_mm", default=JUNCTION_WINDOW_MM, type=float)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--pred_dir", default=None, type=Path)
    ap.add_argument("--circuit_breaker_n", type=int,
                     default=int(os.environ.get("CIRCUIT_BREAKER_N", "3")))
    ap.add_argument("--circuit_breaker_secs", type=float,
                     default=float(os.environ.get("CIRCUIT_BREAKER_SECS", "10.0")))
    ap.add_argument("--splits_file", type=Path, default=None,
                     help="Path to splits_5fold.json (v6+) for 6-way LSTV "
                          "subgroup binning.")
    ap.add_argument("--reaggregate_only", action="store_true",
                     help="Skip TS inference; re-aggregate existing JSONL.")
    ap.add_argument("--record_level", action="store_true",
                     help="Disable patient-level deduplication; report at "
                          "record level (legacy behavior).")

    args = ap.parse_args()
    skip_surface = bool(args.skip_surface or args.skip_hd95)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits_subtypes, splits_schema = load_splits_subtype_map(args.splits_file)

    if args.reaggregate_only:
        return reaggregate_only(args, splits_subtypes, splits_schema)

    pred_dir = args.pred_dir or (args.out_dir / "ts_preds")
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_case_log = args.out_dir / "per_case_partial.jsonl"

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_interface import CTSpinoPelvic1K

    if args.dataset_dir:
        ds = CTSpinoPelvic1K(args.dataset_dir)
    else:
        ds = CTSpinoPelvic1K.from_hub(
            repo_id=args.hf_repo_id,
            token=args.hf_token or os.environ.get("HF_TOKEN"))
    log.info(ds.stats())

    cases = select_cases(ds, args)
    if not cases:
        log.error("No cases found."); sys.exit(1)

    from collections import Counter
    cfg_counts = Counter(c.config for c in cases)
    log.info("Benchmarking %d cases  per-config=%s  splits_schema=v%d  patient_level=%s",
             len(cases), dict(cfg_counts), splits_schema, not args.record_level)

    if args.force_recompute_metrics:
        results, already_done = [], set()
    else:
        if args.retry_failed:
            results, already_done, _, n_fail = load_completed_results(per_case_log)
            if n_fail: log.info("AUTO-RETRY: %d failed records will retry", n_fail)
        else:
            results, already_done = _load_completed_legacy(per_case_log)

    cases_to_run = [c for c in cases
                     if f"{c.token}__{c.config}" not in already_done]
    log.info("Will attempt %d cases (skipping %d already-ok)",
             len(cases_to_run), len(cases) - len(cases_to_run))

    if not args.dataset_dir and cases_to_run:
        try:
            from huggingface_hub import hf_hub_download
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            for case in cases_to_run:
                for rel in (f"ct/{case.ct_path.name}", f"labels/{case.label_path.name}"):
                    target = ds.root / rel
                    if not target.exists():
                        hf_hub_download(repo_id=args.hf_repo_id, repo_type="dataset",
                                          filename=rel, token=hf_token,
                                          local_dir=str(ds.root))
        except Exception as e: log.warning("HF download: %s", e)

    t0 = time.time()
    consecutive_fast_fails = 0
    fast_fail_durations: List[float] = []
    cb_enabled = args.circuit_breaker_n > 0

    for i, case in enumerate(cases_to_run, 1):
        log.info("[%d/%d] token=%-6s config=%-14s lstv=%s",
                 i, len(cases_to_run), case.token, case.config, case.lstv_label)
        case_meta = {
            "config": case.config, "match_type": case.match_type,
            "lstv_label": case.lstv_label, "position": case.position,
            "has_l6": case.has_l6, "lstv_agreement": case.lstv_agreement,
            "lstv_confusion_zone": case.lstv_confusion_zone,
        }
        case_t0 = time.time()
        r = benchmark_one(
            token=case.token, ct_path=case.ct_path, label_path=case.label_path,
            pred_dir=pred_dir, case_meta=case_meta, device=args.device,
            fast=args.fast, skip_surface=skip_surface, force_ts=args.force_ts,
            junction_window=args.window_mm)
        case_dt = time.time() - case_t0
        results.append(r)
        append_result_jsonl(per_case_log, r)
        log.info("  progress %d/%d elapsed=%.0fs case=%.1fs",
                 i, len(cases_to_run), time.time() - t0, case_dt)

        if r.get("ok"):
            consecutive_fast_fails = 0; fast_fail_durations = []
        else:
            if cb_enabled and case_dt < args.circuit_breaker_secs:
                consecutive_fast_fails += 1
                fast_fail_durations.append(case_dt)
                log.warning("  fast failure %d/%d (%.1fs)",
                             consecutive_fast_fails, args.circuit_breaker_n, case_dt)
            else:
                consecutive_fast_fails = 0; fast_fail_durations = []

            if cb_enabled and consecutive_fast_fails >= args.circuit_breaker_n:
                log.error("CIRCUIT BREAKER TRIPPED")
                try:
                    write_final_outputs(results, args, args.out_dir, per_case_log,
                                          t0, splits_subtypes=splits_subtypes,
                                          splits_schema=splits_schema, partial=True,
                                          patient_level=not args.record_level)
                except Exception as e:
                    log.error("partial outputs failed: %s", e)
                sys.exit(EXIT_CIRCUIT_BREAKER)

    write_final_outputs(results, args, args.out_dir, per_case_log, t0,
                         splits_subtypes=splits_subtypes,
                         splits_schema=splits_schema,
                         partial=False, patient_level=not args.record_level)


def _load_completed_legacy(per_case_log: Path) -> Tuple[List[Dict], Set[str]]:
    if not per_case_log.exists(): return [], set()
    by_key = {}
    with open(per_case_log) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            tok, cfg = rec.get("token"), rec.get("config")
            if not tok or not cfg: continue
            by_key[f"{tok}__{cfg}"] = rec
    return list(by_key.values()), set(by_key.keys())


if __name__ == "__main__":
    main()
