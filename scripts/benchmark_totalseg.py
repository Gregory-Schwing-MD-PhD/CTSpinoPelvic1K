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
5. Optional --tokens subset for debugging
6. TS prediction cache key includes config

NOTE: --fast mode is available but NOT the default.  For publication-quality
TotalSegmentator numbers, leave it off.
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


def dice(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    p = pred == cls
    g = gt   == cls
    if not p.any() and not g.any():
        return float("nan")
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (p & g).sum() / denom)


def hausdorff95(pred, gt, cls, vox_spacing=(1.0, 1.0, 1.0)) -> float:
    from scipy.ndimage import binary_erosion
    p = (pred == cls).astype(bool)
    g = (gt   == cls).astype(bool)
    if not p.any() or not g.any():
        return float("nan")
    p_surf = p ^ binary_erosion(p)
    g_surf = g ^ binary_erosion(g)
    p_pts = np.argwhere(p_surf).astype(np.float32) * np.array(vox_spacing)
    g_pts = np.argwhere(g_surf).astype(np.float32) * np.array(vox_spacing)

    def _directed_hd95(src, tgt, chunk=2000):
        dists = []
        for i in range(0, len(src), chunk):
            diff = src[i:i+chunk, None, :] - tgt[None, :, :]
            dists.extend(np.sqrt((diff**2).sum(axis=2)).min(axis=1).tolist())
        return np.percentile(dists, 95)

    return float(max(_directed_hd95(p_pts, g_pts), _directed_hd95(g_pts, p_pts)))


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


def benchmark_one(token, ct_path, label_path, pred_dir, case_meta,
                   device="gpu", fast=False, skip_hd95=False,
                   force_ts=False, junction_window=JUNCTION_WINDOW_MM) -> Dict:
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
        "dice": {}, "hd95": {}, "junction": {},
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
        for cls in FOREGROUND_CLASSES:
            if cls not in scoreable:
                result["dice"][cls] = None
                continue
            d = dice(pred, gt, cls)
            result["dice"][cls] = round(d, 4) if d == d else None
        if result["l6_gt_present"]:
            result["l6_dice_meaningful"] = 0.0
        if not skip_hd95:
            for cls in TS_CAPABLE_CLASSES:
                if cls not in scoreable:
                    result["hd95"][cls] = None
                    continue
                h = hausdorff95(pred, gt, cls, vox_spacing=spacing)
                result["hd95"][cls] = round(h, 4) if h == h else None
        if config == "fused":
            result["junction"] = junction_analysis(pred, gt, affine,
                                                    window_mm=junction_window)
        else:
            result["junction"] = {"error": f"junction_analysis_skipped_config_{config}"}
        result["ok"] = True
        d_str = "  ".join(
            f"{CLASS_NAMES[c]}={(result['dice'].get(c) if result['dice'].get(c) is not None else float('nan')):.3f}"
            for c in [1, 2, 3, 4, 5, 7])
        log.info("  %-6s  cfg=%s  %s  lstv=%s",
                 token, config, d_str, result["lstv_label"])
    except Exception:
        result["error"] = traceback.format_exc()
        log.error("  FAIL token=%s cfg=%s: %s", token, config,
                  result["error"].splitlines()[-1])
    return result


# Aggregation helpers (simplified for brevity)

def _nanmean(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.mean(v)), 4) if v else None

def _nanstd(vals):
    v = [x for x in vals if x is not None and x == x]
    return round(float(np.std(v)), 4) if len(v) > 1 else None


def class_stats(cases):
    stats = {}
    for cls in FOREGROUND_CLASSES:
        dvals = [r["dice"].get(cls) for r in cases]
        hvals = [r["hd95"].get(cls) for r in cases if r.get("hd95")]
        stats[CLASS_NAMES[cls]] = {
            "n_cases":   len(cases),
            "dice_mean": _nanmean(dvals),
            "dice_std":  _nanstd(dvals),
            "hd95_mean": _nanmean(hvals),
            "hd95_std":  _nanstd(hvals),
        }
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
    ok = [r for r in results if r["ok"]]
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
    """Publication-ready per-vertebra Dice table.

    Column layout (fixed widths, space-separated):
      Subgroup (26) | n (3) | L1-L5 (5 ea) | L6 (5, always dash) | Sac/HipL/HipR (5 ea) | JxnDSC (7)
    """
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

        # Build the row piece-by-piece with consistent widths so columns
        # align regardless of which values are "—". Previous version had
        # a no-op str.replace() trying to normalize spacing that did
        # nothing; this version uses format specifiers directly.
        lumbar_cells = "  ".join(
            _f(cls.get(n, {}).get("dice_mean"))
            for n in ("L1", "L2", "L3", "L4", "L5")
        )
        pelvic_cells = "  ".join(
            _f(cls.get(n, {}).get("dice_mean"))
            for n in ("sacrum", "hip_left", "hip_right")
        )
        l6_cell  = f"{'—':>5}"  # TS has no L6 by construction
        jxn_cell = f"{_f(jxn.get('mean_junction_dsc')):>7}"

        lines.append(
            f"  {sg['label']:<26}  {sg['n']:>3}  "
            f"{lumbar_cells}  {l6_cell}  {pelvic_cells}  {jxn_cell}"
        )
    lines.append("")
    lines.append("  † L6 column = '—': TS has no L6 label.")
    return "\n".join(lines)


def write_csv(results: List[Dict], path: Path) -> None:
    import csv
    rows = []
    for r in results:
        if not r["ok"]:
            continue
        row = {"token": r["token"], "config": r["config"],
               "match_type": r["match_type"], "lstv_label": r["lstv_label"],
               "position": r["position"], "has_l6": r["has_l6"],
               "l6_gt_present": r["l6_gt_present"]}
        for cls in FOREGROUND_CLASSES:
            name = CLASS_NAMES[cls]
            row[f"dice_{name}"] = r["dice"].get(cls, "")
            row[f"hd95_{name}"] = r["hd95"].get(cls, "") if r.get("hd95") else ""
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


def select_cases(ds, args) -> List:
    """
    Zero-shot benchmarking — no train/val/test filter.  We run TS on every
    available case and stratify subgroups (config, LSTV class, match_type) only
    at aggregation time.
    """
    if args.config == "all":
        universe = ds.filter(present_only=True)
    else:
        universe = ds.filter(config=args.config, present_only=True)
    if args.tokens.strip():
        toks = {t.strip() for t in args.tokens.split(",") if t.strip()}
        universe = [c for c in universe if c.token in toks]
    return universe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default=None, type=Path)
    ap.add_argument("--hf_repo_id",  default="anonymous-mlhc/CTSpinoPelvic1K")
    ap.add_argument("--hf_token",    default=None)
    ap.add_argument("--config",     default="all",
                    choices=["all", "fused", "spine_only", "pelvic_native"],
                    help="Which dataset config to benchmark. Zero-shot evaluation "
                         "runs on the WHOLE dataset by default — splits are "
                         "irrelevant at inference time.")
    ap.add_argument("--tokens",      default="", type=str,
                    help="Optional comma-separated token subset (debugging).")
    ap.add_argument("--device",    default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--fast",      action="store_true",
                    help="TS --fast mode (NOT recommended for publication)")
    ap.add_argument("--force_ts",  action="store_true")
    ap.add_argument("--skip_hd95", action="store_true")
    ap.add_argument("--window_mm", default=JUNCTION_WINDOW_MM, type=float)
    ap.add_argument("--out_dir",  required=True, type=Path)
    ap.add_argument("--pred_dir", default=None,  type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.pred_dir or (args.out_dir / "ts_preds")
    pred_dir.mkdir(parents=True, exist_ok=True)

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
    log.info("Benchmarking %d cases  config=%s  per-config=%s  device=%s  fast=%s",
             len(cases), args.config, dict(cfg_counts), args.device, args.fast)

    # If using HF and files aren't local, snapshot-download CT + labels for selected cases
    if not args.dataset_dir:
        try:
            from huggingface_hub import hf_hub_download
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            for i, case in enumerate(cases, 1):
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

    results = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d]  token=%-6s  config=%-14s  lstv=%s",
                 i, len(cases), case.token, case.config, case.lstv_label)
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
            skip_hd95       = args.skip_hd95,
            force_ts        = args.force_ts,
            junction_window = args.window_mm,
        )
        results.append(r)
        elapsed = time.time() - t0
        log.info("  progress %d/%d  elapsed=%.0fs", i, len(cases), elapsed)

    summary = aggregate(results)
    t5 = format_table5(summary)
    print(t5)
    (args.out_dir / "paper_tables.txt").write_text(t5)
    (args.out_dir / "benchmark_results.json").write_text(
        json.dumps({"config": vars(args), "summary": summary, "per_case": results},
                   indent=2, default=str))
    (args.out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    write_csv(results, args.out_dir / "benchmark_per_case.csv")

    log.info("DONE  ok=%d  fail=%d  total_time=%.0fs",
             summary["n_ok"], summary["n_fail"], time.time() - t0)


if __name__ == "__main__":
    main()
