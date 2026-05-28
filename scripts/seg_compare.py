"""
seg_compare.py — quantify how much the model's pseudo segmentation disagrees
with the CT-intensity segmentation, to gauge whether a second training run is
worthwhile.

Neither label is ground truth, but the intensity segmentation snaps to actual
bone edges, so where the model and intensity DISAGREE the model's boundaries
are suspect. Low Dice / large surface distance between the two — concentrated in
particular classes — is the quantitative case for retraining (e.g. on the
intensity-refined labels as improved targets). High agreement says the model's
masks are already close and a retrain would buy little.

Compares, per `spine_only` / `pelvic_native` case, in the PSEUDO-filled region
only (identified by diffing the original manual tree, so manual voxels don't
inflate the agreement), for each pseudo class:
  * Dice overlap (model vs intensity)
  * voxel volumes + volume ratio (intensity / model)
  * average symmetric surface distance (ASSD, in voxels) — the boundary metric

Writes a per-case-per-class CSV and prints an aggregate summary.

Usage
-----
  python scripts/seg_compare.py \
      --manual_from data/hf_export \
      --model       data/hf_export_v2 \
      --intensity   data/hf_export_v2_refined \
      --out_csv     data/seg_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.seg_compare")

CLASS_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
               7: "sacrum", 8: "left_hip", 9: "right_hip"}
REGION_CANONICAL = {"spine": (1, 2, 3, 4, 5, 6), "pelvis": (7, 8, 9)}
SCOPE = {"spine_only": "pelvis", "pelvic_native": "spine"}   # pseudo region


# ===========================================================================
# Pure core  (unit-tested in tests/test_seg_compare.py)
# ===========================================================================

def dice_volumes(model, intensity, fillable, classes) -> Dict[int, dict]:
    """Per-class Dice + voxel volumes between model and intensity, restricted
    to `fillable` (the pseudo region). Dice is NaN where neither has the class."""
    import numpy as np
    model = np.asarray(model)
    intensity = np.asarray(intensity)
    fillable = np.asarray(fillable, dtype=bool)
    out: Dict[int, dict] = {}
    for c in classes:
        a = (model == c) & fillable
        b = (intensity == c) & fillable
        na, nb = int(a.sum()), int(b.sum())
        inter = int((a & b).sum())
        dice = (2.0 * inter / (na + nb)) if (na + nb) else float("nan")
        out[int(c)] = {
            "dice": dice, "vol_model": na, "vol_intensity": nb,
            "vol_ratio": (nb / na) if na else float("nan"),
        }
    return out


def surface_distance(a, b) -> float:
    """Average symmetric surface distance (voxels) between two binary masks.
    NaN if either is empty."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() or not b.any():
        return float("nan")
    # Crop to the union bounding box: both masks lie inside it, so EDT distances
    # to the nearest opposite voxel are exact, but the transform runs on a small
    # box instead of the whole 512^3 volume.
    u = a | b
    coords = np.argwhere(u)
    sl = tuple(slice(int(lo), int(hi) + 1)
               for lo, hi in zip(coords.min(0), coords.max(0)))
    a, b = a[sl], b[sl]
    sa = a ^ binary_erosion(a)            # surface voxels of a
    sb = b ^ binary_erosion(b)
    da = distance_transform_edt(~b)[sa]   # a-surface -> nearest b
    db = distance_transform_edt(~a)[sb]   # b-surface -> nearest a
    n = len(da) + len(db)
    return float((da.sum() + db.sum()) / n) if n else float("nan")


# ===========================================================================
# Orchestrator
# ===========================================================================

def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _compare_one(task: dict) -> dict:
    """ProcessPoolExecutor worker: load v1/model/intensity for one case, compute
    per-class Dice + volumes + (optionally) ASSD. Returns a result dict."""
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        v1 = np.asarray(nib.load(task["v1_path"]).dataobj).astype(np.int16)
        mdl = np.asarray(nib.load(task["mdl_path"]).dataobj).astype(np.int16)
        inten = np.asarray(nib.load(task["int_path"]).dataobj).astype(np.int16)
        if not (v1.shape == mdl.shape == inten.shape):
            return {"token": tok, "status": "skip_shape"}
        fillable = ~((v1 >= 1) & (v1 <= 9))
        dv = dice_volumes(mdl, inten, fillable, task["classes"])
        per_class = {}
        for c, m in dv.items():
            assd = float("nan")
            if task["assd"] and (m["vol_model"] or m["vol_intensity"]):
                assd = surface_distance((mdl == c) & fillable,
                                        (inten == c) & fillable)
            per_class[int(c)] = {**m, "assd": assd}
        return {"token": tok, "status": "ok", "config": task["config"],
                "per_class": per_class}
    except Exception as exc:                                  # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual_from", required=True, type=Path,
                    help="Original manual tree (to mask out manual voxels).")
    ap.add_argument("--model", required=True, type=Path,
                    help="Model pseudo tree (pseudolabel.py output).")
    ap.add_argument("--intensity", required=True, type=Path,
                    help="Intensity-refined tree (intensity_refine.py output).")
    ap.add_argument("--out_csv", type=Path, default=Path("seg_compare.csv"))
    ap.add_argument("--no_assd", dest="assd", action="store_false",
                    help="Skip surface-distance (faster; Dice + volumes only).")
    ap.add_argument("--workers", type=int,
                    default=max(1, (__import__("os").cpu_count() or 8) // 2),
                    help="Parallel worker processes (default = nproc/2).")
    ap.add_argument("--limit", type=int, default=0)
    ap.set_defaults(assd=True)
    args = ap.parse_args()

    records = _load_manifest(args.model / "manifest.json")
    scoped = [r for r in records if r.get("config") in SCOPE
              and r.get("label_file")]
    if args.limit:
        scoped = scoped[:args.limit]
    log.info("comparing %d scoped cases  (assd=%s, workers=%d)",
             len(scoped), args.assd, args.workers)

    rows: List[dict] = []
    per_class_dice: Dict[int, list] = {}
    per_class_assd: Dict[int, list] = {}
    per_class_ratio: Dict[int, list] = {}

    tasks = [{
        "token": rec.get("token", "?"), "config": rec["config"],
        "v1_path":  str(args.manual_from / rec["label_file"]),
        "mdl_path": str(args.model       / rec["label_file"]),
        "int_path": str(args.intensity   / rec["label_file"]),
        "classes": REGION_CANONICAL[SCOPE[rec["config"]]],
        "assd": args.assd,
    } for rec in scoped]

    import time
    t0 = time.time()
    log.info("submitting %d tasks (first result usually takes a few minutes) ...",
             len(tasks))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_compare_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tok = r["token"]
            if r["status"] != "ok":
                log.warning("token=%s: %s — skip", tok,
                            r.get("error") or r["status"])
                continue
            cfg = r["config"]
            for c, m in r["per_class"].items():
                assd = m["assd"]
                rows.append({
                    "token": tok, "config": cfg, "class": c,
                    "class_name": CLASS_NAMES.get(c, str(c)),
                    "dice": round(m["dice"], 4) if m["dice"] == m["dice"] else "",
                    "vol_model": m["vol_model"], "vol_intensity": m["vol_intensity"],
                    "vol_ratio": round(m["vol_ratio"], 4) if m["vol_ratio"] == m["vol_ratio"] else "",
                    "assd_vox": round(assd, 3) if assd == assd else "",
                })
                if m["dice"] == m["dice"]:
                    per_class_dice.setdefault(c, []).append(m["dice"])
                if m["vol_ratio"] == m["vol_ratio"]:
                    per_class_ratio.setdefault(c, []).append(m["vol_ratio"])
                if assd == assd:
                    per_class_assd.setdefault(c, []).append(assd)
            if i == 1 or i % 10 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1.0)
                eta = (len(tasks) - i) / rate if rate > 0 else 0.0
                log.info("  [%d/%d] token=%s  elapsed=%dm%02ds  rate=%.2f/s  ETA=%dm",
                         i, len(tasks), tok,
                         int(elapsed) // 60, int(elapsed) % 60,
                         rate, int(eta) // 60)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "config", "class",
                           "class_name", "dice", "vol_model", "vol_intensity",
                           "vol_ratio", "assd_vox"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), args.out_csv)

    log.info("=" * 64)
    log.info("model-vs-intensity agreement (per pseudo class)")
    log.info("  %-10s %6s  %7s %7s  %8s  %8s", "class", "n",
             "Dice", "median", "ASSD vox", "vol_int/mdl")
    for c in sorted(per_class_dice):
        ds = per_class_dice[c]
        assds = per_class_assd.get(c, [])
        ratios = per_class_ratio.get(c, [])
        log.info("  %-10s %6d  %7.3f %7.3f  %8s  %8s",
                 CLASS_NAMES.get(c, str(c)), len(ds),
                 st.mean(ds), st.median(ds),
                 f"{st.mean(assds):.2f}" if assds else "—",
                 f"{st.mean(ratios):.2f}" if ratios else "—")
    all_d = [d for v in per_class_dice.values() for d in v]
    if all_d:
        log.info("  %-10s %6d  %7.3f %7.3f", "ALL", len(all_d),
                 st.mean(all_d), st.median(all_d))
    log.info("=" * 64)
    log.info("Low mean Dice / large ASSD in a class = the model's boundaries "
             "there disagree most with bone intensity → strongest retrain case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
