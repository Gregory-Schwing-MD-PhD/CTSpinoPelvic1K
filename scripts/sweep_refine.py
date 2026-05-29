"""
sweep_refine.py — sweep (percentile, grow_iters) against MANUAL ground truth on
the scoped manual side, pick the best combo, write it for a downstream refine.

Loads each scoped case ONCE (v1, ct, raw pred), then iterates over every
(pctl, grow) combination computing per-class Dice vs manual. The dominant cost
(reading the 500 MB CT from NFS) is paid once per case, not once per setting,
so a 5×3 grid sweeps in roughly the same wall time as one normal eval pass.

Outputs:
  --out_csv:   per-token-per-class-per-combo Dice
  --best_out:  JSON {"best_pctl":…, "best_grow":…, …} consumed by stage 2
  console:     per-combo overall-Dice table sorted descending, BEST: …,
               then per-class breakdown for the winner

Usage
-----
  python scripts/sweep_refine.py \\
      --manual_from   data/hf_export \\
      --preds_dir     data/hf_export_v2_work/preds \\
      --models_config configs/pseudolabel_models.json \\
      --pctl_list     5,10,15,20,30 \\
      --grow_list     0,1,2 \\
      --out_csv       data/sweep_refine.csv \\
      --best_out      data/best_refine_params.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.sweep_refine")

CLASS_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
               7: "sacrum", 8: "left_hip", 9: "right_hip"}
REGION_CANONICAL = {"spine": (1, 2, 3, 4, 5, 6), "pelvis": (7, 8, 9)}
SCOPE_MANUAL_SIDE = {"spine_only": "spine", "pelvic_native": "pelvis"}

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from seg_compare import dice_volumes  # noqa: E402
from eval_vs_manual import (  # noqa: E402
    remap_prediction_array, _align_pred_to_ref, _refine_pred_for_eval,
    _load_manifest, _find_pred,
)


def _sweep_one(task: dict) -> dict:
    """Worker: load one case ONCE, iterate over every (pctl, grow) combination."""
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        v1_img = nib.load(task["v1_path"])
        pred_img = nib.load(task["pred_path"])
        v1 = np.asarray(v1_img.dataobj).astype(np.int16)
        pred_raw = _align_pred_to_ref(v1_img, pred_img)
        pred = remap_prediction_array(pred_raw, task["label_remap"])
        ct = np.asarray(nib.load(task["ct_path"]).dataobj).astype(np.float32)
        if ct.shape[:3] != v1.shape[:3]:
            return {"token": tok, "status": "skip_ct_shape"}

        fillable = np.ones_like(v1, dtype=bool)
        dv_raw = dice_volumes(pred, v1, fillable, task["classes"])
        raw_per_class = {int(c): dv_raw[c]["dice"] for c in task["classes"]}

        combos = []
        for pctl, grow in task["combos"]:
            refined, _ = _refine_pred_for_eval(
                pred, v1, ct, task["classes"],
                mode="clip", percentile=pctl, erode_iter=1,
                fill_holes=True, grow_iters=grow)
            if refined is None:
                continue
            dv_ref = dice_volumes(refined, v1, fillable, task["classes"])
            combos.append({
                "pctl": pctl, "grow": grow,
                "per_class": {int(c): dv_ref[c]["dice"] for c in task["classes"]},
            })
        return {"token": tok, "status": "ok", "config": task["config"],
                "raw_per_class": raw_per_class, "combos": combos}
    except Exception as exc:                                # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual_from", required=True, type=Path)
    ap.add_argument("--preds_dir", required=True, type=Path)
    ap.add_argument("--models_config", type=Path,
                    default=Path("configs/pseudolabel_models.json"))
    ap.add_argument("--pctl_list", type=str, default="5,10,15,20,30")
    ap.add_argument("--grow_list", type=str, default="0,1,2")
    ap.add_argument("--out_csv", type=Path, default=Path("sweep_refine.csv"))
    ap.add_argument("--best_out", type=Path,
                    default=Path("best_refine_params.json"))
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    label_remap = json.loads(args.models_config.read_text())[
        "checkpoints"]["label_remap"]
    pctl_list = [float(x.strip()) for x in args.pctl_list.split(",") if x.strip()]
    grow_list = [int(x.strip()) for x in args.grow_list.split(",") if x.strip()]
    combos = [(p, g) for p in pctl_list for g in grow_list]

    log.info("sweep: %d percentile × %d grow = %d combinations",
             len(pctl_list), len(grow_list), len(combos))
    log.info("       percentiles : %s", pctl_list)
    log.info("       grow_iters  : %s", grow_list)

    records = _load_manifest(args.manual_from / "manifest.json")
    scoped = [r for r in records if r.get("config") in SCOPE_MANUAL_SIDE
              and r.get("label_file")]
    if args.limit:
        scoped = scoped[:args.limit]

    tasks = []
    n_missing = 0
    for rec in scoped:
        cid = Path(rec["label_file"]).name[:-len(".nii.gz")].replace("_label", "")
        pred = (_find_pred(args.preds_dir, cid)
                or _find_pred(args.preds_dir, cid + "_ct")
                or _find_pred(args.preds_dir, cid.replace("_label", "_ct")))
        if pred is None:
            n_missing += 1
            continue
        side = SCOPE_MANUAL_SIDE[rec["config"]]
        tasks.append({
            "token": rec.get("token", "?"), "config": rec["config"],
            "v1_path":   str(args.manual_from / rec["label_file"]),
            "ct_path":   str(args.manual_from / rec["ct_file"]),
            "pred_path": str(pred),
            "classes": REGION_CANONICAL[side],
            "label_remap": label_remap,
            "combos": combos,
        })

    log.info("sweeping %d cases × %d combos  (%d had no cached pred); workers=%d",
             len(tasks), len(combos), n_missing, args.workers)

    combo_class_dices: Dict[tuple, Dict[int, list]] = defaultdict(
        lambda: defaultdict(list))
    raw_class_dices: Dict[int, list] = defaultdict(list)
    rows: List[dict] = []

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_sweep_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["status"] != "ok":
                log.warning("token=%s: %s", r["token"],
                            r.get("error") or r["status"])
                continue
            tok, cfg = r["token"], r["config"]
            for c, d in r["raw_per_class"].items():
                if d == d:
                    raw_class_dices[c].append(d)
            for combo in r["combos"]:
                key = (combo["pctl"], combo["grow"])
                for c, d in combo["per_class"].items():
                    rows.append({
                        "token": tok, "config": cfg, "class": c,
                        "class_name": CLASS_NAMES.get(c, str(c)),
                        "pctl": combo["pctl"], "grow": combo["grow"],
                        "dice_raw": (round(r["raw_per_class"].get(c, float("nan")), 4)
                                     if r["raw_per_class"].get(c, float("nan")) ==
                                        r["raw_per_class"].get(c, float("nan")) else ""),
                        "dice_ref": round(d, 4) if d == d else "",
                    })
                    if d == d:
                        combo_class_dices[key][c].append(d)

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
                           "class_name", "pctl", "grow",
                           "dice_raw", "dice_ref"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), args.out_csv)

    raw_overall = (st.mean([d for ds in raw_class_dices.values() for d in ds])
                   if raw_class_dices else float("nan"))

    log.info("")
    log.info("=" * 72)
    log.info("SWEEP RESULTS — per-combo overall mean Dice  (raw baseline = %.4f)",
             raw_overall)
    log.info("=" * 72)
    log.info("  %-6s %-5s %10s %12s", "pctl", "grow", "mean Dice", "Δ vs raw")
    combo_summary = []
    for (pctl, grow), class_dices in combo_class_dices.items():
        all_d = [d for ds in class_dices.values() for d in ds]
        if not all_d:
            continue
        mean_d = st.mean(all_d)
        combo_summary.append((mean_d, pctl, grow, mean_d - raw_overall))
    combo_summary.sort(reverse=True)
    for mean_d, pctl, grow, delta in combo_summary:
        log.info("  %-6.1f %-5d %10.4f %+12.4f", pctl, grow, mean_d, delta)
    log.info("=" * 72)

    if not combo_summary:
        log.error("no combos produced metrics; cannot pick best.")
        return 1

    best_mean_d, best_pctl, best_grow, best_delta = combo_summary[0]
    log.info("BEST: pctl=%s grow=%s   mean Dice=%.4f   Δ vs raw = %+.4f",
             best_pctl, best_grow, best_mean_d, best_delta)
    log.info("")
    log.info("BEST per-class breakdown:")
    log.info("  %-10s %6s %10s %10s %12s",
             "class", "n", "raw", "best", "Δ")
    best_class = combo_class_dices[(best_pctl, best_grow)]
    for c in sorted(best_class):
        raws = raw_class_dices.get(c, [])
        refs = best_class[c]
        r_mean = st.mean(raws) if raws else float("nan")
        f_mean = st.mean(refs)
        log.info("  %-10s %6d %10.4f %10.4f %+12.4f",
                 CLASS_NAMES.get(c, str(c)), len(refs),
                 r_mean, f_mean, f_mean - r_mean)
    log.info("=" * 72)

    best_data = {
        "best_pctl": best_pctl,
        "best_grow": best_grow,
        "best_mean_dice": round(best_mean_d, 4),
        "raw_baseline_mean_dice": round(raw_overall, 4),
        "improvement": round(best_delta, 4),
        "swept": {"pctl_list": pctl_list, "grow_list": grow_list},
    }
    args.best_out.parent.mkdir(parents=True, exist_ok=True)
    args.best_out.write_text(json.dumps(best_data, indent=2))
    log.info("wrote best params -> %s", args.best_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
