"""
boundary_decomp.py — split the pseudolabel-vs-radiologist Dice gap into the part
that is IRREDUCIBLE boundary/partial-volume noise vs the part that is genuine
INTERIOR / categorical error (the part review + retraining can actually fix).

Runs on the FUSED cases (complete radiologist GT) using the cached model
predictions (from `pseudolabel --predict_fused`). For each class, every
disagreeing voxel (gt XOR pred) is labelled:
  * BOUNDARY  — within --k voxels of the GT surface (sub-voxel edge ambiguity;
                this is where two radiologists also disagree → irreducible),
  * INTERIOR  — deeper than that (a real miss / leak / mixing → fixable).

Headline: mean Dice, the % of disagreement that is boundary (the irreducible
share), and the boundary-tolerant Dice ceiling (Dice if surface-band voxels were
forgiven) — i.e. how close to the radiologist you could get by fixing only the
interior errors. Directly answers "what do I have to do to close the gap."

  python scripts/boundary_decomp.py \
      --manual_from data/hf_export --preds_dir data/hf_export_v2_work/preds \
      --models_config configs/pseudolabel_models.json --out_csv data/boundary_decomp.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from eval_vs_manual import (  # noqa: E402
    ALL_CLASSES, CLASS_NAMES, _align_pred_to_ref, _find_pred, _load_manifest,
    remap_prediction_array)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.boundary_decomp")


# ── pure core (unit-tested) ─────────────────────────────────────────────────

def decompose_class(gt_mask, pred_mask, k: int = 1) -> Dict[str, float]:
    """Boundary/interior decomposition of the gt-vs-pred disagreement for one
    class. `near surface` = a 2k-thick band straddling the GT surface
    (dilate(gt,k) & ~erode(gt,k)). Returns dice, error voxel counts, the
    boundary share of error, and the boundary-tolerant dice (surface-band
    disagreement forgiven)."""
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    gt = np.asarray(gt_mask, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    ng, npd = int(gt.sum()), int(pred.sum())
    inter = int((gt & pred).sum())
    dice = (2 * inter / (ng + npd)) if (ng + npd) else 1.0

    err = gt ^ pred
    n_err = int(err.sum())
    if n_err == 0:
        return {"dice": round(dice, 4), "n_err": 0, "boundary_err": 0,
                "interior_err": 0, "boundary_frac": float("nan"),
                "dice_tolerant": round(dice, 4)}
    if ng:
        near = binary_dilation(gt, iterations=k) & ~binary_erosion(gt, iterations=k)
    else:
        near = binary_dilation(pred, iterations=k) & ~binary_erosion(pred, iterations=k)
    boundary_err = int((err & near).sum())
    interior_err = n_err - boundary_err
    # tolerant dice: forgive surface-band disagreement (count it as agreement).
    inter_tol = inter + boundary_err
    dice_tol = (2 * inter_tol / (ng + npd)) if (ng + npd) else 1.0
    dice_tol = min(dice_tol, 1.0)
    return {"dice": round(dice, 4), "n_err": n_err, "boundary_err": boundary_err,
            "interior_err": interior_err,
            "boundary_frac": round(boundary_err / n_err, 4),
            "dice_tolerant": round(dice_tol, 4)}


# ── orchestrator ────────────────────────────────────────────────────────────

def _one(task: dict) -> dict:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        gt_img = nib.load(task["gt_path"])
        gt = np.asarray(gt_img.dataobj).astype(np.int16)
        pred = remap_prediction_array(
            _align_pred_to_ref(gt_img, nib.load(task["pred_path"])), task["remap"])
        out = {}
        for c in ALL_CLASSES:
            g, p = (gt == c), (pred == c)
            if not g.any() and not p.any():
                continue
            out[c] = decompose_class(g, p, k=task["k"])
        return {"token": tok, "status": "ok", "per_class": out}
    except Exception as exc:                       # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual_from", required=True, type=Path)
    ap.add_argument("--preds_dir", required=True, type=Path)
    ap.add_argument("--models_config", type=Path,
                    default=Path("configs/pseudolabel_models.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("boundary_decomp.csv"))
    ap.add_argument("--k", type=int, default=1, help="surface band half-width (vox)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    remap = json.loads(args.models_config.read_text())["checkpoints"]["label_remap"]
    records = [r for r in _load_manifest(args.manual_from / "manifest.json")
               if r.get("config") == "fused" and r.get("label_file")]
    if args.limit:
        records = records[:args.limit]

    tasks, n_missing = [], 0
    for rec in records:
        cid = Path(rec["label_file"]).name[:-len(".nii.gz")].replace("_label", "")
        pred = (_find_pred(args.preds_dir, cid) or _find_pred(args.preds_dir, cid + "_ct")
                or _find_pred(args.preds_dir, cid.replace("_label", "_ct")))
        if pred is None:
            n_missing += 1
            continue
        tasks.append({"token": rec.get("token", "?"),
                      "gt_path": str(args.manual_from / rec["label_file"]),
                      "pred_path": str(pred), "remap": remap, "k": args.k})
    log.info("decomposing %d fused cases vs GT (%d had no cached pred); k=%d",
             len(tasks), n_missing, args.k)
    if not tasks:
        log.error("no fused predictions found — run `pseudolabel --predict_fused` first.")
        return 1

    rows: List[dict] = []
    agg: Dict[int, Dict[str, list]] = {}
    tot_b = tot_i = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] != "ok":
                log.warning("token=%s: %s", r["token"], r.get("error"))
                continue
            for c, m in r["per_class"].items():
                rows.append({"token": r["token"], "class": c,
                             "class_name": CLASS_NAMES.get(c, str(c)), **m})
                a = agg.setdefault(c, {"dice": [], "dice_tol": [], "bf": []})
                a["dice"].append(m["dice"])
                a["dice_tol"].append(m["dice_tolerant"])
                if m["boundary_frac"] == m["boundary_frac"]:
                    a["bf"].append(m["boundary_frac"])
                tot_b += m["boundary_err"]
                tot_i += m["interior_err"]
            if i % 50 == 0 or i == len(futs):
                log.info("  %d/%d", i, len(futs))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "class", "class_name", "dice",
                           "dice_tolerant", "n_err", "boundary_err", "interior_err",
                           "boundary_frac"])
        w.writeheader()
        w.writerows(rows)

    tot_err = tot_b + tot_i
    log.info("=" * 72)
    log.info("BOUNDARY vs INTERIOR decomposition (fused complete-GT cases, k=%d vox)", args.k)
    log.info("  %-10s %6s  %8s  %8s  %9s", "class", "n", "Dice", "Dice_tol", "bnd_err%")
    for c in sorted(agg):
        a = agg[c]
        log.info("  %-10s %6d  %8.3f  %8.3f  %8.1f%%", CLASS_NAMES.get(c, str(c)),
                 len(a["dice"]), st.mean(a["dice"]), st.mean(a["dice_tol"]),
                 100 * st.mean(a["bf"]) if a["bf"] else float("nan"))
    all_d = [d for a in agg.values() for d in a["dice"]]
    all_t = [d for a in agg.values() for d in a["dice_tol"]]
    if all_d:
        log.info("  %-10s %6d  %8.3f  %8.3f", "ALL", len(all_d),
                 st.mean(all_d), st.mean(all_t))
    if tot_err:
        log.info("-" * 72)
        log.info("Of ALL disagreeing voxels: %.1f%% are within %d vox of the GT "
                 "surface (boundary/partial-volume = IRREDUCIBLE), %.1f%% are "
                 "interior (miss/leak/mixing = FIXABLE).",
                 100 * tot_b / tot_err, args.k, 100 * tot_i / tot_err)
        log.info("Dice %.3f -> %.3f if surface-band disagreement were forgiven: the "
                 "gap to that tolerant ceiling is the part REVIEW can close.",
                 st.mean(all_d), st.mean(all_t))
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
