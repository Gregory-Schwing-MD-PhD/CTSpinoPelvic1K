"""
eval_vs_manual.py — quantify model accuracy vs MANUAL ground truth.

For each scoped case (spine_only / pelvic_native), the model predicted BOTH
spine and pelvis during pseudolabel; the merge then DISCARDED whichever side
was already manual. Those raw cached predictions are still in the pseudolabel
work dir, so we can recover the model's prediction on the MANUAL side and
compare it to ground truth there — the cheapest honest model-quality signal
available (no extra GPU run needed).

For each scoped case in scope:
  * load v1 manual label                       (data/hf_export/labels/<lbl>)
  * load the cached raw model prediction       (work/preds/<group>/<cid>.nii.gz)
  * canonical-remap the prediction with        (configs/pseudolabel_models.json)
  * restrict to the MANUAL SIDE's classes      (spine for spine_only, ...)
  * compute per-class Dice + voxel volumes + ASSD vs manual

Writes a per-case-per-class CSV and prints an aggregate summary. High Dice =
model is good where we have ground truth, so the pseudo labels on the OTHER
side are probably trustworthy. Low Dice = the model is wobbly and intensity
refinement / human review matter more.

Usage
-----
  python scripts/eval_vs_manual.py \
      --manual_from   data/hf_export \
      --preds_dir     data/hf_export_v2_work/preds \
      --models_config configs/pseudolabel_models.json \
      --out_csv       data/eval_vs_manual.csv
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
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.eval_vs_manual")

CLASS_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
               7: "sacrum", 8: "left_hip", 9: "right_hip"}
REGION_CANONICAL = {"spine": (1, 2, 3, 4, 5, 6), "pelvis": (7, 8, 9)}
SCOPE_MANUAL_SIDE = {"spine_only": "spine", "pelvic_native": "pelvis"}


# ===========================================================================
# Pure cores reused from seg_compare (kept dependency-light: same file dir)
# ===========================================================================

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from seg_compare import dice_volumes, surface_distance  # noqa: E402


def remap_prediction_array(pred, label_remap: Dict[str, int]) -> "object":
    """Map nnU-Net output ids to canonical 1..9 (background 0)."""
    import numpy as np
    out = np.zeros_like(pred, dtype=np.int16)
    for src, dst in label_remap.items():
        out[pred == int(src)] = int(dst)
    return out


def _align_pred_to_ref(ref_img, pred_img) -> "object":
    """Nearest-neighbour resample pred onto ref's grid if they differ. Mirrors
    pseudolabel._align_to so the model-vs-manual comparison is voxel-correct
    (the cached pred is in CT-native space; v1 is PIR after export)."""
    import numpy as np
    if (pred_img.shape[:3] == ref_img.shape[:3]
            and np.allclose(pred_img.affine, ref_img.affine, atol=1e-4)):
        return np.asarray(pred_img.dataobj).astype(np.int16)
    from scipy.ndimage import affine_transform
    M = np.linalg.inv(pred_img.affine) @ ref_img.affine
    out = affine_transform(
        np.asarray(pred_img.dataobj).astype(np.float32),
        M[:3, :3], offset=M[:3, 3], output_shape=ref_img.shape[:3],
        order=0, mode="constant", cval=0.0)
    return np.rint(out).astype(np.int16)


# ===========================================================================
# Orchestrator
# ===========================================================================

def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _find_pred(preds_dir: Path, cid: str) -> Optional[Path]:
    """Find the cached prediction for `cid` under preds/<group>/<cid>.nii.gz."""
    for sub in preds_dir.iterdir() if preds_dir.is_dir() else []:
        cand = sub / f"{cid}.nii.gz"
        if cand.exists():
            return cand
    return None


def _eval_one(task: dict) -> dict:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        v1_img = nib.load(task["v1_path"])
        pred_img = nib.load(task["pred_path"])
        v1 = np.asarray(v1_img.dataobj).astype(np.int16)
        pred_raw = _align_pred_to_ref(v1_img, pred_img)   # PIR / native handled
        pred = remap_prediction_array(pred_raw, task["label_remap"])
        # Restrict to the MANUAL side of this case (where ground truth is real).
        # Background mask = neither manual fg there nor pred fg there in those
        # classes is reported per-class via dice_volumes (NaN for empty class).
        # fillable = whole volume (we want global Dice on manual classes).
        fillable = np.ones_like(v1, dtype=bool)
        dv = dice_volumes(pred, v1, fillable, task["classes"])
        per_class = {}
        for c, m in dv.items():
            assd = float("nan")
            if task["assd"] and (m["vol_model"] or m["vol_intensity"]):
                # In this script "model" = pred, "intensity" slot = v1 manual.
                assd = surface_distance(pred == c, v1 == c)
            per_class[int(c)] = {**m, "assd": assd}
        return {"token": tok, "status": "ok", "config": task["config"],
                "per_class": per_class}
    except Exception as exc:                                # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual_from", required=True, type=Path,
                    help="Manual (v1) tree (data/hf_export).")
    ap.add_argument("--preds_dir", required=True, type=Path,
                    help="Pseudolabel work dir's preds/ subdir.")
    ap.add_argument("--models_config", type=Path,
                    default=Path("configs/pseudolabel_models.json"),
                    help="For the model->canonical label_remap.")
    ap.add_argument("--out_csv", type=Path, default=Path("eval_vs_manual.csv"))
    ap.add_argument("--no_assd", dest="assd", action="store_false",
                    help="Skip surface-distance (faster).")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.set_defaults(assd=True)
    args = ap.parse_args()

    label_remap = json.loads(args.models_config.read_text())[
        "checkpoints"]["label_remap"]

    records = _load_manifest(args.manual_from / "manifest.json")
    scoped = [r for r in records if r.get("config") in SCOPE_MANUAL_SIDE
              and r.get("label_file")]
    if args.limit:
        scoped = scoped[:args.limit]

    # build tasks; drop cases with no cached prediction
    tasks: List[dict] = []
    n_missing_pred = 0
    for rec in scoped:
        cid = Path(rec["label_file"]).name[:-len(".nii.gz")].replace("_label", "")
        # cached pred files are keyed by the CT id (e.g. "0017_spine_ct"),
        # which is the label name with "_label" stripped.
        # Some labels end "_ct"? export writes "_label", so strip it.
        cid = cid if not cid.endswith("_ct") else cid
        # Actually labels are "<base>_label.nii.gz" -> cid = "<base>" — but the
        # pred file was named after the CT, "<base>_ct.nii.gz" minus ".nii.gz"
        # i.e. "<base>_ct". Try a couple of variants below.
        pred = (_find_pred(args.preds_dir, cid)
                or _find_pred(args.preds_dir, cid + "_ct")
                or _find_pred(args.preds_dir,
                              cid.replace("_label", "_ct")))
        if pred is None:
            n_missing_pred += 1
            continue
        side = SCOPE_MANUAL_SIDE[rec["config"]]
        tasks.append({
            "token": rec.get("token", "?"), "config": rec["config"],
            "v1_path": str(args.manual_from / rec["label_file"]),
            "pred_path": str(pred),
            "classes": REGION_CANONICAL[side],
            "label_remap": label_remap, "assd": args.assd,
        })

    log.info("evaluating %d scoped cases vs MANUAL (%d had no cached pred); "
             "workers=%d  assd=%s", len(tasks), n_missing_pred, args.workers,
             args.assd)

    rows: List[dict] = []
    per_class_dice: Dict[int, list] = {}
    per_class_assd: Dict[int, list] = {}
    per_class_ratio: Dict[int, list] = {}

    import time
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_eval_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["status"] != "ok":
                log.warning("token=%s: %s", r["token"],
                            r.get("error") or r["status"])
                continue
            for c, m in r["per_class"].items():
                rows.append({
                    "token": r["token"], "config": r["config"], "class": c,
                    "class_name": CLASS_NAMES.get(c, str(c)),
                    "dice": round(m["dice"], 4) if m["dice"] == m["dice"] else "",
                    "vol_pred": m["vol_model"], "vol_manual": m["vol_intensity"],
                    "vol_ratio": round(m["vol_ratio"], 4) if m["vol_ratio"] == m["vol_ratio"] else "",
                    "assd_vox": round(m["assd"], 3) if m["assd"] == m["assd"] else "",
                })
                if m["dice"] == m["dice"]:
                    per_class_dice.setdefault(c, []).append(m["dice"])
                if m["vol_ratio"] == m["vol_ratio"]:
                    per_class_ratio.setdefault(c, []).append(m["vol_ratio"])
                if m["assd"] == m["assd"]:
                    per_class_assd.setdefault(c, []).append(m["assd"])
            if i == 1 or i % 10 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1.0)
                eta = (len(tasks) - i) / rate if rate > 0 else 0.0
                log.info("  [%d/%d] token=%s  elapsed=%dm%02ds  rate=%.2f/s  ETA=%dm",
                         i, len(tasks), r["token"],
                         int(elapsed) // 60, int(elapsed) % 60,
                         rate, int(eta) // 60)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "config", "class",
                           "class_name", "dice", "vol_pred", "vol_manual",
                           "vol_ratio", "assd_vox"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), args.out_csv)

    log.info("=" * 64)
    log.info("model-vs-MANUAL accuracy (per manual-side class)")
    log.info("  %-10s %6s  %7s %7s  %8s  %8s", "class", "n",
             "Dice", "median", "ASSD vox", "vol_pred/man")
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
    log.info("High Dice / low ASSD = model is good where we HAVE ground truth, "
             "so its pseudo labels on the other side are likely trustworthy. "
             "Low Dice = the model is wobbly; intensity refinement and human "
             "review matter more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
