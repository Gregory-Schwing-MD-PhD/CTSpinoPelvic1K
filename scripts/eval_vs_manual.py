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


def _cname(i: int) -> str:
    return "bg" if i == 0 else CLASS_NAMES.get(i, str(i))
REGION_CANONICAL = {"spine": (1, 2, 3, 4, 5, 6), "pelvis": (7, 8, 9)}
SCOPE_MANUAL_SIDE = {"spine_only": "spine", "pelvic_native": "pelvis"}
ALL_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def classes_for_config(config: str, include_fused: bool = False):
    """Canonical classes that have MANUAL ground truth for this config.

    scoped (spine_only/pelvic_native) -> the manually-annotated region only;
    `fused` (both regions manual) -> ALL 9 classes, so the diff is a true
    FULL-SCAN model-vs-GT comparison (incl. the L5-S1 junction). Returns None
    for configs we don't evaluate."""
    if config in SCOPE_MANUAL_SIDE:
        return REGION_CANONICAL[SCOPE_MANUAL_SIDE[config]]
    if config == "fused" and include_fused:
        return ALL_CLASSES
    return None


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


def confusion_counts(pred, v1, gt_classes):
    """For each GT class g in gt_classes, tally how the model labelled those
    TRUE voxels across canonical ids 0..9.

    This pinpoints CLASS MIXING that per-class Dice only implies: an off-diagonal
    sacrum->ilium (or 7<->8<->9) entry is the sacroiliac-joint class bleed, and a
    left_hip<->right_hip entry (8<->9) is voxel-level L/R mixing (complementary to
    structure_qc's whole-structure lr_swap). =bg (0) is under-segmentation; a
    pelvis voxel landing on a spine id (1..6) is a gross error."""
    import numpy as np
    conf = {}
    for g in gt_classes:
        m = v1 == int(g)
        if not m.any():
            conf[int(g)] = [0] * 10
            continue
        conf[int(g)] = np.bincount(np.clip(pred[m], 0, 9).astype(np.int64),
                                   minlength=10).astype(np.int64).tolist()
    return conf


def _refine_pred_for_eval(pred, v1, ct, manual_classes, *,
                          mode: str, percentile: float, erode_iter: int,
                          fill_holes: bool, grow_iters: int):
    """Apply intensity refinement to `pred` restricted to `manual_classes`,
    calibrating the bone threshold from the REAL manual bone in v1. For
    eval only — does NOT preserve manual (we're testing the refinement against
    ground truth, not protecting it)."""
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt
    from intensity_refine import calibrate_threshold, _solid_fill

    manual_mask = (v1 >= 1) & (v1 <= 9)
    if not manual_mask.any():
        return None, None
    thr = calibrate_threshold(ct, manual_mask, percentile=percentile,
                              erode_iter=erode_iter)
    if thr is None:
        return None, None
    pred_classmap = np.where(np.isin(pred, list(manual_classes)),
                              pred, 0).astype(np.int16)
    pred_fg = pred_classmap > 0
    out = np.zeros_like(pred, dtype=np.int16)
    if not pred_fg.any():
        return out, thr
    bone = ct >= thr
    if mode == "clip" and int(grow_iters) > 0:
        candidates = np.zeros(bone.shape, dtype=bool)
        for c in np.unique(pred_classmap[pred_fg]):
            seed = (pred_classmap == int(c)) & bone
            grown = binary_dilation(seed, iterations=int(grow_iters), mask=bone)
            if fill_holes:
                grown = _solid_fill(grown)
            candidates |= grown
        idx = distance_transform_edt(~pred_fg, return_distances=False,
                                     return_indices=True)
        nearest = pred_classmap[tuple(idx)]
        out[candidates] = nearest[candidates].astype(np.int16)
    else:    # pure clip (grow_iters==0)
        for c in np.unique(pred_classmap[pred_fg]):
            pc = pred_classmap == int(c)
            seg_c = pc & bone
            if fill_holes:
                seg_c = _solid_fill(seg_c) & pc
            out[seg_c] = int(c)
    return out, thr


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
        pred_raw = _align_pred_to_ref(v1_img, pred_img)
        pred = remap_prediction_array(pred_raw, task["label_remap"])
        ct = np.asarray(nib.load(task["ct_path"]).dataobj).astype(np.float32)
        if ct.shape[:3] != v1.shape[:3]:
            return {"token": tok, "status": "skip_ct_shape"}

        fillable = np.ones_like(v1, dtype=bool)
        dv_raw = dice_volumes(pred, v1, fillable, task["classes"])

        refined, thr = _refine_pred_for_eval(
            pred, v1, ct, task["classes"],
            mode=task["refine_mode"], percentile=task["refine_pctl"],
            erode_iter=task["refine_erode"], fill_holes=task["refine_fill"],
            grow_iters=task["refine_grow"])
        dv_ref = (dice_volumes(refined, v1, fillable, task["classes"])
                  if refined is not None else None)

        per_class = {}
        for c in task["classes"]:
            c = int(c)
            raw = dv_raw[c]
            ref = dv_ref[c] if dv_ref else None
            assd_raw = float("nan")
            assd_ref = float("nan")
            if task["assd"]:
                if raw["vol_model"] or raw["vol_intensity"]:
                    assd_raw = surface_distance(pred == c, v1 == c)
                if ref is not None and (ref["vol_model"] or ref["vol_intensity"]):
                    assd_ref = surface_distance(refined == c, v1 == c)
            per_class[c] = {
                "dice_raw":   raw["dice"],
                "dice_ref":   ref["dice"] if ref else float("nan"),
                "vol_pred":   raw["vol_model"],
                "vol_ref":    ref["vol_model"] if ref else 0,
                "vol_manual": raw["vol_intensity"],
                "assd_raw":   assd_raw,
                "assd_ref":   assd_ref,
                "thr":        thr if thr is not None else float("nan"),
            }
        return {"token": tok, "status": "ok", "config": task["config"],
                "per_class": per_class,
                "confusion": confusion_counts(pred, v1, task["classes"])}
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
    # Intensity refinement to ALSO evaluate against ground truth — same knobs
    # as intensity_refine. With these defaults the comparison answers
    # "does intensity refinement get CLOSER to ground truth than the raw model?"
    ap.add_argument("--include_fused", action="store_true",
                    help="ALSO evaluate fused cases (both regions manual) as a "
                         "FULL-SCAN, all-9-class diff vs the complete GT. Needs "
                         "their predictions cached — run pseudolabel "
                         "--predict_fused first.")
    ap.add_argument("--refine_mode", choices=("clip", "resegment"), default="clip")
    ap.add_argument("--refine_grow", type=int, default=3)
    ap.add_argument("--refine_pctl", type=float, default=10.0)
    ap.add_argument("--refine_erode", type=int, default=1)
    ap.add_argument("--no_refine_fill", dest="refine_fill", action="store_false")
    ap.set_defaults(assd=True, refine_fill=True)
    args = ap.parse_args()

    label_remap = json.loads(args.models_config.read_text())[
        "checkpoints"]["label_remap"]

    records = _load_manifest(args.manual_from / "manifest.json")
    wanted = set(SCOPE_MANUAL_SIDE) | ({"fused"} if args.include_fused else set())
    scoped = [r for r in records if r.get("config") in wanted
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
        classes = classes_for_config(rec["config"], args.include_fused)
        tasks.append({
            "token": rec.get("token", "?"), "config": rec["config"],
            "v1_path":   str(args.manual_from / rec["label_file"]),
            "ct_path":   str(args.manual_from / rec["ct_file"]),
            "pred_path": str(pred),
            "classes": classes,
            "label_remap": label_remap, "assd": args.assd,
            "refine_mode":  args.refine_mode,
            "refine_grow":  args.refine_grow,
            "refine_pctl":  args.refine_pctl,
            "refine_erode": args.refine_erode,
            "refine_fill":  args.refine_fill,
        })

    log.info("evaluating %d scoped cases vs MANUAL (%d had no cached pred); "
             "workers=%d  assd=%s", len(tasks), n_missing_pred, args.workers,
             args.assd)

    rows: List[dict] = []
    per_class_dice_raw: Dict[int, list] = {}
    per_class_dice_ref: Dict[int, list] = {}
    per_class_assd_raw: Dict[int, list] = {}
    per_class_assd_ref: Dict[int, list] = {}
    agg_conf: Dict[int, List[int]] = {}        # true class -> summed pred 0..9

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
            def _r(v, n=4):
                return round(v, n) if v == v else ""
            for c, m in r["per_class"].items():
                rows.append({
                    "token": r["token"], "config": r["config"], "class": c,
                    "class_name": CLASS_NAMES.get(c, str(c)),
                    "dice_raw":     _r(m["dice_raw"]),
                    "dice_refined": _r(m["dice_ref"]),
                    "vol_pred":     m["vol_pred"],
                    "vol_refined":  m["vol_ref"],
                    "vol_manual":   m["vol_manual"],
                    "assd_raw":     _r(m["assd_raw"], 3),
                    "assd_refined": _r(m["assd_ref"], 3),
                    "threshold":    _r(m["thr"], 1),
                })
                if m["dice_raw"] == m["dice_raw"]:
                    per_class_dice_raw.setdefault(c, []).append(m["dice_raw"])
                if m["dice_ref"] == m["dice_ref"]:
                    per_class_dice_ref.setdefault(c, []).append(m["dice_ref"])
                if m["assd_raw"] == m["assd_raw"]:
                    per_class_assd_raw.setdefault(c, []).append(m["assd_raw"])
                if m["assd_ref"] == m["assd_ref"]:
                    per_class_assd_ref.setdefault(c, []).append(m["assd_ref"])
            for g, vec in r.get("confusion", {}).items():
                cur = agg_conf.setdefault(int(g), [0] * 10)
                for p, x in enumerate(vec):
                    cur[p] += int(x)
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
                           "class_name", "dice_raw", "dice_refined",
                           "vol_pred", "vol_refined", "vol_manual",
                           "assd_raw", "assd_refined", "threshold"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), args.out_csv)

    # ---- class-mixing confusion (true class -> how the model labelled it) ----
    if agg_conf:
        conf_csv = args.out_csv.with_name(args.out_csv.stem + "_confusion.csv")
        with open(conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true_class", "true_name", "n_true_vox"]
                       + [f"pred_{_cname(p)}" for p in range(10)]
                       + ["self_frac", "top_offdiag_class", "top_offdiag_frac"])
            for g in sorted(agg_conf):
                vec = agg_conf[g]
                tot = sum(vec)
                off = sorted(((vec[p], p) for p in range(10) if p != g),
                             reverse=True)
                top_n, top_p = off[0] if off else (0, -1)
                w.writerow([g, _cname(g), tot] + vec
                           + ([round(vec[g] / tot, 4),
                               _cname(top_p) if top_p >= 0 else "",
                               round(top_n / tot, 4)] if tot else ["", "", ""]))
        log.info("wrote class-mixing confusion -> %s", conf_csv)
        log.info("-" * 72)
        log.info("CLASS-MIXING confusion (row-normalized; how the model labelled "
                 "each TRUE voxel)")
        log.info("  %-10s %8s  %7s   %s", "true class", "n_vox", "self",
                 "largest off-diagonal leaks")
        for g in sorted(agg_conf):
            vec = agg_conf[g]
            tot = sum(vec)
            if not tot:
                continue
            off = sorted(((vec[p] / tot, p) for p in range(10)
                          if p != g and vec[p] > 0), reverse=True)[:3]
            leaks = ", ".join(f"{_cname(p)}={fr:.3f}" for fr, p in off) or "—"
            log.info("  %-10s %8d  %7.3f   %s", _cname(g), tot, vec[g] / tot, leaks)
        log.info("  ^ off-diagonal to a neighbour = class mixing; "
                 "left_hip<->right_hip = voxel-level L/R bleed; "
                 "=bg = under-segmentation; =L1..L6 = gross spine/pelvis error.")

    log.info("=" * 72)
    log.info("model-vs-MANUAL accuracy (per manual-side class; raw model vs "
             "intensity-refined model)")
    log.info("  %-10s %6s   %7s %7s   %7s %7s   %5s",
             "class", "n", "Dice raw", "Dice ref", "ASSD raw", "ASSD ref",
             "Δ Dice")
    for c in sorted(per_class_dice_raw):
        raws = per_class_dice_raw[c]
        refs = per_class_dice_ref.get(c, [])
        ar = per_class_assd_raw.get(c, [])
        af = per_class_assd_ref.get(c, [])
        d_raw = st.mean(raws)
        d_ref = st.mean(refs) if refs else float("nan")
        delta = d_ref - d_raw if refs else float("nan")
        log.info("  %-10s %6d   %7.3f %7.3f   %7s %7s   %+5.3f",
                 CLASS_NAMES.get(c, str(c)), len(raws),
                 d_raw, d_ref,
                 f"{st.mean(ar):.2f}" if ar else "—",
                 f"{st.mean(af):.2f}" if af else "—",
                 delta if delta == delta else float("nan"))
    all_raw = [d for v in per_class_dice_raw.values() for d in v]
    all_ref = [d for v in per_class_dice_ref.values() for d in v]
    if all_raw:
        d_raw = st.mean(all_raw)
        d_ref = st.mean(all_ref) if all_ref else float("nan")
        log.info("  %-10s %6d   %7.3f %7.3f                       %+5.3f",
                 "ALL", len(all_raw), d_raw, d_ref,
                 (d_ref - d_raw) if all_ref else float("nan"))
    if args.include_fused:
        f_raw = [float(r["dice_raw"]) for r in rows
                 if r["config"] == "fused" and r["dice_raw"] != ""]
        f_ref = [float(r["dice_refined"]) for r in rows
                 if r["config"] == "fused" and r["dice_refined"] != ""]
        n_fused_cases = len({r["token"] for r in rows if r["config"] == "fused"})
        if f_raw:
            log.info("-" * 72)
            log.info("FUSED FULL-SCAN (complete-GT scans, all 9 classes): "
                     "%d cases  mean Dice raw=%.3f  refined=%.3f",
                     n_fused_cases, st.mean(f_raw),
                     st.mean(f_ref) if f_ref else float("nan"))
            log.info("  ^ the cleanest single pseudolabel-quality number — both "
                     "regions of the same scan, incl. the L5-S1 junction.")
        else:
            log.warning("--include_fused set but NO fused predictions found — run "
                        "`pseudolabel --predict_fused` to cache them first.")
    log.info("=" * 72)
    log.info("Δ Dice > 0 = intensity refinement gets CLOSER to ground truth in "
             "that class (refinement is improving the model). Δ Dice < 0 = the "
             "refinement is hurting (its boundaries differ from truth more "
             "than the raw model's). High raw Dice = model is already good and "
             "refinement is just polish. Low raw Dice + positive Δ = strongest "
             "case for retraining on the refined labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
