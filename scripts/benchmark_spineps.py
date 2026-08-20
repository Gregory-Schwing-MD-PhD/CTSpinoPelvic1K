#!/usr/bin/env python3
"""
benchmark_spineps.py — score SPINEPS (CT mode) against CTSpinoPelvic1K labels.

Consumes what scripts/spineps_ct_pipeline.py wrote (native SPINEPS label space) and compares
the VERTEBRA INSTANCE mask to our GT. No remap is needed on the vertebra range: SPINEPS is
VerSe-numbered (1-7 C1-C7, 8-19 T1-T12, 20-25 L1-L6, 26 sacrum, 28 T13) and so is
label_scheme.py — that is exactly why this comparison is meaningful.

Two things are measured, and they are NOT the same thing:

  segmentation   per-vertebra Dice against the SAME id in GT
  identification whether the GT vertebra's best-overlapping predicted id IS its GT id
                 (the standard VerSe identification rate)

The second is the one that matters here. A method can segment every vertebra beautifully and
still name them all one level off; on an LSTV cohort that off-by-one is the failure mode, so
the per-case OFFSET (pred_id - gt_id, modal over the case's vertebrae) is reported separately.
A case where every vertebra is offset by the same k is a counting error, not a segmentation
error, and gets its own row in the summary.

FOV: our thoracic GT is FOV-limited (lower thoracic only on many cases). Scoring is therefore
restricted to labels PRESENT IN GT — a vertebra SPINEPS finds outside our labelled extent is
neither credited nor penalised.

    python scripts/benchmark_spineps.py \\
        --pred_dir /data/spineps_bench/spineps \\
        --gt_dir   /data/hf_export_v5/labels \\
        --out_dir  results/spineps_bench

Optional:
    --splits_file  splits_5fold.json   subgroup the summary by LSTV phenotype
    --rib_csv      rib_measurements_shard*.csv (merged)  exploratory rib-measurement summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS                                            # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.bench_spineps")

# scoreable ids: the VerSe vertebra range + sacrum + T13. Ribs/hips/femurs are out of scope —
# SPINEPS does not predict them.
VERT_IDS = list(range(1, 26)) + [28]
SACRUM_ID = LS.SACRUM_ID                                             # 26
SCORE_IDS = VERT_IDS + [SACRUM_ID]
NAMES = {v: k for k, v in LS.label_dict().items()}


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    denom = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    return (2.0 * inter / denom) if denom else float("nan")


def _find_gt(gt_dir: Path, case: str) -> Optional[Path]:
    for cand in (f"{case}_label.nii.gz", f"{case}.nii.gz", f"{case}_seg.nii.gz"):
        p = gt_dir / cand
        if p.exists():
            return p
    return None


def score_case(case: str, pred_path: Path, gt_path: Path) -> Optional[Dict]:
    """Per-vertebra Dice + identification for one case; None if the grids don't match."""
    pred_img, gt_img = nib.load(str(pred_path)), nib.load(str(gt_path))
    pred = np.asanyarray(pred_img.dataobj).astype(np.int16)
    gt = np.asanyarray(gt_img.dataobj).astype(np.int16)
    if pred.shape != gt.shape:
        log.warning("%s: shape mismatch pred%s vs gt%s — skipped", case, pred.shape, gt.shape)
        return None

    rows, offsets = [], []
    for g in SCORE_IDS:
        gm = gt == g
        n_gt = int(np.count_nonzero(gm))
        if n_gt == 0:
            continue
        # best-overlapping predicted id inside the GT vertebra (identification)
        vals, counts = np.unique(pred[gm], return_counts=True)
        order = np.argsort(counts)[::-1]
        best, best_n = 0, 0
        for i in order:
            if int(vals[i]) != 0:
                best, best_n = int(vals[i]), int(counts[i]); break
        pm = pred == g
        d = _dice(gm, pm)
        off = (best - g) if (best in SCORE_IDS and g in SCORE_IDS) else None
        if off is not None and g != SACRUM_ID:
            offsets.append(off)
        rows.append({"case": case, "gt_id": g, "gt_name": NAMES.get(g, str(g)),
                     "gt_voxels": n_gt, "pred_voxels": int(np.count_nonzero(pm)),
                     "dice": round(d, 4) if d == d else "",
                     "best_pred_id": best, "best_pred_name": NAMES.get(best, str(best)),
                     "best_pred_frac": round(best_n / n_gt, 4),
                     "identified": int(best == g), "offset": "" if off is None else off,
                     "detected": int(best_n > 0)})

    modal = Counter(offsets).most_common(1)[0][0] if offsets else None
    consistent = bool(offsets) and len(set(offsets)) == 1 and offsets[0] != 0
    dices = [r["dice"] for r in rows if r["dice"] != ""]
    return {"rows": rows,
            "case": {"case": case, "n_gt_vert": len(rows),
                     "mean_dice": round(float(np.mean(dices)), 4) if dices else "",
                     "id_rate": round(sum(r["identified"] for r in rows) / len(rows), 4)
                                if rows else "",
                     "modal_offset": "" if modal is None else modal,
                     "consistent_shift": int(consistent)}}


# ── LSTV subgrouping (splits v6 schema; same source benchmark_totalseg.py uses) ──
def load_subtypes(splits_file: Optional[Path]) -> Dict[str, str]:
    if not splits_file or not splits_file.exists():
        return {}
    try:
        s = json.loads(splits_file.read_text())
    except Exception as exc:
        log.warning("could not read %s (%s) — no subgrouping", splits_file, exc)
        return {}
    sub = s.get("patient_subtypes") or {}
    if sub:
        return {str(k): str(v) for k, v in sub.items()}
    return {str(k): str((v or {}).get("lstv_subtype", "unknown"))
            for k, v in (s.get("token_info") or {}).items()}


def _subgroup_of(case: str, subtypes: Dict[str, str]) -> str:
    if case in subtypes:
        return subtypes[case]
    for tok, st in subtypes.items():                 # case ids embed the patient token
        if tok and tok in case:
            return st
    return "unknown"


def rib_summary(rib_csv: Path, gt_dir: Optional[Path]) -> List[str]:
    """Exploratory summary of the Hendrik-code rib measurements (rib-segmentation repo).

    Reported as-is, NOT as a scored benchmark: his `sr` flag is a stump-rib morphology call on a
    rib assigned to a vertebra, while our GT class 74/75 marks a rib on a LUMBAR vertebra. They
    overlap but are not the same label, so the cross-tab below is a look, not a metric.
    """
    out: List[str] = []
    rows = list(csv.DictReader(rib_csv.open()))
    if not rows:
        return ["(rib csv empty)"]
    lens = [float(r["rib_length"]) for r in rows
            if r.get("rib_length") not in (None, "", "None")]
    sr = [r for r in rows if str(r.get("sr", "")).lower() in ("true", "1")]
    per_case = defaultdict(int)
    for r in rows:
        per_case[r["case"]] += 1
    out.append(f"cases with rib measurements : {len(per_case)}")
    out.append(f"rib rows (vertebra x side)  : {len(rows)}   median {np.median(list(per_case.values())):.0f}/case")
    if lens:
        out.append(f"rib_length mm               : median {np.median(lens):.1f}  "
                   f"IQR {np.percentile(lens, 25):.1f}-{np.percentile(lens, 75):.1f}")
    out.append(f"flagged stump ribs (sr=True): {len(sr)} rows in {len({r['case'] for r in sr})} cases")

    if gt_dir is not None:
        both = neither = only_sr = only_gt = 0
        for case in per_case:
            gp = _find_gt(gt_dir, case)
            if gp is None:
                continue
            lab = np.asanyarray(nib.load(str(gp)).dataobj)
            gt_lum = bool(np.isin(lab, [LS.LUMBAR_RIB_LEFT, LS.LUMBAR_RIB_RIGHT]).any())
            pred_sr = any(r["case"] == case and str(r.get("sr", "")).lower() in ("true", "1")
                          for r in rows)
            both += gt_lum and pred_sr
            neither += (not gt_lum) and (not pred_sr)
            only_sr += pred_sr and not gt_lum
            only_gt += gt_lum and not pred_sr
        out.append(f"vs our GT lumbar rib (74/75), exploratory: both {both} · neither {neither} · "
                   f"sr-only {only_sr} · GT-only {only_gt}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", type=Path, required=True,
                    help="<out>/spineps from spineps_ct_pipeline.py")
    ap.add_argument("--gt_dir", type=Path, required=True, help="dir of <case>_label.nii.gz")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--splits_file", type=Path, default=None)
    ap.add_argument("--rib_csv", type=Path, default=None,
                    help="merged rib_measurements_shard*.csv for the rib summary")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    preds = sorted(a.pred_dir.glob("*_seg-vert_msk.nii.gz"))
    if a.limit:
        preds = preds[:a.limit]
    if not preds:
        log.error("no *_seg-vert_msk.nii.gz in %s", a.pred_dir); return 1
    a.out_dir.mkdir(parents=True, exist_ok=True)
    subtypes = load_subtypes(a.splits_file)

    vert_rows: List[Dict] = []
    case_rows: List[Dict] = []
    n_missing = n_skipped = 0
    for p in preds:
        case = p.name[:-len("_seg-vert_msk.nii.gz")]
        gp = _find_gt(a.gt_dir, case)
        if gp is None:
            n_missing += 1; continue
        res = score_case(case, p, gp)
        if res is None:
            n_skipped += 1; continue
        vert_rows.extend(res["rows"])
        res["case"]["subgroup"] = _subgroup_of(case, subtypes)
        case_rows.append(res["case"])

    if not case_rows:
        log.error("nothing scored (%d without GT, %d shape-mismatched)", n_missing, n_skipped)
        return 1

    with open(a.out_dir / "spineps_per_vertebra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vert_rows[0].keys())); w.writeheader()
        w.writerows(vert_rows)
    with open(a.out_dir / "spineps_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys())); w.writeheader()
        w.writerows(case_rows)

    # ── summary ──────────────────────────────────────────────────────────────
    dice = np.array([r["dice"] for r in vert_rows if r["dice"] != ""], dtype=float)
    ident = np.array([r["identified"] for r in vert_rows], dtype=float)
    det = np.array([r["detected"] for r in vert_rows], dtype=float)
    shifted = sum(r["consistent_shift"] for r in case_rows)

    lines = [f"===== SPINEPS (CT) vs CTSpinoPelvic1K — {len(case_rows)} cases, "
             f"{len(vert_rows)} GT vertebrae =====",
             f"  skipped: {n_missing} without GT, {n_skipped} shape-mismatched",
             f"  Dice (same id)     mean {dice.mean():.3f}  median {np.median(dice):.3f}",
             f"  detection rate     {det.mean():.3f}   (any prediction inside the GT vertebra)",
             f"  identification     {ident.mean():.3f}   (best-overlap pred id == GT id)",
             f"  cases with a CONSISTENT whole-spine offset (miscount, not missegmentation): "
             f"{shifted}/{len(case_rows)}",
             ""]

    off = Counter(r["offset"] for r in vert_rows if r["offset"] != "")
    lines.append("  offset histogram (pred_id - gt_id):")
    for k in sorted(off):
        lines.append(f"     {k:+d} : {off[k]:5d}  ({100 * off[k] / sum(off.values()):.1f}%)")
    lines.append("")

    by_level = defaultdict(list)
    for r in vert_rows:
        by_level[r["gt_name"]].append(r)
    lines.append(f"  {'level':<8}{'n':>5}{'dice':>9}{'ident':>9}")
    for name in sorted(by_level, key=lambda n: by_level[n][0]["gt_id"]):
        rs = by_level[name]
        d = np.array([r["dice"] for r in rs if r["dice"] != ""], dtype=float)
        lines.append(f"  {name:<8}{len(rs):>5}{(d.mean() if d.size else float('nan')):>9.3f}"
                     f"{np.mean([r['identified'] for r in rs]):>9.3f}")

    if subtypes:
        lines += ["", f"  {'LSTV subgroup':<20}{'cases':>7}{'dice':>9}{'ident':>9}{'shift':>7}"]
        by_sub = defaultdict(list)
        for c in case_rows:
            by_sub[c["subgroup"]].append(c)
        for sg in sorted(by_sub):
            cs = by_sub[sg]
            d = np.array([c["mean_dice"] for c in cs if c["mean_dice"] != ""], dtype=float)
            i = np.array([c["id_rate"] for c in cs if c["id_rate"] != ""], dtype=float)
            lines.append(f"  {sg:<20}{len(cs):>7}{(d.mean() if d.size else float('nan')):>9.3f}"
                         f"{(i.mean() if i.size else float('nan')):>9.3f}"
                         f"{sum(c['consistent_shift'] for c in cs):>7}")

    if a.rib_csv:
        csvs = [Path(p) for p in glob.glob(str(a.rib_csv))]
        if len(csvs) == 1 and csvs[0].exists():
            lines += ["", "  ----- rib measurements (Hendrik-code rib-segmentation) -----"]
            lines += ["  " + s for s in rib_summary(csvs[0], a.gt_dir)]
        elif csvs:
            merged = a.out_dir / "rib_measurements_merged.csv"
            allr: List[Dict] = []
            for c in sorted(csvs):
                allr.extend(csv.DictReader(c.open()))
            if allr:
                with open(merged, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(allr[0].keys()), extrasaction="ignore")
                    w.writeheader(); w.writerows(allr)
                lines += ["", f"  ----- rib measurements (merged {len(csvs)} shards) -----"]
                lines += ["  " + s for s in rib_summary(merged, a.gt_dir)]

    text = "\n".join(lines)
    (a.out_dir / "spineps_summary.txt").write_text(text)
    print(text)
    log.info("-> %s", a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
