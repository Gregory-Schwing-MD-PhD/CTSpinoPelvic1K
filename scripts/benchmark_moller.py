"""
scripts/benchmark_moller.py — evaluate Möller's BINARY rib foreground/background classifier on THIS
dataset (as a held-out test set) against the final, corrected rib ground truth. Reports, per case and
aggregate:
    Dice, precision, recall  for the binary rib mask (Möller output vs GT ribs 34-57 binarised)
and compares to Möller's claimed Dice 0.997.

SCOPE / FAIRNESS (read before interpreting):
  * Möller's net is BINARY (rib vs not-rib). This benchmarks DETECTION only. It does NOT test rib
    NUMBERING — TS provides that, and most of our manual corrections were numbering / label-mixing /
    detachment fixes. So a HIGH Dice here would NOT contradict the correction effort; the two are
    different tasks.
  * The v4 pipeline DROPS Möller-only blobs (bowel / vascular calcification false positives). So the
    metric that captures Möller's generalization gap on a new distribution is PRECISION (low precision
    = the false-positive problem). Recall captures missed ribs. Report all three, not just Dice.

Inputs:
    --moller DIR   binary rib masks from Möller's net (one .nii.gz per case, matched by filename)
    --gt DIR       final corrected labels (ribs = ids 34-57, plus 74/75 lumbar ribs, binarised)

    python scripts/benchmark_moller.py --moller MOLLER_DIR --gt FINAL_DIR
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS            # noqa: E402
LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12          # 34..57
LUM = (LS.LUMBAR_RIB_LEFT, LS.LUMBAR_RIB_RIGHT)                    # 74, 75


def _rib_mask(lab):
    return ((lab >= LO) & (lab <= HI)) | np.isin(lab, LUM)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moller", required=True); ap.add_argument("--gt", required=True)
    a = ap.parse_args(argv)
    gt_dir, mo_dir = Path(a.gt), Path(a.moller)
    rows = []
    for gp in sorted(gt_dir.glob("*.nii.gz")):
        mp = mo_dir / gp.name
        if not mp.exists():
            continue
        gt = _rib_mask(np.asanyarray(nib.load(str(gp)).dataobj))
        mo = np.asanyarray(nib.load(str(mp)).dataobj) > 0
        if gt.shape != mo.shape:
            print(f"  ! {gp.name}: shape mismatch — skipped"); continue
        tp = int((gt & mo).sum()); fp = int((~gt & mo).sum()); fn = int((gt & ~mo).sum())
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        rows.append({"case": gp.name, "dice": round(dice, 4), "precision": round(prec, 4),
                     "recall": round(rec, 4), "fp_voxels": fp, "fn_voxels": fn})
    if not rows:
        print("no matched cases (check --moller / --gt filenames)"); return 1
    with open("moller_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "dice", "precision", "recall", "fp_voxels", "fn_voxels"])
        w.writeheader(); w.writerows(rows)
    d = np.array([r["dice"] for r in rows]); p = np.array([r["precision"] for r in rows])
    r = np.array([r["recall"] for r in rows])
    print(f"===== Möller binary rib classifier on THIS dataset ({len(rows)} cases) -> moller_benchmark.csv =====")
    print(f"   Dice      mean {d.mean():.3f}  median {np.median(d):.3f}  (Möller claimed 0.997 on his data)")
    print(f"   Precision mean {p.mean():.3f}  median {np.median(p):.3f}   (low = false positives: bowel/vascular)")
    print(f"   Recall    mean {r.mean():.3f}  median {np.median(r):.3f}   (low = missed ribs)")
    print(f"   cases with Dice < 0.90: {(d < 0.90).sum()}   < 0.80: {(d < 0.80).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
