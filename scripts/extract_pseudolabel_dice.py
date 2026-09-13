"""scripts/extract_pseudolabel_dice.py -- the pelvic pseudolabeller's held-out fold Dice.

WHY THIS EXISTS. The manuscript states that the pelvis on 440 records is a pseudolabel from
the Dataset803 nnU-Net ensemble, predicted out of fold. The number that says how good that
model is on manual labels it never trained on was never written down anywhere: no
validation/summary.json was uploaded with the checkpoints, the training logs were truncated
mid-run, and eval_vs_manual.py was written but never run. The number does exist, though, and
it lives inside the checkpoints themselves: nnU-Net's trainer stores its logger state in every
checkpoint, including the per-epoch, per-class validation Dice on the fold's held-out cases.
This script reads it out and writes it down.

WHAT THE NUMBER IS. nnU-Net's online validation Dice ("pseudo Dice" in its logs): after each
epoch the trainer runs the network on patches sampled from the validation fold, argmaxes, and
accumulates per-class true positives, false positives and false negatives over the epoch;
Dice is computed from those totals. Voxels under the ignore label (the unannotated region of a
partially annotated record) are excluded, so the sacrum and hip Dice are measured only against
manual pelvic labels, on records the fold's model never saw. It is patch-based rather than a
full-volume sliding-window inference, so it is an approximation of the volume Dice, and one
that sits close to it for large structures.

Which epoch: checkpoint_best.pth is saved at the epoch with the best exponential moving
average of the mean foreground Dice, so its logger state ends at that epoch; the reported
value is the per-class Dice at that epoch. The five folds are those of splits_5fold.json.

    python scripts/extract_pseudolabel_dice.py                      # fetch from HF, write CSVs
    python scripts/extract_pseudolabel_dice.py --results <nnUNet_results dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pseudolabel_dice"
HF_REPO = "OpenSpineConsortium/spinopelvic-seg-checkpoints"
TRAINER_DIR = ("Dataset803_SpineSurgCTFullMerged/"
               "nnUNetTrainerWandB_500ep_LSTVOversample__nnUNetResEncUNetPlans_100G__3d_fullres")
# Dataset803 is the merged-label scheme the pseudolabeller was trained on (dataset.json):
# L5 and L6 are one class, "last_lumbar"; the release maps it back to the VerSe ids.
CLASSES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "last_lumbar (L5+L6)",
           6: "sacrum", 7: "left_hip", 8: "right_hip"}


def find_checkpoints(results_dir: str | None) -> list[Path]:
    if results_dir:
        found = sorted(Path(results_dir).glob("Dataset803*/**/fold_*/checkpoint_best.pth"))
        if found:
            return found
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(HF_REPO, f"{TRAINER_DIR}/fold_{f}/checkpoint_best.pth",
                                 repo_type="model")) for f in range(5)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="an nnUNet_results dir holding Dataset803")
    a = ap.parse_args()
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    per_fold, best_rows = [], []
    for p in find_checkpoints(a.results):
        fold = int(str(p).split("fold_")[1][0])
        ck = torch.load(p, map_location="cpu", weights_only=False)
        lg = ck["logging"]
        dpc = np.asarray(lg["dice_per_class_or_region"], dtype=float)   # epochs x classes
        ema = np.asarray(lg["ema_fg_dice"], dtype=float)
        best = int(np.argmax(ema))
        best_rows.append(dpc[best])
        for i, name in CLASSES.items():
            per_fold.append({"fold": fold, "class_id": i, "class": name,
                             "dice_at_selected_epoch": round(float(dpc[best, i - 1]), 4),
                             "selected_epoch": best, "epochs_trained": int(ck["current_epoch"]),
                             "max_dice_any_epoch": round(float(dpc[:, i - 1].max()), 4)})
        print(f"fold {fold}: epoch {best} of {ck['current_epoch']}, "
              f"mean fg Dice {dpc[best].mean():.4f}")

    with open(OUT / "dataset803_fold_validation_dice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        w.writeheader()
        w.writerows(per_fold)
    m, s = np.mean(best_rows, axis=0), np.std(best_rows, axis=0, ddof=0)
    with open(OUT / "dataset803_fold_validation_dice_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "dice_mean_over_folds", "dice_sd_over_folds",
                    "dice_min_fold", "dice_max_fold"])
        for i, name in CLASSES.items():
            col = [r[i - 1] for r in best_rows]
            w.writerow([i, name, f"{m[i-1]:.4f}", f"{s[i-1]:.4f}", f"{min(col):.4f}", f"{max(col):.4f}"])
            print(f"  {name:<20} {m[i-1]:.4f} ± {s[i-1]:.4f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
