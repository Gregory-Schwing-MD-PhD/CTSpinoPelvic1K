# Pelvic pseudolabeller: held-out fold Dice

The pelvis on 440 released records (351 *separate* + 89 *spine_only*) is a pseudolabel from
the Dataset803 nnU-Net v2 model (`OpenSpineConsortium/spinopelvic-seg-checkpoints`,
trainer `nnUNetTrainerWandB_500ep_LSTVOversample`, plans `nnUNetResEncUNetPlans_100G`,
`3d_fullres`), predicted out of fold: each record was completed by the one fold whose
validation set held it, so no record was completed by a model that had trained on it.

These files are that model's validation Dice on the manual labels of each held-out fold,
read out of the five `checkpoint_best.pth` files by `scripts/extract_pseudolabel_dice.py`.
Nothing else records it: no `validation/summary.json` was uploaded with the checkpoints,
the training logs on the grid are truncated mid-run, and `scripts/eval_vs_manual.py` was
written but never run.

## Files

| file | content |
|---|---|
| `dataset803_fold_validation_dice.csv` | one row per fold × class: Dice at the selected epoch, the epoch, epochs trained, the best value any epoch reached |
| `dataset803_fold_validation_dice_summary.csv` | per class: mean, SD (population), min and max over the five folds |

## What the number is, exactly

nnU-Net's online validation Dice. After every epoch the trainer runs the network on patches
sampled from the validation fold, takes the argmax, accumulates per-class TP/FP/FN over the
epoch, and computes Dice from those totals. Voxels under the `ignore` label (the region a
partially annotated record never had a manual label for) are excluded, so sacrum and hip Dice
are measured only against manual CTPelvic1K labels on records the fold's model never saw.

It is patch-based, not full-volume sliding-window inference, so it approximates the volume
Dice; for structures the size of a sacrum or an innominate the two sit close. The selected
epoch is the one `checkpoint_best.pth` was saved at (best EMA of mean foreground Dice, which
in every fold was within three epochs of the end of training).

Dataset803 merges L5 and L6 into one class, `last_lumbar`; the release maps it back to the
VerSe identifiers. Folds are those of the release's `splits_5fold.json`.

## Summary (mean ± SD over five folds, at each fold's selected epoch)

| class | Dice |
|---|---|
| L1 | 0.965 ± 0.017 |
| L2 | 0.959 ± 0.022 |
| L3 | 0.958 ± 0.016 |
| L4 | 0.957 ± 0.016 |
| last lumbar (L5+L6) | 0.957 ± 0.016 |
| sacrum | 0.980 ± 0.003 |
| left hip | 0.970 ± 0.016 |
| right hip | 0.971 ± 0.015 |

The lumbar rows are reported for completeness; the released spine is manual on 782 records
and hand-corrected on the 20 pelvis-only ones, so the lumbar Dice describes the model, not
the release. The sacrum and hip rows are the quality of the 440 pseudolabelled pelves.

## Reproduce

```
python scripts/extract_pseudolabel_dice.py              # downloads the five checkpoints (~1.1 GB each)
python scripts/extract_pseudolabel_dice.py --results <nnUNet_results>   # if they are local
```
