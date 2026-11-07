# April 2026 release: record manifest and the evaluated split

These are the record-level files behind the segmentation results in *Not All Spines Are Created Equal: How
CT Segmenters Fail on Lumbosacral Transitional Vertebrae* (NeurIPS 2026, Evaluations and Datasets Track).

The archived dataset — [10.5281/zenodo.22139642](https://doi.org/10.5281/zenodo.22139642), current version
v11 — holds one volume per patient, 802 of them. The experiments in that paper were run on the **April 2026
release**, which held **1,153 records** over the same 802 patients, because several patients contributed
more than one acquisition. This directory is the map between the two, plus the exact split the reported
numbers were computed over.

| file | what it is |
|---|---|
| `records_1153.csv` | every April record: case id, patient token, LSTV subtype, which side of the split it is on, and the per-record attributes the build recorded |
| `heldout_174.txt` | the 174 records the released ensemble was evaluated on |
| `cv_pool_979.txt` | the 979 records available to cross-validation (the complement) |
| `evaluated_split.json` | the same two lists, with counts, in one file |

## Where the held-out split comes from, and why not from the obvious place

The obvious source is the nnU-Net dataset directory, which carries a `splits_final.json` and an `imagesTs`
holding exactly 174 records. **It is the wrong file, twice over.**

1. Its per-fold case counts disagree with the training logs published with the weights. The logs record
   781/198, 774/205, 786/193, 786/193 and 789/190 training/validation cases for folds 0 to 4; that file says
   783/196, 774/205, 785/194, 790/189 and 784/195. Only fold 1 agrees.
2. Only **34** of its 174 test records are among the 174 the paper actually scored. The other 140 sit in its
   *training* folder.

It is dated 19 May 2026 — a rebuild of a dataset whose weights were trained on 3 May. Publishing it as the
evaluated split would have put a confidently wrong file under a claim of reproducibility.

The split published here is instead **read off the file that produced the numbers**: the per-case Dice table
written on **6 May 2026**, which the paper cites as the ensemble result it reports. Its rows are the records
that were scored, so the split is not reconstructed — it is observed.

## What that reading reproduces

Each of these is stated in the paper independently of this file, so each is a check and not a restatement:

| quantity | these files | the paper |
|---|---|---|
| records in the April release | 1,153 | 1,153 |
| records in the cross-validation pool | 979 | 979 |
| held-out records | 174 | 174 |
| held-out patients | 167 | 167 |
| held-out records with sacral ground truth | 113 | 113 |
| LSTV-tagged held-out records | 7 | 7 |
| subtypes among those 7 | 3 lumbarization, 1 count-style sacralization, 2 full sacralization, 1 ambiguous, 0 semi-sacralization | the same |
| LSTV patients in the release | 33 | 33 |

The record universe and the per-record attributes come from the April build's own `lstv_cases.json`; the
cross-validation pool is the complement of the held-out split within it, and comes to 979 without being
told to.

## What is **not** here

**The five-fold assignment within the 979-record pool.** The only copy of it on disk disagrees with the
training logs published alongside the weights, as above, and nothing available reconstructs the assignment
used on 3 May. Any reader rebuilding the cross-validation from these files will therefore draw their own
folds and will not reproduce the per-fold numbers exactly.

What the folds must satisfy is on record, from the published training logs:

| fold | training cases | validation cases |
|---|---|---|
| 0 | 781 | 198 |
| 1 | 774 | 205 |
| 2 | 786 | 193 |
| 3 | 786 | 193 |
| 4 | 789 | 190 |

The validation counts sum to 979, so every pool record is held out exactly once. The split was grouped by
patient and stratified by LSTV subtype.

**This does not affect the headline results.** The ensemble numbers the paper reports are computed on the
174 held-out records, which are given here exactly; the fold assignment governs only how the five
checkpoints were trained, and those checkpoints are released as weights.

## Related artifacts

- Dataset: [10.5281/zenodo.22139642](https://doi.org/10.5281/zenodo.22139642) · mirror
  [`OpenSpineConsortium/CTSpinoPelvic1K`](https://huggingface.co/datasets/OpenSpineConsortium/CTSpinoPelvic1K)
- Fold checkpoints:
  [`OpenSpineConsortium/spinopelvic-seg-checkpoints`](https://huggingface.co/OpenSpineConsortium/spinopelvic-seg-checkpoints)
  (`Dataset803_SpineSurgCTFullMerged`, training logs dated 3 May 2026)
- Trainer, conversion and inference code:
  [`OpenSpineConsortium/spinopelvic-seg`](https://github.com/OpenSpineConsortium/spinopelvic-seg)
- Construction pipeline, evaluation protocol and the TotalSegmentator benchmark script: this repository

Licence: the annotations and derived tables are CC BY-NC-SA 4.0, as in the Zenodo record of record.
