"""
export_hf.py — Export CTSpinoPelvic1K as separate NIfTI image + label volumes.

INPUT
-----
  data/placed/placed_manifest.json        (from place_fused_masks.py)
  data/tcia_nifti/{series_uid}.nii.gz     reference CTs (dcm2niix output)
  data/placed/spine/{uid}_seg_placed.nii.gz
  data/placed/pelvic/{stem}_pelvic_placed.nii.gz

OUTPUT TREE  (under --out_dir, flat NIfTI layout)
-------------------------------------------------
  ct/<token:04d>_<config>_ct.nii.gz              int16, HU clipped, PIR orientation
  labels/<token:04d>_<config>_label.nii.gz       uint8, 10-class fused label map
  splits/{train,val,test}.json                   patient-level splits
  manifest.json                                  per-record case metadata
  splits_summary.json                            class stratification stats
  README.md                                      dataset card rendered on the Hub

DATASET CONFIGS (3 per-patient variants, derived from placed_manifest)
----------------------------------------------------------------------
  fused           one record per patient with match_type=fused;
                  label is full 10-class (L1-L6 + sacrum + L/R hip)
  spine_only      spine-only patients + the spine half of separate patients;
                  label contains only L1-L6 (background elsewhere)
  pelvic_native   pelvic-only patients + the pelvic half of separate patients;
                  label contains only sacrum + L/R hip

LABEL SCHEME (10-class)
-----------------------
  0 background
  1 L1    2 L2    3 L3    4 L4    5 L5    6 L6 (LSTV / lumbarized S1)
  7 sacrum   8 left_hip   9 right_hip

CT and label share identical affines — no resampling at training time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.export_hf")

SPINE_REMAP: Dict[int, int] = {20:1, 21:2, 22:3, 23:4, 24:5, 25:6}
PELVIC_REMAP: Dict[int, int] = {1:7, 2:8, 3:9}
LABEL_NAMES = [
    "background", "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "left_hip", "right_hip",
]


def _load_nifti(path: Path):
    import nibabel as nib
    from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
    import numpy as np
    img       = nib.load(str(path))
    src_ornt  = nib.io_orientation(img.affine)
    dst_ornt  = axcodes2ornt(("P", "I", "R"))
    xfm       = ornt_transform(src_ornt, dst_ornt)
    data      = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff   = img.affine @ nib.orientations.inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _fuse_labels_fused(spine_arr, pelvic_arr, shape):
    import numpy as np
    fused = np.zeros(shape, dtype=np.uint8)
    if spine_arr is not None:
        mn = tuple(min(a, b) for a, b in zip(shape, spine_arr.shape))
        sl = tuple(slice(0, m) for m in mn)
        sa = spine_arr[sl]
        for src, dst in SPINE_REMAP.items():
            fused[sl][sa == src] = dst
    if pelvic_arr is not None:
        mn = tuple(min(a, b) for a, b in zip(shape, pelvic_arr.shape))
        sl = tuple(slice(0, m) for m in mn)
        pa = pelvic_arr[sl]
        for src, dst in PELVIC_REMAP.items():
            fused[sl][pa == src] = dst
    return fused


def _remap_spine_only(spine_arr, shape):
    import numpy as np
    out = np.zeros(shape, dtype=np.uint8)
    if spine_arr is not None:
        mn = tuple(min(a, b) for a, b in zip(shape, spine_arr.shape))
        sl = tuple(slice(0, m) for m in mn)
        sa = spine_arr[sl]
        for src, dst in SPINE_REMAP.items():
            out[sl][sa == src] = dst
    return out


def _remap_pelvic_only(pelvic_arr, shape):
    import numpy as np
    out = np.zeros(shape, dtype=np.uint8)
    if pelvic_arr is not None:
        mn = tuple(min(a, b) for a, b in zip(shape, pelvic_arr.shape))
        sl = tuple(slice(0, m) for m in mn)
        pa = pelvic_arr[sl]
        for src, dst in PELVIC_REMAP.items():
            out[sl][pa == src] = dst
    return out


def _enumerate_records(cases: List[Dict]) -> List[Dict]:
    """
    Turn manifest cases into per-(patient, config) export records.
      fused       -> 1 'fused' record
      spine_only  -> 1 'spine_only' record
      pelvic_only -> 1 'pelvic_native' record
      separate    -> 1 'spine_only' + 1 'pelvic_native' record
    """
    records = []
    for c in cases:
        token = str(c.get("patient_token", ""))
        if not token:
            continue
        mt = c.get("match_type", "")
        sp = c.get("spine")
        pv = c.get("pelvic")
        base = {
            "patient_token":       token,
            "match_type":          mt,
            "lstv_pelvic":         c.get("lstv_pelvic", ""),
            "lstv_vertebral":      c.get("lstv_vertebral", ""),
            "lstv_agreement":      c.get("lstv_agreement"),
            "lstv_confusion_zone": c.get("lstv_confusion_zone", False),
            "lstv_class":          c.get("lstv_class", 0),
        }
        if mt == "fused" and sp and pv:
            records.append({**base, "config": "fused", "spine": sp, "pelvic": pv})
        elif mt == "spine_only" and sp:
            records.append({**base, "config": "spine_only", "spine": sp, "pelvic": None})
        elif mt == "pelvic_only" and pv:
            records.append({**base, "config": "pelvic_native", "spine": None, "pelvic": pv})
        elif mt == "separate" and sp:
            records.append({**base, "config": "spine_only", "spine": sp, "pelvic": None})
        if mt == "separate" and pv:
            records.append({**base, "config": "pelvic_native", "spine": None, "pelvic": pv})
    return records


def _export_one_record(args):
    rec, nifti_dir_str, ct_out_dir_str, lbl_out_dir_str, hu_range = args
    try:
        import numpy as np
        import nibabel as nib

        nifti_dir   = Path(nifti_dir_str)
        ct_out_dir  = Path(ct_out_dir_str)
        lbl_out_dir = Path(lbl_out_dir_str)

        token  = rec["patient_token"]
        config = rec["config"]
        sp     = rec.get("spine")
        pv     = rec.get("pelvic")

        if config in ("fused", "spine_only") and sp:
            ref_uid = sp["series_uid"]
        elif config == "pelvic_native" and pv:
            ref_uid = pv["series_uid"]
        else:
            return token, config, False, "no_series_uid"

        ct_path = nifti_dir / f"{ref_uid}.nii.gz"
        if not ct_path.exists():
            return token, config, False, f"ct_missing:{ref_uid}"

        try:
            stem = f"{int(token):04d}_{config}"
        except ValueError:
            stem = f"{token}_{config}"

        ct_out_path  = ct_out_dir  / f"{stem}_ct.nii.gz"
        lbl_out_path = lbl_out_dir / f"{stem}_label.nii.gz"
        if ct_out_path.exists() and lbl_out_path.exists():
            return token, config, True, "skip"

        ct_arr, affine = _load_nifti(ct_path)
        shape = ct_arr.shape
        ct_arr = np.clip(ct_arr, hu_range[0], hu_range[1]).astype(np.int16)

        spine_arr = None
        if sp and sp.get("placed") and Path(sp["placed"]).exists():
            spine_arr, _ = _load_nifti(Path(sp["placed"]))
            spine_arr = spine_arr.astype(np.int32)
        pelvic_arr = None
        if pv and pv.get("placed") and Path(pv["placed"]).exists():
            pelvic_arr, _ = _load_nifti(Path(pv["placed"]))
            pelvic_arr = pelvic_arr.astype(np.int32)

        if config == "fused":
            label = _fuse_labels_fused(spine_arr, pelvic_arr, shape)
        elif config == "spine_only":
            label = _remap_spine_only(spine_arr, shape)
        elif config == "pelvic_native":
            label = _remap_pelvic_only(pelvic_arr, shape)
        else:
            return token, config, False, f"unknown_config:{config}"

        # Identical affines for CT and label — this is the alignment guarantee.
        ct_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(ct_arr, affine.astype(np.float32)), str(ct_out_path))
        nib.save(nib.Nifti1Image(label,  affine.astype(np.float32)), str(lbl_out_path))
        return token, config, True, stem
    except Exception:
        import traceback
        return rec.get("patient_token","?"), rec.get("config","?"), False, traceback.format_exc()[-400:]


def _coalesce_rare_strata(strata: List[str], min_count: int) -> List[str]:
    """Collapse '<match>|<lstv>' strata that are too thin for splitting."""
    counts = Counter(strata)
    rare = {s for s, c in counts.items() if c < min_count}
    if not rare:
        return strata
    out = []
    for s in strata:
        if s in rare and "|" in s:
            out.append(s.split("|", 1)[0])
        else:
            out.append(s)
    return out


def _build_strata_labels(tokens: List[str],
                         patient_keys: Dict[str, Tuple]) -> List[str]:
    return [f"{patient_keys[t][1]}|{patient_keys[t][0]}" for t in tokens]


def _stratify_test_holdout_and_kfold(
        patient_keys: Dict[str, Tuple],
        test_fraction: float,
        n_folds: int,
        seed: int,
) -> Dict:
    """
    Patient-level stratification by (match_type | lstv_class).
    Holds out `test_fraction` as a fixed test set, then runs StratifiedKFold
    on the remaining trainval pool.

    Returns a dict shaped like generate_5fold_splits.py schema v3.
    """
    import warnings as _w
    _w.filterwarnings("ignore",
                      message="The least populated class in y has only",
                      category=UserWarning)
    try:
        from sklearn.model_selection import (StratifiedShuffleSplit,
                                              StratifiedKFold, ShuffleSplit, KFold)
    except ImportError as e:
        raise RuntimeError("scikit-learn required for 5-fold splitting. "
                           "pip install scikit-learn") from e

    tokens  = sorted(patient_keys.keys())
    strata  = _build_strata_labels(tokens, patient_keys)

    min_count_test = max(n_folds, int(round(1.0 / max(test_fraction, 1e-6))))
    strata_test    = _coalesce_rare_strata(strata, min_count_test)

    # Holdout
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction,
                                      random_state=seed)
        tv_idx, te_idx = next(sss.split(tokens, strata_test))
    except ValueError as e:
        log.warning("Stratified test holdout failed (%s); falling back.", e)
        ss = ShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        tv_idx, te_idx = next(ss.split(tokens))

    trainval_tokens = [tokens[i] for i in tv_idx]
    test_tokens     = sorted(tokens[i] for i in te_idx)
    trainval_strata = [strata[i] for i in tv_idx]
    trainval_strata_safe = _coalesce_rare_strata(trainval_strata, n_folds)

    # KFold on trainval
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_splits = list(skf.split(trainval_tokens, trainval_strata_safe))
    except ValueError as e:
        log.warning("StratifiedKFold failed (%s); falling back to KFold.", e)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(trainval_tokens))

    folds = []
    for tr_i, va_i in fold_splits:
        folds.append({
            "train_tokens": sorted(trainval_tokens[j] for j in tr_i),
            "val_tokens":   sorted(trainval_tokens[j] for j in va_i),
        })

    # Sanity: fold val sets are a disjoint partition of trainval_tokens
    val_union: set = set()
    for f in folds:
        s = set(f["val_tokens"])
        if val_union & s:
            raise RuntimeError("Fold val sets are not disjoint")
        val_union |= s
    if val_union != set(trainval_tokens):
        raise RuntimeError("Fold val union != trainval pool")

    from datetime import datetime, timezone
    stratum_by_token = dict(zip(tokens, strata))
    def _cnt(tok_list): return dict(Counter(stratum_by_token[t] for t in tok_list))

    token_info = {
        t: {"match_type": patient_keys[t][1],
            "lstv_class": int(patient_keys[t][0])}
        for t in tokens
    }

    return {
        "schema_version":    3,
        "created_at":        datetime.now(timezone.utc)
                              .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source":            "export_hf.py",
        "test_fraction":     test_fraction,
        "n_folds":           n_folds,
        "kfold_seed":        seed,
        "strata_scheme":     "match_type_x_lstv",
        "n_patients_total":  len(tokens),
        "n_patients_test":   len(test_tokens),
        "n_patients_trainval": len(trainval_tokens),
        "test_tokens":       test_tokens,
        "folds":             folds,
        "strata_counts_total":        _cnt(tokens),
        "strata_counts_test":         _cnt(test_tokens),
        "strata_counts_per_fold_val": [_cnt(f["val_tokens"]) for f in folds],
        "token_info":        token_info,
    }


def _write_readme(out_dir: Path, cv: Dict,
                  n_records_by_config: Dict[str, int]) -> None:
    n_total = sum(n_records_by_config.values())
    n_test     = cv["n_patients_test"]
    n_trainval = cv["n_patients_trainval"]
    n_folds    = cv["n_folds"]
    readme = f"""---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
tags:
- medical
- ct
- lumbar-spine
- pelvis
- colonography
- lstv
size_categories:
- 1K<n<10K
---

# CTSpinoPelvic1K

Fused spine + pelvis CT segmentation dataset via patient-level crosswalk of
TCIA CT COLONOGRAPHY × CTSpine1K × CTPelvic1K.

## Statistics

- Total records:     {n_total}
- fused:             {n_records_by_config.get('fused', 0)}
- spine_only:        {n_records_by_config.get('spine_only', 0)}
- pelvic_native:     {n_records_by_config.get('pelvic_native', 0)}
- Test patients:     {n_test}  (fixed, {int(cv['test_fraction']*100)}% holdout)
- Trainval patients: {n_trainval}  ({n_folds}-fold stratified CV)

## Labels (10-class)

| ID | Name       |
|----|------------|
|  0 | background |
|  1 | L1         |
|  2 | L2         |
|  3 | L3         |
|  4 | L4         |
|  5 | L5         |
|  6 | L6 / LSTV  |
|  7 | sacrum     |
|  8 | left hip   |
|  9 | right hip  |

## Layout

```
ct/<token:04d>_<config>_ct.nii.gz         int16 HU, PIR orientation
labels/<token:04d>_<config>_label.nii.gz  uint8 0..9, identical affine
manifest.json                              per-record metadata
splits/
  test.json         fixed test holdout (list of patient tokens)
  cv_5fold.json     stratified 5-fold CV on the trainval pool (schema v3)
```

CT and label share the same 4×4 affine — no resampling at training time.

## Splits

**Design:** a fixed test holdout (never touched during model selection) and a
5-fold stratified CV on the remaining trainval pool. Both stratified at the
*patient* level by `(match_type × lstv_class)`.

### `splits/test.json`
A flat list of patient tokens for the held-out test set. Use this for final
reporting only — never for hyperparameter tuning.

### `splits/cv_5fold.json`
Schema v3 matching the output of `scripts/generate_5fold_splits.py`:
```json
{{
  "schema_version": 3,
  "n_folds": {n_folds},
  "test_fraction": {cv['test_fraction']},
  "strata_scheme": "match_type_x_lstv",
  "test_tokens": [...],
  "folds": [
    {{"train_tokens": [...], "val_tokens": [...]}},
    ... {n_folds} folds total ...
  ],
  "token_info": {{"<tok>": {{"match_type": "...", "lstv_class": 0}}, ...}}
}}
```

A `separate` patient produces two records (`spine_only` + `pelvic_native`);
both always land in the same fold / test split — no cross-split leak.

## Configs

- `fused`: spine + pelvis on one CT (10-class)
- `spine_only`: spine-only patients + spine half of separate (6-class, L1-L6)
- `pelvic_native`: pelvic-only patients + pelvic half of separate (3-class, sacrum + hips)

## License

- Source datasets retain their own licenses (TCIA CT COLONOGRAPHY, CTSpine1K, CTPelvic1K).
- Derivative fused labels, splits, and code: **CC BY-NC 4.0**.
"""
    (out_dir / "README.md").write_text(readme)


def _push_to_hub(out_dir: Path, hf_repo: str) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        log.error("huggingface_hub not installed.")
        sys.exit(1)
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN env var not set.")
        sys.exit(1)
    api = HfApi(token=token)
    create_repo(repo_id=hf_repo, repo_type="dataset",
                exist_ok=True, private=False, token=token)
    api.upload_folder(folder_path=str(out_dir), repo_id=hf_repo,
                       repo_type="dataset",
                       commit_message="CTSpinoPelvic1K release")
    log.info("Pushed → https://huggingface.co/datasets/%s", hf_repo)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest",  required=True, type=Path)
    p.add_argument("--nifti_dir", required=True, type=Path)
    p.add_argument("--out_dir",   required=True, type=Path)
    p.add_argument("--workers",   default=16, type=int)
    p.add_argument("--seed",      default=42, type=int)
    p.add_argument("--test_fraction", default=0.15, type=float,
                   help="Fraction of patients held out as the fixed test set")
    p.add_argument("--n_folds",   default=5, type=int,
                   help="K for StratifiedKFold on the trainval pool")
    p.add_argument("--hu_min",    default=-1024, type=int)
    p.add_argument("--hu_max",    default= 2048, type=int)
    p.add_argument("--push",      action="store_true")
    p.add_argument("--hf_repo",   default="", type=str)
    args = p.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    manifest = json.loads(args.manifest.read_text())
    cases    = manifest.get("cases", [])
    log.info("Manifest: %d cases", len(cases))

    records = _enumerate_records(cases)
    n_records_by_config = Counter(r["config"] for r in records)
    log.info("Records per config: %s", dict(n_records_by_config))

    # Patient-level stratification on (lstv_class, match_type)
    patient_info: Dict[str, Tuple] = {}
    for r in records:
        tok = r["patient_token"]
        if tok not in patient_info:
            patient_info[tok] = (r.get("lstv_class", 0), r.get("match_type", ""))
    log.info("Unique patients: %d", len(patient_info))

    log.info("Stratifying: %d%% test holdout + %d-fold CV  (strata: match_type × lstv_class)",
             int(args.test_fraction * 100), args.n_folds)
    cv = _stratify_test_holdout_and_kfold(
        patient_keys  = patient_info,
        test_fraction = args.test_fraction,
        n_folds       = args.n_folds,
        seed          = args.seed,
    )
    test_tokens     = set(cv["test_tokens"])
    trainval_tokens = set(t for f in cv["folds"] for t in f["train_tokens"]) \
                      | set(t for f in cv["folds"] for t in f["val_tokens"])
    log.info("  test=%d patients   trainval=%d patients (across %d folds)",
             len(test_tokens), len(trainval_tokens), args.n_folds)

    token_to_split: Dict[str, str] = {
        **{t: "test"     for t in test_tokens},
        **{t: "trainval" for t in trainval_tokens},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ct_dir  = args.out_dir / "ct"
    lbl_dir = args.out_dir / "labels"
    ct_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)
    (args.out_dir / "splits").mkdir(exist_ok=True)

    # Fixed test holdout (simple token list for downstream convenience)
    (args.out_dir / "splits" / "test.json").write_text(
        json.dumps(sorted(test_tokens), indent=2))
    # Full 5-fold CV structure on trainval pool (schema v3, matches
    # generate_5fold_splits.py — stable across both entry points)
    (args.out_dir / "splits" / "cv_5fold.json").write_text(
        json.dumps(cv, indent=2))

    work = [(r, str(args.nifti_dir), str(ct_dir), str(lbl_dir),
             (args.hu_min, args.hu_max)) for r in records]

    log.info("Exporting %d NIfTI pairs (workers=%d) ...", len(work), args.workers)
    t0 = time.time()
    n_ok = n_fail = n_skip = 0
    ok_set = set()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_export_one_record, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            tok, cfg, ok, msg = fut.result()
            if msg == "skip":
                n_skip += 1; ok_set.add((tok, cfg))
            elif ok:
                n_ok += 1;   ok_set.add((tok, cfg))
            else:
                n_fail += 1
                log.warning("  FAIL token=%s config=%s: %s", tok, cfg,
                            msg.strip().splitlines()[-1] if msg else "?")
            if i % 25 == 0 or i == len(work):
                elapsed = time.time() - t0
                eta     = elapsed / i * (len(work) - i) if i < len(work) else 0
                log.info("  [%d/%d] ok=%d skip=%d fail=%d  %.0fs ETA=%.0fs",
                         i, len(work), n_ok, n_skip, n_fail, elapsed, eta)

    log.info("Export done: ok=%d skip=%d fail=%d", n_ok, n_skip, n_fail)

    # Master manifest.json with one record per exported NIfTI pair
    master_records = []
    for r in records:
        key = (r["patient_token"], r["config"])
        if key not in ok_set:
            continue
        try:
            stem = f"{int(r['patient_token']):04d}_{r['config']}"
        except ValueError:
            stem = f"{r['patient_token']}_{r['config']}"
        sp = r.get("spine") or {}
        pv = r.get("pelvic") or {}
        master_records.append({
            "token":             r["patient_token"],
            "config":            r["config"],
            "match_type":        r["match_type"],
            "ct_file":           f"ct/{stem}_ct.nii.gz",
            "label_file":        f"labels/{stem}_label.nii.gz",
            "split":             token_to_split.get(r["patient_token"], "unknown"),
            "lstv_label":        r.get("lstv_pelvic") or r.get("lstv_vertebral") or "",
            "lstv_pelvic":       r.get("lstv_pelvic", ""),
            "lstv_vertebral":    r.get("lstv_vertebral", ""),
            "lstv_agreement":    r.get("lstv_agreement"),
            "lstv_confusion_zone": r.get("lstv_confusion_zone", False),
            "lstv_class":        r.get("lstv_class", 0),
            "has_l6":           bool(("lumbar" in str(r.get("lstv_pelvic","")).lower())
                                      or ("lumbar" in str(r.get("lstv_vertebral","")).lower())),
            "position":          sp.get("patient_position") or pv.get("patient_position") or "unknown",
            "spine_series_uid":  sp.get("series_uid"),
            "pelvic_series_uid": pv.get("series_uid"),
            "spine_bone_pct":    sp.get("bone_pct"),
            "pelvic_bone_pct":   pv.get("bone_pct"),
        })

    master = {
        "dataset_name": "CTSpinoPelvic1K",
        "n_records":    len(master_records),
        "n_patients":   len(set(r["token"] for r in master_records)),
        "label_scheme": LABEL_NAMES,
        "records":      master_records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(master, indent=2, default=str))
    log.info("Master manifest → %s  (%d records)",
             args.out_dir / "manifest.json", len(master_records))

    stats = {
        "n_records":       len(master_records),
        "n_patients":      len(set(r["token"] for r in master_records)),
        "records_by_config": dict(n_records_by_config),
        "records_by_split":  {s: sum(1 for r in master_records if r["split"] == s)
                               for s in ("trainval", "test")},
        "patients_by_split":  {"test":     len(test_tokens),
                                "trainval": len(trainval_tokens)},
        "n_folds":          args.n_folds,
        "test_fraction":    args.test_fraction,
        "strata_scheme":    cv["strata_scheme"],
        "lstv_classes":     dict(Counter(r["lstv_class"] for r in master_records)),
        "match_types":      dict(Counter(r["match_type"] for r in master_records)),
        "export_stats":     {"ok": n_ok, "skip": n_skip, "fail": n_fail},
    }
    (args.out_dir / "splits_summary.json").write_text(json.dumps(stats, indent=2))

    _write_readme(args.out_dir, cv, n_records_by_config)
    log.info("Wrote README.md and splits_summary.json → %s", args.out_dir)

    if args.push:
        if not args.hf_repo:
            log.error("--push requires --hf_repo USER/REPO")
            sys.exit(1)
        _push_to_hub(args.out_dir, args.hf_repo)


if __name__ == "__main__":
    main()
