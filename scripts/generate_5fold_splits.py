"""
generate_5fold_splits.py — 5-fold CV splits with LSTV stratification.

SCHEMA VERSION 6 (Apr 2026)
===========================
6-way LSTV subtype taxonomy, derived from per-patient signals in the HF
manifests (manifest_train.json, manifest_validation.json, manifest_test.json):

  Subtype             n_expected   Resolution rule
  ------------------- -----------  ----------------------------------------
  normal              ~769         lstv_label all NORMAL across patient records
  lumb                14           any record with n_lumbar_labels==6 or has_l6=True
  sacr_count          9            vert=SACR + n_lumb=4 (annotator drew 4 lumbars)
  semisacralization   2            any record with lstv_label=SEMI_SACRAL or
                                   lstv_pelvic=SEMI_SACRALIZATION (tokens 22, 120)
  sacralization       6            lstv_label=SACRALIZATION + n_lumb in {0, 5} or
                                   pelvic-only with sacralization filename
  ambiguous           2            cross-anatomy disagreement (tokens 4, 67)

Pre-fix (schema v5) and earlier had only a 4-way scheme that lost the
semi-sacralization vs full-sacralization distinction (substring-match bug
in mask_index.py). After the parser fix this schema correctly distinguishes
all 6 clinically meaningful subtypes for stratified CV.

PATIENT-LEVEL SPLITS
====================
Splits are at the patient_token level (each patient -> single fold). Within
a fold, all records belonging to a patient go together.

OUTPUT
======
splits_5fold.json with the following structure:
  {
    "schema_version": 6,
    "n_patients": int,
    "subtype_counts": {subtype: n, ...},
    "folds": [
      {"fold": 0, "train": [tok, ...], "val": [tok, ...]},
      ...
    ],
    "patient_subtypes": {tok: subtype, ...},
    "patient_attrs": {tok: {n_lumb_max, has_l6_any, lstv_label, lstv_pelvic,
                            lstv_vertebral, n_records}, ...}
  }
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.generate_5fold_splits")

SCHEMA_VERSION = 6

SUBTYPES = (
    "normal",
    "lumb",
    "sacr_count",
    "semisacralization",
    "sacralization",
    "ambiguous",
)


def _load_manifest_records(hf_export_dir: Path) -> List[Dict]:
    """Load all records from train/val/test manifests in the HF export dir."""
    records: List[Dict] = []
    for fn in ("manifest_train.json", "manifest_validation.json", "manifest_test.json"):
        p = hf_export_dir / fn
        if not p.exists():
            log.warning("manifest not found: %s", p)
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            log.warning("failed to load %s: %s", p, e)
            continue
        if isinstance(data, dict):
            data = data.get("records", data.get("cases", list(data.values())))
        if isinstance(data, list):
            records.extend(r for r in data if isinstance(r, dict))
            log.info("loaded %d records from %s", len(data), fn)
    return records


def _aggregate_per_patient(records: List[Dict]) -> Dict[str, Dict]:
    """Collapse multiple records per patient into a single attribute dict."""
    by_token: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        tok = str(r.get("token") or r.get("patient_token") or "")
        if not tok:
            continue
        by_token[tok].append(r)

    attrs: Dict[str, Dict] = {}
    for tok, recs in by_token.items():
        n_lumb_max = max(
            (int(r.get("n_lumbar_labels", 0) or 0) for r in recs),
            default=0,
        )
        has_l6_any = any(bool(r.get("has_l6", False)) for r in recs)

        # lstv_label (general): pick the most informative value across records
        lstv_labels = [
            str(r.get("lstv_label", "")).upper() for r in recs
            if r.get("lstv_label")
        ]
        lstv_label = ""
        for candidate in ("AMBIGUOUS", "LUMBARIZATION", "SEMI_SACRAL",
                           "SEMI_SACRALIZATION", "SACRALIZATION", "NORMAL"):
            if candidate in lstv_labels:
                lstv_label = candidate
                break

        # lstv_pelvic / lstv_vertebral: first non-empty across records
        lstv_pelvic = ""
        lstv_vertebral = ""
        for r in recs:
            if not lstv_pelvic and r.get("lstv_pelvic"):
                lstv_pelvic = str(r.get("lstv_pelvic", "")).upper()
            if not lstv_vertebral and r.get("lstv_vertebral"):
                lstv_vertebral = str(r.get("lstv_vertebral", "")).upper()

        # lstv_class: max across records (consistent w/ existing _resolve fns)
        lstv_class = max(
            (int(r.get("lstv_class", 0) or 0) for r in recs),
            default=0,
        )

        attrs[tok] = {
            "n_lumb_max":      n_lumb_max,
            "has_l6_any":      has_l6_any,
            "lstv_label":      lstv_label,
            "lstv_pelvic":     lstv_pelvic,
            "lstv_vertebral":  lstv_vertebral,
            "lstv_class":      lstv_class,
            "n_records":       len(recs),
        }
    return attrs


def _resolve_subtype(a: Dict) -> str:
    """
    Resolve 6-way LSTV subtype from per-patient aggregated attributes.

    Resolution order (first match wins):
      1. ambiguous           cross-anatomy disagreement
      2. lumb                n_lumb_max == 6 OR has_l6
      3. semisacralization   any SEMI_* label hit
      4. sacr_count          vert=SACR + n_lumb_max == 4
      5. sacralization       SACRALIZATION label + n_lumb in {0, 5} OR pelvic-only
      6. normal              fallthrough
    """
    label = a.get("lstv_label", "")
    pel   = a.get("lstv_pelvic", "")
    vert  = a.get("lstv_vertebral", "")
    n_lumb = int(a.get("n_lumb_max", 0) or 0)
    has_l6 = bool(a.get("has_l6_any", False))

    is_lumb_lstv = lambda s: s in ("LUMBARIZATION",)
    is_sacr_lstv = lambda s: s in ("SACRALIZATION", "SEMI_SACRAL", "SEMI_SACRALIZATION")
    is_semi      = lambda s: s in ("SEMI_SACRAL", "SEMI_SACRALIZATION")

    # 1. Ambiguous: pelvic and vertebral signals point to opposite LSTV directions
    if pel and vert and is_lumb_lstv(vert) and is_sacr_lstv(pel):
        return "ambiguous"
    if pel and vert and is_sacr_lstv(vert) and is_lumb_lstv(pel):
        return "ambiguous"
    if label == "AMBIGUOUS":
        return "ambiguous"

    # 2. Lumbarization: extra L6 either by count or by has_l6 flag
    if n_lumb == 6 or has_l6 or is_lumb_lstv(label) or is_lumb_lstv(pel) or is_lumb_lstv(vert):
        return "lumb"

    # 3. Semi-sacralization: any semi flag
    if is_semi(label) or is_semi(pel):
        return "semisacralization"

    # 4. Sacralization-by-count: spine annotator drew 4 lumbars
    if is_sacr_lstv(vert) and n_lumb == 4:
        return "sacr_count"

    # 5. Sacralization-by-morphology / pelvic-flagged
    if is_sacr_lstv(label) or is_sacr_lstv(pel):
        return "sacralization"

    # 6. Default: normal
    return "normal"


def _stratified_kfold_split(
    tokens: List[str],
    subtypes_by_token: Dict[str, str],
    n_folds: int,
    seed: int,
) -> List[Dict[str, List[str]]]:
    """
    Per-stratum round-robin assignment of tokens to folds. Sklearn's
    StratifiedKFold can't handle strata with n < n_folds; we round-robin
    instead so each fold gets at least 1 token from every stratum that
    has at least n_folds tokens, and small strata are distributed as
    evenly as possible.
    """
    import random
    rng = random.Random(seed)

    by_subtype: Dict[str, List[str]] = defaultdict(list)
    for tok in tokens:
        st = subtypes_by_token.get(tok, "normal")
        by_subtype[st].append(tok)

    # Shuffle each stratum so cross-fold assignment is randomized but seed-deterministic.
    for st in by_subtype:
        rng.shuffle(by_subtype[st])

    fold_train: List[List[str]] = [[] for _ in range(n_folds)]
    fold_val:   List[List[str]] = [[] for _ in range(n_folds)]

    # Round-robin assign each stratum's tokens to folds (val = current fold,
    # train = the other folds). To produce a proper k-fold partition, we
    # assign each token to exactly one fold's val list, then build train
    # lists by aggregating the rest.
    val_assignments: Dict[str, int] = {}
    for st, toks in by_subtype.items():
        for i, tok in enumerate(toks):
            val_assignments[tok] = i % n_folds

    for tok, vfold in val_assignments.items():
        for f in range(n_folds):
            if f == vfold:
                fold_val[f].append(tok)
            else:
                fold_train[f].append(tok)

    return [
        {"fold": f, "train": sorted(fold_train[f]), "val": sorted(fold_val[f])}
        for f in range(n_folds)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_dir", required=True, type=Path,
                   help="HF export directory containing manifest_*.json files")
    p.add_argument("--out",    required=True, type=Path,
                   help="Output path for splits_5fold.json")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    if not args.hf_dir.exists():
        log.error("HF export dir not found: %s", args.hf_dir)
        raise SystemExit(1)

    records = _load_manifest_records(args.hf_dir)
    if not records:
        log.error("No records found in %s", args.hf_dir)
        raise SystemExit(1)
    log.info("Loaded %d records total", len(records))

    attrs = _aggregate_per_patient(records)
    log.info("Aggregated to %d unique patients", len(attrs))

    subtypes_by_token: Dict[str, str] = {}
    for tok, a in attrs.items():
        subtypes_by_token[tok] = _resolve_subtype(a)

    counts = Counter(subtypes_by_token.values())
    log.info("Subtype counts:")
    for st in SUBTYPES:
        log.info("  %-20s %d", st, counts.get(st, 0))
    other = sum(c for st, c in counts.items() if st not in SUBTYPES)
    if other:
        log.warning("  (unrecognised subtypes total: %d)", other)

    # Sanity check: expected counts
    expected = {"lumb": 14, "semisacralization": 2, "sacralization": 6,
                "sacr_count": 9, "ambiguous": 2}
    for st, exp in expected.items():
        actual = counts.get(st, 0)
        if actual != exp:
            log.warning("  subtype '%s': got %d, expected %d (Apr 2026 fix target)",
                        st, actual, exp)

    tokens_sorted = sorted(
        attrs.keys(),
        key=lambda t: (0, int(t)) if t.isdigit() else (1, t),
    )
    folds = _stratified_kfold_split(
        tokens_sorted, subtypes_by_token, args.n_folds, args.seed,
    )

    # Per-fold subtype distribution
    log.info("Per-fold subtype distribution:")
    log.info("  %-3s  %-7s  %-7s  %s",
             "fold", "n_train", "n_val", "  ".join(f"{st[:8]:>8}" for st in SUBTYPES))
    for fold in folds:
        f = fold["fold"]
        val_sub_counts = Counter(subtypes_by_token[t] for t in fold["val"])
        log.info("  %-3d  %-7d  %-7d  %s",
                 f, len(fold["train"]), len(fold["val"]),
                 "  ".join(f"{val_sub_counts.get(st, 0):>8}" for st in SUBTYPES))

    out_data = {
        "schema_version": SCHEMA_VERSION,
        "n_patients":     len(attrs),
        "n_folds":        args.n_folds,
        "seed":           args.seed,
        "subtypes":       list(SUBTYPES),
        "subtype_counts": dict(counts),
        "folds":          folds,
        "patient_subtypes": subtypes_by_token,
        "patient_attrs":  attrs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, default=str))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
