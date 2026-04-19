#!/usr/bin/env python3
"""
generate_5fold_splits.py -- Stratified 5-fold CV splits at the patient-token level.

Reads placed_manifest.json (canonical, from place_fused_masks.py) and writes
data/splits_5fold.json with a test holdout + 5-fold CV split on the remaining
train pool, stratified on (match_type × has_lstv).

Schema version 3:
    {
      "schema_version": 3,
      "source_manifest":       "/data/.../placed_manifest.json",
      "source_manifest_kind":  "placed" | "hf",
      "test_fraction":         0.15,
      "n_folds":               5,
      "kfold_seed":            42,
      "strata_scheme":         "match_type_x_lstv",
      "n_tokens_total":        987,
      "n_tokens_test":         148,
      "n_tokens_trainval":     839,
      "test_tokens":           ["123", ...],
      "folds": [
        {"train_tokens": [...], "val_tokens": [...]},
        ...
      ],
      "strata_counts_total":   {"fused|no_lstv": 500, ...},
      "token_info":            {"<token>": {"match_type": ..., "has_lstv": ...}, ...}
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message="The least populated class in y has only",
    category=UserWarning,
)

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold
except ImportError:
    print("ERROR: scikit-learn required. `pip install scikit-learn`", file=sys.stderr)
    sys.exit(1)


_LSTV_NEGATIVE = {
    "", "none", "n/a", "na", "normal", "absent", "no", "false", "0",
    "negative", "normal_variant",
}


def is_lstv(label) -> bool:
    if label is None:
        return False
    s = str(label).strip().lower()
    return s not in _LSTV_NEGATIVE


def read_placed_manifest(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not cases:
        raise RuntimeError(f"No cases in {path}")
    out = []
    for c in cases:
        token = str(c.get("patient_token") or c.get("token") or "")
        if not token:
            continue
        match_type = c.get("match_type") or "unknown"
        # placed_manifest.json (new schema) stores LSTV at top level
        lstv_label = (
            c.get("lstv_pelvic")
            or c.get("lstv_vertebral")
            or c.get("lstv_label")
            or (c.get("spine") or {}).get("lstv_label")
            or None
        )
        out.append({
            "token":      token,
            "match_type": match_type,
            "lstv_label": lstv_label,
        })
    by_token: Dict[str, Dict] = {}
    for r in out:
        if r["token"] in by_token:
            prev = by_token[r["token"]]
            if prev.get("lstv_label") is None and r.get("lstv_label") is not None:
                prev["lstv_label"] = r["lstv_label"]
        else:
            by_token[r["token"]] = r
    return list(by_token.values())


_TOKEN_KEYS = ("token", "patient_token", "case_id", "id", "case", "name")
_MATCH_KEYS = ("match_type", "matchType", "match", "type")
_LSTV_KEYS  = ("lstv_label", "has_lstv", "lstv", "is_lstv", "LSTV", "lstv_class",
               "castellvi", "castellvi_type")


def _first(d: Dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default


def read_hf_export_manifest(path: Path) -> List[Dict]:
    raw = json.loads(path.read_text())

    if isinstance(raw, dict):
        unwrapped = None
        for k in ("cases", "records", "items", "data", "entries", "manifest"):
            if k in raw and isinstance(raw[k], list):
                unwrapped = raw[k]
                break
        if unwrapped is not None:
            raw = unwrapped
        elif raw and all(isinstance(v, dict) for v in raw.values()):
            raw = [{"token": k, **v} for k, v in raw.items()]
        else:
            raise RuntimeError(
                f"{path}: unrecognized manifest shape")

    if not isinstance(raw, list):
        raise RuntimeError(f"{path}: manifest root is {type(raw).__name__}")
    if not raw:
        raise RuntimeError(f"{path}: manifest is empty")

    out: List[Dict] = []
    n_missing_match = 0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        token = _first(entry, _TOKEN_KEYS)
        if token is None:
            continue
        token = str(token)
        if token.endswith("_0000"):
            token = token[:-5]
        match_type = _first(entry, _MATCH_KEYS)
        if match_type is None:
            match_type = "unknown"
            n_missing_match += 1
        lstv_raw = _first(entry, _LSTV_KEYS)
        out.append({
            "token":      token,
            "match_type": str(match_type),
            "lstv_label": lstv_raw,
        })

    by_token: Dict[str, Dict] = {}
    for r in out:
        if r["token"] in by_token:
            prev = by_token[r["token"]]
            if prev.get("lstv_label") is None and r.get("lstv_label") is not None:
                prev["lstv_label"] = r["lstv_label"]
        else:
            by_token[r["token"]] = r

    records = list(by_token.values())
    if n_missing_match > 0:
        print(f"WARN: {n_missing_match} records missing match_type",
              file=sys.stderr)
    return records


def _primary_stratum(match_type: str) -> str:
    return (match_type or "unknown").lower()


def compute_strata(records: List[Dict]) -> List[str]:
    strata = []
    any_lstv_info = any(r.get("lstv_label") is not None for r in records)
    for r in records:
        prim = _primary_stratum(r["match_type"])
        if any_lstv_info:
            lstv_tag = "lstv" if is_lstv(r["lstv_label"]) else "no_lstv"
            strata.append(f"{prim}|{lstv_tag}")
        else:
            strata.append(prim)
    return strata


def coalesce_rare_strata(strata: List[str], min_count: int) -> List[str]:
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


def existing_splits_still_valid(out_path: Path, current_tokens: List[str],
                                  n_folds: int) -> bool:
    if not out_path.exists():
        return False
    try:
        j = json.loads(out_path.read_text())
    except Exception:
        return False
    if j.get("n_folds") != n_folds:
        return False
    if j.get("schema_version", 0) < 3:
        return False
    tokens_in_splits = set(j.get("test_tokens", []))
    for f in j.get("folds", []):
        tokens_in_splits.update(f.get("train_tokens", []))
        tokens_in_splits.update(f.get("val_tokens", []))
    return tokens_in_splits == set(current_tokens)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate stratified 5-fold CV splits at patient-token level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--placed_manifest", type=Path, default=None,
                    help="Canonical placed_manifest.json (from place_fused_masks.py).")
    ap.add_argument("--hf_manifest", type=Path, default=None,
                    help="Explicit path to HF export manifest.json (fallback).")
    ap.add_argument("--hf_export_dir", type=Path, default=Path("data/hf_export"),
                    help="HF export dir; manifest.json here is last-resort fallback.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output splits JSON path.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest_path: Optional[Path] = None
    manifest_source: Optional[str] = None
    candidates = [
        (args.placed_manifest,                           "placed"),
        (args.hf_manifest,                               "hf"),
        (args.hf_export_dir / "manifest.json" if args.hf_export_dir else None, "hf"),
    ]
    for cand, kind in candidates:
        if cand and cand.exists():
            manifest_path, manifest_source = cand, kind
            break

    if manifest_path is None:
        print("ERROR: no manifest found. Tried:", file=sys.stderr)
        for cand, kind in candidates:
            if cand is not None:
                print(f"  [{kind}] {cand}  (exists={cand.exists()})", file=sys.stderr)
        return 1

    if not 0.0 < args.test_fraction < 0.5:
        print(f"ERROR: --test_fraction must be in (0, 0.5)", file=sys.stderr)
        return 1
    if args.n_folds < 2:
        print(f"ERROR: --n_folds must be >= 2", file=sys.stderr)
        return 1

    if manifest_source == "placed":
        print(f"Reading canonical placed_manifest: {manifest_path}")
        records = read_placed_manifest(manifest_path)
    else:
        print(f"Reading HF export manifest (fallback): {manifest_path}")
        records = read_hf_export_manifest(manifest_path)
    print(f"  found {len(records)} unique tokens")

    tokens = [r["token"] for r in records]

    if not args.overwrite and existing_splits_still_valid(
            args.out, tokens, args.n_folds):
        print(f"Existing {args.out} is valid. No-op. Use --overwrite to regenerate.")
        return 0

    strata_raw = compute_strata(records)
    min_count = max(args.n_folds, int(round(1.0 / args.test_fraction)))
    strata = coalesce_rare_strata(strata_raw, min_count)

    scheme = ("match_type_x_lstv"
              if any(r.get("lstv_label") is not None for r in records)
              else "match_type_only")
    print(f"  strata scheme: {scheme}")
    print(f"  strata counts: {dict(Counter(strata))}")

    print(f"\nHolding out {args.test_fraction * 100:.0f}% as test set...")
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_fraction, random_state=args.seed)
        trainval_idx, test_idx = next(sss.split(tokens, strata))
    except ValueError as exc:
        print(f"  WARN: stratified test split failed ({exc}); falling back.",
              file=sys.stderr)
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=args.test_fraction,
                          random_state=args.seed)
        trainval_idx, test_idx = next(ss.split(tokens))

    test_tokens  = sorted(tokens[i] for i in test_idx)
    trainval_tokens = [tokens[i] for i in trainval_idx]
    trainval_strata = [strata[i] for i in trainval_idx]
    print(f"  test:      {len(test_tokens)} tokens")
    print(f"  trainval:  {len(trainval_tokens)} tokens")

    print(f"\nStratified {args.n_folds}-fold on trainval pool...")
    trainval_strata_safe = coalesce_rare_strata(trainval_strata, args.n_folds)
    try:
        skf = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(skf.split(trainval_tokens, trainval_strata_safe))
    except ValueError as exc:
        print(f"  WARN: StratifiedKFold failed ({exc}); falling back to KFold.",
              file=sys.stderr)
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kf.split(trainval_tokens))

    folds_out = []
    for i, (tr_i, va_i) in enumerate(fold_splits):
        tr_tokens = sorted(trainval_tokens[j] for j in tr_i)
        va_tokens = sorted(trainval_tokens[j] for j in va_i)
        folds_out.append({"train_tokens": tr_tokens, "val_tokens": va_tokens})
        print(f"  fold {i}: train={len(tr_tokens)}  val={len(va_tokens)}")

    val_union = set()
    for f in folds_out:
        s = set(f["val_tokens"])
        if val_union & s:
            raise RuntimeError("Fold val sets not disjoint")
        val_union |= s
    if val_union != set(trainval_tokens):
        raise RuntimeError("Fold val union != trainval pool")

    stratum_by_token = dict(zip(tokens, strata))

    def _counts(token_list):
        return dict(Counter(stratum_by_token[t] for t in token_list))

    token_info = {
        r["token"]: {
            "match_type": r["match_type"],
            "has_lstv":   is_lstv(r.get("lstv_label")),
        }
        for r in records
    }

    doc = {
        "schema_version":      3,
        "created_at":          datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_manifest":     str(manifest_path.resolve()),
        "source_manifest_kind": manifest_source,
        "source_manifest_mtime":
            datetime.fromtimestamp(
                manifest_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "test_fraction":       args.test_fraction,
        "n_folds":              args.n_folds,
        "kfold_seed":          args.seed,
        "strata_scheme":       scheme,
        "n_tokens_total":      len(tokens),
        "n_tokens_test":       len(test_tokens),
        "n_tokens_trainval":   len(trainval_tokens),
        "test_tokens":         test_tokens,
        "folds":                folds_out,
        "strata_counts_total":  _counts(tokens),
        "strata_counts_test":   _counts(test_tokens),
        "strata_counts_per_fold_val":
            [_counts(f["val_tokens"]) for f in folds_out],
        "token_info":          token_info,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\nWrote {args.out}")
    print(f"  total tokens : {len(tokens)}")
    print(f"  test         : {len(test_tokens)}")
    print(f"  folds        : {args.n_folds}  (stratified on '{scheme}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
