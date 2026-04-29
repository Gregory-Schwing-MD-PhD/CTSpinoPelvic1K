(base) [go2432@warrior spinesurg-ct-nnunet]$ cat ~/CTSpinoPelvic1K/scripts/generate_5fold_splits.py
#!/usr/bin/env python3
"""
generate_5fold_splits.py -- Stratified 5-fold CV splits at the patient-token level.

Reads placed_manifest.json (canonical, from place_fused_masks.py) and writes
data/splits_5fold.json with a test holdout + 5-fold CV split on the remaining
train pool, stratified on (match_type × has_lstv).

BULLETPROOFING
--------------
This script treats the splits file as safety-critical: bad splits (overlap
between test and trainval, empty folds, unseen tokens at inference) silently
invalidate every downstream result. It therefore:

  * validates input manifest shape + required fields up front, with clear
    error messages identifying what's wrong and where
  * logs the input manifest's schema_version (v2.0+ from place_fused_masks.py)
    and warns if too many records are missing key fields
  * validates output invariants after generation:
      - test ∩ trainval = ∅
      - pairwise fold-val sets disjoint
      - union of fold-val sets == trainval pool
      - every fold has ≥1 train token and ≥1 val token
    and refuses to write if any invariant fails
  * atomic write (tmp file + os.replace) so an interrupted run can never
    leave a corrupt splits file that partially-overwrites a good one
  * cache-skip when an existing splits file already covers the same token
    set with the same n_folds — re-run with --overwrite to force
  * records whether stratification actually ran — the strata_scheme field
    gets an `_unstratified_fallback` suffix if sklearn bailed to plain KFold
    so downstream code knows the folds may be class-imbalanced

Schema version 3 (unchanged; all new fields are additive):
    {
      "schema_version":            3,
      "input_manifest_schema":     "2.0" | "2.1" | "1.x-legacy" | "unknown",
      "source_manifest":           "/data/.../placed_manifest.json",
      "source_manifest_kind":      "placed" | "hf",
      "test_fraction":             0.15,
      "n_folds":                   5,
      "kfold_seed":                42,
      "strata_scheme":             "match_type_x_lstv" | "match_type_only"
                                    | "<scheme>_unstratified_fallback",
      "n_tokens_total":            987,
      "n_tokens_test":             148,
      "n_tokens_trainval":         839,
      "test_tokens":               ["123", ...],
      "folds": [
        {"train_tokens": [...], "val_tokens": [...]},
        ...
      ],
      "strata_counts_total":       {"fused|no_lstv": 500, ...},
      "strata_counts_test":        {...},
      "strata_counts_per_fold_val":[{...}, ...],
      "lstv_counts":               {"total": {...}, "test": {...}, "folds_val": [{...}]},
      "position_counts":           {"total": {...}, "test": {...}, "folds_val": [{...}]},
      "invariants_validated":      true,
      "token_info":                {"<token>": {...}, ...}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ── Constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 3

_LSTV_NEGATIVE = {
    "", "none", "n/a", "na", "normal", "absent", "no", "false", "0",
    "negative", "normal_variant",
}

_TOKEN_KEYS = ("token", "patient_token", "case_id", "id", "case", "name")
_MATCH_KEYS = ("match_type", "matchType", "match", "type")
_LSTV_KEYS  = ("lstv_label", "has_lstv", "lstv", "is_lstv", "LSTV", "lstv_class",
               "castellvi", "castellvi_type")

# Warn the user if more than this fraction of records lack a given field.
# LSTV threshold is tighter because LSTV agreement is the paper's key
# stratification variable — silent degradation there is much more costly
# than for match_type.
_WARN_MISSING_FRAC      = 0.10
_WARN_MISSING_FRAC_LSTV = 0.05


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON via tmp file + os.replace. Non-atomic writes risk
    corrupting a good existing splits file if the job is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(str(tmp), str(path))


def _first(d: Dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default


def is_lstv(label) -> bool:
    if label is None:
        return False
    s = str(label).strip().lower()
    return s not in _LSTV_NEGATIVE


# ── Manifest loading ────────────────────────────────────────────────────────

def _warn_missing(records: List[Dict], field: str,
                  log_prefix: str = "",
                  threshold: float = _WARN_MISSING_FRAC) -> None:
    n = len(records)
    if n == 0:
        return
    n_missing = sum(1 for r in records
                    if r.get(field) in (None, "", "unknown"))
    frac = n_missing / n
    if frac > threshold:
        print(f"WARN: {log_prefix}{n_missing}/{n} records "
              f"({frac*100:.1f}%) missing '{field}'  "
              f"[threshold {threshold*100:.0f}%]",
              file=sys.stderr)


def read_placed_manifest(path: Path) -> Tuple[List[Dict], str]:
    """
    Read canonical placed_manifest.json (from place_fused_masks.py).

    Returns (records, input_schema_version).
    Accepts both the v2.0 dict-with-cases shape and a legacy flat-list shape.
    """
    raw = json.loads(path.read_text())

    schema = "unknown"
    if isinstance(raw, list):
        print(f"WARN: {path} is a flat list (legacy). Wrapping.",
              file=sys.stderr)
        cases = raw
        schema = "1.x-legacy"
    elif isinstance(raw, dict):
        schema = str(raw.get("schema_version", "unknown"))
        cases = raw.get("cases", [])
        if not isinstance(cases, list):
            raise ValueError(
                f"{path}: 'cases' must be a list, got {type(cases).__name__}")
    else:
        raise ValueError(
            f"{path}: manifest root is {type(raw).__name__}; "
            "expected dict or list")

    if not cases:
        raise RuntimeError(f"{path}: no cases found")

    out = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        token = str(c.get("patient_token") or c.get("token") or "")
        if not token:
            continue
        match_type = c.get("match_type") or "unknown"

        # placed_manifest.json (v2.0+ schema) stores LSTV at top level
        lstv_label = (
            c.get("lstv_pelvic")
            or c.get("lstv_vertebral")
            or c.get("lstv_label")
            or (c.get("spine") or {}).get("lstv_label")
            or None
        )
        # Position field is v2.0+; fall back to unknown on older manifests
        position = c.get("position") or "unknown"

        out.append({
            "token":      token,
            "match_type": match_type,
            "lstv_label": lstv_label,
            "position":   position,
        })

    # Deduplicate by token — a placed_manifest where the same token appears
    # multiple times would otherwise yield splits containing the same patient
    # in both train and val, silently violating patient-level separation.
    by_token: Dict[str, Dict] = {}
    n_dup = 0
    for r in out:
        tok = r["token"]
        if tok in by_token:
            n_dup += 1
            prev = by_token[tok]
            # Fill in missing fields from the duplicate
            if prev.get("lstv_label") is None and r.get("lstv_label") is not None:
                prev["lstv_label"] = r["lstv_label"]
            if prev.get("position", "unknown") == "unknown" \
                    and r.get("position", "unknown") != "unknown":
                prev["position"] = r["position"]
        else:
            by_token[tok] = dict(r)
    if n_dup:
        print(f"INFO: placed manifest had {n_dup} duplicate token records "
              f"(collapsed to {len(by_token)} unique).", file=sys.stderr)

    records = list(by_token.values())
    _warn_missing(records, "lstv_label", "placed manifest: ",
                  threshold=_WARN_MISSING_FRAC_LSTV)
    return records, schema


def read_hf_export_manifest(path: Path) -> Tuple[List[Dict], str]:
    """
    Fallback reader for the HF export manifest (scripts/export_hf.py output).

    Returns (records, input_schema_version). The HF manifest has no
    schema_version field, so returns "hf_export" as a kind-of-schema tag.
    """
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
            raise RuntimeError(f"{path}: unrecognized manifest shape")

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
            token = token[:-5]  # nnU-Net-style channel suffix
        match_type = _first(entry, _MATCH_KEYS)
        if match_type is None:
            match_type = "unknown"
            n_missing_match += 1
        lstv_raw = _first(entry, _LSTV_KEYS)
        position = entry.get("position") or "unknown"
        out.append({
            "token":      token,
            "match_type": str(match_type),
            "lstv_label": lstv_raw,
            "position":   position,
        })

    by_token: Dict[str, Dict] = {}
    for r in out:
        if r["token"] in by_token:
            prev = by_token[r["token"]]
            if prev.get("lstv_label") is None and r.get("lstv_label") is not None:
                prev["lstv_label"] = r["lstv_label"]
            if prev.get("position", "unknown") == "unknown" \
                    and r.get("position", "unknown") != "unknown":
                prev["position"] = r["position"]
        else:
            by_token[r["token"]] = r

    records = list(by_token.values())
    if n_missing_match > 0:
        print(f"WARN: {n_missing_match} records missing match_type",
              file=sys.stderr)
    _warn_missing(records, "lstv_label", "hf manifest: ",
                  threshold=_WARN_MISSING_FRAC_LSTV)
    return records, "hf_export"


# ── Stratification ──────────────────────────────────────────────────────────

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


def _warn_small_strata(strata: List[str], n_folds: int,
                        label: str = "") -> None:
    """Emit a heads-up when any stratum has fewer samples than n_folds.
    StratifiedKFold will either silently imbalance or raise here — we
    want an explicit log line so the reason is in the job output."""
    counts = Counter(strata)
    small = {s: c for s, c in counts.items() if c < n_folds}
    if small:
        print(f"WARN: {label}strata below n_folds={n_folds} (stratification will "
              f"be degraded or fall back to plain KFold): {small}",
              file=sys.stderr)


# ── Post-split invariant validation ─────────────────────────────────────────

def _validate_invariants(
    all_tokens:       List[str],
    test_tokens:      List[str],
    trainval_tokens:  List[str],
    folds_out:        List[Dict[str, List[str]]],
) -> None:
    """
    Raise RuntimeError if any split invariant is violated.

    Checked invariants:
      1. Every all_tokens entry is unique
      2. test ∩ trainval = ∅
      3. test ∪ trainval = all_tokens  (no tokens dropped, none invented)
      4. Every fold has ≥1 train token and ≥1 val token
      5. Each fold's val ⊆ trainval
      6. Pairwise fold-val disjoint
      7. Union of fold-val = trainval
    """
    all_set      = set(all_tokens)
    test_set     = set(test_tokens)
    trainval_set = set(trainval_tokens)

    # 1. Uniqueness
    if len(all_tokens) != len(all_set):
        dups = [t for t, c in Counter(all_tokens).items() if c > 1]
        raise RuntimeError(f"Duplicate tokens in input: {dups[:10]}")

    # 2. Disjoint test / trainval
    overlap = test_set & trainval_set
    if overlap:
        raise RuntimeError(
            f"Test and trainval overlap ({len(overlap)} tokens): "
            f"{sorted(overlap)[:10]}")

    # 3. No dropped/invented tokens
    covered = test_set | trainval_set
    missing = all_set - covered
    extra   = covered - all_set
    if missing:
        raise RuntimeError(
            f"{len(missing)} tokens dropped from splits: "
            f"{sorted(missing)[:10]}")
    if extra:
        raise RuntimeError(
            f"{len(extra)} tokens in splits but not in input: "
            f"{sorted(extra)[:10]}")

    # 4-7. Fold-level invariants
    seen_val = set()
    for i, fold in enumerate(folds_out):
        tr = set(fold["train_tokens"])
        va = set(fold["val_tokens"])

        if not tr or not va:
            raise RuntimeError(
                f"Fold {i} has empty side: |train|={len(tr)}  |val|={len(va)}")

        if not va.issubset(trainval_set):
            raise RuntimeError(
                f"Fold {i} val contains tokens outside trainval: "
                f"{sorted(va - trainval_set)[:10]}")

        if not tr.issubset(trainval_set):
            raise RuntimeError(
                f"Fold {i} train contains tokens outside trainval: "
                f"{sorted(tr - trainval_set)[:10]}")

        if tr & va:
            raise RuntimeError(
                f"Fold {i} has train ∩ val overlap: {sorted(tr & va)[:10]}")

        if va & seen_val:
            raise RuntimeError(
                f"Fold {i} val overlaps with earlier folds: "
                f"{sorted(va & seen_val)[:10]}")
        seen_val |= va

    # Union of val sets should equal trainval (standard k-fold property)
    if seen_val != trainval_set:
        missing_val = trainval_set - seen_val
        extra_val   = seen_val - trainval_set
        raise RuntimeError(
            f"Union of fold val sets != trainval pool. "
            f"missing_from_any_val={len(missing_val)} "
            f"extra_in_val_but_not_trainval={len(extra_val)}")


# ── Existing-file short-circuit ─────────────────────────────────────────────

def existing_splits_still_valid(
    out_path:        Path,
    current_tokens:  List[str],
    n_folds:         int,
) -> bool:
    """True iff a prior splits file covers the exact same token set with
    the same n_folds (schema v3+). Used to skip regeneration on re-runs."""
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


# ── Main ────────────────────────────────────────────────────────────────────

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

    # ── Pick manifest ───────────────────────────────────────────────────────
    manifest_path: Optional[Path] = None
    manifest_source: Optional[str] = None
    candidates = [
        (args.placed_manifest, "placed"),
        (args.hf_manifest,     "hf"),
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
                print(f"  [{kind}] {cand}  (exists={cand.exists()})",
                      file=sys.stderr)
        return 1

    # ── Validate args ───────────────────────────────────────────────────────
    if not 0.0 < args.test_fraction < 0.5:
        print(f"ERROR: --test_fraction must be in (0, 0.5)", file=sys.stderr)
        return 1
    if args.n_folds < 2:
        print(f"ERROR: --n_folds must be >= 2", file=sys.stderr)
        return 1

    # ── Load manifest ───────────────────────────────────────────────────────
    try:
        if manifest_source == "placed":
            print(f"Reading canonical placed_manifest: {manifest_path}")
            records, input_schema = read_placed_manifest(manifest_path)
        else:
            print(f"Reading HF export manifest (fallback): {manifest_path}")
            records, input_schema = read_hf_export_manifest(manifest_path)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"  input schema: {input_schema}")
    print(f"  unique tokens: {len(records)}")

    tokens = [r["token"] for r in records]

    # Guard against not-enough-tokens-to-split.
    min_required = max(args.n_folds, int(round(1.0 / args.test_fraction))) + 1
    if len(tokens) < min_required:
        print(f"ERROR: need at least {min_required} tokens for "
              f"{args.n_folds}-fold × test_fraction={args.test_fraction}, "
              f"got {len(tokens)}.", file=sys.stderr)
        return 1

    # ── Skip if existing splits still valid ────────────────────────────────
    if not args.overwrite and existing_splits_still_valid(
            args.out, tokens, args.n_folds):
        print(f"Existing {args.out} is valid. No-op. Use --overwrite to regenerate.")
        return 0

    # ── Strata ──────────────────────────────────────────────────────────────
    strata_raw = compute_strata(records)
    min_count  = max(args.n_folds, int(round(1.0 / args.test_fraction)))
    strata     = coalesce_rare_strata(strata_raw, min_count)

    scheme = ("match_type_x_lstv"
              if any(r.get("lstv_label") is not None for r in records)
              else "match_type_only")
    print(f"  strata scheme: {scheme}")
    print(f"  strata counts: {dict(Counter(strata))}")
    _warn_small_strata(strata, args.n_folds, label="post-coalesce: ")

    # Track whether stratification actually held all the way through.
    # When sklearn bails to plain KFold, we record that in strata_scheme
    # so downstream readers know per-fold class balance isn't guaranteed.
    test_was_stratified  = True
    folds_were_stratified = True

    # ── Test holdout ────────────────────────────────────────────────────────
    print(f"\nHolding out {args.test_fraction * 100:.0f}% as test set...")
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_fraction, random_state=args.seed)
        trainval_idx, test_idx = next(sss.split(tokens, strata))
    except ValueError as exc:
        print(f"  WARN: stratified test split failed ({exc}); falling back.",
              file=sys.stderr)
        test_was_stratified = False
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=args.test_fraction,
                          random_state=args.seed)
        trainval_idx, test_idx = next(ss.split(tokens))

    test_tokens     = sorted(tokens[i] for i in test_idx)
    trainval_tokens = [tokens[i] for i in trainval_idx]
    trainval_strata = [strata[i] for i in trainval_idx]
    print(f"  test:      {len(test_tokens)} tokens")
    print(f"  trainval:  {len(trainval_tokens)} tokens")

    # ── 5-fold CV ───────────────────────────────────────────────────────────
    print(f"\nStratified {args.n_folds}-fold on trainval pool...")
    trainval_strata_safe = coalesce_rare_strata(trainval_strata, args.n_folds)
    _warn_small_strata(trainval_strata_safe, args.n_folds,
                        label="trainval post-coalesce: ")
    try:
        skf = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(skf.split(trainval_tokens, trainval_strata_safe))
    except ValueError as exc:
        print(f"  WARN: StratifiedKFold failed ({exc}); falling back to KFold.",
              file=sys.stderr)
        folds_were_stratified = False
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kf.split(trainval_tokens))

    # Record the effective scheme: if either stratification step fell
    # back to the unstratified shuffler, append a suffix so the caller
    # can tell just by reading the JSON.
    if test_was_stratified and folds_were_stratified:
        effective_scheme = scheme
    else:
        suffix_parts = []
        if not test_was_stratified:
            suffix_parts.append("test")
        if not folds_were_stratified:
            suffix_parts.append("folds")
        effective_scheme = f"{scheme}_unstratified_fallback_{'+'.join(suffix_parts)}"
        print(f"  WARN: effective strata_scheme={effective_scheme}",
              file=sys.stderr)

    folds_out = []
    for i, (tr_i, va_i) in enumerate(fold_splits):
        tr_tokens = sorted(trainval_tokens[j] for j in tr_i)
        va_tokens = sorted(trainval_tokens[j] for j in va_i)
        folds_out.append({"train_tokens": tr_tokens, "val_tokens": va_tokens})
        print(f"  fold {i}: train={len(tr_tokens)}  val={len(va_tokens)}")

    # ── Validate invariants (refuse to write on failure) ────────────────────
    try:
        _validate_invariants(
            all_tokens=tokens,
            test_tokens=test_tokens,
            trainval_tokens=trainval_tokens,
            folds_out=folds_out,
        )
        print("  ✓ all split invariants validated")
    except RuntimeError as exc:
        print(f"ERROR: split invariant violated: {exc}", file=sys.stderr)
        print("Refusing to write splits file.", file=sys.stderr)
        return 1

    # ── Build output ────────────────────────────────────────────────────────
    stratum_by_token = dict(zip(tokens, strata))
    lstv_by_token    = {r["token"]: is_lstv(r.get("lstv_label")) for r in records}
    pos_by_token     = {r["token"]: (r.get("position") or "unknown")
                        for r in records}

    def _counts_stratum(token_list):
        return dict(Counter(stratum_by_token[t] for t in token_list))

    def _counts_lstv(token_list):
        return dict(Counter(
            "lstv" if lstv_by_token.get(t, False) else "no_lstv"
            for t in token_list
        ))

    def _counts_position(token_list):
        return dict(Counter(pos_by_token.get(t, "unknown") for t in token_list))

    token_info = {
        r["token"]: {
            "match_type": r["match_type"],
            "has_lstv":   is_lstv(r.get("lstv_label")),
            "position":   r.get("position", "unknown"),
        }
        for r in records
    }

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest_mtime_iso = datetime.fromtimestamp(
        manifest_path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    doc = {
        # Provenance
        "schema_version":          SCHEMA_VERSION,
        "input_manifest_schema":   input_schema,
        "created_at":              now_iso,
        "source_manifest":         str(manifest_path.resolve()),
        "source_manifest_kind":    manifest_source,
        "source_manifest_mtime":   manifest_mtime_iso,
        # Parameters
        "test_fraction":           args.test_fraction,
        "n_folds":                 args.n_folds,
        "kfold_seed":              args.seed,
        "strata_scheme":           effective_scheme,
        "strata_scheme_intended":  scheme,
        "test_was_stratified":     test_was_stratified,
        "folds_were_stratified":   folds_were_stratified,
        # Counts
        "n_tokens_total":          len(tokens),
        "n_tokens_test":           len(test_tokens),
        "n_tokens_trainval":       len(trainval_tokens),
        # Splits
        "test_tokens":             test_tokens,
        "folds":                   folds_out,
        # Per-split breakdowns
        "strata_counts_total":     _counts_stratum(tokens),
        "strata_counts_test":      _counts_stratum(test_tokens),
        "strata_counts_per_fold_val": [_counts_stratum(f["val_tokens"])
                                        for f in folds_out],
        "lstv_counts": {
            "total":     _counts_lstv(tokens),
            "test":      _counts_lstv(test_tokens),
            "folds_val": [_counts_lstv(f["val_tokens"]) for f in folds_out],
        },
        "position_counts": {
            "total":     _counts_position(tokens),
            "test":      _counts_position(test_tokens),
            "folds_val": [_counts_position(f["val_tokens"]) for f in folds_out],
        },
        # Validation
        "invariants_validated":    True,
        # Token metadata
        "token_info":              token_info,
    }

    # ── Atomic write ────────────────────────────────────────────────────────
    _atomic_write_json(args.out, doc)
    print(f"\nWrote {args.out}")
    print(f"  total tokens : {len(tokens)}")
    print(f"  test         : {len(test_tokens)}")
    print(f"  folds        : {args.n_folds}  (stratified on '{effective_scheme}')")
    print(f"  input schema : {input_schema}")
    print(f"  invariants   : validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
