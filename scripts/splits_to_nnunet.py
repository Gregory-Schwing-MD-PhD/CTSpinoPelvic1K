#!/usr/bin/env python3
"""
splits_to_nnunet.py — Translate CTSpinoPelvic1K splits_5fold.json into
nnU-Net's splits_final.json.

WHY THIS EXISTS
---------------
Our splits_5fold.json holds TRAIN/VAL/TEST at the patient-TOKEN level
(because the same patient may appear as 1-3 cases: fused, spine_only,
pelvic_native, depending on which masks were available). nnU-Net, on the
other hand, expects splits at the CASE level and wants a specific on-disk
layout:

    nnUNet_preprocessed/Dataset<ID>_<name>/splits_final.json

The file is a JSON list of {"train": [...], "val": [...]} dicts, one per
fold; elements are nnU-Net case IDs (the stems of the CT files under
imagesTr, e.g. "CTSpinoPelvic_0142_fused" without the "_0000" channel
suffix).

ANATOMY-LEAK GUARD
------------------
A single patient token T expands to up to 3 cases. If those cases landed
on different sides of a train/val split, the network would train on one
view of T's anatomy and validate on another view of the SAME anatomy —
a textbook form of data leakage that silently inflates val metrics.
This script asserts that every expanded case for a given token lands
on the same side of each fold's split. If the assertion fires, something
upstream is wrong (most likely someone split at the case level instead
of the token level) and training is halted.

TEST TOKENS are expanded and stored separately as `test_cases.json`
alongside `splits_final.json`. They are deliberately kept OUT of
nnU-Net's fold structure so that no fold's val set ever sees a test
patient.

HF MANIFEST SHAPE
-----------------
Reads `hf_export/manifest.json` (produced by export_hf.py) and expects
each record to carry, at minimum:

    {
      "token":    "142",                     # patient token
      "config":   "fused" | "spine_only" | "pelvic_native",
      "case_id":  "CTSpinoPelvic_0142_fused" # nnU-Net-style stem
    }

`case_id` is the authoritative nnU-Net ID. If it's missing, we
synthesize one from token + config using a deterministic template:

    f"{dataset_name}_{int(token):04d}_{config}"

(Override the template via --case_id_template if export_hf.py uses a
different convention.)

USAGE
=====
    # Default: read splits_5fold.json + manifest.json from hf_export/
    python splits_to_nnunet.py \\
        --splits           data/hf_export/splits_5fold.json \\
        --hf_manifest      data/hf_export/manifest.json \\
        --nnunet_raw_dir   $nnUNet_preprocessed/Dataset501_CTSpinoPelvic \\
        --dataset_name     CTSpinoPelvic

    # Custom case-ID template (if export_hf.py uses a different one)
    python splits_to_nnunet.py \\
        --splits           data/hf_export/splits_5fold.json \\
        --hf_manifest      data/hf_export/manifest.json \\
        --nnunet_raw_dir   /path/to/Dataset501_CTSpinoPelvic \\
        --case_id_template "{dataset_name}_{token_int:04d}_{config}"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON via tmp file + os.replace so an interrupted run can never
    leave a half-written splits_final.json — nnU-Net would crash on parse
    and training would need to be redone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(str(tmp), str(path))


def _load_json(path: Path):
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path}: invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)


# ── HF manifest parsing ──────────────────────────────────────────────────────

def _unwrap_manifest(raw) -> List[dict]:
    """HF manifest may be a list, a {'records': [...]} / {'cases': [...]} dict,
    or a {token: {...}} dict. Normalize to a flat list of records."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for k in ("cases", "records", "items", "data", "entries", "manifest"):
            if k in raw and isinstance(raw[k], list):
                return raw[k]
        # {token: record} shape
        if raw and all(isinstance(v, dict) for v in raw.values()):
            return [{"token": k, **v} for k, v in raw.items()]
    raise RuntimeError(
        f"Unrecognized HF manifest shape: root is {type(raw).__name__}")


def _synth_case_id(token: str, config: str,
                   dataset_name: str, template: str) -> str:
    """Synthesize a case ID when the HF manifest doesn't provide one.
    Safe for non-numeric tokens — falls back to zero padding the string."""
    try:
        token_int = int(token)
    except ValueError:
        token_int = None
    # Provide both token_int (for numeric tokens) and token (always) so
    # templates can choose whichever makes sense.
    fmt_args = {
        "dataset_name": dataset_name,
        "token":        token,
        "config":       config,
    }
    if token_int is not None:
        fmt_args["token_int"] = token_int
    try:
        return template.format(**fmt_args)
    except KeyError as exc:
        # Template references a field we don't have (e.g. token_int when
        # token is non-numeric). Re-raise with context.
        raise RuntimeError(
            f"Case-ID template {template!r} references {exc} but token={token!r} "
            f"does not provide it. Use a template that only references "
            f"{{dataset_name}}, {{token}}, {{config}} for non-numeric tokens."
        ) from exc


def build_token_to_cases(
    manifest_records: List[dict],
    dataset_name:     str,
    case_id_template: str,
    strict:           bool,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns: {token: [(case_id, config), ...]}

    `case_id` is the authoritative nnU-Net ID — either read from the
    record's `case_id` field if present, or synthesized from the template.

    If `strict`, we abort on any record missing `token` or `config`;
    otherwise we skip + warn.
    """
    token_to_cases: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    seen_case_ids: Set[str] = set()
    n_dropped = 0
    duplicate_case_ids: List[str] = []

    for entry in manifest_records:
        if not isinstance(entry, dict):
            continue

        tok = entry.get("token") or entry.get("patient_token") or entry.get("id")
        cfg = entry.get("config") or entry.get("match_type")

        if tok is None or cfg is None:
            n_dropped += 1
            if strict:
                raise RuntimeError(
                    f"Manifest record missing token and/or config: {entry}")
            continue
        tok = str(tok)
        cfg = str(cfg)

        # nnU-Net-style channel suffix accidentally in token -> strip it
        if tok.endswith("_0000"):
            tok = tok[:-5]

        # Trust the manifest's case_id if present; otherwise synthesize.
        case_id = entry.get("case_id") or entry.get("nnunet_case_id")
        if not case_id:
            case_id = _synth_case_id(tok, cfg, dataset_name, case_id_template)

        # Duplicate case IDs mean something is wrong upstream — collect
        # them all so the user can see every collision in one run.
        if case_id in seen_case_ids:
            duplicate_case_ids.append(case_id)
        seen_case_ids.add(case_id)

        token_to_cases[tok].append((case_id, cfg))

    if n_dropped:
        print(f"WARN: skipped {n_dropped} manifest records missing token/config",
              file=sys.stderr)
    if duplicate_case_ids:
        raise RuntimeError(
            f"Duplicate nnU-Net case IDs in manifest — training would be "
            f"corrupt: {sorted(set(duplicate_case_ids))[:10]}")

    return dict(token_to_cases)


# ── Split translation ───────────────────────────────────────────────────────

def expand_tokens(
    tokens:         List[str],
    token_to_cases: Dict[str, List[Tuple[str, str]]],
    label:          str,
) -> Tuple[List[str], Set[str]]:
    """Expand a list of patient tokens into (case_ids, unmapped_tokens).

    Unmapped tokens are tokens present in splits_5fold.json but absent
    from the HF manifest — typically because the case was excluded at
    export time. We return them as a set so the caller can decide
    whether to abort or warn (we default to abort if they're training
    or val tokens, since fold completeness matters).
    """
    out: List[str] = []
    unmapped: Set[str] = set()
    for t in tokens:
        cases = token_to_cases.get(t)
        if not cases:
            unmapped.add(t)
            continue
        for case_id, _cfg in cases:
            out.append(case_id)
    return sorted(out), unmapped


def translate(
    splits_doc:     dict,
    token_to_cases: Dict[str, List[Tuple[str, str]]],
    strict_unmapped: bool,
) -> Tuple[List[Dict[str, List[str]]], List[str], Dict]:
    """
    Returns:
      nnunet_folds      list[{"train": [...], "val": [...]}]
      test_case_ids     list[str]
      diagnostics       dict of per-fold expansion counts + warnings
    """
    test_tokens  = list(splits_doc.get("test_tokens", []))
    folds_input  = list(splits_doc.get("folds", []))

    if not folds_input:
        raise RuntimeError("splits file has no 'folds' list")

    # ── Test set (held out of all folds) ────────────────────────────────────
    test_case_ids, test_unmapped = expand_tokens(
        test_tokens, token_to_cases, "test")
    if test_unmapped and strict_unmapped:
        raise RuntimeError(
            f"{len(test_unmapped)} test tokens have no nnU-Net cases "
            f"in the HF manifest: {sorted(test_unmapped)[:10]}")
    elif test_unmapped:
        print(f"WARN: {len(test_unmapped)} test tokens unmapped (dropped): "
              f"{sorted(test_unmapped)[:10]}", file=sys.stderr)

    # ── Folds ───────────────────────────────────────────────────────────────
    nnunet_folds: List[Dict[str, List[str]]] = []
    per_fold_diag = []
    test_set = set(test_case_ids)

    for i, fold in enumerate(folds_input):
        tr_tokens = list(fold.get("train_tokens", []))
        va_tokens = list(fold.get("val_tokens",   []))

        tr_cases, tr_unmapped = expand_tokens(tr_tokens, token_to_cases, f"fold{i}/train")
        va_cases, va_unmapped = expand_tokens(va_tokens, token_to_cases, f"fold{i}/val")

        # Every token with mask files should have made it to the HF
        # manifest. Missing tokens in train/val are almost always a bug
        # upstream (export_hf.py dropped the case without updating the
        # splits file), so we abort unless explicitly told to tolerate it.
        if (tr_unmapped or va_unmapped):
            msg = (f"fold {i}: unmapped tokens — "
                   f"train={sorted(tr_unmapped)[:5]} (+{max(0,len(tr_unmapped)-5)}), "
                   f"val={sorted(va_unmapped)[:5]} (+{max(0,len(va_unmapped)-5)})")
            if strict_unmapped:
                raise RuntimeError(msg)
            else:
                print(f"WARN: {msg}", file=sys.stderr)

        # ── Anatomy-leak assertion (CRITICAL) ──────────────────────────
        # Every case of a given token must land on the same side of this
        # fold. If this fires, splits_5fold.json split at the case level
        # instead of the token level, or the manifest has duplicate
        # token entries under different configs pointing at the same
        # patient.
        tr_cases_set = set(tr_cases)
        va_cases_set = set(va_cases)

        # Build token -> sides-seen-in-this-fold map from the expansion.
        token_sides: Dict[str, Set[str]] = defaultdict(set)
        for t in tr_tokens:
            if t in token_to_cases:
                token_sides[t].add("train")
        for t in va_tokens:
            if t in token_to_cases:
                token_sides[t].add("val")
        leaking = {t: s for t, s in token_sides.items() if len(s) > 1}
        if leaking:
            raise RuntimeError(
                f"ANATOMY LEAK in fold {i}: {len(leaking)} tokens appear in "
                f"both train and val: {sorted(leaking)[:10]}. "
                f"Upstream splits are broken — refusing to write splits_final.json.")

        # ── Test-set contamination check ───────────────────────────────
        # No test case should appear in any fold's train or val.
        train_test_leak = tr_cases_set & test_set
        val_test_leak   = va_cases_set & test_set
        if train_test_leak or val_test_leak:
            raise RuntimeError(
                f"TEST-SET CONTAMINATION in fold {i}: "
                f"train∩test={sorted(train_test_leak)[:5]}  "
                f"val∩test={sorted(val_test_leak)[:5]}")

        # ── Intra-fold train/val disjointness ──────────────────────────
        tv_overlap = tr_cases_set & va_cases_set
        if tv_overlap:
            raise RuntimeError(
                f"fold {i} train∩val overlap: {sorted(tv_overlap)[:10]}")

        nnunet_folds.append({"train": tr_cases, "val": va_cases})
        per_fold_diag.append({
            "fold":            i,
            "n_tokens_train":  len(tr_tokens),
            "n_tokens_val":    len(va_tokens),
            "n_cases_train":   len(tr_cases),
            "n_cases_val":     len(va_cases),
            "n_unmapped_train": len(tr_unmapped),
            "n_unmapped_val":   len(va_unmapped),
        })

    diag = {
        "n_test_tokens":      len(test_tokens),
        "n_test_cases":       len(test_case_ids),
        "n_test_unmapped":    len(test_unmapped),
        "folds":              per_fold_diag,
    }
    return nnunet_folds, test_case_ids, diag


def verify_cases_exist_on_disk(
    nnunet_folds:   List[Dict[str, List[str]]],
    test_case_ids:  List[str],
    nnunet_raw_dir: Path,
    strict:         bool,
) -> None:
    """Verify every case referenced by the splits exists as a preprocessed
    case on disk. nnU-Net will fail later at dataloader time if a case is
    missing; catching it here is much easier to debug."""
    expected = set(test_case_ids)
    for f in nnunet_folds:
        expected.update(f["train"])
        expected.update(f["val"])

    # nnU-Net v2 preprocesses into nnUNet_preprocessed/Dataset<ID>_<name>/
    # nnUNetPlans_3d_fullres/ with files like CTSpinoPelvic_0142_fused.npz.
    # We look for any file matching the case stem in the raw imagesTr/ or
    # in any preprocessed subdirectory. If nnunet_raw_dir is the
    # nnUNet_raw Dataset<ID>_<name>/ directory, check imagesTr/.
    images_tr = nnunet_raw_dir / "imagesTr"
    search_dirs = []
    if images_tr.is_dir():
        search_dirs.append(images_tr)
    # Also look directly inside the directory in case the user passed the
    # preprocessed Dataset dir.
    for sub in nnunet_raw_dir.iterdir() if nnunet_raw_dir.is_dir() else []:
        if sub.is_dir() and sub.name.startswith("nnUNetPlans"):
            search_dirs.append(sub)

    if not search_dirs:
        print(f"INFO: --nnunet_raw_dir={nnunet_raw_dir} has no imagesTr/ or "
              f"nnUNetPlans* subdirs; skipping on-disk verification.",
              file=sys.stderr)
        return

    found: Set[str] = set()
    for d in search_dirs:
        for p in d.iterdir():
            name = p.name
            # imagesTr: CTSpinoPelvic_0142_fused_0000.nii.gz
            # preprocessed: CTSpinoPelvic_0142_fused.npz
            for suffix in ("_0000.nii.gz", ".nii.gz", ".npz", ".pkl"):
                if name.endswith(suffix):
                    stem = name[:-len(suffix)]
                    found.add(stem)
                    break

    missing = expected - found
    if missing:
        msg = (f"{len(missing)} case IDs referenced by splits but not found "
               f"in {nnunet_raw_dir}: {sorted(missing)[:10]}")
        if strict:
            raise RuntimeError(msg)
        print(f"WARN: {msg}", file=sys.stderr)
    else:
        print(f"  ✓ all {len(expected)} case IDs verified on disk")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--splits", type=Path, required=True,
                    help="Path to splits_5fold.json (schema v3+).")
    ap.add_argument("--hf_manifest", type=Path, required=True,
                    help="HF export manifest.json (maps tokens to case IDs).")
    ap.add_argument("--nnunet_raw_dir", type=Path, default=None,
                    help="nnUNet_raw/Dataset<ID>_<name>/ directory. When provided, "
                         "splits_final.json is written here and case existence "
                         "is verified against imagesTr/.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Explicit output path for splits_final.json. "
                         "Defaults to <nnunet_raw_dir>/splits_final.json.")
    ap.add_argument("--test_cases_out", type=Path, default=None,
                    help="Where to write the test_cases.json sidecar. "
                         "Defaults to <out>.parent/test_cases.json.")
    ap.add_argument("--dataset_name", default="CTSpinoPelvic",
                    help="Dataset name used in synthesized case IDs.")
    ap.add_argument("--case_id_template",
                    default="{dataset_name}_{token_int:04d}_{config}",
                    help="Template for synthesizing case IDs when the HF "
                         "manifest lacks a `case_id` field. Supports "
                         "{dataset_name}, {token}, {token_int}, {config}.")
    ap.add_argument("--allow_unmapped", action="store_true",
                    help="Warn instead of abort when tokens in splits_5fold.json "
                         "are absent from the HF manifest.")
    ap.add_argument("--skip_disk_verify", action="store_true",
                    help="Skip verifying case IDs exist on disk under "
                         "--nnunet_raw_dir.")
    args = ap.parse_args()

    if args.out is None:
        if args.nnunet_raw_dir is None:
            print("ERROR: must provide either --out or --nnunet_raw_dir",
                  file=sys.stderr)
            return 1
        args.out = args.nnunet_raw_dir / "splits_final.json"
    if args.test_cases_out is None:
        args.test_cases_out = args.out.parent / "test_cases.json"

    # ── Load inputs ─────────────────────────────────────────────────────────
    splits_doc = _load_json(args.splits)
    if splits_doc.get("schema_version", 0) < 3:
        print(f"WARN: splits schema_version={splits_doc.get('schema_version')} "
              f"(expected >=3); field names may differ.", file=sys.stderr)

    manifest_raw = _load_json(args.hf_manifest)
    manifest_records = _unwrap_manifest(manifest_raw)
    print(f"HF manifest:  {len(manifest_records)} records")

    # ── Build token -> case_id mapping ──────────────────────────────────────
    token_to_cases = build_token_to_cases(
        manifest_records,
        dataset_name=args.dataset_name,
        case_id_template=args.case_id_template,
        strict=not args.allow_unmapped,
    )
    config_dist = Counter(
        cfg for cases in token_to_cases.values() for _cid, cfg in cases
    )
    n_expanded = sum(len(v) for v in token_to_cases.values())
    print(f"  unique tokens in manifest : {len(token_to_cases)}")
    print(f"  total expanded cases      : {n_expanded}")
    print(f"  config distribution       : {dict(config_dist)}")

    # ── Translate ───────────────────────────────────────────────────────────
    nnunet_folds, test_case_ids, diag = translate(
        splits_doc, token_to_cases,
        strict_unmapped=not args.allow_unmapped,
    )

    print("\nFold expansion:")
    for d in diag["folds"]:
        print(f"  fold {d['fold']}:  "
              f"train_tokens={d['n_tokens_train']:>3} -> {d['n_cases_train']:>3} cases   "
              f"val_tokens={d['n_tokens_val']:>3} -> {d['n_cases_val']:>3} cases")
    print(f"  test_tokens={diag['n_test_tokens']} -> "
          f"{diag['n_test_cases']} cases")

    # ── On-disk verification (optional) ─────────────────────────────────────
    if args.nnunet_raw_dir and not args.skip_disk_verify:
        verify_cases_exist_on_disk(
            nnunet_folds, test_case_ids, args.nnunet_raw_dir,
            strict=False,  # warn-only; nnU-Net will error later if needed
        )

    # ── Write outputs ───────────────────────────────────────────────────────
    _atomic_write_json(args.out, nnunet_folds)
    print(f"\nWrote nnU-Net splits: {args.out}")

    test_doc = {
        "note": ("Test cases held out of all training folds. "
                 "Pass these through the trained model for final evaluation; "
                 "do NOT let them leak into any fold's train or val."),
        "source_splits":  str(args.splits.resolve()),
        "source_manifest": str(args.hf_manifest.resolve()),
        "n_test_cases":   len(test_case_ids),
        "test_cases":     test_case_ids,
    }
    _atomic_write_json(args.test_cases_out, test_doc)
    print(f"Wrote test cases:     {args.test_cases_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
