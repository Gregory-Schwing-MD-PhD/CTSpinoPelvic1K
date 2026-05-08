"""
prune_sample_repo.py — Prune an existing HF dataset repo down to N cases.

Use case: anonymous-neurips-ED/CTSpinoPelvic1K-Sample is currently a full
copy of the parent dataset. We want it small (~10 cases). Rather than
re-uploading from scratch, this script:

  1. Pulls metadata from the Sample repo (manifest.json + splits_5fold.json).
  2. Stratified-samples N records by lstv_class.
  3. Lists every file under ct/ and labels/ on the remote.
  4. Computes the delete set = (remote ct/labels files) - (sampled ct/labels).
  5. Issues ONE atomic create_commit that:
       - deletes the non-sampled NIfTIs
       - overwrites manifest.json + manifest.csv with the filtered subset
       - overwrites splits_5fold.json with surviving tokens
       - filters legacy split files in place (data_splits.json, splits/test.json,
         splits/cv_5fold.json) — these may be load-bearing for the HF dataset
         viewer's split-filter UI, so we rewrite rather than delete
       - regenerates splits_summary.json so stats reflect the sample
       - writes SAMPLE_NOTES.md describing the sampling

Atomic in the sense that the HF Hub commit lands all operations together
or none of them. No mid-prune inconsistent state.

Usage:
  HF_TOKEN=hf_xxx python prune_sample_repo.py \
      --repo anonymous-neurips-ED/CTSpinoPelvic1K-Sample \
      --n    10 \
      --dry_run

  HF_TOKEN=hf_xxx python prune_sample_repo.py \
      --repo anonymous-neurips-ED/CTSpinoPelvic1K-Sample \
      --n    10 \
      --yes
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

log = logging.getLogger("prune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# -- Stratified sampler -------------------------------------------------------

def stratified_sample(records: List[dict], n: int, seed: int = 42) -> List[dict]:
    """Sample n records stratified by lstv_class with ≥1 per non-empty class."""
    rng = random.Random(seed)
    by_class: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_class[int(r.get("lstv_class", 0) or 0)].append(r)
    for cls in by_class:
        rng.shuffle(by_class[cls])

    total = len(records)
    if n >= total:
        return list(records)

    alloc: Dict[int, int] = {
        cls: max(1, (len(g) * n) // total) if g else 0
        for cls, g in by_class.items()
    }
    while sum(alloc.values()) > n:
        cand = max((c for c in alloc if alloc[c] > 1),
                   key=lambda c: alloc[c], default=None)
        if cand is None:
            break
        alloc[cand] -= 1
    while sum(alloc.values()) < n:
        cand = max(by_class, key=lambda c: len(by_class[c]) - alloc[c])
        alloc[cand] += 1

    sampled: List[dict] = []
    for cls, k in alloc.items():
        sampled.extend(by_class[cls][:min(k, len(by_class[cls]))])
    return sampled


# -- Splits filter ------------------------------------------------------------

def filter_splits_5fold(doc: dict, sampled_tokens: Set[str], repo: str,
                         n_sampled: int) -> dict:
    out = dict(doc)
    if "test_tokens" in out:
        out["test_tokens"] = [t for t in out["test_tokens"]
                              if str(t) in sampled_tokens]
    if "folds" in out:
        new_folds = []
        for fold in out["folds"]:
            nf = dict(fold)
            for key in ("train_tokens", "val_tokens"):
                if key in nf:
                    nf[key] = [t for t in nf[key] if str(t) in sampled_tokens]
            new_folds.append(nf)
        out["folds"] = new_folds
    out["pruned_from"] = repo
    out["n_sampled"]   = n_sampled
    return out


def filter_data_splits(doc: dict, kept_ct_files: Set[str]) -> dict:
    """Filter data_splits.json (the earliest format: ct_file lists per split).

    The HF dataset viewer may key its split-filter UI off this file, so we
    rewrite rather than delete. Entries pointing to deleted CT files are
    dropped from each side.
    """
    out: Dict[str, list] = {}
    for side in ("train", "val", "test"):
        out[side] = [f for f in (doc.get(side) or []) if f in kept_ct_files]
    return out


def filter_test_tokens_list(tokens: list, sampled_tokens: Set[str]) -> list:
    """Filter splits/test.json (flat list of unique test patient tokens)."""
    return [t for t in tokens if str(t) in sampled_tokens]


def filter_cv_5fold_legacy(doc: dict, sampled_tokens: Set[str]) -> dict:
    """Filter splits/cv_5fold.json (legacy 5-fold pre-unification)."""
    out = dict(doc)
    if "folds" in out:
        new_folds = []
        for fold in out["folds"]:
            nf = dict(fold)
            for key in ("train_tokens", "val_tokens"):
                if key in nf:
                    nf[key] = [t for t in nf[key] if str(t) in sampled_tokens]
            new_folds.append(nf)
        out["folds"] = new_folds
    return out


def regenerate_splits_summary(sampled: List[dict], n_folds: int,
                               n_test: int, seed: int) -> dict:
    """Build a fresh splits_summary.json reflecting the sampled subset."""
    return {
        "seed": seed,
        "pruned_to_sample": True,
        "n_records": {
            "sampled":  len(sampled),
            "test":     n_test,
            "trainval": len(sampled) - n_test,
        },
        "n_tokens": {
            "sampled": len({str(r["token"]) for r in sampled}),
        },
        "n_folds": n_folds,
        "note": "This is a sample subset; counts do not reflect the parent dataset.",
    }


# -- Metadata fetch ----------------------------------------------------------

def fetch_metadata(repo: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=repo, repo_type="dataset", token=token,
        allow_patterns=[
            "manifest.json", "manifest.csv",
            "splits_5fold.json", "splits/**", "data_splits.json",
            "splits_summary.json", "README.md", "dataset_interface.py",
        ],
    ))


def list_remote_tree(repo: str, token: str | None) -> Dict[str, int]:
    """Return {path: size_bytes} for every file in the repo. Folders are
    skipped. Used to estimate the post-prune dataset size up front."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    tree = api.list_repo_tree(repo_id=repo, repo_type="dataset",
                              recursive=True)
    return {item.path: int(item.size) for item in tree
            if hasattr(item, "size")}


# -- Main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo",
                    default="anonymous-neurips-ED/CTSpinoPelvic1K-Sample")
    ap.add_argument("--n",       type=int, default=10)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--token",   default=None,
                    help="HF token; falls back to HF_TOKEN env var")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan and exit without committing")
    ap.add_argument("--yes",     action="store_true",
                    help="Skip the interactive confirmation prompt")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        log.error("HF_TOKEN required (set env var or --token), or use --dry_run")
        sys.exit(2)

    try:
        from huggingface_hub import (CommitOperationAdd, CommitOperationDelete,
                                      HfApi)
    except ImportError:
        log.error("pip install 'huggingface_hub>=0.20'")
        sys.exit(2)

    # ── 1. Pull metadata ─────────────────────────────────────────────────
    log.info("Fetching metadata from %s ...", args.repo)
    meta_dir = fetch_metadata(args.repo, token)
    raw = json.loads((meta_dir / "manifest.json").read_text())
    records = raw.get("records", []) if isinstance(raw, dict) else raw
    log.info("  %d records in current manifest", len(records))

    # ── 2. Sample ────────────────────────────────────────────────────────
    sampled = stratified_sample(records, args.n, seed=args.seed)
    sampled_tokens = {str(r["token"]) for r in sampled}
    cls_dist = Counter(int(r.get("lstv_class", 0) or 0) for r in sampled)
    cfg_dist = Counter(r.get("config", "") for r in sampled)
    log.info("Sampled %d records  lstv_class=%s  config=%s",
             len(sampled), dict(cls_dist), dict(cfg_dist))

    # ── 3. Compute keep / delete sets ────────────────────────────────────
    keep_files: Set[str] = set()
    for r in sampled:
        for key in ("ct_file", "label_file"):
            v = r.get(key, "") or ""
            if v:
                keep_files.add(v)

    log.info("Listing remote files ...")
    sizes = list_remote_tree(args.repo, token)
    remote_files = list(sizes.keys())
    remote_data = {f for f in remote_files
                   if f.startswith("ct/") or f.startswith("labels/")}
    delete_files = sorted(remote_data - keep_files)
    keep_size_b   = sum(sizes.get(f, 0) for f in keep_files)
    delete_size_b = sum(sizes.get(f, 0) for f in delete_files)
    log.info("  remote ct+labels: %d  (%.2f GB total)",
             len(remote_data),
             (keep_size_b + delete_size_b) / 1e9)
    log.info("  to keep:          %d  (%.2f GB)",
             len(keep_files), keep_size_b / 1e9)
    log.info("  to delete:        %d  (%.2f GB)",
             len(delete_files), delete_size_b / 1e9)

    if keep_size_b > 4 * 1024**3:
        log.warning("Post-prune size %.2f GiB exceeds 4 GiB. "
                    "Reduce --n if a 4 GiB cap matters for the venue.",
                    keep_size_b / 1024**3)

    missing = sorted(keep_files - remote_data)
    if missing:
        log.error("Sampled %d files are NOT on the remote — aborting:",
                  len(missing))
        for m in missing[:10]:
            log.error("    %s", m)
        sys.exit(3)

    # ── 4. Build the new metadata blobs ──────────────────────────────────
    new_manifest = json.dumps(sampled, indent=2).encode("utf-8")

    # Filter manifest.csv if present.
    new_manifest_csv: bytes | None = None
    src_csv = meta_dir / "manifest.csv"
    if src_csv.exists():
        # Filter by (token, config) tuple, not token alone — separate-mode
        # tokens have a spine_only row AND a pelvic_native row, and the
        # sampler operates at record granularity. A token-only filter
        # would let an unsampled partner row through (and that partner's
        # NIfTIs are not in keep_files, so the CSV would point to files
        # that the same commit deletes).
        sampled_keys = {(str(r.get("token", "")), str(r.get("config", "")))
                        for r in sampled}
        with open(src_csv) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = [row for row in reader
                    if (str(row.get("token", "")),
                        str(row.get("config", ""))) in sampled_keys]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        new_manifest_csv = buf.getvalue().encode("utf-8")
        log.info("  manifest.csv: %d -> %d rows",
                 sum(1 for _ in open(src_csv)) - 1, len(rows))

    # Filter splits_5fold.json if present.
    new_splits_5fold: bytes | None = None
    src_splits = meta_dir / "splits_5fold.json"
    if src_splits.exists():
        filtered = filter_splits_5fold(
            json.loads(src_splits.read_text()),
            sampled_tokens, args.repo, len(sampled),
        )
        new_splits_5fold = json.dumps(filtered, indent=2).encode("utf-8")

    # Filter data_splits.json (legacy; HF viewer may use it for the
    # split-filter UI).
    new_data_splits: bytes | None = None
    src_data_splits = meta_dir / "data_splits.json"
    if src_data_splits.exists():
        kept_ct = {r.get("ct_file", "") for r in sampled if r.get("ct_file")}
        filtered_ds = filter_data_splits(
            json.loads(src_data_splits.read_text()), kept_ct,
        )
        new_data_splits = json.dumps(filtered_ds, indent=2).encode("utf-8")
        log.info("  data_splits.json filtered: train=%d val=%d test=%d",
                 len(filtered_ds["train"]), len(filtered_ds["val"]),
                 len(filtered_ds["test"]))

    # Filter splits/test.json (legacy; flat token list).
    new_splits_test: bytes | None = None
    src_splits_test = meta_dir / "splits" / "test.json"
    if src_splits_test.exists():
        filtered_test = filter_test_tokens_list(
            json.loads(src_splits_test.read_text()), sampled_tokens,
        )
        new_splits_test = json.dumps(filtered_test, indent=2).encode("utf-8")
        log.info("  splits/test.json filtered: %d tokens", len(filtered_test))

    # Filter splits/cv_5fold.json (legacy; pre-unification 5-fold doc).
    new_cv_5fold: bytes | None = None
    src_cv_5fold = meta_dir / "splits" / "cv_5fold.json"
    if src_cv_5fold.exists():
        filtered_cv = filter_cv_5fold_legacy(
            json.loads(src_cv_5fold.read_text()), sampled_tokens,
        )
        new_cv_5fold = json.dumps(filtered_cv, indent=2).encode("utf-8")

    # Regenerate splits_summary.json from the sample.
    n_test_records = sum(
        1 for r in sampled
        if str(r.get("token", "")) in sampled_tokens and (
            # If splits_5fold is present, use its test_tokens; else fall
            # back to the earlier per-record `split` field if any.
            (new_splits_5fold and str(r["token"]) in
             set(json.loads(new_splits_5fold).get("test_tokens", [])))
            or r.get("split") == "test"
        )
    )
    new_splits_summary = json.dumps(
        regenerate_splits_summary(
            sampled,
            n_folds=len(json.loads(new_splits_5fold).get("folds", []))
                    if new_splits_5fold else 0,
            n_test=n_test_records,
            seed=args.seed,
        ),
        indent=2,
    ).encode("utf-8")

    notes = (
        f"# CTSpinoPelvic1K Sample\n\n"
        f"This is a {len(sampled)}-case stratified sample of the parent "
        f"CTSpinoPelvic1K dataset, intended for interface inspection and "
        f"reproducibility checks. It is NOT large enough for any "
        f"segmentation training or meaningful evaluation.\n\n"
        f"## Sampling provenance\n\n"
        f"- Pruned from: `{args.repo}` (in place)\n"
        f"- Random seed: `{args.seed}`\n"
        f"- Strategy: stratified by `lstv_class` "
        f"(≥1 case per non-empty class)\n"
        f"- Records kept: `{len(sampled)}` of `{len(records)}`\n"
        f"- `lstv_class` distribution: `{dict(cls_dist)}`\n"
        f"- `config` distribution: `{dict(cfg_dist)}`\n\n"
        f"## Splits caveat\n\n"
        f"`splits_5fold.json` is preserved for surviving tokens but folds "
        f"may be sparse or empty. Do not run cross-validation on the "
        f"sample; use the parent dataset for that.\n"
    ).encode("utf-8")

    # ── 5. Print plan, optionally exit ───────────────────────────────────
    log.info("Plan summary:")
    log.info("  delete:  %d data files (ct/ + labels/)", len(delete_files))
    log.info("  upload:")
    log.info("    manifest.json        (%d B)", len(new_manifest))
    if new_manifest_csv is not None:
        log.info("    manifest.csv         (%d B)", len(new_manifest_csv))
    if new_splits_5fold is not None:
        log.info("    splits_5fold.json    (%d B)", len(new_splits_5fold))
    if new_data_splits is not None:
        log.info("    data_splits.json     (%d B)", len(new_data_splits))
    if new_splits_test is not None:
        log.info("    splits/test.json     (%d B)", len(new_splits_test))
    if new_cv_5fold is not None:
        log.info("    splits/cv_5fold.json (%d B)", len(new_cv_5fold))
    log.info("    splits_summary.json  (%d B)", len(new_splits_summary))
    log.info("    SAMPLE_NOTES.md      (%d B)", len(notes))

    if args.dry_run:
        log.info("--dry_run set; exiting without committing.")
        log.info("Sampled tokens: %s", sorted(sampled_tokens))
        return

    if not args.yes:
        if not sys.stdin.isatty():
            log.error("Non-interactive shell. Re-run with --yes to confirm.")
            sys.exit(2)
        log.warning("=" * 60)
        log.warning("ABOUT TO PRUNE %s", args.repo)
        log.warning("This will delete %d files. IRREVERSIBLE.",
                    len(delete_files))
        log.warning("=" * 60)
        ans = input(f"Type the repo name to confirm: ").strip()
        if ans != args.repo:
            log.error("Aborted: typed '%s', expected '%s'", ans, args.repo)
            sys.exit(2)

    # ── 6. Atomic commit ─────────────────────────────────────────────────
    operations = []
    for f in delete_files:
        operations.append(CommitOperationDelete(path_in_repo=f))

    operations.append(CommitOperationAdd(
        path_in_repo="manifest.json", path_or_fileobj=new_manifest))
    if new_manifest_csv is not None:
        operations.append(CommitOperationAdd(
            path_in_repo="manifest.csv", path_or_fileobj=new_manifest_csv))
    if new_splits_5fold is not None:
        operations.append(CommitOperationAdd(
            path_in_repo="splits_5fold.json", path_or_fileobj=new_splits_5fold))
    if new_data_splits is not None:
        operations.append(CommitOperationAdd(
            path_in_repo="data_splits.json", path_or_fileobj=new_data_splits))
    if new_splits_test is not None:
        operations.append(CommitOperationAdd(
            path_in_repo="splits/test.json", path_or_fileobj=new_splits_test))
    if new_cv_5fold is not None:
        operations.append(CommitOperationAdd(
            path_in_repo="splits/cv_5fold.json", path_or_fileobj=new_cv_5fold))
    operations.append(CommitOperationAdd(
        path_in_repo="splits_summary.json", path_or_fileobj=new_splits_summary))
    operations.append(CommitOperationAdd(
        path_in_repo="SAMPLE_NOTES.md", path_or_fileobj=notes))

    log.info("Committing %d operations to %s ...", len(operations), args.repo)
    api = HfApi(token=token)
    api.create_commit(
        repo_id=args.repo, repo_type="dataset",
        operations=operations,
        commit_message=f"Prune to {len(sampled)}-case sample (seed={args.seed})",
    )
    log.info("Done -> https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
