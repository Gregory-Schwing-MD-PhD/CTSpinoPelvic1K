"""
gather_lstv_review.py — bundle LSTV cases (CT + label) into a flat folder for
local ITK-SNAP review. Pure file ops; no nibabel needed.

For each selected case it places the pair under
  <out>/<token>__<config>/{ct.nii.gz, label.nii.gz}
(symlink by default — no disk duplication; --copy to materialize), writes
review_index.csv, and prints an rsync command to pull the bundle to your laptop.

Open in ITK-SNAP locally: load ct.nii.gz as the main image, then
"Segmentation > Open Segmentation" -> label.nii.gz.

Selection (default: non-normal patient_subtypes from splits_5fold.json — the
LSTV set, ~50 cases):
  --splits PATH   non-normal subtypes (default <hf_dir>/splits_5fold.json)
  --has_l6        cases whose manifest has_l6 is True (the L6/lumbarization set)
  --tokens PATH   explicit token list (one per line, or a CSV with a 'token' col)

Run AFTER refresh_lstv_v3.sh + re-split so the selection reflects the corrected
labels (otherwise stale has_l6 / subtypes would miss the newly-found L6 cases).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set


def _load_records(manifest_path: Path) -> List[dict]:
    data = json.loads(manifest_path.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _tokens_from_splits(splits_path: Path) -> Set[str]:
    s = json.loads(splits_path.read_text())
    subs: Dict[str, str] = s.get("patient_subtypes", {})
    return {tok for tok, sub in subs.items() if sub and sub != "normal"}


def _tokens_from_file(p: Path) -> Set[str]:
    text = p.read_text()
    if "," in text.splitlines()[0] if text.splitlines() else False:
        rows = list(csv.DictReader(p.open(newline="")))
        if rows and "token" in rows[0]:
            return {str(r["token"]).strip() for r in rows if r.get("token")}
    return {ln.strip() for ln in text.splitlines() if ln.strip()
            and not ln.startswith("#")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_dir", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default <hf_dir>/manifest.json")
    ap.add_argument("--splits", type=Path, default=None,
                    help="default <hf_dir>/splits_5fold.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <hf_dir>/_lstv_review")
    ap.add_argument("--has_l6", action="store_true",
                    help="select manifest has_l6==True instead of non-normal subtypes")
    ap.add_argument("--tokens", type=Path, default=None,
                    help="explicit token list (overrides --splits/--has_l6)")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (default: symlink)")
    args = ap.parse_args()

    manifest_path = args.manifest or (args.hf_dir / "manifest.json")
    out = args.out or (args.hf_dir / "_lstv_review")
    records = _load_records(manifest_path)
    subs_map: Dict[str, str] = {}

    # ── resolve the token selection ──────────────────────────────────────────
    if args.tokens:
        selected = _tokens_from_file(args.tokens)
        how = f"token list {args.tokens}"
    elif args.has_l6:
        selected = {str(r.get("token")) for r in records if r.get("has_l6")}
        how = "manifest has_l6==True"
    else:
        splits_path = args.splits or (args.hf_dir / "splits_5fold.json")
        if not splits_path.exists():
            raise SystemExit(f"no splits at {splits_path}; pass --splits, "
                             f"--tokens, or --has_l6")
        selected = _tokens_from_splits(splits_path)
        subs_map = json.loads(splits_path.read_text()).get("patient_subtypes", {})
        how = f"non-normal subtypes in {splits_path.name}"

    print(f"selection: {how} -> {len(selected)} token(s)")
    out.mkdir(parents=True, exist_ok=True)

    index: List[dict] = []
    n_pairs = n_missing = 0
    for rec in records:
        tok = str(rec.get("token") or rec.get("patient_token") or "")
        if tok not in selected:
            continue
        cfg = str(rec.get("config", "")) or "na"
        ct_rel = rec.get("ct_file") or rec.get("ct")
        lb_rel = rec.get("label_file") or rec.get("label")
        if not ct_rel or not lb_rel:
            continue
        ct_src = (args.hf_dir / ct_rel).resolve()
        lb_src = (args.hf_dir / lb_rel).resolve()
        if not ct_src.exists() or not lb_src.exists():
            print(f"  WARN missing files for {tok}__{cfg}")
            n_missing += 1
            continue
        dst_dir = out / f"{tok}__{cfg}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src, name in ((ct_src, "ct.nii.gz"), (lb_src, "label.nii.gz")):
            dst = dst_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
        index.append({"token": tok, "config": cfg,
                      "subtype": subs_map.get(tok, ""),
                      "has_l6": rec.get("has_l6", ""),
                      "ct_src": str(ct_src), "label_src": str(lb_src)})
        n_pairs += 1

    idx_path = out / "review_index.csv"
    with idx_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "config", "subtype",
                                          "has_l6", "ct_src", "label_src"])
        w.writeheader(); w.writerows(index)

    print(f"bundled {n_pairs} case(s) ({'copied' if args.copy else 'symlinked'}) "
          f"into {out}  (missing: {n_missing})")
    print(f"index: {idx_path}")
    print("\nPull to your laptop (follow symlinks with -L), then open in ITK-SNAP:")
    print(f"  rsync -avL <user>@<host>:{out.resolve()}/ ./lstv_review/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
