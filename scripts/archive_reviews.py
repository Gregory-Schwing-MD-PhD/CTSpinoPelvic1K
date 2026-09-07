"""scripts/archive_reviews.py — the three student-review ledgers, pseudonymised, for the build archive.

The ledgers (reviews-ribs, reviews-spine, reviews-triaged on Hugging Face) are the student
corrections: for every reviewed case a JSON record of who claimed which slot and what they
decided, plus the corrected label volume per slot and the finalised label the merge used.
They are the one input to v5 that no script can regenerate, so they go in the archive.

Reviewer handles are personal Hugging Face usernames. They are replaced throughout by
annotator_01 .. annotator_NN (the lead becomes "lead"), consistently across all three
ledgers, and the mapping is written OUTSIDE the archive so it is never deposited.

    ~/mambaforge/envs/hfup/bin/python scripts/archive_reviews.py \
        --out ~/build_archive_stage/04_reviews --private-map ~/build_archive_private/annotator_map.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPOS = {
    "reviews_ribs": "anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs",
    "reviews_spine": "anonymous-mlhc/CTSpinoPelvic1K-reviews-spine",
    "reviews_triaged": "gregoryschwingmdphd/CTSpinoPelvic1K-reviews-triaged",
}
LEAD = "gregoryschwingmdphd"
TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".yaml", ".yml"}


def _tokens() -> list[str]:
    out = []
    for p in ("~/.hf_org_token", "~/.cache/huggingface/token", "~/.hf_osc_token"):
        f = Path(os.path.expanduser(p))
        if f.exists():
            t = f.read_text().strip()
            if t and t not in out:
                out.append(t)
    if os.environ.get("HF_TOKEN"):
        out.insert(0, os.environ["HF_TOKEN"])
    return out


def _download(repo: str, dst: Path) -> None:
    from huggingface_hub import snapshot_download

    last = None
    for tok in _tokens():
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(dst), token=tok)
            return
        except Exception as e:  # noqa: BLE001 — try the next token
            last = e
    raise SystemExit(f"could not download {repo}: {last}")


def _collect_handles(root: Path) -> set[str]:
    handles: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "reviewer" and isinstance(v, str) and v:
                    handles.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for p in root.rglob("*.json"):
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — not every json is a ledger record
            pass
    return handles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--private-map", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name, repo in REPOS.items():
        print(f"  downloading {repo}", flush=True)
        _download(repo, out / name)
        n = sum(1 for _ in (out / name).rglob("*") if _.is_file())
        print(f"    {n} files", flush=True)
        for junk in (out / name / ".cache", out / name / ".git"):
            if junk.exists():
                shutil.rmtree(junk)

    handles = _collect_handles(out)
    handles.discard(LEAD)
    mapping = {LEAD: "lead"}
    for i, h in enumerate(sorted(handles), 1):
        mapping[h] = f"annotator_{i:02d}"
    print(f"  {len(mapping) - 1} annotator handles pseudonymised")

    # longest handle first so one handle that is a prefix of another cannot half-match
    order = sorted(mapping, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(h) for h in order))
    n_files = 0
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        s = p.read_text(encoding="utf-8")
        t = pat.sub(lambda m: mapping[m.group(0)], s)
        if t != s:
            p.write_text(t, encoding="utf-8")
            n_files += 1
    # and any path that carries a handle
    for p in sorted(out.rglob("*"), key=lambda q: len(str(q)), reverse=True):
        new = pat.sub(lambda m: mapping[m.group(0)], p.name)
        if new != p.name:
            p.rename(p.with_name(new))
    print(f"  {n_files} text files rewritten")

    leftover = [str(p) for p in out.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
                and pat.search(p.read_text(encoding="utf-8"))]
    if leftover:
        print("  ! handles still present in:", leftover[:5])
        return 1

    pm = Path(a.private_map)
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    os.chmod(pm, 0o600)
    print(f"  private mapping -> {pm} (NOT part of the archive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
