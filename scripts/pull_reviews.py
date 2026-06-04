"""pull_reviews.py — pull finalized reviews from the REVIEW_REPO and build the
reduce_to_v3 inputs + a status report.

Downloads cases/ and reviews/ from the private review ledger dataset, computes
each case's status from the double-review + adjudication state machine, and writes
  <out>/review_repo/    the downloaded cases/ + reviews/ (labels_root for reduce)
  <out>/finals.json     {case_id: case['final']} for FINALIZED + EXCLUDED cases
then prints a status summary (finalized / needs_adjudication / in_review / ...),
so you know the adjudication backlog before reducing to v3.

  REVIEW_REPO=org/CTSpinoPelvic1K-reviews-triaged HF_TOKEN=hf_xxx \
      python scripts/pull_reviews.py --out data/reviews_pull

build_finals() is a pure function over the case dicts (unit-tested-friendly).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "review"))
import schema  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.pull_reviews")

_REPORT_ORDER = ("finalized", "needs_adjudication", "in_review",
                 "unassigned", "excluded")


def build_finals(cases: List[dict]) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """From the case dicts, return (finals_index, status_counts).

    finals_index maps case_id -> case['final'] for every case that has a final
    (finalized or excluded). status_counts is derive_status() over all cases."""
    finals: Dict[str, dict] = {}
    counts: Dict[str, int] = {}
    for case in cases:
        st = schema.derive_status(case)
        counts[st] = counts.get(st, 0) + 1
        if case.get("final"):
            finals[str(case.get("case_id"))] = case["final"]
    return finals, counts


def _load_cases(cases_dir: Path) -> List[dict]:
    out = []
    for cf in sorted(cases_dir.glob("*.json")):
        try:
            out.append(json.loads(cf.read_text()))
        except Exception as exc:                       # noqa: BLE001
            log.warning("skipping %s (%s)", cf.name, exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("REVIEW_REPO"),
                    help="review ledger dataset (or REVIEW_REPO env)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token with read access (or HF_TOKEN env)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir: writes review_repo/ + finals.json")
    args = ap.parse_args()
    if not args.repo:
        sys.exit("need --repo or REVIEW_REPO env (the review ledger dataset).")

    from huggingface_hub import snapshot_download
    local = args.out / "review_repo"
    local.mkdir(parents=True, exist_ok=True)
    log.info("pulling cases/ + reviews/ from %s ...", args.repo)
    snapshot_download(repo_id=args.repo, repo_type="dataset", token=args.token,
                      local_dir=str(local), allow_patterns=["cases/*", "reviews/*"])

    cases_dir = local / "cases"
    if not cases_dir.exists():
        sys.exit(f"no cases/ in {args.repo} — nothing to pull "
                 "(is REVIEW_REPO the ledger, not the dataset?).")
    cases = _load_cases(cases_dir)
    finals, counts = build_finals(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "finals.json").write_text(json.dumps(finals, indent=2))

    total = sum(counts.values())
    log.info("=" * 60)
    log.info("REVIEW STATUS  (%d cases in %s)", total, args.repo)
    for st in _REPORT_ORDER:
        if counts.get(st):
            log.info("  %-18s %d", st, counts[st])
    for st, n in counts.items():                        # anything unexpected
        if st not in _REPORT_ORDER:
            log.info("  %-18s %d", st, n)
    log.info("  -> finals.json: %d finalized/excluded case(s)", len(finals))
    na = counts.get("needs_adjudication", 0)
    if na:
        log.info("")
        log.info("  %d case(s) NEED ADJUDICATION before they're final.", na)
        log.info("  Resolve (senior, interactive): python -m reviewtool adjudicate")
        log.info("  then re-run pull-reviews to pick up the resolutions.")
    log.info("  NEXT: make reduce-v3   (build the corrected v3 tree)")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
