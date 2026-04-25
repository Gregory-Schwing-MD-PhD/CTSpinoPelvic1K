#!/usr/bin/env python3
"""
merge_benchmark_shards.py — Merge sharded TotalSegmentator benchmark logs.

Reads every per-shard JSONL log under <results_base>/shard_*/per_case_partial.jsonl,
deduplicates by (token, config) keeping the latest record, and runs the
existing aggregation + table formatting from benchmark_totalseg.py to
produce a unified set of outputs in <out_dir> (default:
<results_base>/_merged/).

Outputs (under <out_dir>):
  per_case_partial.jsonl     deduplicated combined log (latest wins)
  benchmark_results.json     full per-case + summary
  benchmark_summary.json     aggregated subgroups only
  paper_tables.txt           Table 5 (Dice) + Table 6 (surface metrics)
  benchmark_per_case.csv     per-case CSV with all metrics

This script does NOT re-run TS inference or metrics. It just aggregates.

Usage
-----
  # Default: walk results/totalseg_bench/shard_*/ and write _merged/
  python scripts/merge_benchmark_shards.py

  # Custom paths:
  python scripts/merge_benchmark_shards.py \\
      --results_base $HOME/CTSpinoPelvic1K/results/totalseg_bench \\
      --out_dir      $HOME/CTSpinoPelvic1K/results/totalseg_bench/_merged

  # Limit which shards to merge (e.g. only ones that finished cleanly):
  python scripts/merge_benchmark_shards.py \\
      --shard_glob 'shard_[0-3]_of_8'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ts_merge")


def _import_benchmark_module():
    """
    Import benchmark_totalseg.py from sibling scripts/ dir so we can
    reuse its aggregation + formatting helpers without copy-paste.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import benchmark_totalseg as bt  # type: ignore
    return bt


def collect_shard_logs(results_base: Path, shard_glob: str) -> List[Path]:
    matches = sorted(results_base.glob(shard_glob))
    logs = []
    for d in matches:
        log_path = d / "per_case_partial.jsonl"
        if log_path.exists():
            logs.append(log_path)
        else:
            log.warning("  %s has no per_case_partial.jsonl (in-progress or failed)",
                         d.name)
    return logs


def merge_jsonl(logs: List[Path]) -> List[Dict]:
    """
    Read every JSONL log, dedupe by (token, config). Last write wins
    (so a re-run that recomputed metrics overrides earlier entries).
    """
    by_key: Dict[str, Dict] = {}
    n_lines_total = n_bad = 0
    per_shard_counts: List[Tuple[str, int]] = []

    for path in logs:
        n_lines = n_kept = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                n_lines_total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                tok = rec.get("token")
                cfg = rec.get("config")
                if not tok or not cfg:
                    n_bad += 1
                    continue
                by_key[f"{tok}__{cfg}"] = rec
                n_kept += 1
        per_shard_counts.append((path.parent.name, n_lines))

    log.info("Read %d lines across %d shard logs (bad=%d)",
             n_lines_total, len(logs), n_bad)
    for name, n in per_shard_counts:
        log.info("  %-40s %5d lines", name, n)
    log.info("After dedup: %d unique (token, config) records", len(by_key))

    return list(by_key.values())


def write_merged_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Wrote merged log -> %s (%d records)", out_path, len(records))


def aggregate_and_format(records: List[Dict], out_dir: Path) -> None:
    """
    Run aggregate() + format_table5() + format_table_surface() +
    write_csv() from benchmark_totalseg.py, exactly as that script does
    at the end of its main loop.
    """
    bt = _import_benchmark_module()

    summary = bt.aggregate(records)
    t5 = bt.format_table5(summary)
    t6 = bt.format_table_surface(summary)
    print(t5)
    print(t6)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_tables.txt").write_text(t5 + "\n" + t6)
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    (out_dir / "benchmark_results.json").write_text(
        json.dumps({"summary": summary, "per_case": records},
                   indent=2, default=str))
    bt.write_csv(records, out_dir / "benchmark_per_case.csv")
    log.info("Wrote merged outputs to %s/", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo_default = Path(os.environ.get(
        "REPO_ROOT", str(Path.home() / "CTSpinoPelvic1K")))
    ap.add_argument(
        "--results_base", type=Path,
        default=repo_default / "results" / "totalseg_bench",
        help="Directory containing shard_*/ subdirectories.",
    )
    ap.add_argument(
        "--shard_glob", type=str, default="shard_*",
        help="Glob pattern (relative to --results_base) selecting which "
             "shard directories to merge.",
    )
    ap.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory for merged results. "
             "Default: <results_base>/_merged",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_base / "_merged")

    log.info("Searching for shards under %s/%s", args.results_base, args.shard_glob)
    logs = collect_shard_logs(args.results_base, args.shard_glob)
    if not logs:
        log.error("No per_case_partial.jsonl files found. Nothing to merge.")
        return 1
    log.info("Found %d shard logs", len(logs))

    records = merge_jsonl(logs)
    if not records:
        log.error("No valid records after merge.")
        return 1

    write_merged_jsonl(records, out_dir / "per_case_partial.jsonl")
    aggregate_and_format(records, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
