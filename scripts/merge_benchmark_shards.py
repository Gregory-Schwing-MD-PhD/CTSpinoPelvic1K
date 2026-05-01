#!/usr/bin/env python3
"""
merge_benchmark_shards.py — Merge sharded TotalSegmentator benchmark logs.

Walks <results_base>/shard_*/per_case_partial.jsonl, deduplicates by
(token, config), and runs the splits-aware patient-level aggregation
from benchmark_totalseg.py.

Apr 2026 v6 update — 6-way LSTV subgroups
==========================================
Default behavior: aggregation uses the 6-way taxonomy from
splits_5fold.json schema v6:

  normal | lumb | sacr_count | semisacralization | sacralization | ambiguous

Pass --record_level to disable patient-level dedup (legacy behavior).

Pass --splits_file pointing at splits_5fold.json (v6+) to enable 6-way
LSTV subgroup binning. v5 splits fall back to 4-way; missing/older falls
back to 3-way.

Outputs (under <out_dir>):
  paper_tables.txt                    Tables 5+6, patient-level
  benchmark_summary.json              patient-level summary
  benchmark_results.json              full per-case + summary
  benchmark_per_case.csv              one row per patient
  per_case_partial.jsonl              deduplicated combined log

  paper_tables_record_level.txt       Tables 5+6, record-level (supplement)
  benchmark_summary_record_level.json record-level summary
  benchmark_per_case_record_level.csv one row per record

This script does NOT re-run TS inference. Pure JSONL aggregation.

Usage
-----
  # Default: walk results/totalseg_bench/shard_*/, write _merged/
  python scripts/merge_benchmark_shards.py \\
      --splits_file data/hf_export/splits_5fold.json

  # Force record-level (legacy):
  python scripts/merge_benchmark_shards.py \\
      --splits_file data/hf_export/splits_5fold.json \\
      --record_level

  # Custom paths:
  python scripts/merge_benchmark_shards.py \\
      --results_base $HOME/CTSpinoPelvic1K/results/totalseg_bench_35900384 \\
      --out_dir      $HOME/CTSpinoPelvic1K/results/totalseg_bench_35900384/_merged_v6 \\
      --splits_file  $HOME/CTSpinoPelvic1K/data/hf_export/splits_5fold.json
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
            log.warning("  %s has no per_case_partial.jsonl", d.name)
    return logs


def merge_jsonl(logs: List[Path]) -> List[Dict]:
    by_key: Dict[str, Dict] = {}
    n_lines_total = n_bad = 0
    per_shard: List[Tuple[str, int]] = []

    for path in logs:
        n_lines = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                n_lines += 1
                n_lines_total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1; continue
                tok, cfg = rec.get("token"), rec.get("config")
                if not tok or not cfg:
                    n_bad += 1; continue
                by_key[f"{tok}__{cfg}"] = rec
        per_shard.append((path.parent.name, n_lines))

    log.info("Read %d lines across %d shard logs (bad=%d)",
             n_lines_total, len(logs), n_bad)
    for name, n in per_shard:
        log.info("  %-40s %5d lines", name, n)
    log.info("After dedup: %d unique (token, config) records", len(by_key))
    return list(by_key.values())


def write_merged_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Wrote merged log -> %s (%d records)", out_path, len(records))


def aggregate_and_format(records: List[Dict], out_dir: Path,
                          splits_subtypes: Dict[str, str],
                          splits_schema: int,
                          patient_level: bool) -> None:
    """Run benchmark_totalseg's splits-aware aggregation + table formatters."""
    bt = _import_benchmark_module()

    # ── Main outputs (patient-level by default) ──────────────────────
    summary = bt.aggregate(records, splits_subtypes=splits_subtypes,
                            splits_schema=splits_schema,
                            patient_level=patient_level)
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
    bt.write_csv(records, out_dir / "benchmark_per_case.csv",
                  splits_subtypes=splits_subtypes, patient_level=patient_level)

    # ── Supplementary record-level outputs ─────────────────────────────
    if patient_level:
        rec_summary = bt.aggregate(records, splits_subtypes=splits_subtypes,
                                     splits_schema=splits_schema,
                                     patient_level=False)
        rec_t5 = bt.format_table5(rec_summary)
        rec_t6 = bt.format_table_surface(rec_summary)
        (out_dir / "paper_tables_record_level.txt").write_text(rec_t5 + "\n" + rec_t6)
        (out_dir / "benchmark_summary_record_level.json").write_text(
            json.dumps(rec_summary, indent=2, default=str))
        bt.write_csv(records, out_dir / "benchmark_per_case_record_level.csv",
                      splits_subtypes=splits_subtypes, patient_level=False)
        log.info("Also wrote record-level supplementary outputs.")

    log.info("Wrote merged outputs to %s/", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_default = Path(os.environ.get("REPO_ROOT", str(Path.home() / "CTSpinoPelvic1K")))
    ap.add_argument(
        "--results_base", type=Path,
        default=repo_default / "results" / "totalseg_bench",
        help="Directory containing shard_*/ subdirectories.")
    ap.add_argument(
        "--shard_glob", type=str, default="shard_*",
        help="Glob (relative to --results_base) selecting shard dirs.")
    ap.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output dir. Default: <results_base>/_merged")
    ap.add_argument(
        "--splits_file", type=Path, default=None,
        help="Path to splits_5fold.json (v6+) for 6-way LSTV subgroup "
             "binning. If omitted, falls back to per-record lstv_label "
             "(3-way).")
    ap.add_argument(
        "--record_level", action="store_true",
        help="Disable patient-level deduplication; aggregate at the "
             "record level (legacy behavior). Default is patient-level.")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_base / "_merged")

    log.info("Searching for shards under %s/%s", args.results_base, args.shard_glob)
    logs = collect_shard_logs(args.results_base, args.shard_glob)
    if not logs:
        log.error("No per_case_partial.jsonl files found.")
        return 1
    log.info("Found %d shard logs", len(logs))

    records = merge_jsonl(logs)
    if not records:
        log.error("No valid records after merge.")
        return 1

    bt = _import_benchmark_module()
    splits_subtypes, splits_schema = bt.load_splits_subtype_map(args.splits_file)

    write_merged_jsonl(records, out_dir / "per_case_partial.jsonl")
    aggregate_and_format(records, out_dir, splits_subtypes, splits_schema,
                          patient_level=not args.record_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
