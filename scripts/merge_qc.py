"""
merge_qc.py — join the per-case QC CSVs into ONE ranked triage worklist.

Takes any of the three GT-free QC outputs (neighbour-mixing, bone-leak,
structure) and joins them on (token, config) into a single master table with a
combined `needs_review` flag and `n_flags` severity, sorted worst-first — the
single list to hand students.

  python scripts/merge_qc.py \
      --mixing    data/qc_pseudo.csv \
      --leak      data/leak_pseudo.csv \
      --structure data/struct_pseudo.csv \
      --out       data/qc_master.csv

Any of --mixing/--leak/--structure may be omitted; only the provided checks
contribute. Pure join logic (build_master) is unit-tested.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ctspinopelvic1k.merge_qc")

# Which columns to carry from each source (besides token/config), and the
# per-source overall flag column.
PULL = {
    "mixing": (["off_main_frac", "n_order_inversions", "n_nonadjacent_touch"],
               "mixing_flag"),
    "leak": (["off_bone_frac", "bg_leak_frac", "under_seg_frac"], "leak_flag"),
    "structure": (["lr_swap", "lr_same_side", "vertebra_gap",
                   "pelvis_incomplete", "duplication_flag"], "struct_flag"),
}


def _key(row: dict) -> Tuple[str, str]:
    return (str(row.get("token", "")), str(row.get("config", "")))


def build_master(sources: Dict[str, List[dict]]) -> List[dict]:
    """Join QC source rows on (token, config). `sources` maps a source name in
    PULL to its list of row dicts. Returns master rows with each source's flag +
    pulled columns, a combined needs_review, and n_flags, sorted worst-first."""
    keys: List[Tuple[str, str]] = []
    indexed: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for name, rows in sources.items():
        idx = {}
        for r in rows:
            k = _key(r)
            idx[k] = r
            if k not in keys:
                keys.append(k)
        indexed[name] = idx

    master: List[dict] = []
    for k in keys:
        out = {"token": k[0], "config": k[1]}
        n_flags = 0
        for name, (cols, flag_col) in PULL.items():
            row = indexed.get(name, {}).get(k)
            flag = int(float(row[flag_col])) if (row and row.get(flag_col) not in
                                                 (None, "")) else 0
            out[flag_col] = flag
            n_flags += 1 if flag else 0
            for c in cols:
                out[c] = (row.get(c, "") if row else "")
        out["n_flags"] = n_flags
        out["needs_review"] = int(n_flags > 0)
        master.append(out)

    def _sev(r):
        def f(c):
            try:
                return float(r.get(c) or 0)
            except (TypeError, ValueError):
                return 0.0
        return (r["needs_review"], r["n_flags"],
                f("off_bone_frac") + f("off_main_frac"))
    master.sort(key=_sev, reverse=True)
    return master


_FIELDS = (["token", "config", "needs_review", "n_flags",
            "mixing_flag", "leak_flag", "struct_flag"]
           + PULL["mixing"][0] + PULL["leak"][0] + PULL["structure"][0])


def _read(path: Optional[Path]) -> List[dict]:
    if not path:
        return []
    if not path.exists():
        log.warning("missing %s — skipping", path)
        return []
    return list(csv.DictReader(open(path)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixing", type=Path, default=None)
    ap.add_argument("--leak", type=Path, default=None)
    ap.add_argument("--structure", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sources = {k: _read(p) for k, p in
               (("mixing", args.mixing), ("leak", args.leak),
                ("structure", args.structure)) if p}
    if not any(sources.values()):
        log.error("no QC CSVs provided/found — nothing to merge")
        return 1

    master = build_master(sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(master)

    n = len(master)
    flagged = sum(r["needs_review"] for r in master)
    by = {fc: sum(r.get(fc, 0) for r in master)
          for fc in ("mixing_flag", "leak_flag", "struct_flag")}
    log.info("=" * 60)
    log.info("MASTER TRIAGE: %d cases, %d need review (%.1f%%)",
             n, flagged, 100.0 * flagged / n if n else 0.0)
    log.info("  by check: mixing=%d  leak=%d  structure=%d",
             by["mixing_flag"], by["leak_flag"], by["struct_flag"])
    log.info("  wrote ranked worklist -> %s", args.out)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
