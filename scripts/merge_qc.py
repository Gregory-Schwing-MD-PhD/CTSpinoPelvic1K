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
    "mixing": (["off_main_frac", "n_order_inversions", "n_nonadjacent_touch",
                "split_classes"], "mixing_flag"),
    "leak": (["off_bone_frac", "bg_leak_frac", "under_seg_frac"], "leak_flag"),
    "structure": (["lr_swap", "lr_same_side", "vertebra_gap",
                   "pelvis_incomplete", "duplication_flag", "dup_classes"],
                  "struct_flag"),
}

# source name -> its overall flag column. Lets --exclude name a CHECK to keep
# measuring (its columns stay) but drop from the review trigger (needs_review).
_SRC_TO_FLAG = {name: flag for name, (_cols, flag) in PULL.items()}
_ALL_FLAGS = ("mixing_flag", "leak_flag", "struct_flag")


def _key(row: dict) -> Tuple[str, str]:
    return (str(row.get("token", "")), str(row.get("config", "")))


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(vals, p: float) -> float:
    """Linear-interpolated p-th percentile of a list (ignores blanks)."""
    xs = sorted(x for x in vals if x is not None)
    if not xs:
        return float("inf")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# Continuous metric(s) that get a radiologist-percentile threshold, and which
# per-source flag they drive. Only LEAK is baselined: radiologists genuinely
# leak ~0.02-0.05 at the cortex (partial volume), so a fixed tol mis-fires.
# Mixing is NOT baselined — radiologists have ~zero label-islands, so p95≈0
# would over-flag; its absolute 0.005 tol is already discriminating.
_BASELINED = {"off_bone_frac": "leak_flag"}


def recalibrate(master: List[dict], baseline: List[dict], pct: float,
                exclude_flags=frozenset()):
    """Re-flag the LEAK check relative to the radiologist baseline: a case trips
    leak_flag only if off_bone_frac exceeds the p-th percentile of the baseline
    (gold) rows. Mixing keeps its absolute tol and the categorical struct checks
    (duplication / L-R swap / gap) are left as-is — both already discriminate.
    Recomputes needs_review + n_flags. `exclude_flags` (flag-column names) are
    still measured but do NOT contribute to needs_review. Returns thresholds."""
    thr = {col: _pct([_to_float(r.get(col)) for r in baseline], pct)
           for col in _BASELINED}
    counted = [f for f in _ALL_FLAGS if f not in exclude_flags]
    for r in master:
        for col, flag in _BASELINED.items():
            v = _to_float(r.get(col))
            r[flag] = int(v is not None and v > thr[col])
        r["n_flags"] = sum(int(r.get(f, 0)) for f in counted)
        r["needs_review"] = int(r["n_flags"] > 0)
    master.sort(key=lambda r: (r["needs_review"], r["n_flags"]), reverse=True)
    return thr


def build_master(sources: Dict[str, List[dict]],
                 exclude_flags=frozenset()) -> List[dict]:
    """Join QC source rows on (token, config). `sources` maps a source name in
    PULL to its list of row dicts. Returns master rows with each source's flag +
    pulled columns, a combined needs_review, and n_flags, sorted worst-first.
    `exclude_flags` (flag-column names) stay as columns but don't count toward
    needs_review (e.g. keep leak measured but not a review trigger)."""
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
            n_flags += 1 if (flag and flag_col not in exclude_flags) else 0
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
    ap.add_argument("--baseline", type=Path, default=None,
                    help="a radiologist (manual) merged-QC CSV; re-flags the "
                         "continuous checks at the --pct percentile of THIS set, "
                         "so a flag means 'worse than a radiologist'.")
    ap.add_argument("--pct", type=float, default=95.0,
                    help="baseline percentile for the continuous thresholds "
                         "(default 95).")
    ap.add_argument("--exclude", default="",
                    help="comma list of CHECKS to keep measuring but drop from "
                         "the review trigger: any of mixing,leak,structure "
                         "(e.g. --exclude leak — off-bone leak is too hard to fix, "
                         "so record it but don't put leak-only cases on the worklist).")
    args = ap.parse_args()

    bad = [n for n in (s.strip() for s in args.exclude.split(",") if s.strip())
           if n not in _SRC_TO_FLAG]
    if bad:
        log.error("--exclude: unknown check(s) %s (choose from %s)",
                  bad, list(_SRC_TO_FLAG))
        return 1
    exclude_flags = frozenset(_SRC_TO_FLAG[n] for n in
                              (s.strip() for s in args.exclude.split(",")) if n)

    sources = {k: _read(p) for k, p in
               (("mixing", args.mixing), ("leak", args.leak),
                ("structure", args.structure)) if p}
    if not any(sources.values()):
        log.error("no QC CSVs provided/found — nothing to merge")
        return 1

    master = build_master(sources, exclude_flags=exclude_flags)
    if exclude_flags:
        log.info("excluded from review trigger (still measured): %s",
                 sorted(exclude_flags))
    if args.baseline:
        if not args.baseline.exists():
            log.error("baseline CSV not found: %s", args.baseline)
            return 1
        thr = recalibrate(master, list(csv.DictReader(open(args.baseline))),
                          args.pct, exclude_flags=exclude_flags)
        log.info("calibrated to radiologist p%.0f: %s", args.pct,
                 {k: round(v, 4) for k, v in thr.items()})
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
