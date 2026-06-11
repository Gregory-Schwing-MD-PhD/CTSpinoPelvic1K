"""
refresh_lstv_from_labels.py — recompute has_l6 / n_lumbar_labels from the
ACTUAL label voxels and (optionally) rewrite the manifest.

Why this exists
---------------
has_l6 / n_lumbar_labels are computed ONLY at the original export_hf.py run
(export_hf.py: `uniq = {v in label if 1<=v<=6}`; has_l6 = 6 in uniq;
n_lumbar_labels = len(uniq)). Neither reduce_to_v3.py (which swaps in
corrected/pseudolabelled labels) nor refresh_hf_manifests.py (which updates
lstv_class from placed_manifest) recomputes these from the new label voxels.

So if an L6 first appears in a corrected/pseudolabelled label (e.g. the extra
L6 cases that arose in review), the manifest still says has_l6=False — and a
later generate_5fold_splits run would then mislabel those cases as `normal`.
This script re-reads each label NIfTI and fixes has_l6 / n_lumbar_labels.

Report-first: a DRY RUN by default. It prints exactly which tokens flip
False->True (the newly-found L6 cases) and, if given the finalized-reviews
index (--finals), whether each came from a reviewer correction. Re-run with
--write to apply and rewrite manifest.json/csv + manifest_{train,val,test}.json
+ splits (via export_hf's own writers, so output matches an end-to-end run).

Usage
-----
  # 1) audit v3 (dry run) — does NOT modify anything
  python scripts/refresh_lstv_from_labels.py --hf_dir data/hf_export_v3 \
      --finals reviews/finalized_index.json

  # 2) apply the fix, then re-split
  python scripts/refresh_lstv_from_labels.py --hf_dir data/hf_export_v3 --write
  python scripts/generate_5fold_splits.py --hf_dir data/hf_export_v3 \
      --out data/hf_export_v3/splits_5fold.json
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)   # stream progress in real time
except (AttributeError, ValueError):
    pass
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)             # land in the SLURM .out file
log = logging.getLogger("refresh_lstv_from_labels")


def lumbar_from_label(label_path: Path) -> Tuple[int, bool, List[int]]:
    """Mirror export_hf.py:855-857 exactly."""
    import nibabel as nib
    arr = np.asarray(nib.load(str(label_path)).dataobj)
    uniq = {int(v) for v in np.unique(arr) if 1 <= v <= 6}
    return len(uniq), (6 in uniq), sorted(uniq)


def _scan_task(task: Tuple[int, str, str, str]):
    """Worker: (idx, token, config, label_path) -> result tuple. Module-level
    so it is picklable for multiprocessing. Each label is independent, so the
    scan is embarrassingly parallel."""
    idx, tok, cfg, path = task
    try:
        if not os.path.exists(path):
            return (idx, tok, cfg, 0, False, [], False)
        n, has6, uniq = lumbar_from_label(Path(path))
        return (idx, tok, cfg, n, has6, uniq, True)
    except Exception:
        return (idx, tok, cfg, 0, False, [], False)


def _load_records(manifest_path: Path) -> List[dict]:
    data = json.loads(manifest_path.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _finals_by_token(finals_path: Optional[Path]) -> Dict[str, List[Tuple[str, str]]]:
    """token -> [(config, decision), ...] from a finalized-reviews index keyed
    by '<token>__<config>'."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    if not finals_path:
        return out
    finals = json.loads(finals_path.read_text())
    for case_id, entry in finals.items():
        tok, _, cfg = case_id.partition("__")
        out.setdefault(tok, []).append((cfg, str(entry.get("decision", "?"))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_dir", required=True, type=Path,
                    help="export tree (e.g. data/hf_export_v3) — labels resolve under it")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest.json (default <hf_dir>/manifest.json)")
    ap.add_argument("--finals", type=Path, default=None,
                    help="finalized-reviews index JSON, to attribute flips to corrections")
    ap.add_argument("--write", action="store_true",
                    help="apply the fix and rewrite manifest + splits (default: dry run)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel label-scan workers (0 = min(8, cpu_count))")
    ap.add_argument("--spine_authoritative", action="store_true",
                    help="trust the SPINE-bearing record (fused/spine_only) for "
                         "has_l6; NEUTRALISE pelvic_native pseudolabels "
                         "(has_l6=False, n_lumbar=0) so they can't promote a "
                         "patient to lumb on an unreliable pelvic-view count. "
                         "Pelvic-ONLY patients whose pseudolabel drew an L6 are "
                         "listed for review (and kept out of lumb unless you "
                         "confirm them via --keep_pelvic).")
    ap.add_argument("--keep_pelvic", default="",
                    help="comma-separated pelvic-only tokens you CONFIRMED have a "
                         "real L6 (after review) — promote these to has_l6=True "
                         "despite spine-authoritative neutralisation. e.g. "
                         "--keep_pelvic 140,46")
    ap.add_argument("--exclude_tokens", default="",
                    help="comma-separated tokens to LEAVE UNCHANGED (don't write "
                         "has_l6 for them) — e.g. a known class-mixing GT error "
                         "whose stray value-6 voxels are a mislabel, not a real "
                         "L6. Correct the label instead; this just keeps the "
                         "bogus L6 out of the manifest meanwhile. e.g. "
                         "--exclude_tokens 103")
    args = ap.parse_args()

    keep_pelvic = {t.strip() for t in args.keep_pelvic.split(",") if t.strip()}
    exclude_tokens = {t.strip() for t in args.exclude_tokens.split(",") if t.strip()}

    manifest_path = args.manifest or (args.hf_dir / "manifest.json")
    if not manifest_path.exists():
        log.error("manifest not found: %s", manifest_path)
        return 1
    records = _load_records(manifest_path)
    finals = _finals_by_token(args.finals)
    log.info("scanning %d records under %s", len(records), args.hf_dir)

    # Build the scan task list (records that have a label file).
    tasks: List[Tuple[int, str, str, str]] = []
    for i, rec in enumerate(records):
        tok = str(rec.get("token") or rec.get("patient_token") or "")
        cfg = str(rec.get("config", ""))
        lf = rec.get("label_file") or rec.get("label")
        if lf:
            tasks.append((i, tok, cfg, str(args.hf_dir / lf)))

    workers = args.workers if args.workers > 0 else min(8, mp.cpu_count() or 1)
    log.info("scanning %d labels with %d worker(s) ...", len(tasks), workers)

    results = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for j, r in enumerate(pool.imap_unordered(_scan_task, tasks, chunksize=8), 1):
                results.append(r)
                if j % 200 == 0:
                    log.info("  scanned %d/%d ...", j, len(tasks))
    else:
        for j, t in enumerate(tasks, 1):
            results.append(_scan_task(t))
            if j % 200 == 0:
                log.info("  scanned %d/%d ...", j, len(tasks))

    # Per-patient label-derived L6 from spine-bearing vs pelvic records, used by
    # spine-authoritative mode + conflict detection.
    from collections import defaultdict as _dd
    by_tok: Dict[str, list] = _dd(list)
    for (idx, tok, cfg, n_new, has_new, uniq, ok) in results:
        if ok:
            by_tok[tok].append((cfg.lower(), has_new, idx))

    redundant: List[str] = []        # spine already has L6 AND pelvic has L6
    conflicts: List[str] = []        # has spine record, spine=no-L6 but pelvic=L6
    pelvic_only_l6: List[str] = []   # pelvic-only patient, pseudolabel drew L6
    for tok, items in by_tok.items():
        spine = [h for c, h, _ in items if c in ("fused", "spine_only")]
        pelv  = [h for c, h, _ in items if c == "pelvic_native"]
        if spine:
            if any(pelv) and any(spine):
                redundant.append(tok)
            elif any(pelv) and not any(spine):
                conflicts.append(tok)
        elif any(pelv):
            pelvic_only_l6.append(tok)

    flips_true:  List[Tuple[str, str, List[int]]] = []   # has_l6 False -> True
    flips_false: List[Tuple[str, str, List[int]]] = []   # True -> False
    nlumb_changes: List[Tuple[str, int, int]] = []
    missing: List[str] = []
    n_scanned = 0
    n_neutralised = 0

    for (idx, tok, cfg, n_new, has_new, uniq, ok) in results:
        if not ok:
            missing.append(f"{tok}__{cfg}")
            continue
        rec = records[idx]
        has_old = bool(rec.get("has_l6", False))
        n_old = int(rec.get("n_lumbar_labels", 0) or 0)

        if tok in exclude_tokens:
            n_scanned += 1   # leave has_l6/n_lumbar exactly as they are
            continue

        # Resolve the value to WRITE. Default = factual (v3 label content, which
        # already includes reviewer corrections to spine masks). Spine-
        # authoritative mode neutralises pelvic_native pseudolabels so they
        # can't promote a patient to lumb on an unreliable pelvic-view count,
        # unless the token was confirmed via --keep_pelvic.
        write_has, write_n = has_new, n_new
        if args.spine_authoritative and cfg.lower() == "pelvic_native":
            if tok in keep_pelvic:
                write_has, write_n = True, max(6, n_new)
            else:
                write_has, write_n = False, 0
                if has_new:
                    n_neutralised += 1

        if write_has and not has_old:
            flips_true.append((tok, cfg, uniq))
        elif has_old and not write_has:
            flips_false.append((tok, cfg, uniq))
        if write_n != n_old:
            nlumb_changes.append((tok, n_old, write_n))
        if args.write:
            rec["has_l6"] = write_has
            rec["n_lumbar_labels"] = write_n
        n_scanned += 1

    n_missing = len(missing)
    if args.spine_authoritative:
        log.info("spine-authoritative: %d pelvic-view pseudolabel L6 flag(s) "
                 "dropped from patient subtyping (kept %d via --keep_pelvic):",
                 n_neutralised, len(keep_pelvic))
        log.info("  %d REDUNDANT  -- patient already has L6 in the spine mask "
                 "(already lumb; nothing lost)", len(redundant))
        log.info("  %d CONFLICT   -- spine mask has NO L6 but pelvic pseudolabel "
                 "drew one  -> REVIEW (see list)", len(conflicts))
        log.info("  %d PELVIC-ONLY-- no spine mask; pseudolabel is the only "
                 "source -> REVIEW (see list)", len(pelvic_only_l6))
    if missing:
        log.warning("%d record(s) had missing/unreadable labels: %s",
                    n_missing, ", ".join(missing[:10]) + (" ..." if n_missing > 10 else ""))

    log.info("=" * 64)
    log.info("scanned=%d  missing_labels=%d", n_scanned, n_missing)
    log.info("has_l6 False->True : %d   <-- newly-found L6 (the cases in question)",
             len(flips_true))
    log.info("has_l6 True->False : %d", len(flips_false))
    log.info("n_lumbar_labels changed on %d records", len(nlumb_changes))
    if flips_true:
        log.info("-" * 64)
        log.info("Newly-found L6 cases:")
        for tok, cfg, uniq in flips_true:
            attribution = ""
            if finals:
                decs = finals.get(tok, [])
                if decs:
                    attribution = "  reviews=" + ",".join(f"{c}:{d}" for c, d in decs)
                else:
                    attribution = "  reviews=<none for token>"
            log.info("  token=%s config=%s lumbar_labels=%s%s",
                     tok, cfg, uniq, attribution)
        if finals:
            n_corrected = sum(1 for tok, _, _ in flips_true
                              if any(d == "corrected" for _, d in finals.get(tok, [])))
            log.info("-> %d/%d of the newly-found L6 cases were reviewer-CORRECTED",
                     n_corrected, len(flips_true))
            log.info("   (confirms whether these arose from review label corrections)")

    def _sk(t):
        return (0, int(t)) if t.isdigit() else (1, t)

    if args.spine_authoritative:
        log.info("-" * 64)
        log.info("REVIEW QUEUE (these did NOT auto-update; eyeball them):")
        log.info("  PELVIC-ONLY L6 candidates (no spine mask; pseudolabel only): %d",
                 len(pelvic_only_l6))
        for tok in sorted(pelvic_only_l6, key=_sk):
            log.info("    token=%s%s  -> if a real L6, include with --keep_pelvic",
                     tok, "  [KEPT]" if tok in keep_pelvic else "")
        log.info("  CONFLICTS (spine mask=no-L6, pelvic pseudolabel=L6): %d",
                 len(conflicts))
        for tok in sorted(conflicts, key=_sk):
            log.info("    token=%s  -> if the radiologist MISSED an L6, correct the "
                     "spine_only label (then re-run; has_l6 follows). else leave.",
                     tok)
        if not pelvic_only_l6 and not conflicts:
            log.info("  (empty -- nothing to review)")
        if exclude_tokens:
            log.info("excluded (left unchanged): %s", ",".join(sorted(exclude_tokens)))

    if not args.write:
        log.info("=" * 64)
        log.info("DRY RUN -- nothing written. Re-run with --write to apply, then:")
        log.info("  python scripts/generate_5fold_splits.py --hf_dir %s --out %s/splits_5fold.json",
                 args.hf_dir, args.hf_dir)
        return 0

    # Apply: rewrite manifest + splits using export_hf's own writers.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_hf import write_manifest, write_splits
    out_records = [{**r, "ok": r.get("ok", True)} for r in records]
    log.info("re-writing manifest.json/csv + manifest_{train,val,test}.json + splits")
    write_manifest(out_records, args.hf_dir)
    write_splits(out_records, args.hf_dir)
    log.info("done. NEXT: re-run generate_5fold_splits.py on %s", args.hf_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
