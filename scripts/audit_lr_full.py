#!/usr/bin/env python3
"""
audit_lr_full.py — Full-dataset L/R hip alignment audit, parallel.

For every hip-bearing record in data/hf_export/manifest.json (configs:
fused, pelvic_native), compare cached TotalSegmentator predictions in
results/totalseg_bench_*/ts_preds/ to the GT labels via two pairings:

    no-swap:  TS hip_left (77) vs GT hip_left (8)
              TS hip_right(78) vs GT hip_right(9)
    swap:     TS hip_left (77) vs GT hip_right(9)
              TS hip_right(78) vs GT hip_left (8)

A record is FLIPPED when the swap pairing wins decisively (avg_swap > 0.5
and avg_noswap < 0.1) and OK when no-swap wins. Anything else is MIXED
(partial FOV, cropped hip, weirdness worth eyeballing).

Outputs:
  - Console summary (counts, flip rate, hip Dice distribution)
  - {out_csv}: per-record table with token, config, all four Dice values,
    verdict, and label / pred paths

The 9 manually-flipped tokens (configs/flip_list.json) get AP-flipped in
Step C of the pipeline. The AP flip is orthogonal to L/R, so those tokens
should still audit as OK — TS predicts hips correctly on the AP-flipped
CT, and the GT labels were also AP-flipped along with the CT, so both
live in the same frame. Any FLIPPED result is a real concern regardless
of flip_list membership.

Usage (inside the TS container):
    python3 audit_lr_full.py \\
        --hf_export /data/hf_export \\
        --ts_glob   "/results/totalseg_bench_*/ts_preds" \\
        --workers   16 \\
        --out_csv   /data/audit_lr_full.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple


def _dice(a, b) -> float:
    i = int((a & b).sum())
    s = int(a.sum()) + int(b.sum())
    return float("nan") if s == 0 else 2 * i / s


def _audit_one(work) -> Dict:
    """
    Worker function: load TS pred + GT label for one record, resample GT
    onto TS grid, compute the four hip-pair Dice values, return a dict.
    """
    tok, cfg, label_path, pred_path = work

    try:
        import numpy as np
        import SimpleITK as sitk

        ts = sitk.ReadImage(pred_path, sitk.sitkInt32)
        gt = sitk.ReadImage(label_path, sitk.sitkInt32)

        rs = sitk.ResampleImageFilter()
        rs.SetReferenceImage(ts)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        rs.SetTransform(sitk.Transform())
        gt_r = rs.Execute(gt)

        ts_arr = sitk.GetArrayFromImage(ts)
        gt_arr = sitk.GetArrayFromImage(gt_r)

        nL = _dice(ts_arr == 77, gt_arr == 8)   # TS L vs GT L
        nR = _dice(ts_arr == 78, gt_arr == 9)   # TS R vs GT R
        sL = _dice(ts_arr == 77, gt_arr == 9)   # TS L vs GT R
        sR = _dice(ts_arr == 78, gt_arr == 8)   # TS R vs GT L

        all_nan = all(np.isnan(v) for v in (nL, nR, sL, sR))
        if all_nan:
            return dict(token=tok, config=cfg, status="no_hips",
                        nL=None, nR=None, sL=None, sR=None,
                        verdict="SKIP", label_path=label_path,
                        pred_path=pred_path)

        nL = 0.0 if np.isnan(nL) else float(nL)
        nR = 0.0 if np.isnan(nR) else float(nR)
        sL = 0.0 if np.isnan(sL) else float(sL)
        sR = 0.0 if np.isnan(sR) else float(sR)
        avg_n = (nL + nR) / 2
        avg_s = (sL + sR) / 2

        if avg_n > 0.5 and avg_s < 0.1:
            verdict = "OK"
        elif avg_s > 0.5 and avg_n < 0.1:
            verdict = "FLIPPED"
        else:
            verdict = "MIXED"

        return dict(token=tok, config=cfg, status="audited",
                    nL=round(nL, 4), nR=round(nR, 4),
                    sL=round(sL, 4), sR=round(sR, 4),
                    avg_n=round(avg_n, 4), avg_s=round(avg_s, 4),
                    verdict=verdict, label_path=label_path,
                    pred_path=pred_path)

    except Exception as exc:
        return dict(token=tok, config=cfg, status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    verdict="ERROR", label_path=label_path, pred_path=pred_path)


def _build_work_items(hf_export: Path, ts_glob_pattern: str) -> Tuple[List[Tuple], Dict]:
    manifest_path = hf_export / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found at {manifest_path}")

    m = json.loads(manifest_path.read_text())
    records = m if isinstance(m, list) else m.get("records", [])

    work: List[Tuple] = []
    n_skip_label = n_skip_pred = n_not_hip_cfg = 0

    for r in records:
        cfg = r.get("config", "")
        if cfg not in ("fused", "pelvic_native"):
            n_not_hip_cfg += 1
            continue

        tok = str(r.get("token", "")).strip()
        if not tok:
            continue

        label_rel = r.get("label_file") or ""
        label_path = (hf_export / label_rel).as_posix()
        if not Path(label_path).exists():
            n_skip_label += 1
            continue

        pred_paths = sorted(glob(f"{ts_glob_pattern}/{tok}_{cfg}/segmentation.nii.gz"))
        if not pred_paths:
            n_skip_pred += 1
            continue

        # Use the latest available cached prediction
        pred_path = pred_paths[-1]
        work.append((tok, cfg, label_path, pred_path))

    stats = dict(
        total_records=len(records),
        not_hip_config=n_not_hip_cfg,
        skip_label_missing=n_skip_label,
        skip_no_pred=n_skip_pred,
        work_items=len(work),
    )
    return work, stats


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hf_export", type=Path, default=Path("/data/hf_export"))
    p.add_argument("--ts_glob",   type=str,
                   default="/results/totalseg_bench_*/ts_preds")
    p.add_argument("--workers",   type=int,
                   default=int(os.environ.get(
                       "AUDIT_WORKERS",
                       os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    p.add_argument("--out_csv",   type=Path,
                   default=Path("/data/audit_lr_full.csv"))
    p.add_argument("--flip_list", type=Path,
                   default=Path("/workspace/configs/flip_list.json"),
                   help="If present, FLIPPED tokens are cross-referenced "
                        "and tagged in the CSV. Note: AP flips (Step C) do "
                        "not affect L/R, so pipeline-flipped tokens should "
                        "still audit as OK. Any FLIPPED result is a real "
                        "concern regardless of flip_list membership.")
    args = p.parse_args()

    print("=" * 78)
    print("L/R Hip Alignment Audit — Full Dataset")
    print("=" * 78)
    print(f"  HF export   : {args.hf_export}")
    print(f"  TS preds    : {args.ts_glob}")
    print(f"  Workers     : {args.workers}")
    print(f"  Output CSV  : {args.out_csv}")
    print(f"  Flip list   : {args.flip_list if args.flip_list.exists() else '(none)'}")
    print("=" * 78)

    # Discover all available TS run directories so we can report what we found
    ts_dirs = sorted(glob(args.ts_glob))
    print(f"\nFound {len(ts_dirs)} TS prediction directories:")
    for d in ts_dirs[:5]:
        n = len(list(Path(d).glob("*/segmentation.nii.gz")))
        print(f"  {d}  ({n} segmentations)")
    if len(ts_dirs) > 5:
        print(f"  ...and {len(ts_dirs) - 5} more")

    # Build work
    work, stats = _build_work_items(args.hf_export, args.ts_glob)
    print(f"\nWork queue:")
    print(f"  Manifest records       : {stats['total_records']}")
    print(f"  Not hip config (skip)  : {stats['not_hip_config']}")
    print(f"  Label missing (skip)   : {stats['skip_label_missing']}")
    print(f"  No TS pred (skip)      : {stats['skip_no_pred']}")
    print(f"  Audit work items       : {stats['work_items']}")

    if not work:
        sys.exit("\nERROR: no work items — no auditable records found.")

    # Optional: load flip list for cross-reference
    flipped_in_pipeline = set()
    if args.flip_list.exists():
        try:
            fl = json.loads(args.flip_list.read_text())
            flipped_in_pipeline = {str(f.get("token")) for f in fl.get("flips", [])}
            print(f"  Pipeline-flipped tokens: {sorted(flipped_in_pipeline)}")
        except Exception as exc:
            print(f"  WARN: could not parse flip list: {exc}")

    # Run parallel
    print(f"\nRunning {len(work)} audits with {args.workers} workers...\n")
    t0 = time.time()
    results: List[Dict] = []

    n_done = 0
    n_ok = n_flipped = n_mixed = n_nohip = n_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_audit_one, w) for w in work]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1

            v = r["verdict"]
            if   v == "OK":      n_ok += 1
            elif v == "FLIPPED": n_flipped += 1
            elif v == "MIXED":   n_mixed += 1
            elif v == "SKIP":    n_nohip += 1
            else:                n_err += 1

            if n_done % 50 == 0 or n_done == len(work):
                rate = n_done / max(1e-3, time.time() - t0)
                print(f"  ...{n_done}/{len(work)}  "
                      f"ok={n_ok}  flipped={n_flipped}  mixed={n_mixed}  "
                      f"no_hip={n_nohip}  err={n_err}  ({rate:.1f}/s)")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({len(work)/max(1e-3,elapsed):.1f} records/s)")

    # Sort: errors first, then flipped, mixed, ok, skip
    rank = {"ERROR": 0, "FLIPPED": 1, "MIXED": 2, "OK": 3, "SKIP": 4}
    results.sort(key=lambda r: (rank.get(r["verdict"], 99), r["token"], r["config"]))

    # Hip Dice distribution on OK records
    ok_results = [r for r in results if r["verdict"] == "OK"]
    if ok_results:
        import numpy as np
        nLs = [r["nL"] for r in ok_results]
        nRs = [r["nR"] for r in ok_results]
        print(f"\nHip Dice distribution (OK records only, n={len(ok_results)}):")
        print(f"  L:  mean={np.mean(nLs):.3f}  median={np.median(nLs):.3f}  "
              f"min={np.min(nLs):.3f}  p5={np.percentile(nLs, 5):.3f}  "
              f"p95={np.percentile(nLs, 95):.3f}")
        print(f"  R:  mean={np.mean(nRs):.3f}  median={np.median(nRs):.3f}  "
              f"min={np.min(nRs):.3f}  p5={np.percentile(nRs, 5):.3f}  "
              f"p95={np.percentile(nRs, 95):.3f}")

    # Final summary
    print(f"\n{'=' * 78}")
    print("FINAL RESULT")
    print(f"{'=' * 78}")
    print(f"  Total audited            : {len(results)}")
    print(f"  OK                       : {n_ok}")
    print(f"  FLIPPED                  : {n_flipped}")
    print(f"  MIXED (partial / weird)  : {n_mixed}")
    print(f"  No hips in label (skip)  : {n_nohip}")
    print(f"  Errors                   : {n_err}")

    n_classified = n_ok + n_flipped + n_mixed
    if n_classified:
        print(f"\n  Classification rate     : {n_classified}/{len(results)} "
              f"({100*n_classified/len(results):.1f}%)")
        print(f"  Flip rate               : {100*n_flipped/n_classified:.2f}%")
        print(f"  Mixed rate              : {100*n_mixed/n_classified:.2f}%")

    # Show flipped records, noting which are also AP-flipped in the pipeline
    # (the AP flip doesn't affect L/R, so flip_list membership doesn't excuse
    # an L/R inversion — it's just a useful cross-reference)
    flipped_recs = [r for r in results if r["verdict"] == "FLIPPED"]
    if flipped_recs:
        print(f"\n  FLIPPED records ({len(flipped_recs)}):  ALL ARE REAL CONCERNS")
        for r in flipped_recs[:60]:
            tok = r["token"]
            tag = "  [also AP-flipped]" if tok in flipped_in_pipeline else ""
            print(f"    {tok}/{r['config']:<14}  "
                  f"no-swap={r['nL']:.3f}/{r['nR']:.3f}  "
                  f"swap={r['sL']:.3f}/{r['sR']:.3f}{tag}")
        if len(flipped_recs) > 60:
            print(f"    ...and {len(flipped_recs) - 60} more")
        if flipped_in_pipeline:
            n_also_ap = sum(1 for r in flipped_recs
                            if r["token"] in flipped_in_pipeline)
            n_ap_only = len(flipped_recs) - n_also_ap
            print(f"\n  Cross-reference with flip_list.json:")
            print(f"    L/R-flipped AND AP-flipped : {n_also_ap}")
            print(f"    L/R-flipped only           : {n_ap_only}")

    # Show mixed records — manual look territory
    mixed_recs = [r for r in results if r["verdict"] == "MIXED"]
    if mixed_recs:
        print(f"\n  MIXED records ({len(mixed_recs)}):")
        for r in mixed_recs[:30]:
            print(f"    {r['token']}/{r['config']:<14}  "
                  f"no-swap={r['nL']:.3f}/{r['nR']:.3f}  "
                  f"swap={r['sL']:.3f}/{r['sR']:.3f}")
        if len(mixed_recs) > 30:
            print(f"    ...and {len(mixed_recs) - 30} more")

    # Errors
    err_recs = [r for r in results if r["verdict"] == "ERROR"]
    if err_recs:
        print(f"\n  ERROR records ({len(err_recs)}):")
        for r in err_recs[:30]:
            print(f"    {r['token']}/{r['config']:<14}  {r.get('error','?')}")
        if len(err_recs) > 30:
            print(f"    ...and {len(err_recs) - 30} more")

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "token", "config", "verdict", "status",
            "noswap_L", "noswap_R", "swap_L", "swap_R",
            "avg_noswap", "avg_swap",
            "pipeline_flipped", "label_path", "pred_path", "error",
        ])
        for r in results:
            w.writerow([
                r["token"], r["config"], r["verdict"], r.get("status", ""),
                r.get("nL", ""), r.get("nR", ""),
                r.get("sL", ""), r.get("sR", ""),
                r.get("avg_n", ""), r.get("avg_s", ""),
                "Y" if r["token"] in flipped_in_pipeline else "",
                r.get("label_path", ""), r.get("pred_path", ""),
                r.get("error", ""),
            ])
    print(f"\n  Per-record CSV: {args.out_csv}")
    print("=" * 78)


if __name__ == "__main__":
    main()
