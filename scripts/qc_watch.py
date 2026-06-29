"""qc_watch.py — stream rib-QC failures as build_v4_ribs writes them, instead of waiting
for the whole run to finish and aggregating.

Polls <v4_dir>/_v4ribs_done/*.json, and for each NEW case prints a line only if it FAILS
(numbering gap or duplicate id) or warrants a CHECK (large false-positive drop, which may
mean real bone was filtered). Keeps a running tally. Stdlib-only -> run on the login node.

  python scripts/qc_watch.py --v4_dir data/hf_export_v4          # poll until Ctrl-C
  python scripts/qc_watch.py --v4_dir data/hf_export_v4 --once   # one pass then exit

Tip: clear stale records first (rm _v4ribs_done/*.json) so the list reflects only this run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4_dir", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=10.0, help="poll seconds")
    ap.add_argument("--fp_comps", type=int, default=3,
                    help="flag CHECK if a case drops >= this many components (possible over-filter)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    done = a.v4_dir / "_v4ribs_done"

    seen: set = set()
    n = nfail = ncheck = fp_comps_tot = fp_vox_tot = 0
    while True:
        for p in sorted(done.glob("*.json")):
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                r = json.loads(p.read_text())
            except Exception:                                # noqa: BLE001 (half-written file)
                seen.discard(p.name)                         # retry next pass
                continue
            n += 1
            fp_comps_tot += int(r.get("n_dropped_fp", 0))
            fp_vox_tot += int(r.get("dropped_fp_vox", 0))
            ct = r.get("ct", p.stem)
            problems = []
            if r.get("left_gaps") or r.get("right_gaps"):
                problems.append(f"gaps L{r.get('left_gaps')} R{r.get('right_gaps')}")
            if r.get("duplicate_rib_ids"):
                problems.append(f"dup={r['duplicate_rib_ids']}")
            if problems:
                nfail += 1
                print(f"FAIL  {ct:>10}: {'  '.join(problems)}  (fp_drop={r.get('n_dropped_fp',0)})",
                      flush=True)
            elif int(r.get("n_dropped_fp", 0)) >= a.fp_comps:
                ncheck += 1
                print(f"CHECK {ct:>10}: dropped {r.get('n_dropped_fp')} comp "
                      f"({r.get('dropped_fp_vox')} vox) -- confirm it was bowel, not rib", flush=True)
        print(f"[qc_watch] {n} done | {nfail} FAIL (gaps/dups) | {ncheck} CHECK | "
              f"filtered {fp_comps_tot} comps / {fp_vox_tot} vox", flush=True)
        if a.once:
            break
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
