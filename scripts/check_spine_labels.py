"""
check_spine_labels.py — is the rib anchor already in the CTSpine1K ground truth?

The anchor is "the vertebra directly above GT L1." In the VerSe / CTSpine1K
label convention the export uses (export_hf.VERSE_TO_10CLASS):

    1-7  C1-C7      8-19  T1-T12      20-25  L1-L6      26  sacrum      28  T13

so **label 19 = T12 = the vertebra above L1 = the anchor**, and the
supernumerary case the limited-FOV CT can't adjudicate is already encoded as
**label 28 (T13)** and **L6 = 25**. The export keeps only 20-26, dropping the
thoracic labels — but the placed spine masks (NN-resampled into CT space, BEFORE
that remap) still carry them. This scans those masks and reports, across the
dataset:

  * how often T12 (19) is present  -> how free the anchor is (just relabel it)
  * how often T13 (28) is present  -> supernumerary cases CTSpine1K already called
  * how often L6  (25) is present  -> lumbarization already labelled
  * the full raw-label histogram   -> sanity on the scheme

Presence of label 19 in a PLACED mask means T12 is both labelled AND in the CT
FOV (placement only keeps what overlaps the volume).

Run inside the project container (needs numpy + nibabel):

  singularity exec --bind "$(pwd):/workspace,$DATA_DIR:/data" "$SIF_PATH" \
    python3 /workspace/scripts/check_spine_labels.py --spine_dir /data/placed/spine
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np      # noqa: E402
import nibabel as nib   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("check_spine_labels")

# VerSe / CTSpine1K names for the values that matter here.
VERSE_NAMES = {19: "T12 (anchor)", 28: "T13", 20: "L1", 24: "L5",
               25: "L6", 26: "sacrum"}


def _labels_in(path_str: str) -> Optional[Tuple[str, list]]:
    """Worker (process-pool, so it takes/returns picklable plain types)."""
    try:
        arr = np.asarray(nib.load(path_str).dataobj)
        vals = sorted(int(v) for v in np.unique(arr) if v != 0)
        return Path(path_str).name, vals
    except Exception:                                    # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine_dir", required=True, type=Path,
                    help="dir of placed spine masks (e.g. data/placed/spine).")
    ap.add_argument("--glob", default="*_seg_placed.nii.gz",
                    help="filename glob (default placed spine masks).")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or os.cpu_count() or 16),
                    help="process-pool size (default: $SLURM_CPUS_PER_TASK or all cores).")
    a = ap.parse_args()

    files = sorted(a.spine_dir.glob(a.glob))
    if not files:
        files = sorted(a.spine_dir.glob("*.nii.gz"))
    if not files:
        log.error("no masks under %s", a.spine_dir)
        return 1
    total = len(files)
    log.info("scanning %d spine masks under %s  (workers=%d)",
             total, a.spine_dir, a.workers)

    hist: Counter = Counter()          # label value -> # masks containing it
    n_ok = 0
    n_t12 = n_t13 = n_l6 = n_l1 = 0
    remapped_seen = False
    t0 = time.time()
    step = max(1, total // 40)         # ~40 progress lines
    done = 0
    # Processes, not threads: nib load + np.unique is CPU-bound (GIL). Stream
    # results as they finish so progress is visible on a long pool.
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_labels_in, str(p)) for p in files]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            res = fut.result()
            if res is not None:
                _, vals = res
                n_ok += 1
                for v in vals:
                    hist[v] += 1
                n_t12 += int(19 in vals)
                n_t13 += int(28 in vals)
                n_l6 += int(25 in vals)
                n_l1 += int(20 in vals)
                if vals and max(vals) <= 10:
                    remapped_seen = True
            if done % step == 0 or done == total:
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                log.info("  %5d/%-5d (%4.1f%%)  %.1f/s  elapsed %.0fs  eta %.0fs"
                         "   [T12 so far %d]",
                         done, total, 100 * done / total, rate, el, eta, n_t12)

    log.info("=" * 60)
    log.info("SPINE-LABEL CHECK   masks read: %d / %d", n_ok, len(files))
    log.info("-" * 60)
    if remapped_seen and not any(v >= 20 for v in hist):
        log.info("WARNING: labels are all <=10 — this dir holds the REMAPPED "
                 "scheme (thoracic already dropped). Point --spine_dir at the "
                 "PLACED masks (data/placed/spine) to see T12/T13.")
    pct = lambda n: 100 * n / max(n_ok, 1)               # noqa: E731
    log.info("L1  (20) present : %5d  (%.1f%%)   <- sanity, expect ~all", n_l1, pct(n_l1))
    log.info("T12 (19) present : %5d  (%.1f%%)   <- ANCHOR free from GT", n_t12, pct(n_t12))
    log.info("T13 (28) present : %5d  (%.1f%%)   <- supernumerary, already called", n_t13, pct(n_t13))
    log.info("L6  (25) present : %5d  (%.1f%%)   <- lumbarization, already labelled", n_l6, pct(n_l6))
    log.info("-" * 60)
    log.info("Full raw-label histogram (value: #masks  name):")
    for v in sorted(hist):
        log.info("  %3d : %5d   %s", v, hist[v], VERSE_NAMES.get(v, ""))
    log.info("=" * 60)
    log.info("READ: high T12(19)%% => the anchor is mostly FREE — retain 19->class"
             " 11, no re-segmentation. Where 19 is absent (T12 out of FOV or "
             "unlabelled), it becomes the actual annotation work. T13(28)/L6(25)"
             " counts are the transitional cases CTSpine1K already adjudicated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
