"""
download_tcia_colonog.py — Download TCIA CT COLONOG series.

Downloads every CT series in the CT COLONOGRAPHY collection (~3,451 series)
into one subfolder per SeriesInstanceUID containing its DICOMs:

    out_dir/{series_uid}/*.dcm

Implementation: tcia_utils.getSeries() + downloadSeries(), parallelised via
ThreadPoolExecutor.  Retry logic: exponential backoff up to 3 tries per series.

Notes on tcia_utils quirks handled here:

    1) downloadSeries() SILENTLY no-ops when its target series directory
       already exists — even if empty.  So we must NOT pre-create the
       series directory, and we must rmtree any pre-existing (partial) dir
       before each attempt so tcia_utils can populate it cleanly.

    2) A connection drop mid-series can leave a partial DICOM set on disk.
       We therefore verify dcm_count >= ImageCount from the getSeries()
       response before declaring success.  Series missing from the metadata
       fall back to "any .dcm present" success.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.download_tcia")

COLLECTION = "CT COLONOGRAPHY"

# Populated in main() from TCIA getSeries() metadata; used by the worker
# to verify completeness after each download attempt.
_EXPECTED_COUNTS: Dict[str, int] = {}


def _get_tcia_client():
    try:
        from tcia_utils import nbia
    except ImportError:
        log.error("tcia_utils not installed.  pip install tcia_utils")
        sys.exit(1)
    return nbia


def _dcm_count(series_dir: Path) -> int:
    if not series_dir.is_dir():
        return 0
    return sum(1 for _ in series_dir.glob("*.dcm"))


def _series_complete(series_uid: str, series_dir: Path) -> bool:
    """
    True iff the series directory holds at least the expected number of
    DICOMs (from TCIA metadata), or — if no expected count is known — at
    least one DICOM.
    """
    n_have = _dcm_count(series_dir)
    if n_have == 0:
        return False
    n_expect = _EXPECTED_COUNTS.get(series_uid, 0)
    if n_expect > 0:
        return n_have >= n_expect
    return True  # unknown expectation: accept any non-empty dir


def _download_one_series(series_uid: str, out_dir: Path, max_retry: int = 3) -> bool:
    nbia = _get_tcia_client()
    series_dir = out_dir / series_uid

    # Fast path: already complete per expected DICOM count.
    if _series_complete(series_uid, series_dir):
        return True

    # CRITICAL: tcia_utils.downloadSeries() silently no-ops when the target
    # series directory already exists (even if empty or partial).  Any
    # pre-existing dir here will cause every retry to skip the download and
    # report failure with no error.  Clear it before each attempt.
    #
    # Do NOT mkdir series_dir — tcia_utils creates it itself during download.
    if series_dir.exists():
        shutil.rmtree(series_dir)

    for attempt in range(1, max_retry + 1):
        try:
            nbia.downloadSeries(
                series_data=[{"SeriesInstanceUID": series_uid}],
                path=str(out_dir),
                format="df",
                csv_filename=None,
            )
            if _series_complete(series_uid, series_dir):
                return True

            # Some versions of tcia_utils drop the download into a
            # slightly-renamed folder; move it into place if so.
            dl_dirs = [d for d in out_dir.iterdir()
                       if d.is_dir() and series_uid in d.name]
            if dl_dirs and dl_dirs[0] != series_dir:
                for f in dl_dirs[0].glob("*"):
                    f.rename(series_dir / f.name)
                dl_dirs[0].rmdir()
                if _series_complete(series_uid, series_dir):
                    return True

            # Log partial-download case so we know why the retry is happening.
            n_have   = _dcm_count(series_dir)
            n_expect = _EXPECTED_COUNTS.get(series_uid, 0)
            if n_have and n_expect:
                log.warning("  [attempt %d/%d] %s: partial %d/%d DICOMs",
                            attempt, max_retry, series_uid, n_have, n_expect)
        except Exception as exc:
            log.warning("  [attempt %d/%d] %s: %s",
                        attempt, max_retry, series_uid, str(exc)[:200])

        # Clear whatever partial/empty dir the failed attempt left behind
        # so the next retry starts clean (see no-op bug above).
        if series_dir.exists():
            shutil.rmtree(series_dir)

        time.sleep(2 ** attempt + random.random())

    return False


def _all_colonog_series_df():
    nbia = _get_tcia_client()
    log.info("Querying TCIA for all CT COLONOGRAPHY CT series ...")
    df = nbia.getSeries(collection=COLLECTION, modality="CT", format="df")
    log.info("  -> %d series", len(df))
    return df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Target root — each series gets its own subfolder")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download threads (default 8)")
    p.add_argument("--max_series", type=int, default=0,
                   help="Cap total series downloaded (0 = no cap). Useful for testing")
    p.add_argument("--dry_run", action="store_true",
                   help="Print how many series would be downloaded and exit")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pull the full metadata DataFrame so we can extract both SeriesInstanceUID
    # and the canonical ImageCount for completeness verification.
    df = _all_colonog_series_df()
    uids: List[str] = df["SeriesInstanceUID"].astype(str).tolist()

    if "ImageCount" in df.columns:
        global _EXPECTED_COUNTS
        _EXPECTED_COUNTS = {
            str(row["SeriesInstanceUID"]): int(row["ImageCount"])
            for _, row in df.iterrows()
            if row.get("ImageCount") and int(row["ImageCount"]) > 0
        }
        log.info("Expected DICOM counts loaded for %d / %d series",
                 len(_EXPECTED_COUNTS), len(uids))
    else:
        log.warning("getSeries() response has no ImageCount column — "
                    "completeness will fall back to 'any DICOM present'.")

    if args.max_series > 0:
        uids = uids[:args.max_series]
        log.info("Capped to first %d series", len(uids))

    already = sum(1 for u in uids if _series_complete(u, args.out_dir / u))
    log.info("Series to evaluate: %d   already complete: %d   remaining: %d",
             len(uids), already, len(uids) - already)

    if args.dry_run:
        log.info("--dry_run set, exiting without downloading.")
        return

    if len(uids) - already == 0:
        log.info("Everything already present — nothing to do.")
        return

    t0 = time.time()
    n_ok = n_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one_series, u, args.out_dir): u for u in uids}
        for i, fut in enumerate(as_completed(futs), 1):
            uid = futs[fut]
            try:
                ok = fut.result()
            except Exception as exc:
                log.warning("  crash %s: %s", uid, exc)
                ok = False
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                log.warning("  FAIL: %s", uid)
            if i % 25 == 0 or i == len(uids):
                elapsed = time.time() - t0
                rate    = i / max(elapsed, 1e-6)
                eta     = (len(uids) - i) / max(rate, 1e-6)
                log.info("  [%d/%d]  ok=%d  fail=%d  elapsed=%.0fs  ETA=%.0fs  (%.1f s/series)",
                         i, len(uids), n_ok, n_fail, elapsed, eta, 1/max(rate, 1e-6))

    log.info("=" * 60)
    log.info("Download complete: ok=%d  fail=%d  total=%d",
             n_ok, n_fail, len(uids))
    log.info("Output root: %s", args.out_dir)
    if n_fail:
        log.warning("Re-run the same command to retry failed series.")


if __name__ == "__main__":
    main()
