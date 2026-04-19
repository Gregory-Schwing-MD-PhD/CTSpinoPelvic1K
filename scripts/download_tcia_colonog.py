"""
download_tcia_colonog.py — Download TCIA CT COLONOG series.

Two modes:
  --scope all       All ~3,451 CT COLONOG series       (default)
  --scope filtered  Only the ~1,194 patients annotated in CTPelvic1K COLONOG
                    (reads a series-list from the pelvic dataset or a provided
                    JSON of canonical PatientIDs).

Output tree: one subfolder per SeriesInstanceUID containing its DICOMs.
  out_dir/{series_uid}/*.dcm

Implementation: tcia_utils.getSeries() + getImageSRIES for DICOM bytes.
Retry logic: exponential backoff up to 3 tries per series.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.download_tcia")

COLLECTION = "CT COLONOGRAPHY"


def _get_tcia_client():
    try:
        from tcia_utils import nbia
    except ImportError:
        log.error("tcia_utils not installed.  pip install tcia_utils")
        sys.exit(1)
    return nbia


def _series_already_downloaded(series_dir: Path) -> bool:
    if not series_dir.is_dir():
        return False
    return any(series_dir.glob("*.dcm"))


def _download_one_series(series_uid: str, out_dir: Path, max_retry: int = 3) -> bool:
    nbia = _get_tcia_client()
    series_dir = out_dir / series_uid
    if _series_already_downloaded(series_dir):
        return True

    series_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retry + 1):
        try:
            nbia.downloadSeries(
                series_data=[{"SeriesInstanceUID": series_uid}],
                path=str(out_dir),
                format="df",
                csv_filename=None,
            )
            if _series_already_downloaded(series_dir):
                return True
            dl_dirs = [d for d in out_dir.iterdir()
                       if d.is_dir() and series_uid in d.name]
            if dl_dirs and dl_dirs[0] != series_dir:
                for f in dl_dirs[0].glob("*"):
                    f.rename(series_dir / f.name)
                dl_dirs[0].rmdir()
                if _series_already_downloaded(series_dir):
                    return True
        except Exception as exc:
            log.warning("  [attempt %d/%d] %s: %s",
                        attempt, max_retry, series_uid, str(exc)[:200])
        time.sleep(2 ** attempt + random.random())

    return False


def _all_colonog_series() -> List[str]:
    nbia = _get_tcia_client()
    log.info("Querying TCIA for all CT COLONOGRAPHY CT series ...")
    df = nbia.getSeries(collection=COLLECTION, modality="CT", format="df")
    uids = df["SeriesInstanceUID"].astype(str).tolist()
    log.info("  → %d series", len(uids))
    return uids


def _filter_by_patient_uids(all_uids: List[str], wanted: Set[str]) -> List[str]:
    nbia = _get_tcia_client()
    log.info("Filtering %d series against %d target PatientIDs ...",
             len(all_uids), len(wanted))
    df = nbia.getSeries(collection=COLLECTION, modality="CT", format="df")
    df_filt = df[df["PatientID"].astype(str).str.strip().isin(wanted)]
    uids = df_filt["SeriesInstanceUID"].astype(str).tolist()
    log.info("  → %d series matching target patients", len(uids))
    return uids


def _load_wanted_patient_uids(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    if not path.exists():
        log.error("PatientID list not found: %s", path)
        sys.exit(1)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {str(x).strip() for x in data}
    if isinstance(data, dict):
        if "patient_uids" in data:
            return {str(x).strip() for x in data["patient_uids"]}
        return {str(k).strip() for k in data.keys()}
    log.error("Unrecognised patient list format in %s", path)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Target root — each series gets its own subfolder")
    p.add_argument("--scope", choices=("all", "filtered"), default="all",
                   help="'all' downloads every CT COLONOG series; "
                        "'filtered' restricts to --patient_list PatientIDs")
    p.add_argument("--patient_list", type=Path, default=None,
                   help="JSON list of canonical PatientIDs (required if --scope filtered)")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download threads (default 8)")
    p.add_argument("--max_series", type=int, default=0,
                   help="Cap total series downloaded (0 = no cap). Useful for testing")
    p.add_argument("--dry_run", action="store_true",
                   help="Print how many series would be downloaded and exit")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.scope == "filtered" and args.patient_list is None:
        log.error("--scope filtered requires --patient_list")
        sys.exit(1)

    if args.scope == "all":
        uids = _all_colonog_series()
    else:
        wanted = _load_wanted_patient_uids(args.patient_list)
        all_uids = _all_colonog_series()
        uids = _filter_by_patient_uids(all_uids, wanted)

    if args.max_series > 0:
        uids = uids[:args.max_series]
        log.info("Capped to first %d series", len(uids))

    already = sum(1 for u in uids if _series_already_downloaded(args.out_dir / u))
    log.info("Series to evaluate: %d   already downloaded: %d   remaining: %d",
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
                         i, len(uids), n_ok, n_fail, elapsed, eta, 1/max(rate,1e-6))

    log.info("=" * 60)
    log.info("Download complete: ok=%d  fail=%d  total=%d",
             n_ok, n_fail, len(uids))
    log.info("Output root: %s", args.out_dir)
    if n_fail:
        log.warning("Re-run the same command to retry failed series.")


if __name__ == "__main__":
    main()
