"""
refresh_hf_manifests.py — Refresh manifest files in data/hf_export/ from
updated placed_manifest.json WITHOUT re-exporting NIfTIs.

Use case
--------
The parser bug fix in mask_index.py changed lstv_class for tokens 22 and 120
(class 3 -> class 2). The CT and label NIfTI files on disk are unchanged
(those depend on placement, not LSTV labels), so re-running export_hf.py
end-to-end would be ~800 wasted file copies.

This script:
  1. Reads the updated placed_manifest.json
  2. Reads existing per-record metadata from manifest.json (whatever
     export_hf.py wrote last time — ct_file, label_file, alignment_ok,
     n_lumbar_labels, has_l6, etc. all unchanged by the parser fix)
  3. Updates the LSTV fields on each record from the new placed_manifest
  4. Re-writes manifest.json, manifest.csv, manifest_train/val/test.json,
     data_splits.json, splits/test.json, splits_summary.json

Does NOT touch ct/, labels/, qc/ subdirectories. Cheap and fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Use export_hf.py's helpers so behavior matches end-to-end run.
sys.path.insert(0, str(Path(__file__).parent))
from export_hf import (
    write_manifest,
    write_splits,
    _OPTIONAL_NUMERIC_FIELDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("refresh_hf_manifests")


# Mirror export_hf.py build_work LSTV resolution exactly so refreshed
# records match what an end-to-end run would have produced.
_CLS_TO_LABEL = {
    0: "normal",
    1: "LUMBARIZATION",
    2: "SEMI_SACRALIZATION",
    3: "SACRALIZATION",
    4: "SACRALIZATION",
}


def _resolve_lstv_label(case: dict) -> str:
    """Mirror of build_work's lstv resolution from export_hf.py."""
    cls = int(case.get("lstv_class", 0) or 0)
    if cls > 0:
        return _CLS_TO_LABEL.get(cls, "normal")

    pel = (case.get("lstv_pelvic") or "").strip()
    vert = (case.get("lstv_vertebral") or "").strip()
    if pel and pel.lower() not in ("unknown", "", "normal"):
        return pel
    if vert and vert.lower() not in ("unknown", "", "normal"):
        return vert
    return "normal"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placed_manifest", required=True, type=Path,
                    help="Updated placed_manifest.json with corrected lstv_class")
    ap.add_argument("--hf_dir", required=True, type=Path,
                    help="data/hf_export/ — the dir to refresh in-place")
    args = ap.parse_args()

    if not args.placed_manifest.exists():
        log.error("placed_manifest not found: %s", args.placed_manifest)
        sys.exit(1)
    existing_manifest = args.hf_dir / "manifest.json"
    if not existing_manifest.exists():
        log.error("existing manifest.json not found: %s", existing_manifest)
        log.error("run export_hf.py end-to-end first to populate hf_dir")
        sys.exit(1)

    placed = json.loads(args.placed_manifest.read_text())
    placed_by_token = {}
    for c in placed.get("cases", []):
        tok = str(c.get("patient_token", ""))
        if tok:
            placed_by_token[tok] = c
    log.info("Loaded placed_manifest: %d cases  schema=%s",
             len(placed_by_token), placed.get("schema_version", "?"))

    existing_records = json.loads(existing_manifest.read_text())
    log.info("Loaded existing manifest.json: %d records", len(existing_records))

    n_updated = 0
    n_class_changed = 0
    refreshed = []
    for rec in existing_records:
        tok = str(rec.get("token", ""))
        case = placed_by_token.get(tok)
        if not case:
            log.warning("token=%s present in HF manifest but not in placed_manifest",
                        tok)
            refreshed.append(rec)
            continue

        old_class = int(rec.get("lstv_class", 0) or 0)
        new_class = int(case.get("lstv_class", 0) or 0)
        new_label = _resolve_lstv_label(case)

        rec_new = dict(rec)
        rec_new["lstv_class"] = new_class
        rec_new["lstv_label"] = new_label
        rec_new["lstv_pelvic"] = case.get("lstv_pelvic", "") or ""
        rec_new["lstv_vertebral"] = case.get("lstv_vertebral", "") or ""
        rec_new["lstv_agreement"] = case.get("lstv_agreement")
        rec_new["lstv_confusion_zone"] = bool(case.get("lstv_confusion_zone", False))

        # Mark ok=True so write_manifest/write_splits keep the record.
        # The parser fix doesn't invalidate anything; previously-failed
        # exports stay failed (ok stays False from the prior run).
        rec_new["ok"] = rec.get("ok", True)
        rec_new["error"] = rec.get("error")

        refreshed.append(rec_new)
        n_updated += 1
        if old_class != new_class:
            n_class_changed += 1
            log.info("  token=%s lstv_class %d -> %d  label=%s",
                     tok, old_class, new_class, new_label)

    log.info("Updated %d/%d records; lstv_class changed on %d records",
             n_updated, len(existing_records), n_class_changed)

    log.info("Re-writing manifest.json + manifest.csv")
    write_manifest(refreshed, args.hf_dir)

    log.info("Re-writing manifest_train/validation/test.json + splits")
    write_splits(refreshed, args.hf_dir)

    log.info("Done. data/hf_export/ refreshed.")


if __name__ == "__main__":
    main()
