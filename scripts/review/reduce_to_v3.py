"""
scripts/review/reduce_to_v3.py — fold finalized reviews into a v3 tree.

Takes the v2 export (pseudo) + a finalized-reviews index and produces the
v3 export (pseudo_corrected): manual labels untouched, reviewed regions
flipped pseudo -> pseudo_corrected, corrected label NIfTIs swapped in,
rejected cases dropped. The result stages/pushes through the SAME machinery
as v1/v2:  make hf-stage(implicit, already a tree) -> HF_REVISION=v3 make hf-push.

The provenance/record reduction is a PURE function (apply_reviews_to_records)
and is unit-tested; main() is just the file I/O around it.

finalized-reviews index (JSON), one entry per finalized case_id:
  { "<token>__<config>": {
        "decision": "corrected" | "accept" | "reject",
        "prov_after": {"spine": "...", "pelvis": "..."},
        "label_rel": "reviews/<...>/final_label.nii.gz",   # omitted for accept/reject
        "final_review_id": "..." }, ... }
`accept` keeps the v2 label as-is (only provenance flips); `corrected`
swaps in `label_rel`; `reject` excludes the case from v3.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.review.reduce")


def apply_reviews_to_records(
    records: List[dict], finals: Dict[str, dict]
) -> Tuple[List[dict], Dict[str, str], List[str]]:
    """Pure reduction: v2 records + finalized index -> v3 records.

    Returns (new_records, label_swaps, dropped):
      new_records  — v3 manifest records (rejected cases removed; reviewed
                     regions' prov flipped per the stored prov_after).
      label_swaps  — {label_file_rel: final_label_rel} for cases whose
                     decision == "corrected" (caller copies these in).
      dropped      — case_ids excluded (decision == "reject").

    Records for cases not in `finals` pass through unchanged (not yet
    reviewed / out of scope). manual provenance is never altered.
    """
    new_records: List[dict] = []
    label_swaps: Dict[str, str] = {}
    dropped: List[str] = []

    for rec in records:
        cid = schema.case_id(rec.get("token"), rec.get("config"))
        f = finals.get(cid)
        if not f:
            new_records.append(rec)
            continue
        if f.get("decision") == "reject":
            dropped.append(cid)
            continue
        r = dict(rec)
        pa = f.get("prov_after") or {}
        if "spine" in pa:
            r["prov_spine"] = pa["spine"]
        if "pelvis" in pa:
            r["prov_pelvis"] = pa["pelvis"]
        if f.get("decision") == "corrected" and f.get("label_rel") \
                and r.get("label_file"):
            label_swaps[r["label_file"]] = f["label_rel"]
        new_records.append(r)
    return new_records, label_swaps, dropped


def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", required=True, type=Path,
                    help="v2 export tree (data/hf_export_v2).")
    ap.add_argument("--finals", required=True, type=Path,
                    help="finalized-reviews index JSON.")
    ap.add_argument("--labels_root", type=Path, default=None,
                    help="Local root the index's label_rel paths resolve "
                         "against (e.g. a pulled review repo).")
    ap.add_argument("--out", required=True, type=Path,
                    help="v3 tree to create (data/hf_export_v3).")
    ap.add_argument("--labels_only", action="store_true",
                    help="skip copying CT volumes — labels + manifest only "
                         "(fast, and enough to run QC on the corrected labels; "
                         "do a full run before pushing the release).")
    args = ap.parse_args()

    import nibabel as nib  # noqa: F401  (validate availability early)

    v2, out = args.v2, args.out
    records = _load_manifest(v2 / "manifest.json")
    finals = json.loads(args.finals.read_text())
    labels_root = args.labels_root or args.finals.parent

    new_records, label_swaps, dropped = apply_reviews_to_records(records, finals)
    log.info("reduce: %d v2 records -> %d v3 records (corrected=%d, "
             "dropped/excluded=%d)", len(records), len(new_records),
             len(label_swaps), len(dropped))

    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        if (v2 / extra).exists():
            shutil.copy2(str(v2 / extra), str(out / extra))

    n_ct = n_lbl = n_swap = 0
    for rec in new_records:
        rels = ([rec.get("label_file")] if args.labels_only
                else [rec.get("ct_file"), rec.get("label_file")])
        for rel in rels:
            if rel and (v2 / rel).exists() and not (out / rel).exists():
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(v2 / rel), str(out / rel))
                n_ct += rel.startswith("ct/")
                n_lbl += rel.startswith("labels/")
    for label_rel, final_rel in label_swaps.items():
        src = labels_root / final_rel
        if not src.exists():
            log.warning("corrected label missing: %s — keeping v2 label", src)
            continue
        (out / label_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(out / label_rel))
        n_swap += 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
    from export_hf import write_manifest
    write_manifest([{**r, "ok": True} for r in new_records], out)

    log.info("=" * 60)
    log.info("v3 tree -> %s", out)
    log.info("  records      : %d", len(new_records))
    log.info("  ct/labels    : %d / %d copied", n_ct, n_lbl)
    log.info("  corrected    : %d labels swapped in", n_swap)
    log.info("  excluded     : %d", len(dropped))
    log.info("=" * 60)
    log.info("Publish:  HF_TOKEN=... HF_REPO_ID=org/Name HF_REVISION=v3 \\")
    log.info("            HF_EXPORT_DIR=%s make hf-push", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
