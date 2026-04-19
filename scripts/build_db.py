"""
build_db.py — Orchestrates the full PatientDB build pipeline.

PIPELINE
--------
Step 1  Index TCIA series dirs by PatientID (ground truth)
Step 2  Parse spine mask files (patient UID + geometry from filename/NIfTI)
Step 3  Parse pelvic mask files (patient UID + metadata from filename/NIfTI)
Step 4  Match masks to patients — patient identity check
Step 5  Assign ranked series candidates within each patient
Step 6  Cross-check assignments (NZ bounds, SeriesNumber, etc.)
Step 7  Determine fusion status for complete patients
Step 8  Compute stats and serialize DB

PATIENT IDENTITY GUARANTEE
---------------------------
A mask is only associated with a patient when:
  canonical_uid(mask_filename_uid) == canonical_uid(DICOM_PatientID)

This is a string equality check — not an affine match, not a heuristic.
Masks whose patient UID has no matching TCIA series are flagged UNRESOLVED.

OUTPUTS
-------
data/patient_db.json              full DB (audit + inspect)
data/patient_db.pkl               fast-load for downstream code
data/patient_db_summary.txt       human-readable summary
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from patient_db import (
    Confidence,
    CONFIDENCE_SCORES,
    DBMetadata,
    FusionStatus,
    PatientDB,
    PatientRecord,
    canonical_uid,
    patient_token,
)
from tcia_index     import build_tcia_patient_index
from mask_index     import parse_spine_masks, parse_pelvic_masks
from series_assigner import (
    assign_all_candidates,
    cross_check_patient,
    determine_fusion_status,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_patient_records(
    tcia_index:   Dict,
    study_groups: Dict,
    spine_masks:  List,
    pelvic_masks: List,
    workers:      int,
) -> Dict[str, PatientRecord]:
    all_uids = set(tcia_index.keys())
    for m in spine_masks:
        all_uids.add(m.patient_uid)
    for m in pelvic_masks:
        all_uids.add(m.patient_uid)

    patients: Dict[str, PatientRecord] = {}
    for uid in sorted(all_uids, key=lambda u: (0, int(u.rsplit(".", 1)[-1]))
                      if not u.startswith("CTC") and u.rsplit(".", 1)[-1].isdigit()
                      else (1, u)):
        tok = patient_token(uid)
        patients[uid] = PatientRecord(
            patient_uid   = uid,
            patient_token = tok,
            tcia_series   = tcia_index.get(uid, []),
            studies       = study_groups.get(uid, []),
        )

    n_spine_orphan  = 0
    n_pelvic_orphan = 0

    for m in spine_masks:
        uid = m.patient_uid
        if uid not in patients:
            log.warning("Spine mask has unknown patient UID: %s  (file: %s)",
                        uid, Path(m.mask_file).name)
            n_spine_orphan += 1
            continue
        patients[uid].spine_masks.append(m)

    for m in pelvic_masks:
        uid = m.patient_uid
        if uid not in patients:
            log.warning("Pelvic mask has unknown patient UID: %s  (file: %s)",
                        uid, Path(m.mask_file).name)
            n_pelvic_orphan += 1
            continue
        patients[uid].pelvic_masks.append(m)

    if n_spine_orphan or n_pelvic_orphan:
        log.warning("Orphan masks (no TCIA match): spine=%d  pelvic=%d",
                    n_spine_orphan, n_pelvic_orphan)

    log.info("Assigning series candidates for %d patients ...", len(patients))
    n_assigned = n_unresolved = 0
    all_warnings: List[str] = []

    for uid, rec in patients.items():
        if not rec.spine_masks and not rec.pelvic_masks:
            continue

        series = rec.tcia_series
        if not series:
            for m in rec.spine_masks + rec.pelvic_masks:
                mask_name = Path(m.mask_file).name
                log.warning(
                    "No TCIA series found for patient uid=%s  token=%s  mask=%s",
                    uid, rec.patient_token, mask_name,
                )
            n_unresolved += len(rec.spine_masks) + len(rec.pelvic_masks)
            continue

        assign_all_candidates(rec, series)
        n_assigned += len(rec.spine_masks) + len(rec.pelvic_masks)

        warns = cross_check_patient(rec, series)
        for w in warns:
            msg = f"token={rec.patient_token}  {w}"
            all_warnings.append(msg)
            log.warning("  CROSS-CHECK: %s", msg)

    log.info("Series assignment: %d masks assigned  %d unresolved",
             n_assigned, n_unresolved)

    log.info("Determining fusion status for complete patients ...")
    for rec in patients.values():
        if rec.is_complete:
            status, reasons = determine_fusion_status(rec)
            rec.fusion_status  = status
            rec.fusion_reasons = reasons

    return patients


def compute_metadata(
    patients:     Dict[str, PatientRecord],
    tcia_dir:     Path,
    spine_root:   Path,
    pelvis_root:  Path,
) -> DBMetadata:
    conf_counts: Counter = Counter()

    for rec in patients.values():
        for m in rec.spine_masks + rec.pelvic_masks:
            conf_counts[m.best_confidence] += 1

    fusion_counts = Counter(
        rec.fusion_status for rec in patients.values() if rec.is_complete
    )

    n_ambiguous_or_unresolved = sum(
        1 for rec in patients.values()
        for m in rec.spine_masks + rec.pelvic_masks
        if m.best_confidence in (Confidence.AMBIGUOUS, Confidence.UNRESOLVED)
    )

    return DBMetadata(
        n_patients             = len(patients),
        n_tcia_series_total    = sum(len(r.tcia_series) for r in patients.values()),
        n_spine_masks          = sum(len(r.spine_masks)  for r in patients.values()),
        n_pelvic_masks         = sum(len(r.pelvic_masks) for r in patients.values()),
        n_complete_patients    = sum(1 for r in patients.values() if r.is_complete),
        n_fusion               = fusion_counts.get(FusionStatus.FUSION,    0),
        n_separate             = fusion_counts.get(FusionStatus.SEPARATE,  0),
        n_ambiguous_assignment = n_ambiguous_or_unresolved,
        n_unresolved           = conf_counts.get(Confidence.UNRESOLVED, 0),
        confidence_breakdown   = dict(conf_counts),
        build_timestamp        = datetime.datetime.now().isoformat(),
        tcia_dir               = str(tcia_dir),
        spine_root             = str(spine_root),
        pelvis_root            = str(pelvis_root),
    )


def print_patient_table(patients: Dict[str, PatientRecord], max_rows: int = 30) -> None:
    hdr = (
        f"{'tok':>5}  {'#tcia':>5}  {'spine_conf':<20}  "
        f"{'pelvic_conf':<20}  {'fusion':<10}  {'warns'}"
    )
    print(hdr)
    print("-" * len(hdr))
    shown = 0
    for uid, rec in sorted(patients.items(),
                           key=lambda kv: (0, int(kv[1].patient_token))
                           if kv[1].patient_token.isdigit() else (1, kv[1].patient_token)):
        if shown >= max_rows:
            print(f"  ... and {len(patients) - shown} more (see JSON for full details)")
            break
        spine_conf  = rec.spine_masks[0].best_confidence  if rec.spine_masks  else "—"
        pelvic_conf = rec.pelvic_masks[0].best_confidence if rec.pelvic_masks else "—"
        fusion      = rec.fusion_status if rec.is_complete else "—"
        warns = []
        for m in rec.spine_masks + rec.pelvic_masks:
            for c in m.candidates[:1]:
                for r in c.reasons:
                    if any(k in r for k in ("MISMATCH", "OVERFLOW", "NZ_SHORT")):
                        warns.append(r[:40])
        warn_str = "; ".join(warns[:2]) if warns else "ok"
        print(
            f"{rec.patient_token:>5}  {rec.n_tcia_series:>5}  "
            f"{spine_conf:<20}  {pelvic_conf:<20}  {fusion:<10}  {warn_str}"
        )
        shown += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build CTSpinoPelvic1K PatientDB — patient-anchored mask→series resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--spine_root",  type=Path, required=True,
                   help="CTSpine1K root (contains rawdata/labels/COLONOG)")
    p.add_argument("--pelvis_root", type=Path, required=True,
                   help="CTPelvic1K root (contains masks/...)")
    p.add_argument("--tcia_dir",    type=Path, required=True,
                   help="TCIA download root")
    p.add_argument("--out_dir",     type=Path, default=Path("data"),
                   help="Output directory for patient_db.json/pkl")
    p.add_argument("--workers",     type=int,  default=32)
    p.add_argument("--rebuild_tcia_index", action="store_true")
    p.add_argument("--rebuild_mask_cache", action="store_true")
    p.add_argument("--debug_n",     type=int,  default=None,
                   help="Limit to first N masks per type (quick test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("build_db.py — CTSpinoPelvic1K PatientDB")
    log.info("  spine_root  : %s", args.spine_root)
    log.info("  pelvis_root : %s", args.pelvis_root)
    log.info("  tcia_dir    : %s", args.tcia_dir)
    log.info("  out_dir     : %s", args.out_dir)
    log.info("  workers     : %d", args.workers)
    log.info("=" * 70)

    for p in (args.spine_root, args.pelvis_root, args.tcia_dir):
        if not p.exists():
            log.error("Path not found: %s", p)
            sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 1: Building TCIA patient index ...")
    tcia_index, study_groups = build_tcia_patient_index(
        args.tcia_dir,
        workers       = args.workers,
        force_rebuild = args.rebuild_tcia_index,
    )
    log.info("  → %d unique patients in TCIA", len(tcia_index))

    log.info("Step 2: Parsing spine mask files ...")
    spine_masks = parse_spine_masks(
        args.spine_root,
        workers       = min(args.workers, 16),
        debug_n       = args.debug_n,
        rebuild_cache = args.rebuild_mask_cache,
    )
    log.info("  → %d spine masks parsed", len(spine_masks))

    log.info("Step 3: Parsing pelvic mask files ...")
    pelvic_masks = parse_pelvic_masks(
        args.pelvis_root,
        workers       = min(args.workers, 16),
        debug_n       = args.debug_n,
        rebuild_cache = args.rebuild_mask_cache,
    )
    log.info("  → %d pelvic masks parsed", len(pelvic_masks))

    log.info("Steps 4–7: Assembling patients and assigning candidates ...")
    patients = build_patient_records(tcia_index, study_groups, spine_masks, pelvic_masks, args.workers)
    log.info("  → %d patient records", len(patients))

    log.info("Step 8: Computing stats and serializing ...")
    metadata = compute_metadata(patients, args.tcia_dir, args.spine_root, args.pelvis_root)
    db       = PatientDB(patients=patients, metadata=metadata)

    json_path   = args.out_dir / "patient_db.json"
    pkl_path    = args.out_dir / "patient_db.pkl"
    summ_path   = args.out_dir / "patient_db_summary.txt"

    db.to_json(json_path)
    db.to_pickle(pkl_path)
    summ = db.summary()
    summ_path.write_text(summ)

    log.info("  JSON   : %s  (%.1f MB)", json_path, json_path.stat().st_size / 1e6)
    log.info("  Pickle : %s  (%.1f MB)", pkl_path,  pkl_path.stat().st_size  / 1e6)

    log.info("")
    print(summ)
    print()
    print("Resolution table (first 40 patients):")
    print_patient_table(patients, max_rows=40)

    problem_patients = [
        rec for rec in patients.values()
        if any(m.best_confidence in (Confidence.AMBIGUOUS, Confidence.UNRESOLVED)
               for m in rec.spine_masks + rec.pelvic_masks)
    ]
    if problem_patients:
        log.warning(
            "\n%d patients have AMBIGUOUS or UNRESOLVED assignments:", len(problem_patients)
        )
        for rec in problem_patients[:20]:
            for m in rec.spine_masks + rec.pelvic_masks:
                if m.best_confidence in (Confidence.AMBIGUOUS, Confidence.UNRESOLVED):
                    log.warning(
                        "  token=%-6s  type=%s  conf=%s  n_tcia_series=%d",
                        rec.patient_token,
                        "spine" if m in rec.spine_masks else "pelvic",
                        m.best_confidence,
                        rec.n_tcia_series,
                    )

    log.info("=" * 70)
    log.info("COMPLETE — next step: place_fused_masks.py (via slurm/create_dataset.sh)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
