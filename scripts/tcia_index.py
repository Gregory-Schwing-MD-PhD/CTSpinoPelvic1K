"""
tcia_index.py — Index all TCIA series directories, grouped by PatientID.

GROUND TRUTH
------------
The DICOM PatientID tag is the authoritative patient identifier.  Every TCIA
series directory contains DICOMs whose PatientID tag encodes the canonical
patient UID (e.g. "1.3.6.1.4.1.9328.50.4.17").

This module reads one DICOM header per series directory (fast — no pixel data),
extracts PatientID, canonicalizes it, and groups all series by patient.

The result is a Dict[str, List[TciaSeriesRecord]] where the key is the
canonical patient UID.  This grouping is used downstream to restrict series
candidate search to the correct patient, making it impossible to assign a mask
to the wrong patient's series.

SPATIAL RECONSTRUCTION
----------------------
Uses cross-product slice normal from IOP (immune to filename sort-order issues)
with IPP-derived slice spacing from adjacent DICOMs (matches dcm2niix).
Caches to .tcia_patient_index.json for fast subsequent loads (~0.5s vs ~12s).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from patient_db import (
    TciaSeriesRecord,
    SpatialInfo,
    PATIENT_POSITION_MAP,
    canonical_uid,
    patient_token,
)

log = logging.getLogger(__name__)

try:
    import numpy as np
    import pydicom
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    log.error("pydicom + numpy required for TCIA indexing.")

_CACHE_FNAME = ".tcia_patient_index.json"
_ANY_TCIA_RE = re.compile(r"^1\.3\.6\.1\.4\.1\.9328\.50\.(4|81)\.")
_MIN_CT_DCM  = 50

_SCOUT_KEYWORDS = (
    "scout", "topogram", "localizer", "surview", "scoutview",
    "scanogram", "survey", "loc ", "loc_",
)


def _is_scout(desc: str) -> bool:
    s = desc.lower()
    return any(k in s for k in _SCOUT_KEYWORDS)


def _read_spatial(dcm_files: List[Path], n: int) -> Optional[SpatialInfo]:
    """Reconstruct RAS spatial info from DICOM headers."""
    try:
        mid_idx = n // 2
        ds_mid  = pydicom.dcmread(str(dcm_files[mid_idx]),
                                  stop_before_pixels=True, force=True)

        ipp = [float(x) for x in ds_mid.ImagePositionPatient]
        iop = [float(x) for x in ds_mid.ImageOrientationPatient]
        ps  = [float(x) for x in ds_mid.PixelSpacing]

        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:])
        normal  = np.cross(row_cos, col_cos)
        nrm_len = np.linalg.norm(normal)
        if nrm_len < 1e-9:
            return None
        normal /= nrm_len

        slice_sp: Optional[float] = None
        if n >= 2:
            adj_idx = (mid_idx + 1) % n
            ds_adj  = pydicom.dcmread(str(dcm_files[adj_idx]),
                                      stop_before_pixels=True, force=True)
            ipp_adj = np.array([float(x) for x in ds_adj.ImagePositionPatient])
            proj    = abs(float(np.dot(ipp_adj - np.array(ipp), normal)))
            if proj > 1e-9:
                slice_sp = proj

        if slice_sp is None:
            for tag in ("SpacingBetweenSlices", "SliceThickness"):
                val = getattr(ds_mid, tag, None)
                if val is not None:
                    v = float(val)
                    if v > 1e-9:
                        slice_sp = v
                        break

        if not slice_sp:
            return None

        aff_lps = np.eye(4, dtype=float)
        aff_lps[:3, 0] = row_cos * ps[1]
        aff_lps[:3, 1] = col_cos * ps[0]
        aff_lps[:3, 2] = normal  * slice_sp
        aff_lps[:3, 3] = ipp

        aff_ras        = aff_lps.copy()
        aff_ras[:2, :] *= -1

        spacing = np.array([np.linalg.norm(aff_ras[:3, i]) for i in range(3)])
        if np.any(spacing < 1e-9):
            return None
        direction = aff_ras[:3, :3] / spacing[np.newaxis, :]

        return SpatialInfo(
            origin    = aff_ras[:3, 3].tolist(),
            spacing   = spacing.tolist(),
            direction = direction.flatten().tolist(),
            nz        = n,
        )
    except Exception as exc:
        log.debug("_read_spatial failed: %s", exc)
        return None


def _read_one_series_dir(d: Path) -> Optional[TciaSeriesRecord]:
    dcm_files = sorted(d.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(d.rglob("*.dcm"))
    n = len(dcm_files)
    if not dcm_files:
        return None

    try:
        mid_idx = n // 2
        ds = pydicom.dcmread(str(dcm_files[mid_idx]),
                             stop_before_pixels=True, force=True)

        pid         = str(getattr(ds, "PatientID",         "") or "").strip()
        series_uid  = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        study_uid   = str(getattr(ds, "StudyInstanceUID",  "") or "").strip()
        series_num  = int(getattr(ds, "SeriesNumber",      0)  or 0)
        raw_pos     = str(getattr(ds, "PatientPosition",   "") or "").strip().upper()
        series_desc = str(getattr(ds, "SeriesDescription", "") or "").strip()

        if not pid or not series_uid:
            return None

        canon_pid = canonical_uid(pid)
        tok       = patient_token(canon_pid)
        position  = PATIENT_POSITION_MAP.get(raw_pos, "UNKNOWN")
        scout     = _is_scout(series_desc)
        ct_qual   = n >= _MIN_CT_DCM and not scout

        spatial = _read_spatial(dcm_files, n) if not scout else None

        return TciaSeriesRecord(
            patient_uid        = canon_pid,
            patient_token      = tok,
            series_uid         = canonical_uid(series_uid),
            study_uid          = study_uid,
            series_dir         = str(d),
            series_number      = series_num,
            patient_position   = position,
            series_description = series_desc,
            n_dcm              = n,
            spatial            = spatial,
            is_scout           = scout,
            is_ct_quality      = ct_qual,
        )

    except Exception as exc:
        log.debug("_read_one_series_dir failed for %s: %s", d.name, exc)
        return None


def _save_cache(records: List[TciaSeriesRecord], cache_path: Path) -> None:
    data = [r.to_dict() for r in records]
    cache_path.write_text(json.dumps(data))
    log.info("TCIA index cached → %s  (%d series)", cache_path, len(data))


def _load_cache(cache_path: Path) -> Optional[List[TciaSeriesRecord]]:
    try:
        data = json.loads(cache_path.read_text())
        records = [TciaSeriesRecord.from_dict(d) for d in data]
        log.info("TCIA index loaded from cache: %d series", len(records))
        return records
    except Exception as exc:
        log.warning("TCIA cache load failed (%s) — will rescan.", exc)
        return None


def build_tcia_patient_index(
    tcia_dir:       Path,
    workers:        int  = 32,
    force_rebuild:  bool = False,
) -> Tuple[Dict[str, List[TciaSeriesRecord]], Dict[str, List]]:
    """
    Scan all TCIA series directories and group records by canonical PatientID.

    Returns (patient_index, study_groups).
    """
    if not HAS_DEPS:
        log.error("pydicom + numpy required.")
        return {}, {}

    cache_path = tcia_dir / _CACHE_FNAME

    if not force_rebuild and cache_path.exists():
        cached = _load_cache(cache_path)
        if cached is not None:
            grouped      = _group_by_patient(cached)
            study_groups = build_study_groups(grouped)
            log.info("TCIA index (from cache): %d patients | %d series",
                     len(grouped), len(cached))
            _log_patient_index_stats(grouped, study_groups)
            return grouped, study_groups

    dirs = [d for d in tcia_dir.iterdir()
            if d.is_dir() and _ANY_TCIA_RE.match(d.name)]
    log.info("TCIA index: scanning %d series dirs ...", len(dirs))

    records: List[TciaSeriesRecord] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(_read_one_series_dir, dirs):
            if rec is not None:
                records.append(rec)

    log.info("TCIA index: %d / %d dirs indexed.", len(records), len(dirs))

    no_pid = [r for r in records if not r.patient_uid]
    if no_pid:
        log.warning("  %d records have empty patient_uid — excluded.", len(no_pid))
        records = [r for r in records if r.patient_uid]

    _save_cache(records, cache_path)
    grouped      = _group_by_patient(records)
    study_groups = build_study_groups(grouped)

    n_ct = sum(1 for rs in grouped.values() for r in rs if r.is_ct_quality)
    log.info(
        "TCIA index: %d patients | %d total series | %d CT-quality series",
        len(grouped), len(records), n_ct,
    )
    _log_patient_index_stats(grouped, study_groups)
    return grouped, study_groups


def _group_by_patient(
    records: List[TciaSeriesRecord],
) -> Dict[str, List[TciaSeriesRecord]]:
    grouped: Dict[str, List[TciaSeriesRecord]] = {}
    for r in records:
        grouped.setdefault(r.patient_uid, []).append(r)
    for uid in grouped:
        grouped[uid].sort(key=lambda r: (r.series_number, r.n_dcm))
    return grouped


def build_study_groups(
    grouped: Dict[str, List[TciaSeriesRecord]],
) -> Dict[str, List]:
    from patient_db import TciaStudyRecord, characterize_study

    study_groups: Dict[str, List] = {}

    for patient_uid, series_list in grouped.items():
        by_study: Dict[str, List] = {}
        for s in series_list:
            by_study.setdefault(s.study_uid, []).append(s)

        def _visit_order(study_uid: str) -> int:
            return min((s.series_number for s in by_study[study_uid]), default=0)

        visit_records = []
        for idx, study_uid in enumerate(sorted(by_study.keys(), key=_visit_order), 1):
            tok = patient_token(patient_uid)
            rec = TciaStudyRecord(
                study_uid     = study_uid,
                patient_uid   = patient_uid,
                patient_token = tok,
                visit_index   = idx,
                series        = sorted(by_study[study_uid], key=lambda s: s.series_number),
            )
            characterize_study(rec)
            visit_records.append(rec)

        study_groups[patient_uid] = visit_records

    return study_groups


def _log_patient_index_stats(
    grouped:       Dict[str, List[TciaSeriesRecord]],
    study_groups:  Dict[str, List],
) -> None:
    from collections import Counter

    visits_per_patient = Counter(len(v) for v in study_groups.values())
    log.info("  Visits (StudyUIDs) per patient:")
    for n_visits, count in sorted(visits_per_patient.items()):
        note = " ← standard (single colonoscopy exam)" if n_visits == 1 else \
               " ← multiple exam dates (rare)" if n_visits > 1 else ""
        log.info("    %d visit(s) : %d patients%s", n_visits, count, note)

    comp_counter: Counter = Counter()
    for visits in study_groups.values():
        for v in visits:
            comp_counter[v.composition_label] += 1

    log.info("  Study composition (what each patient's visit contains):")
    log.info("    %-45s  %s", "composition", "count")
    log.info("    %s", "-" * 55)
    for label, count in sorted(comp_counter.items(), key=lambda x: -x[1]):
        pct  = count / max(sum(comp_counter.values()), 1) * 100
        note = ""
        if "prone+supine" in label and "scout" in label:
            note = "  ← standard COLONOG (prone CT + supine CT + scouts)"
        elif "prone+supine" in label and "extra-kernel" in label:
            note = "  ← extra reconstruction kernels downloaded"
        elif "prone+supine" in label:
            note = "  ← standard (scouts excluded from download)"
        elif label == "empty":
            note = "  ← all series were scouts/tiny (check download)"
        log.info("    %-45s  %4d  (%4.1f%%)%s", label, count, pct, note)

    all_series = [s for series_list in grouped.values() for s in series_list]
    n_total    = len(all_series)
    n_ct       = sum(1 for s in all_series if s.is_ct_quality)
    n_scout    = sum(1 for s in all_series if s.is_scout)
    n_prone    = sum(1 for s in all_series if s.is_ct_quality and s.patient_position == "PRONE")
    n_supine   = sum(1 for s in all_series if s.is_ct_quality and s.patient_position == "SUPINE")
    n_unknown  = sum(1 for s in all_series if s.is_ct_quality and s.patient_position == "UNKNOWN")

    log.info("  Series type breakdown (all %d series):", n_total)
    log.info("    CT-quality (≥50 DICOMs, non-scout) : %d", n_ct)
    log.info("      ├─ PRONE                         : %d", n_prone)
    log.info("      ├─ SUPINE                        : %d", n_supine)
    log.info("      └─ UNKNOWN position              : %d", n_unknown)
    log.info("    Scout/localizer (<50 DICOMs or desc): %d", n_scout)
    log.info("    Other (tiny, non-scout)             : %d", n_total - n_ct - n_scout)
