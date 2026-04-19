"""
series_assigner.py — Assign TCIA series candidates to each mask.

ARCHITECTURE
------------
Patient identity is already certain at this point (from mask_index.py).
This module's only job is: given a mask and its patient's TCIA series,
rank those series by how likely they are to be the source scan.

ALL MATCHING IS PATIENT-LOCAL.  We never compare a mask against series from
other patients.  This eliminates the entire class of cross-patient assignment
errors that plagued the previous affine-search approach.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from patient_db import (
    Confidence,
    CONFIDENCE_SCORES,
    FusionStatus,
    PatientRecord,
    PelvicMaskRecord,
    SeriesCandidate,
    SpineMaskRecord,
    TciaSeriesRecord,
)

log = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

NZ_TOLERANCE   = 5
DIR_TOLERANCE  = 1e-3
SPC_TOLERANCE  = 0.01


def _directions_match(d1, d2, tol: float = DIR_TOLERANCE) -> bool:
    for i in range(3):
        dot = abs(float(np.dot(d1[:, i], d2[:, i])))
        if dot < 1.0 - tol:
            return False
    return True


def _affine_matches_series(
    mask_affine_flat: List[float],
    series:           TciaSeriesRecord,
) -> bool:
    if not HAS_NUMPY or series.spatial is None:
        return False
    try:
        mask_aff = np.array(mask_affine_flat).reshape(4, 4)
        m_spacing = np.array([
            np.linalg.norm(mask_aff[:3, i]) for i in range(3)
        ])
        if np.any(m_spacing < 1e-9):
            return False
        m_dir = mask_aff[:3, :3] / m_spacing[np.newaxis, :]

        s_dir = series.spatial.direction_matrix()
        s_spc = np.array(series.spatial.spacing)

        return (
            _directions_match(m_dir, s_dir)
            and np.abs(m_spacing - s_spc).max() < SPC_TOLERANCE
        )
    except Exception:
        return False


def _make_candidate(
    series: TciaSeriesRecord,
    rank:   int,
    conf:   str,
    reasons: List[str],
) -> SeriesCandidate:
    return SeriesCandidate(
        rank               = rank,
        series_uid         = series.series_uid,
        series_dir         = series.series_dir,
        confidence         = conf,
        confidence_score   = CONFIDENCE_SCORES.get(conf, 0.0),
        patient_position   = series.patient_position,
        n_dcm              = series.n_dcm,
        series_number      = series.series_number,
        series_description = series.series_description,
        reasons            = reasons,
    )


def assign_spine_candidates(
    mask:            SpineMaskRecord,
    patient_series:  List[TciaSeriesRecord],
) -> List[SeriesCandidate]:
    if not patient_series:
        return []

    ct_series = [s for s in patient_series if s.is_ct_quality]
    if not ct_series:
        ct_series = [s for s in patient_series if not s.is_scout]
    if not ct_series:
        ct_series = patient_series

    if len(ct_series) == 1:
        s = ct_series[0]
        reasons = [
            f"Only 1 CT-quality series for this patient  (n_dcm={s.n_dcm})",
            f"position={s.patient_position}  series_num={s.series_number}",
        ]
        nz = mask.nifti_nz()
        if nz and abs(s.n_dcm - nz) <= NZ_TOLERANCE:
            reasons.append(f"NZ confirmed: nifti_nz={nz} ≈ n_dcm={s.n_dcm}")
            conf = Confidence.CERTAIN
        elif nz and abs(s.n_dcm - nz) > NZ_TOLERANCE:
            reasons.append(f"NZ mismatch: nifti_nz={nz}  n_dcm={s.n_dcm} — flagged")
            conf = Confidence.LOW
        else:
            conf = Confidence.CERTAIN
        return [_make_candidate(s, 1, conf, reasons)]

    if mask.nifti_affine and HAS_NUMPY:
        aff_matches = [
            s for s in ct_series
            if _affine_matches_series(mask.nifti_affine, s)
        ]
        nz = mask.nifti_nz()
        if aff_matches:
            if nz:
                nz_ok = [s for s in aff_matches if abs(s.n_dcm - nz) <= NZ_TOLERANCE]
                if nz_ok:
                    aff_matches = nz_ok
            if len(aff_matches) == 1:
                s = aff_matches[0]
                return [_make_candidate(s, 1, Confidence.HIGH, [
                    "Affine direction+spacing exact match (unique)",
                    f"nifti_nz={nz}  n_dcm={s.n_dcm}",
                ])]

    nz = mask.nifti_nz()
    if nz:
        nz_exact = [s for s in ct_series if abs(s.n_dcm - nz) <= NZ_TOLERANCE]
        if len(nz_exact) == 1:
            s = nz_exact[0]
            return [_make_candidate(s, 1, Confidence.MEDIUM, [
                f"NZ exact match: nifti_nz={nz}  n_dcm={s.n_dcm}",
                f"position={s.patient_position}",
            ])]
        if nz_exact:
            ct_series = nz_exact

    if ct_series:
        sorted_by_ndcm = sorted(ct_series, key=lambda s: -s.n_dcm)
        best = sorted_by_ndcm[0]
        if len(sorted_by_ndcm) > 1:
            candidates = []
            for rank, s in enumerate(sorted_by_ndcm, 1):
                conf    = Confidence.AMBIGUOUS
                reasons = [
                    f"Ambiguous: {len(sorted_by_ndcm)} CT series for this patient",
                    f"rank by n_dcm: n_dcm={s.n_dcm}  position={s.patient_position}",
                    f"series_num={s.series_number}  desc={s.series_description[:60]}",
                ]
                if rank == 1:
                    reasons.append("Selected as best by most-DICOMs (patient-local)")
                candidates.append(_make_candidate(s, rank, conf, reasons))
            return candidates
        else:
            return [_make_candidate(best, 1, Confidence.LOW, [
                f"Single CT series after scout filter  n_dcm={best.n_dcm}",
                f"position={best.patient_position}",
            ])]

    return []


def assign_pelvic_candidates(
    mask:           PelvicMaskRecord,
    patient_series: List[TciaSeriesRecord],
) -> List[SeriesCandidate]:
    if not patient_series:
        return []

    ct_series = [s for s in patient_series if s.is_ct_quality]
    if not ct_series:
        ct_series = [s for s in patient_series if not s.is_scout]
    if not ct_series:
        ct_series = patient_series

    sernum  = mask.series_number_from_fname
    nz_fname = mask.nz_from_fname
    pos_fname = mask.position_from_fname

    if len(ct_series) == 1:
        s = ct_series[0]
        reasons = [f"Only 1 CT-quality series for this patient  (n_dcm={s.n_dcm})"]
        conf    = Confidence.CERTAIN

        if sernum is not None and s.series_number == sernum:
            reasons.append(f"SeriesNumber confirmed: {sernum}")
        elif sernum is not None:
            reasons.append(f"SeriesNumber MISMATCH: expected {sernum}  got {s.series_number}")
            conf = Confidence.LOW

        if nz_fname and abs(s.n_dcm - nz_fname) <= NZ_TOLERANCE:
            reasons.append(f"NZ confirmed: filename_nz={nz_fname} ≈ n_dcm={s.n_dcm}")
        elif nz_fname:
            reasons.append(f"NZ mismatch: filename_nz={nz_fname}  n_dcm={s.n_dcm}")
            if conf == Confidence.CERTAIN:
                conf = Confidence.LOW

        return [_make_candidate(s, 1, conf, reasons)]

    if sernum is not None:
        sn_matches = [s for s in ct_series if s.series_number == sernum]
        if len(sn_matches) == 1:
            s = sn_matches[0]
            reasons = [
                f"SeriesNumber exact match: {sernum} (unique within patient)",
                f"n_dcm={s.n_dcm}  nz_from_fname={nz_fname}  position={s.patient_position}",
            ]
            conf = Confidence.HIGH
            if nz_fname and abs(s.n_dcm - nz_fname) <= NZ_TOLERANCE:
                reasons.append(f"NZ confirmed: filename_nz={nz_fname} ≈ n_dcm={s.n_dcm}")
                conf = Confidence.CERTAIN
            elif nz_fname and abs(s.n_dcm - nz_fname) > NZ_TOLERANCE:
                reasons.append(
                    f"NZ disagreement: filename_nz={nz_fname}  n_dcm={s.n_dcm} — check scan"
                )
                conf = Confidence.MEDIUM
            return [_make_candidate(s, 1, conf, reasons)]

        if sn_matches:
            ct_series = sn_matches

    if mask.nifti_affine and HAS_NUMPY:
        aff_matches = [
            s for s in ct_series
            if _affine_matches_series(mask.nifti_affine, s)
        ]
        if aff_matches and nz_fname:
            nz_ok = [s for s in aff_matches
                     if nz_fname <= s.n_dcm + NZ_TOLERANCE]
            if nz_ok:
                aff_matches = nz_ok
        if len(aff_matches) == 1:
            s = aff_matches[0]
            return [_make_candidate(s, 1, Confidence.HIGH, [
                "Affine direction+spacing exact match (unique within patient)",
                f"nz_from_fname={nz_fname}  n_dcm={s.n_dcm}",
            ])]

    if nz_fname:
        nz_ok = [s for s in ct_series if nz_fname <= s.n_dcm + NZ_TOLERANCE]
        if len(nz_ok) == 1:
            s = nz_ok[0]
            return [_make_candidate(s, 1, Confidence.MEDIUM, [
                f"NZ subvolume match (unique): mask_nz={nz_fname} ≤ n_dcm={s.n_dcm}",
                f"position={s.patient_position}",
            ])]
        if nz_ok:
            ct_series = nz_ok

    if pos_fname in ("PRONE", "SUPINE"):
        pos_ok = [s for s in ct_series if s.patient_position == pos_fname]
        if len(pos_ok) == 1:
            s = pos_ok[0]
            return [_make_candidate(s, 1, Confidence.MEDIUM, [
                f"Position match (unique): {pos_fname}",
                f"n_dcm={s.n_dcm}  series_num={s.series_number}",
            ])]
        if pos_ok:
            ct_series = pos_ok

    sorted_by_ndcm = sorted(ct_series, key=lambda s: -s.n_dcm)
    candidates = []
    for rank, s in enumerate(sorted_by_ndcm, 1):
        reasons = [
            f"Ambiguous: {len(sorted_by_ndcm)} candidates remain after all filters",
            f"n_dcm={s.n_dcm}  position={s.patient_position}",
            f"series_num={s.series_number}  desc={s.series_description[:60]}",
        ]
        if sernum is not None:
            reasons.append(f"Expected SeriesNumber={sernum}  got {s.series_number}")
        candidates.append(_make_candidate(s, rank, Confidence.AMBIGUOUS, reasons))
    return candidates


def assign_all_candidates(
    patient:        PatientRecord,
    patient_series: List[TciaSeriesRecord],
) -> None:
    for mask in patient.spine_masks:
        mask.candidates = assign_spine_candidates(mask, patient_series)
    for mask in patient.pelvic_masks:
        mask.candidates = assign_pelvic_candidates(mask, patient_series)


def determine_fusion_status(patient: PatientRecord) -> Tuple[str, List[str]]:
    if not patient.is_complete:
        return FusionStatus.SINGLE, ["Patient has only one annotation type."]

    spine_uid  = patient.spine_masks[0].best_series_uid   if patient.spine_masks  else None
    pelvic_uid = patient.pelvic_masks[0].best_series_uid  if patient.pelvic_masks else None

    spine_conf  = patient.spine_masks[0].best_confidence  if patient.spine_masks  else Confidence.UNRESOLVED
    pelvic_conf = patient.pelvic_masks[0].best_confidence if patient.pelvic_masks else Confidence.UNRESOLVED

    if (spine_conf in (Confidence.AMBIGUOUS, Confidence.UNRESOLVED) or
        pelvic_conf in (Confidence.AMBIGUOUS, Confidence.UNRESOLVED)):
        reasons = [
            f"spine_conf={spine_conf}  pelvic_conf={pelvic_conf}",
            "Cannot determine fusion status with ambiguous/unresolved assignments.",
        ]
        if spine_uid and pelvic_uid and spine_uid == pelvic_uid:
            reasons.append(
                f"Note: both point to same series {spine_uid[-10:]} "
                "despite low confidence — likely FUSION but unverified."
            )
        return FusionStatus.AMBIGUOUS, reasons

    if spine_uid is None or pelvic_uid is None:
        return FusionStatus.AMBIGUOUS, [
            f"spine_uid={spine_uid}  pelvic_uid={pelvic_uid}",
            "One or both series UIDs are None.",
        ]

    if spine_uid == pelvic_uid:
        spine_pos  = patient.spine_masks[0].best_candidate.patient_position  if patient.spine_masks[0].best_candidate  else "?"
        return FusionStatus.FUSION, [
            f"Spine and pelvic both assign to series ...{spine_uid[-10:]}",
            f"Spine conf={spine_conf}  Pelvic conf={pelvic_conf}",
            f"Both masks annotate different anatomy on the same {spine_pos} scan.",
        ]
    else:
        spine_pos  = patient.spine_masks[0].best_candidate.patient_position  if patient.spine_masks[0].best_candidate  else "?"
        pelvic_pos = patient.pelvic_masks[0].best_candidate.patient_position if patient.pelvic_masks[0].best_candidate else "?"
        return FusionStatus.SEPARATE, [
            f"Spine → series ...{spine_uid[-10:]}  ({spine_pos})",
            f"Pelvic → series ...{pelvic_uid[-10:]}  ({pelvic_pos})",
            f"Spine conf={spine_conf}  Pelvic conf={pelvic_conf}",
            "Different scans (typical COLONOG: prone for spine, supine for pelvis).",
        ]


def cross_check_patient(
    patient:        PatientRecord,
    patient_series: List[TciaSeriesRecord],
) -> List[str]:
    warnings: List[str] = []
    series_by_uid = {s.series_uid: s for s in patient_series}

    for mask in patient.pelvic_masks:
        best = mask.best_candidate
        if best is None:
            warnings.append(f"PELVIC_UNRESOLVED: mask={Path(mask.mask_file).name}")
            continue

        assigned = series_by_uid.get(best.series_uid)
        if assigned is None:
            warnings.append(
                f"PELVIC_SERIES_NOT_IN_INDEX: uid=...{best.series_uid[-10:]}  "
                f"mask={Path(mask.mask_file).name}"
            )
            continue

        if (mask.series_number_from_fname is not None and
                assigned.series_number != mask.series_number_from_fname and
                best.confidence in (Confidence.CERTAIN, Confidence.HIGH)):
            warnings.append(
                f"PELVIC_SERNUM_MISMATCH: expected={mask.series_number_from_fname}  "
                f"got={assigned.series_number}  "
                f"mask={Path(mask.mask_file).name}  conf={best.confidence}"
            )

        if mask.nz_from_fname and mask.nz_from_fname > assigned.n_dcm + NZ_TOLERANCE:
            warnings.append(
                f"PELVIC_NZ_OVERFLOW: mask_nz={mask.nz_from_fname}  "
                f"series_n_dcm={assigned.n_dcm}  "
                f"mask={Path(mask.mask_file).name}"
            )

    for mask in patient.spine_masks:
        best = mask.best_candidate
        if best is None:
            warnings.append(f"SPINE_UNRESOLVED: mask={Path(mask.mask_file).name}")
            continue
        nz = mask.nifti_nz()
        if nz and best.n_dcm < nz - NZ_TOLERANCE:
            warnings.append(
                f"SPINE_NZ_SHORT: series_n_dcm={best.n_dcm} < nifti_nz={nz}  "
                f"(TCIA may have fewer DICOMs than expected)  "
                f"conf={best.confidence}"
            )

    return warnings
