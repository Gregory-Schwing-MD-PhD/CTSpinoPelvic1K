"""
patient_db.py — Core schema for the CTSpinoPelvic1K patient/series/mask database.

DESIGN PHILOSOPHY
-----------------
Patient identity is established SOLELY from the DICOM PatientID tag and the
UID encoded in each mask filename.  Both encode the same COLONOG/CTC patient
UID.  Canonicalization (strip zero-padding) makes them directly comparable.
This is 100% certain — no affine matching is required for identity.

Series assignment (which TCIA series a mask was derived from) is the only
uncertain step.  It is handled by series_assigner.py which produces a ranked
candidate list with explicit confidence levels and reasons.

SERIALIZATION
-------------
PatientDB serialises to:
  - JSON  (audit / inspect)
  - Pickle (fast training-time load)

The JSON schema uses snake_case dicts that round-trip cleanly through
dataclasses; Path objects are serialised as strings.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── UID canonicalization ──────────────────────────────────────────────────────

def canonical_uid(uid: str) -> str:
    """
    Strip zero-padding from the last numeric component of a COLONOG/CTC UID.

    Examples
    --------
    "1.3.6.1.4.1.9328.50.4.0017"  → "1.3.6.1.4.1.9328.50.4.17"
    "1.3.6.1.4.1.9328.50.4.17"    → "1.3.6.1.4.1.9328.50.4.17"  (no-op)
    "CTC-3105759107"               → "CTC-3105759107"             (CTC, no-op)
    """
    uid = uid.strip()
    if uid.startswith("CTC-"):
        return uid
    parts = uid.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0] + "." + str(int(parts[1]))
    return uid


def patient_token(uid: str) -> str:
    """
    Derive a short display token from a patient UID.

    COLONOG: "1.3.6.1.4.1.9328.50.4.17"  → "17"
    CTC:     "CTC-3105759107"             → "CTC-3105759107"
    """
    canon = canonical_uid(uid)
    if canon.startswith("CTC-"):
        return canon
    return canon.rsplit(".", 1)[-1]


# ── Confidence levels ─────────────────────────────────────────────────────────

class Confidence(str, Enum):
    CERTAIN    = "certain"
    HIGH       = "high"
    MEDIUM     = "medium"
    LOW        = "low"
    AMBIGUOUS  = "ambiguous"
    UNRESOLVED = "unresolved"

CONFIDENCE_SCORES: Dict[str, float] = {
    Confidence.CERTAIN:    1.00,
    Confidence.HIGH:       0.90,
    Confidence.MEDIUM:     0.75,
    Confidence.LOW:        0.55,
    Confidence.AMBIGUOUS:  0.30,
    Confidence.UNRESOLVED: 0.00,
}


# ── LSTV label constants ──────────────────────────────────────────────────────

class LSTV(str, Enum):
    NORMAL           = "NORMAL"
    SACRALIZATION    = "SACRALIZATION"
    SEMI_SACRAL      = "SEMI_SACRALIZATION"
    LUMBARIZATION    = "LUMBARIZATION"
    AMBIGUOUS        = "AMBIGUOUS"
    INCOMPLETE_SCAN  = "INCOMPLETE_SCAN"
    UNKNOWN          = "UNKNOWN"

# CTSpine1K / VerSe label IDs for lumbar vertebrae
LUMBAR_LABEL_IDS: Dict[int, str] = {
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"
}

PATIENT_POSITION_MAP: Dict[str, str] = {
    "HFP": "PRONE",  "FFP": "PRONE",
    "HFS": "SUPINE", "FFS": "SUPINE",
    "PRONE": "PRONE", "SUPINE": "SUPINE",
}


# ── TCIA series record ────────────────────────────────────────────────────────

@dataclass
class SpatialInfo:
    origin:    List[float]
    spacing:   List[float]
    direction: List[float]
    nz:        int

    def direction_matrix(self):
        import numpy as np
        return np.array(self.direction, dtype=float).reshape(3, 3)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SpatialInfo":
        return cls(**d)


@dataclass
class TciaSeriesRecord:
    patient_uid:       str
    patient_token:     str
    series_uid:        str
    study_uid:         str
    series_dir:        str
    series_number:     int
    patient_position:  str
    series_description: str
    n_dcm:             int
    spatial:           Optional[SpatialInfo] = None
    is_scout:          bool = False
    is_ct_quality:     bool = True

    def series_dir_path(self) -> Path:
        return Path(self.series_dir)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.spatial:
            d["spatial"] = self.spatial.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TciaSeriesRecord":
        sp = d.pop("spatial", None)
        obj = cls(**d)
        obj.spatial = SpatialInfo.from_dict(sp) if sp else None
        return obj


@dataclass
class TciaStudyRecord:
    study_uid:          str
    patient_uid:        str
    patient_token:      str
    visit_index:        int
    series:             List[Any]
    n_series:           int  = 0
    n_ct_quality:       int  = 0
    n_scouts:           int  = 0
    n_extra_kernels:    int  = 0
    has_prone_ct:       bool = False
    has_supine_ct:      bool = False
    has_unknown_pos_ct: bool = False
    composition_label:  str  = ""

    def to_dict(self) -> Dict:
        d = {k: v for k, v in asdict(self).items() if k != "series"}
        d["series"] = [s.to_dict() for s in self.series]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TciaStudyRecord":
        series_data = d.pop("series", [])
        obj = cls(**d, series=[])
        obj.series = [TciaSeriesRecord.from_dict(s) for s in series_data]
        return obj


def characterize_study(study: TciaStudyRecord) -> None:
    ct   = [s for s in study.series if s.is_ct_quality]
    scou = [s for s in study.series if s.is_scout]
    study.n_series        = len(study.series)
    study.n_ct_quality    = len(ct)
    study.n_scouts        = len(scou)

    positions = [s.patient_position for s in ct]
    study.has_prone_ct       = "PRONE"   in positions
    study.has_supine_ct      = "SUPINE"  in positions
    study.has_unknown_pos_ct = "UNKNOWN" in positions

    study.n_extra_kernels = max(0, study.n_ct_quality - (
        int(study.has_prone_ct) + int(study.has_supine_ct)
        + int(study.has_unknown_pos_ct and "UNKNOWN" in positions)
    ))

    parts = []
    if study.has_prone_ct:       parts.append("prone")
    if study.has_supine_ct:      parts.append("supine")
    if study.has_unknown_pos_ct: parts.append(f"{positions.count('UNKNOWN')}×unknown-pos")
    if study.n_scouts:           parts.append(f"{study.n_scouts}×scout")
    if study.n_extra_kernels:    parts.append(f"{study.n_extra_kernels}×extra-kernel")
    study.composition_label = "+".join(parts) if parts else "empty"


@dataclass
class SeriesCandidate:
    rank:               int
    series_uid:         str
    series_dir:         str
    confidence:         str
    confidence_score:   float
    patient_position:   str
    n_dcm:              int
    series_number:      int
    series_description: str
    reasons:            List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SeriesCandidate":
        return cls(**d)

    @property
    def is_certain(self) -> bool:
        return self.confidence == Confidence.CERTAIN

    @property
    def is_high_or_better(self) -> bool:
        return self.confidence_score >= CONFIDENCE_SCORES[Confidence.HIGH]


@dataclass
class SpineMaskRecord:
    mask_file:          str
    image_file:         Optional[str]
    patient_uid:        str
    patient_token:      str
    nifti_shape:        Optional[List[int]]
    nifti_affine:       Optional[List[float]]
    lstv_label:         str
    n_lumbar_found:     int
    lumbar_labels:      List[str]
    candidates:         List[SeriesCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> Optional[SeriesCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def best_series_uid(self) -> Optional[str]:
        c = self.best_candidate
        return c.series_uid if c else None

    @property
    def best_confidence(self) -> str:
        c = self.best_candidate
        return c.confidence if c else Confidence.UNRESOLVED

    def nifti_nz(self) -> Optional[int]:
        if self.nifti_shape and len(self.nifti_shape) >= 3:
            return self.nifti_shape[2]
        return None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SpineMaskRecord":
        cands = [SeriesCandidate.from_dict(c) for c in d.pop("candidates", [])]
        obj = cls(**d)
        obj.candidates = cands
        return obj


@dataclass
class PelvicMaskRecord:
    mask_file:               str
    patient_uid:             str
    patient_token:           str
    series_number_from_fname: Optional[int]
    nz_from_fname:            Optional[int]
    position_from_fname:      str
    nifti_shape:              Optional[List[int]]
    nifti_affine:             Optional[List[float]]
    lstv_label:               str
    lstv_qualifier_block:     str
    lstv_flags:               Dict[str, bool]
    candidates:               List[SeriesCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> Optional[SeriesCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def best_series_uid(self) -> Optional[str]:
        c = self.best_candidate
        return c.series_uid if c else None

    @property
    def best_confidence(self) -> str:
        c = self.best_candidate
        return c.confidence if c else Confidence.UNRESOLVED

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PelvicMaskRecord":
        cands = [SeriesCandidate.from_dict(c) for c in d.pop("candidates", [])]
        obj = cls(**d)
        obj.candidates = cands
        return obj


class FusionStatus(str, Enum):
    FUSION    = "fusion"
    SEPARATE  = "separate"
    AMBIGUOUS = "ambiguous"
    SINGLE    = "single"


@dataclass
class PatientRecord:
    patient_uid:    str
    patient_token:  str
    tcia_series:    List[TciaSeriesRecord]   = field(default_factory=list)
    studies:        List[TciaStudyRecord]    = field(default_factory=list)
    spine_masks:    List[SpineMaskRecord]    = field(default_factory=list)
    pelvic_masks:   List[PelvicMaskRecord]   = field(default_factory=list)
    fusion_status:  str  = "single"
    fusion_reasons: List[str] = field(default_factory=list)

    @property
    def has_spine(self) -> bool:  return bool(self.spine_masks)
    @property
    def has_pelvic(self) -> bool: return bool(self.pelvic_masks)
    @property
    def is_complete(self) -> bool: return self.has_spine and self.has_pelvic
    @property
    def n_tcia_series(self) -> int: return len(self.tcia_series)
    @property
    def n_visits(self) -> int: return len(self.studies)

    @property
    def ct_series(self) -> List[TciaSeriesRecord]:
        return [s for s in self.tcia_series if not s.is_scout and s.is_ct_quality]

    @property
    def prone_ct_series(self) -> List[TciaSeriesRecord]:
        return [s for s in self.ct_series if s.patient_position == "PRONE"]

    @property
    def supine_ct_series(self) -> List[TciaSeriesRecord]:
        return [s for s in self.ct_series if s.patient_position == "SUPINE"]

    def spine_best_uid(self) -> Optional[str]:
        return self.spine_masks[0].best_series_uid if self.spine_masks else None

    def pelvic_best_uids(self) -> List[str]:
        return [m.best_series_uid for m in self.pelvic_masks if m.best_series_uid]

    def to_dict(self) -> Dict:
        return {
            "patient_uid":    self.patient_uid,
            "patient_token":  self.patient_token,
            "fusion_status":  self.fusion_status,
            "fusion_reasons": self.fusion_reasons,
            "tcia_series":    [s.to_dict() for s in self.tcia_series],
            "studies":        [s.to_dict() for s in self.studies],
            "spine_masks":    [m.to_dict() for m in self.spine_masks],
            "pelvic_masks":   [m.to_dict() for m in self.pelvic_masks],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PatientRecord":
        return cls(
            patient_uid    = d["patient_uid"],
            patient_token  = d["patient_token"],
            fusion_status  = d.get("fusion_status", "single"),
            fusion_reasons = d.get("fusion_reasons", []),
            tcia_series    = [TciaSeriesRecord.from_dict(s) for s in d.get("tcia_series", [])],
            studies        = [TciaStudyRecord.from_dict(s)  for s in d.get("studies", [])],
            spine_masks    = [SpineMaskRecord.from_dict(m)  for m in d.get("spine_masks", [])],
            pelvic_masks   = [PelvicMaskRecord.from_dict(m) for m in d.get("pelvic_masks", [])],
        )


@dataclass
class DBMetadata:
    generated_by:           str  = "build_db.py"
    n_patients:             int  = 0
    n_tcia_series_total:    int  = 0
    n_spine_masks:          int  = 0
    n_pelvic_masks:         int  = 0
    n_complete_patients:    int  = 0
    n_fusion:               int  = 0
    n_separate:             int  = 0
    n_ambiguous_assignment: int  = 0
    n_unresolved:           int  = 0
    confidence_breakdown:   Dict[str, int] = field(default_factory=dict)
    build_timestamp:        str  = ""
    tcia_dir:               str  = ""
    spine_root:             str  = ""
    pelvis_root:            str  = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class PatientDB:
    """
    The full patient database.

    Primary key: patient_uid (canonical).
    Secondary key: patient_token (short integer string for display).

    Usage
    -----
    db = PatientDB.build(...)                       # see build_db.py
    db = PatientDB.from_json("patient_db.json")
    db = PatientDB.from_pickle("patient_db.pkl")

    rec = db.patients["1.3.6.1.4.1.9328.50.4.17"]
    rec = db.by_token["17"]

    for rec in db.complete_patients():
        spine_dir  = Path(rec.spine_masks[0].best_candidate.series_dir)
        pelvic_dir = Path(rec.pelvic_masks[0].best_candidate.series_dir)
    """

    def __init__(self, patients: Dict[str, PatientRecord], metadata: DBMetadata):
        self.patients = patients
        self.metadata = metadata

    @property
    def by_token(self) -> Dict[str, PatientRecord]:
        return {r.patient_token: r for r in self.patients.values()}

    def complete_patients(self) -> List[PatientRecord]:
        return [r for r in self.patients.values() if r.is_complete]

    def spine_only_patients(self) -> List[PatientRecord]:
        return [r for r in self.patients.values() if r.has_spine and not r.has_pelvic]

    def pelvic_only_patients(self) -> List[PatientRecord]:
        return [r for r in self.patients.values() if r.has_pelvic and not r.has_spine]

    def high_confidence_complete(self) -> List[PatientRecord]:
        out = []
        for r in self.complete_patients():
            spine_ok  = any(
                CONFIDENCE_SCORES.get(m.best_confidence, 0) >= CONFIDENCE_SCORES[Confidence.HIGH]
                for m in r.spine_masks
            )
            pelvic_ok = any(
                CONFIDENCE_SCORES.get(m.best_confidence, 0) >= CONFIDENCE_SCORES[Confidence.HIGH]
                for m in r.pelvic_masks
            )
            if spine_ok and pelvic_ok:
                out.append(r)
        return out

    def to_dict(self) -> Dict:
        return {
            "metadata":  self.metadata.to_dict(),
            "patients":  {uid: rec.to_dict() for uid, rec in self.patients.items()},
        }

    def to_json(self, path: Path, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent, default=str))

    def to_pickle(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_dict(cls, d: Dict) -> "PatientDB":
        meta = DBMetadata(**d.get("metadata", {}))
        patients = {
            uid: PatientRecord.from_dict(rec)
            for uid, rec in d.get("patients", {}).items()
        }
        return cls(patients=patients, metadata=meta)

    @classmethod
    def from_json(cls, path: Path) -> "PatientDB":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_pickle(cls, path: Path) -> "PatientDB":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PatientDB Summary",
            "=" * 60,
            f"  Patients total          : {len(self.patients)}",
            f"  TCIA series total       : {self.metadata.n_tcia_series_total}",
            f"  Spine masks             : {self.metadata.n_spine_masks}",
            f"  Pelvic masks            : {self.metadata.n_pelvic_masks}",
            f"  Complete (both types)   : {self.metadata.n_complete_patients}",
            f"  Fusion (same series)    : {self.metadata.n_fusion}",
            f"  Separate (diff series)  : {self.metadata.n_separate}",
            f"  Ambiguous assignment    : {self.metadata.n_ambiguous_assignment}",
            f"  Unresolved              : {self.metadata.n_unresolved}",
            "",
            "  Confidence breakdown:",
        ]
        for level, count in sorted(
            self.metadata.confidence_breakdown.items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"    {level:<20} : {count}")
        lines.append("=" * 60)
        return "\n".join(lines)
