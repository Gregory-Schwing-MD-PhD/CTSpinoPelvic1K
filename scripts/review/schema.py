"""
scripts/review/schema.py — record schemas, validation, provenance
transitions, and the double-review + adjudication state machine for the
distributed pseudo-label review pipeline.

Pure stdlib (no numpy/nibabel) so it is a trivially-testable contract shared
by the review service (Phase 2) and the reviewtool client (Phase 3).

Provenance ladder (mirrors place_fused_masks.py / export_hf.py):
    manual            source-dataset annotation — IMMUTABLE, never reviewed here
    pseudo            model-filled (v2) — the thing reviewers correct
    pseudo_corrected  human-reviewed (accepted or edited) — the v3 outcome
    null              region absent / not applicable
A review only ever moves a region pseudo -> pseudo_corrected. manual / null
are never touched. A `reject` decision excludes the case (handled at the
case level), not via provenance.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

SCHEMA_VERSION = 1

# Canonical 10-class scheme (mirrors export_hf.CLASS_NAMES; IGNORE=10).
CLASS_NAMES = {
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6", 7: "sacrum", 8: "left_hip", 9: "right_hip",
}
IGNORE_LABEL = 10
SPINE_CLASSES = frozenset({1, 2, 3, 4, 5, 6})
PELVIC_CLASSES = frozenset({7, 8, 9})
REGION_CLASSES: Dict[str, frozenset] = {"spine": SPINE_CLASSES,
                                        "pelvis": PELVIC_CLASSES}
REGIONS = ("spine", "pelvis")

PROV_VALUES = ("manual", "pseudo", "pseudo_corrected", None)
DECISIONS = ("accept", "corrected", "reject")
ROLES = ("primary", "adjudicator")
CASE_STATUS = ("unassigned", "in_review", "needs_adjudication",
               "finalized", "excluded")

N_PRIMARY = 2          # double review
PRIMARY_SLOTS = ("1", "2")
ADJ_SLOT = "adj"


# ── small helpers ────────────────────────────────────────────────────────────

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s)).strip("_") or "unknown"


def case_id(token, config) -> str:
    return f"{safe_id(token)}__{config}"


def review_id(token, config, reviewer_id, round: int = 1) -> str:
    return f"{case_id(token, config)}__{safe_id(reviewer_id)}__r{round}"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── provenance transition ────────────────────────────────────────────────────

def provenance_after(prov_before: Dict[str, Optional[str]],
                     region: Optional[str], decision: str
                     ) -> Dict[str, Optional[str]]:
    """Provenance of {spine,pelvis} after a review of `region`.

    Only `region`'s `pseudo` becomes `pseudo_corrected` (whether the
    reviewer edited or merely confirmed). `manual`, `pseudo_corrected` and
    `null` are immutable. `reject` changes nothing here (the case is
    excluded at the case level). Unknown region / missing key -> unchanged.
    """
    out: Dict[str, Optional[str]] = {"spine": prov_before.get("spine"),
                                     "pelvis": prov_before.get("pelvis")}
    if decision == "reject":
        return out
    if region == "both":                         # fused gold case: re-check both
        for r in REGIONS:
            if out.get(r) == "pseudo":
                out[r] = "pseudo_corrected"
        return out
    if region not in REGIONS:
        return out
    if out.get(region) == "pseudo":
        out[region] = "pseudo_corrected"
    return out


# ── review record ────────────────────────────────────────────────────────────

@dataclass
class ReviewRecord:
    review_id: str
    token: str
    config: str
    source_revision: str                       # e.g. "v2"
    source_label_sha256: str                   # the EXACT pseudo label reviewed
    reviewer_id: str
    role: str = "primary"
    tool: str = "itk-snap"
    tool_version: str = ""
    round: int = 1
    claimed_at: str = ""
    submitted_at: str = ""
    edit_seconds: Optional[int] = None
    decision: str = "accept"                   # accept | corrected | reject
    region_reviewed: Optional[str] = None      # spine | pelvis (pseudo-filled side)
    diff: dict = field(default_factory=dict)   # from review.diff.label_diff
    prov_before: Dict[str, Optional[str]] = field(default_factory=dict)
    prov_after: Dict[str, Optional[str]] = field(default_factory=dict)
    corrected_label_sha256: Optional[str] = None
    artifact: Optional[str] = None             # path in the review repo
    flags: List[str] = field(default_factory=list)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewRecord":
        known = set(cls.__dataclass_fields__)          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def validate_review_record(rec) -> List[str]:
    """Return a list of human-readable validation errors ([] == valid)."""
    d = rec.to_dict() if isinstance(rec, ReviewRecord) else dict(rec)
    errs: List[str] = []

    for k in ("review_id", "token", "config", "source_label_sha256",
              "reviewer_id", "decision"):
        if not d.get(k):
            errs.append(f"missing required field: {k}")

    if d.get("decision") not in DECISIONS:
        errs.append(f"decision {d.get('decision')!r} not in {DECISIONS}")
    if d.get("role") not in ROLES:
        errs.append(f"role {d.get('role')!r} not in {ROLES}")

    region = d.get("region_reviewed")
    valid_regions = REGIONS + ("both",)          # "both" = a fused gold re-review
    if region is not None and region not in valid_regions:
        errs.append(f"region_reviewed {region!r} not in {valid_regions}")
    if d.get("decision") in ("accept", "corrected") and region not in valid_regions:
        errs.append(f"region_reviewed {region!r} must be one of {valid_regions} "
                    f"for decision={d.get('decision')!r}")

    if d.get("decision") == "corrected":
        if not d.get("corrected_label_sha256"):
            errs.append("corrected decision requires corrected_label_sha256")
        if not d.get("artifact"):
            errs.append("corrected decision requires artifact (label path)")
        if not d.get("diff"):
            errs.append("corrected decision requires a diff")

    # prov_after must equal the transition of prov_before (no hand-editing).
    pb = d.get("prov_before") or {}
    if pb:
        expect = provenance_after(pb, region, d.get("decision", ""))
        pa = d.get("prov_after") or {}
        if {k: pa.get(k) for k in ("spine", "pelvis")} != expect:
            errs.append(f"prov_after {pa} != expected transition {expect}")

    for v in ("prov_before", "prov_after"):
        for reg, val in (d.get(v) or {}).items():
            if reg in REGIONS and val not in PROV_VALUES:
                errs.append(f"{v}[{reg}]={val!r} not in {PROV_VALUES}")
    return errs


# ── double-review + adjudication state machine ───────────────────────────────
#
# A `case` is a plain dict (JSON-friendly, what the service persists):
#   {"case_id","token","config","stratum","priority",
#    "slots": {"1": {...}, "2": {...}, "adj": {...}},   # slot -> claim/submit
#    "final": {...} | None}
# A slot value: {"reviewer": id, "claimed_at","expires_at",
#                "review_id": id|None, "done": bool, "decision": str|None}

def claimable_primary_slot(case: dict, reviewer_id: str,
                           now: Optional[str] = None) -> Optional[str]:
    """Which primary slot (\"1\"/\"2\") this reviewer may claim, or None.

    Enforces double-review distinctness (a reviewer cannot hold both slots)
    and reclaims expired, not-yet-submitted claims. Returns None if no slot
    is open to this reviewer.
    """
    now = now or utcnow()
    slots = case.get("slots", {})
    # A reviewer already attached to this case (any slot) can't take another.
    if any(s.get("reviewer") == reviewer_id for s in slots.values()):
        return None
    for k in PRIMARY_SLOTS:
        s = slots.get(k)
        if s is None:
            return k
        # reclaim an abandoned (expired + not submitted) claim
        if not s.get("done") and s.get("expires_at") and s["expires_at"] < now:
            return k
    return None


def primary_done(case: dict) -> List[dict]:
    slots = case.get("slots", {})
    return [slots[k] for k in PRIMARY_SLOTS
            if slots.get(k, {}).get("done")]


def derive_status(case: dict, agree: Optional[bool] = None) -> str:
    """Compute case status from its slots/final.

    `agree` (IRR result, from review.diff) is only consulted once both
    primaries are submitted: True -> finalized (auto), False ->
    needs_adjudication, None -> in_review (agreement not computed yet).
    """
    final = case.get("final")
    if final:
        return "excluded" if final.get("decision") == "reject" else "finalized"

    slots = case.get("slots", {})
    if slots.get(ADJ_SLOT, {}).get("done"):
        return "finalized"

    done = len(primary_done(case))
    claimed = any(k in slots for k in PRIMARY_SLOTS)

    if done >= N_PRIMARY:
        # `agree` may be passed explicitly, else read the persisted IRR
        # outcome the service stored on the case at evaluation time.
        a = agree if agree is not None else case.get("agree")
        if a is True:
            return "finalized"
        if a is False:
            return "needs_adjudication"
        return "in_review"          # both in, IRR not yet evaluated
    return "in_review" if claimed else "unassigned"
