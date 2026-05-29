"""Unit tests for review_service.admin.plan_reset (pure slot-reset logic).
The HF commit glue isn't unit-tested (network + write token); the decision of
WHAT to remove and the resulting case status is."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "review_service", _ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import admin  # noqa: E402  (review_service/admin.py)
from review import schema  # noqa: E402


def _case_one_accept(reviewer="gregoryschwingmdphd"):
    cid = "101__spine_only"
    return {
        "case_id": cid, "token": "101", "config": "spine_only",
        "slots": {"1": {
            "reviewer": reviewer, "done": True, "decision": "accept",
            "review_id": f"{cid}__{reviewer}__r1",
            "label_path": f"reviews/{cid}/1_label.nii.gz",
        }},
        "final": None,
    }


def test_reset_by_reviewer_returns_unassigned():
    case = _case_one_accept()
    new, removed = admin.plan_reset(case, "gregoryschwingmdphd", None)
    assert [r["slot"] for r in removed] == ["1"]
    assert "1" not in new["slots"]
    assert schema.derive_status(new) == "unassigned"
    # input must not be mutated
    assert "1" in case["slots"]


def test_reset_is_case_insensitive():
    case = _case_one_accept()
    new, removed = admin.plan_reset(case, "GregorySchwingMDPhD", None)
    assert removed and "1" not in new["slots"]


def test_reset_no_match_is_noop():
    case = _case_one_accept()
    new, removed = admin.plan_reset(case, "someone_else", None)
    assert removed == []
    assert "1" in new["slots"]
    assert schema.derive_status(new) == "in_review"


def test_reviewer_and_slot_is_AND():
    case = _case_one_accept()
    # right reviewer, wrong slot -> no removal
    _, removed = admin.plan_reset(case, "gregoryschwingmdphd", "2")
    assert removed == []
    # right reviewer, right slot -> removed
    _, removed2 = admin.plan_reset(case, "gregoryschwingmdphd", "1")
    assert [r["slot"] for r in removed2] == ["1"]


def test_orphan_files_paths():
    case = _case_one_accept()
    _, removed = admin.plan_reset(case, "gregoryschwingmdphd", None)
    files = admin._orphan_files("101__spine_only", removed)
    assert "reviews/101__spine_only/1_label.nii.gz" in files
    assert "reviews/101__spine_only/101__spine_only__gregoryschwingmdphd__r1.json" in files
