"""
Unit tests for the Phase-1 review core (scripts/review/): diff + IRR,
provenance transitions, validation, the double-review state machine, and
the v3 reducer. Pure functions only — no cloud/ITK-SNAP/NIfTI needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from review import diff, labels_descriptor, reduce_to_v3, schema  # noqa: E402


# --------------------------------------------------------------------------- #
# diff + IRR
# --------------------------------------------------------------------------- #

def _vol():
    return np.zeros((6, 6, 6), dtype=np.int16)


def test_label_diff_added_removed_dice_regions():
    pseudo = _vol()
    pseudo[0:2, 0, 0] = 7          # 2 sacrum voxels
    corrected = _vol()
    corrected[0:3, 0, 0] = 7       # grew to 3 (added 1)
    d = diff.label_diff(pseudo, corrected)
    pc = d["per_class"]["7"]
    assert pc["added"] == 1 and pc["removed"] == 0
    assert d["n_voxels_changed"] == 1
    assert d["regions_touched"] == ["pelvis"]
    assert 0.0 < pc["dice"] < 1.0
    assert d["changed_bbox_pir"] is not None


def test_per_class_dice_identical_is_one_and_one_sided_is_zero():
    a = _vol(); a[0, 0, 0] = 5
    assert diff.per_class_dice(a, a) == {"5": 1.0}
    b = _vol()                      # b has no class 5
    assert diff.per_class_dice(a, b)["5"] == 0.0


def test_irr_per_class_min_blocks_on_any_disagreeing_class():
    # A and B agree perfectly on sacrum(7) but differ on L5(5).
    a = _vol(); a[0:4, 0, 0] = 5; a[0:4, 1, 0] = 7
    b = _vol(); b[0:1, 0, 0] = 5; b[0:4, 1, 0] = 7    # L5 barely overlaps
    r = diff.irr(a, b, tau=0.9, mode="per_class_min")
    assert r["per_class_dice"]["7"] == 1.0           # sacrum agrees
    assert r["min_class_dice"] < 0.9                 # L5 drags the min down
    assert r["agree"] is False                       # -> needs adjudication


def test_irr_identical_labels_agree():
    a = _vol(); a[0:3, 0, 0] = 5; a[0:3, 1, 0] = 8
    assert diff.irr(a, a, tau=0.9)["agree"] is True


def test_irr_both_empty_agree():
    z = _vol()
    r = diff.irr(z, z, tau=0.9)
    assert r["agree"] is True and r["metric"] == 1.0


def test_label_diff_shape_mismatch_raises():
    with pytest.raises(ValueError):
        diff.label_diff(_vol(), np.zeros((4, 4, 4), dtype=np.int16))


# --------------------------------------------------------------------------- #
# provenance transitions
# --------------------------------------------------------------------------- #

def test_provenance_flips_only_reviewed_pseudo_region():
    pb = {"spine": "manual", "pelvis": "pseudo"}
    assert schema.provenance_after(pb, "pelvis", "corrected") == \
        {"spine": "manual", "pelvis": "pseudo_corrected"}
    # reviewing the manual side never changes it
    assert schema.provenance_after(pb, "spine", "corrected") == \
        {"spine": "manual", "pelvis": "pseudo"}


def test_provenance_accept_also_flips_pseudo():
    pb = {"spine": "pseudo", "pelvis": "manual"}
    assert schema.provenance_after(pb, "spine", "accept")["spine"] == \
        "pseudo_corrected"


def test_provenance_reject_and_null_unchanged():
    pb = {"spine": "manual", "pelvis": "pseudo"}
    assert schema.provenance_after(pb, "pelvis", "reject") == pb
    pb2 = {"spine": "pseudo", "pelvis": None}
    assert schema.provenance_after(pb2, "pelvis", "corrected")["pelvis"] is None


# --------------------------------------------------------------------------- #
# review-record validation
# --------------------------------------------------------------------------- #

def _good_corrected() -> schema.ReviewRecord:
    pb = {"spine": "manual", "pelvis": "pseudo"}
    return schema.ReviewRecord(
        review_id=schema.review_id("6", "spine_only", "rev_03"),
        token="6", config="spine_only", source_revision="v2",
        source_label_sha256="abc", reviewer_id="rev_03",
        decision="corrected", region_reviewed="pelvis",
        diff={"n_voxels_changed": 5}, corrected_label_sha256="def",
        artifact="reviews/6_spine/rev03_r1.nii.gz",
        prov_before=pb, prov_after=schema.provenance_after(pb, "pelvis", "corrected"),
    )


def test_valid_corrected_record_passes():
    assert schema.validate_review_record(_good_corrected()) == []


def test_corrected_missing_artifact_fails():
    r = _good_corrected(); r.artifact = None
    errs = schema.validate_review_record(r)
    assert any("artifact" in e for e in errs)


def test_bad_decision_and_region_fail():
    r = _good_corrected(); r.decision = "approve"; r.region_reviewed = "leg"
    errs = schema.validate_review_record(r)
    assert any("decision" in e for e in errs)
    assert any("region_reviewed" in e for e in errs)


def test_prov_after_inconsistent_with_transition_fails():
    r = _good_corrected()
    r.prov_after = {"spine": "pseudo_corrected", "pelvis": "pseudo_corrected"}
    assert any("prov_after" in e for e in schema.validate_review_record(r))


def test_review_record_roundtrips_dict():
    r = _good_corrected()
    assert schema.ReviewRecord.from_dict(r.to_dict()).to_dict() == r.to_dict()


# --------------------------------------------------------------------------- #
# double-review state machine
# --------------------------------------------------------------------------- #

def test_claimable_slot_distinctness_and_fill_order():
    case: dict = {"slots": {}}
    assert schema.claimable_primary_slot(case, "rev_a") == "1"
    case["slots"]["1"] = {"reviewer": "rev_a", "done": False,
                          "expires_at": "9999"}
    assert schema.claimable_primary_slot(case, "rev_b") == "2"   # 2nd slot
    assert schema.claimable_primary_slot(case, "rev_a") is None  # A != B
    case["slots"]["2"] = {"reviewer": "rev_b", "done": False,
                          "expires_at": "9999"}
    assert schema.claimable_primary_slot(case, "rev_c") is None  # full


def test_claimable_slot_reclaims_expired():
    case = {"slots": {"1": {"reviewer": "rev_a", "done": False,
                            "expires_at": "2000-01-01T00:00:00+00:00"}}}
    # rev_b can take slot 1 back (expired, not submitted)
    assert schema.claimable_primary_slot(case, "rev_b", now="2026-01-01") == "1"


def test_derive_status_lifecycle():
    assert schema.derive_status({"slots": {}}) == "unassigned"
    assert schema.derive_status(
        {"slots": {"1": {"reviewer": "a", "done": False}}}) == "in_review"
    both = {"slots": {"1": {"reviewer": "a", "done": True},
                      "2": {"reviewer": "b", "done": True}}}
    assert schema.derive_status(both, agree=None) == "in_review"
    assert schema.derive_status(both, agree=True) == "finalized"
    assert schema.derive_status(both, agree=False) == "needs_adjudication"
    assert schema.derive_status(
        {"slots": {}, "final": {"decision": "reject"}}) == "excluded"
    assert schema.derive_status(
        {"slots": {}, "final": {"decision": "corrected"}}) == "finalized"
    assert schema.derive_status(
        {"slots": {"adj": {"reviewer": "snr", "done": True}}}) == "finalized"


# --------------------------------------------------------------------------- #
# v3 reducer
# --------------------------------------------------------------------------- #

def test_apply_reviews_flips_prov_swaps_labels_drops_rejects():
    records = [
        {"token": "6", "config": "pelvic_native", "prov_spine": "pseudo",
         "prov_pelvis": "manual", "ct_file": "ct/0006_pelvic_ct.nii.gz",
         "label_file": "labels/0006_pelvic_label.nii.gz"},
        {"token": "6", "config": "spine_only", "prov_spine": "manual",
         "prov_pelvis": "pseudo", "ct_file": "ct/0006_spine_ct.nii.gz",
         "label_file": "labels/0006_spine_label.nii.gz"},
        {"token": "7", "config": "fused", "prov_spine": "manual",
         "prov_pelvis": "manual", "ct_file": "ct/0007_ct.nii.gz",
         "label_file": "labels/0007_label.nii.gz"},          # not reviewed
        {"token": "9", "config": "spine_only", "prov_spine": "manual",
         "prov_pelvis": "pseudo", "ct_file": "ct/0009_spine_ct.nii.gz",
         "label_file": "labels/0009_spine_label.nii.gz"},     # rejected
    ]
    finals = {
        "6__pelvic_native": {"decision": "corrected",
                             "prov_after": {"spine": "pseudo_corrected",
                                            "pelvis": "manual"},
                             "label_rel": "reviews/a/final.nii.gz"},
        "6__spine_only": {"decision": "accept",
                          "prov_after": {"spine": "manual",
                                         "pelvis": "pseudo_corrected"}},
        "9__spine_only": {"decision": "reject",
                          "prov_after": {"spine": "manual", "pelvis": "pseudo"}},
    }
    new, swaps, dropped = reduce_to_v3.apply_reviews_to_records(records, finals)

    by = {schema.case_id(r["token"], r["config"]): r for r in new}
    assert "9__spine_only" not in by and dropped == ["9__spine_only"]
    # corrected: spine pseudo->pseudo_corrected, manual pelvis untouched, swap
    assert by["6__pelvic_native"]["prov_spine"] == "pseudo_corrected"
    assert by["6__pelvic_native"]["prov_pelvis"] == "manual"
    assert swaps == {"labels/0006_pelvic_label.nii.gz": "reviews/a/final.nii.gz"}
    # accept: pelvis pseudo->pseudo_corrected, no label swap, manual untouched
    assert by["6__spine_only"]["prov_pelvis"] == "pseudo_corrected"
    assert by["6__spine_only"]["prov_spine"] == "manual"
    assert "labels/0006_spine_label.nii.gz" not in swaps
    # unreviewed fused passes through unchanged
    assert by["7__fused"]["prov_spine"] == "manual"


# --------------------------------------------------------------------------- #
# label descriptor
# --------------------------------------------------------------------------- #

def test_label_descriptor_locks_all_classes(tmp_path):
    txt = labels_descriptor.descriptor_text()
    for name in ("L1", "L5", "L6", "sacrum", "left_hip", "right_hip",
                 "Clear Label", "IGNORE"):
        assert f'"{name}"' in txt
    p = labels_descriptor.write_label_descriptor(tmp_path / "labels.txt")
    assert p.exists() and p.read_text() == txt
