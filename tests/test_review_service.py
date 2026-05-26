"""
End-to-end tests for the review service core (review_service/service.py)
on the LocalBackend — no FastAPI, no HF, no ITK-SNAP. Exercises the full
double-review + adjudication lifecycle, A≠B distinctness, IRR-driven
auto-finalize vs adjudication, reject, and the finals index.

Labels are tiny .npy volumes (the service's _load_label_array supports
.npy for exactly this) so IRR is computed without nibabel.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "scripts", _ROOT / "review_service"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import service as svc          # noqa: E402
import store as store_mod      # noqa: E402
from review import schema      # noqa: E402


def _npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.int16))
    return buf.getvalue()


def _label(sacrum_voxels: int):
    """A pelvis label (region reviewed for a spine_only case): `sacrum_voxels`
    of class 7 plus fixed hips, so IRR varies with sacrum_voxels."""
    a = np.zeros((6, 6, 6), dtype=np.int16)
    a[0:4, 0, 0] = 5            # manual spine (identical both reviewers)
    a[0:sacrum_voxels, 1, 0] = 7
    a[0:4, 2, 0] = 8
    a[0:4, 3, 0] = 9
    return a


def _service(tmp_path, **kw):
    st = store_mod.ReviewStore(store_mod.LocalBackend(tmp_path / "repo"))
    rec = {"token": "6", "config": "spine_only", "prov_spine": "manual",
           "prov_pelvis": "pseudo", "ct_file": "ct/0006_spine_ct.nii.gz",
           "label_file": "labels/0006_spine_label.nii.gz",
           "lstv_label": "normal"}
    store_mod.init_cases_from_manifest(st, [rec])
    return svc.ReviewService(st, v2_repo="org/CTSpinoPelvic1K", **kw), st


def _rec(decision="corrected", changed=2):
    return {"decision": decision, "source_label_sha256": "deadbeef",
            "diff": {"n_voxels_changed": changed}}


def test_init_cases_only_scoped_and_sets_region(tmp_path):
    st = store_mod.ReviewStore(store_mod.LocalBackend(tmp_path / "r"))
    n = store_mod.init_cases_from_manifest(st, [
        {"token": "6", "config": "spine_only", "prov_spine": "manual",
         "prov_pelvis": "pseudo", "ct_file": "ct/a.nii.gz",
         "label_file": "labels/a.nii.gz"},
        {"token": "7", "config": "fused", "ct_file": "ct/b.nii.gz",
         "label_file": "labels/b.nii.gz"},      # fused -> skipped
    ])
    assert n == 1
    c = st.get_case("6__spine_only")
    assert c["region_to_review"] == "pelvis"


def test_double_review_agreement_autofinalizes(tmp_path):
    s, st = _service(tmp_path)
    a = s.claim("rev_a"); b = s.claim("rev_b")
    assert a["slot"] != b["slot"]                      # two distinct slots
    assert s.claim("rev_c") is None                    # case full

    lab = _label(4)
    s.submit(a["claim_token"], _rec(changed=2), _npy(lab), "label.npy")
    out = s.submit(b["claim_token"], _rec(changed=5), _npy(lab), "label.npy")

    assert out["status"] == "finalized"
    assert out["irr"]["agree"] is True
    case = st.get_case("6__spine_only")
    assert case["final"]["decision"] == "corrected"
    # conservative pick: reviewer who changed fewer voxels (2 < 5)
    assert case["final"]["prov_after"]["pelvis"] == "pseudo_corrected"
    assert case["final"]["prov_after"]["spine"] == "manual"
    assert st.get_label_bytes(case["final"]["label_rel"]) is not None


def test_disagreement_routes_to_adjudication_then_finalizes(tmp_path):
    s, st = _service(tmp_path)
    a = s.claim("rev_a"); b = s.claim("rev_b")
    # very different sacrum extents -> low per-class Dice on class 7
    s.submit(a["claim_token"], _rec(), _npy(_label(4)), "label.npy")
    out = s.submit(b["claim_token"], _rec(), _npy(_label(1)), "label.npy")
    assert out["status"] == "needs_adjudication"
    assert out["irr"]["agree"] is False

    adj = s.adjudication_next("snr_1")
    assert adj is not None and adj["case_id"] == "6__spine_only"
    assert len(adj["reviews"]) == 2
    res = s.adjudicate(adj["claim_token"], "corrected",
                       _npy(_label(3)), "label.npy", notes="took A's sacrum")
    assert res["status"] == "finalized"
    case = st.get_case("6__spine_only")
    assert case["final"]["by"] == "snr_1"
    assert case["final"]["prov_after"]["pelvis"] == "pseudo_corrected"


def test_adjudicator_reject_excludes_case(tmp_path):
    s, st = _service(tmp_path)
    a = s.claim("rev_a"); b = s.claim("rev_b")
    s.submit(a["claim_token"], _rec(), _npy(_label(4)), "label.npy")
    s.submit(b["claim_token"], _rec(), _npy(_label(1)), "label.npy")
    adj = s.adjudication_next("snr_1")
    res = s.adjudicate(adj["claim_token"], "reject", notes="scan unusable")
    assert res["status"] == "excluded"
    assert st.get_case("6__spine_only")["final"]["decision"] == "reject"


def test_accept_path_finalizes_as_pseudo_corrected(tmp_path):
    s, st = _service(tmp_path)
    a = s.claim("rev_a"); b = s.claim("rev_b")
    lab = _label(4)
    s.submit(a["claim_token"], _rec("accept", 0), _npy(lab), "label.npy")
    out = s.submit(b["claim_token"], _rec("accept", 0), _npy(lab), "label.npy")
    assert out["status"] == "finalized"
    case = st.get_case("6__spine_only")
    assert case["final"]["decision"] == "accept"        # nobody edited
    assert case["final"]["prov_after"]["pelvis"] == "pseudo_corrected"


def test_submit_rejects_bad_token(tmp_path):
    s, _ = _service(tmp_path)
    s.claim("rev_a")
    with pytest.raises(svc.ReviewError):
        s.submit("bogus::1::xyz", _rec(), _npy(_label(4)), "label.npy")


def test_double_submit_same_content_is_idempotent(tmp_path):
    """A retry of the SAME work (lost-response recovery / `reviewtool resume`)
    succeeds as a duplicate; a resubmit with DIFFERENT content still errors."""
    s, _ = _service(tmp_path)
    a = s.claim("rev_a")
    lab = _label(4)
    s.submit(a["claim_token"], _rec(), _npy(lab), "label.npy")
    again = s.submit(a["claim_token"], _rec(), _npy(lab), "label.npy")
    assert again.get("duplicate") is True
    with pytest.raises(svc.ReviewError):
        s.submit(a["claim_token"], _rec(changed=99), _npy(_label(7)), "label.npy")


def test_build_finals_and_status_summary(tmp_path):
    s, st = _service(tmp_path)
    a = s.claim("rev_a"); b = s.claim("rev_b")
    lab = _label(4)
    s.submit(a["claim_token"], _rec(changed=1), _npy(lab), "label.npy")
    s.submit(b["claim_token"], _rec(changed=1), _npy(lab), "label.npy")
    finals = s.build_finals()
    assert "6__spine_only" in finals
    assert finals["6__spine_only"]["prov_after"]["pelvis"] == "pseudo_corrected"
    summ = s.status_summary()
    assert summ["by_status"].get("finalized") == 1
    assert summ["reviews_by_reviewer"]["rev_a"] == 1
    assert summ["irr_mean"] is not None
