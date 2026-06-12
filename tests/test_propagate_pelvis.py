"""Unit tests for propagate_pelvis pure QC/gating math (no ANTs needed).

These cover the bone-safety mechanism: a deformable warp that compresses rigid
bone (det J != 1) must be flagged, a warped bone floating off target bone-HU must
be flagged, and the per-case gate must reject if ANY bone fails."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import propagate_pelvis as P  # noqa: E402


def test_jacobian_clean_bone_passes():
    jac = np.ones((10, 10, 10), np.float32)
    mask = np.zeros((10, 10, 10), bool); mask[2:8, 2:8, 2:8] = True
    med, bad, n = P.jacobian_stats_in_mask(jac, mask, tol=0.3)
    assert abs(med - 1.0) < 1e-6 and bad == 0.0 and n == mask.sum()


def test_jacobian_squashed_bone_flagged():
    """A deformable field compressing bone to half volume -> all voxels bad."""
    jac = np.ones((10, 10, 10), np.float32)
    mask = np.zeros((10, 10, 10), bool); mask[2:8, 2:8, 2:8] = True
    jac[mask] = 0.5
    _, bad, _ = P.jacobian_stats_in_mask(jac, mask, tol=0.3)
    assert bad == 1.0


def test_jacobian_empty_mask_is_nan():
    jac = np.ones((4, 4, 4), np.float32)
    med, bad, n = P.jacobian_stats_in_mask(jac, np.zeros((4, 4, 4), bool))
    assert n == 0 and med != med and bad != bad     # NaN


def test_bone_fit_fraction():
    ct = np.full((6, 6, 6), -100.0, np.float32); ct[0:3] = 300
    on = np.zeros((6, 6, 6), bool); on[0:3] = True
    off = np.zeros((6, 6, 6), bool); off[3:6] = True
    assert P.bone_fit_fraction(ct, on) == 1.0
    assert P.bone_fit_fraction(ct, off) == 0.0


def test_volume_ratio():
    wb = np.zeros((4, 4, 4), bool); wb[:2] = True       # 32 voxels
    assert abs(P.volume_ratio(wb, 32) - 1.0) < 1e-6
    assert abs(P.volume_ratio(wb, 64) - 0.5) < 1e-6     # FOV-truncated
    assert P.volume_ratio(wb, 0) != P.volume_ratio(wb, 0)  # NaN


def test_gate_accepts_clean_case():
    good = {7: {"n_warped": 100, "bone_fit": 0.95, "jac_bad": 0.0, "vol_ratio": 1.0},
            8: {"n_warped": 80, "bone_fit": 0.9, "jac_bad": 0.02, "vol_ratio": 0.98},
            9: {"n_warped": 80, "bone_fit": 0.9, "jac_bad": 0.02, "vol_ratio": 1.02}}
    g = P.gate_case(good, min_fit=0.8, max_jac_bad=0.1, vol_lo=0.7, vol_hi=1.3)
    assert g["accept"] and g["reasons"] == []


def test_gate_rejects_lowfit_bonewarp_and_truncation():
    bad = {7: {"n_warped": 100, "bone_fit": 0.4, "jac_bad": 0.0, "vol_ratio": 1.0},
           8: {"n_warped": 80, "bone_fit": 0.95, "jac_bad": 0.5, "vol_ratio": 1.0},
           9: {"n_warped": 30, "bone_fit": 0.95, "jac_bad": 0.0, "vol_ratio": 0.3}}
    g = P.gate_case(bad, min_fit=0.8, max_jac_bad=0.1, vol_lo=0.7, vol_hi=1.3)
    assert not g["accept"]
    assert any("sacrum_lowfit" in r for r in g["reasons"])
    assert any("left_hip_bonewarp" in r for r in g["reasons"])
    assert any("right_hip_volratio" in r for r in g["reasons"])


def test_to_canonical_pelvis_remaps_raw_and_passes_canonical():
    raw = np.array([[0, 1, 2, 3]], np.int16)
    out = P._to_canonical_pelvis(raw)
    assert list(out.ravel()) == [0, 7, 8, 9]
    already = np.array([[0, 7, 8, 9]], np.int16)
    assert list(P._to_canonical_pelvis(already).ravel()) == [0, 7, 8, 9]


def test_manifest_case_is_place_fused_style():
    """The propagated pelvis is emitted as a placed-mask case on the SPINE series
    (fused-like), tagged manual_propagated, carrying native-vs-propagated overlap."""
    case = {"patient_token": "17", "position": "PRONE", "sex": "M",
            "spine": {"series_uid": "1.2.3", "placed": "/d/spine.nii.gz",
                      "bone_pct": 88.0, "position": "PRONE"},
            "pelvic": {"series_uid": "9.9.9", "placed": "/d/pelv.nii.gz",
                       "bone_pct": 90.0}}
    row = {"token": "17", "spine_uid": "1.2.3", "pelvic_uid": "9.9.9",
           "out_file": "/d/1.2.3_pelvic_propagated.nii.gz", "accept": 1,
           "reasons": "", "prop_bone_pct": 89.4, "src_bone_pct": 90.0,
           "bone_pct_drop": 0.6, "native_bone_pct_manifest": 90.0,
           "sacrum_bonepct": 91.0, "sacrum_jacmed": 1.0, "sacrum_jacbad": 0.0,
           "sacrum_volratio": 1.0}
    mc = P._build_manifest_case(case, row)
    assert mc["match_type"] == "propagated"
    assert mc["prov_spine"] == "manual" and mc["prov_pelvis"] == "manual_propagated"
    # propagated pelvis lives on the SPINE series -> spine & pelvic share series_uid
    assert mc["pelvic"]["series_uid"] == "1.2.3"
    assert mc["pelvic"]["source_series_uid"] == "9.9.9"   # where the GT came from
    assert mc["pelvic"]["bone_pct_before"] == 90.0
    assert mc["pelvic"]["bone_pct_after"] == 89.4
    assert mc["position"] == "PRONE" and mc["sex"] == "M"   # carried demographics
    assert mc["propagation"]["accept"] == 1
    assert mc["propagation"]["seed"] == P.SEED
    assert "sacrum" in mc["propagation"]["per_bone"]
