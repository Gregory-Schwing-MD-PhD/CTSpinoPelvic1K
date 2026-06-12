"""
Unit tests for the model-INDEPENDENT pseudo-label merge core.

Inference (nnU-Net) is intentionally NOT exercised — only the parts that
encode the hard provenance contract, which must hold regardless of which
model / label scheme is finally trained:

  * a manual voxel (1..9) is NEVER overwritten by a prediction;
  * pseudo only fills background / IGNORE_LABEL voxels;
  * remaining IGNORE collapses to background (case no longer partial);
  * only the FILLED region's provenance flips to "pseudo";
  * a prediction class outside the model's supplied canonical set is dropped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pseudolabel import (  # noqa: E402
    IGNORE_LABEL,
    build_heldout_fold_map,
    load_propagated_map,
    merge_pseudo_into_manual,
    remap_prediction,
    updated_record,
)


# --------------------------------------------------------------------------- #
# remap_prediction
# --------------------------------------------------------------------------- #

def test_remap_keeps_only_supplied_canonical_and_drops_rest():
    pred = np.array([0, 1, 2, 3, 99], dtype=np.int16)
    # model-output 1->7, 2->8, 3->9 ; 99 not in map -> dropped
    out = remap_prediction(pred, {"1": 7, "2": 8, "3": 9}, [7, 8, 9])
    assert out.tolist() == [0, 7, 8, 9, 0]


def test_remap_discards_target_outside_supplied_set():
    pred = np.array([1, 2], dtype=np.int16)
    # 2 -> 5 but 5 not in supplied (defensive: spine class into pelvis model)
    out = remap_prediction(pred, {"1": 7, "2": 5}, [7, 8, 9])
    assert out.tolist() == [7, 0]


# --------------------------------------------------------------------------- #
# merge_pseudo_into_manual — the hard contract
# --------------------------------------------------------------------------- #

def test_manual_voxels_are_never_overwritten():
    manual = np.array([3, 0, IGNORE_LABEL, 7], dtype=np.int16)   # 3 & 7 manual
    pred   = np.array([9, 8, 9, 1], dtype=np.int16)              # tries to overwrite
    merged = merge_pseudo_into_manual(manual, pred)
    # manual 3 and 7 untouched; bg/IGNORE filled from pred
    assert merged.tolist() == [3, 8, 9, 7]


def test_unfilled_ignore_collapses_to_background():
    manual = np.array([IGNORE_LABEL, IGNORE_LABEL, 4], dtype=np.int16)
    pred   = np.array([0, 7, 0], dtype=np.int16)   # only middle voxel filled
    merged = merge_pseudo_into_manual(manual, pred)
    # first IGNORE not filled -> 0 ; second -> 7 ; manual 4 kept
    assert merged.tolist() == [0, 7, 4]
    assert not (merged == IGNORE_LABEL).any()


def test_zero_prediction_does_not_fabricate():
    manual = np.array([0, 0, 5], dtype=np.int16)
    pred   = np.zeros(3, dtype=np.int16)
    merged = merge_pseudo_into_manual(manual, pred)
    assert merged.tolist() == [0, 0, 5]


def test_merge_returns_copy_not_view():
    manual = np.array([0, 2], dtype=np.int16)
    pred   = np.array([7, 0], dtype=np.int16)
    merged = merge_pseudo_into_manual(manual, pred)
    assert merged is not manual
    assert manual.tolist() == [0, 2]   # input untouched


# --------------------------------------------------------------------------- #
# updated_record — provenance only flips the filled side
# --------------------------------------------------------------------------- #

def _rec():
    return {"token": "0042", "config": "spine_only",
            "prov_spine": "manual", "prov_pelvis": None,
            "partial_annotation": True,
            "n_voxels_ignore": 9, "n_voxels_fg": 1, "n_voxels_bg": 0}


def test_fill_pelvis_flips_only_pelvis_prov():
    merged = np.array([5, 7, 8, 0], dtype=np.int16)   # spine 5 manual; 7,8 pseudo
    out = updated_record(_rec(), "pelvis", merged)
    assert out["prov_pelvis"] == "pseudo"
    assert out["prov_spine"] == "manual"          # never downgraded
    assert out["partial_annotation"] is False
    assert out["n_voxels_ignore"] == 0
    assert out["n_voxels_fg"] == 3
    assert out["n_voxels_bg"] == 1


def test_fill_pelvis_with_propagated_prov():
    merged = np.array([1, 7, 8, 9], dtype=np.int16)
    out = updated_record(_rec(), "pelvis", merged, prov="manual_propagated")
    assert out["prov_pelvis"] == "manual_propagated"   # real GT, not "pseudo"


def test_fill_spine_flips_only_spine_prov():
    rec = {"token": "1", "config": "pelvic_native",
           "prov_spine": None, "prov_pelvis": "manual",
           "partial_annotation": True}
    merged = np.array([1, 2, 7], dtype=np.int16)
    out = updated_record(rec, "spine", merged)
    assert out["prov_spine"] == "pseudo"
    assert out["prov_pelvis"] == "manual"


def test_updated_record_does_not_mutate_input():
    rec = _rec()
    updated_record(rec, "pelvis", np.array([7], dtype=np.int16))
    assert rec["prov_pelvis"] is None and rec["partial_annotation"] is True


_SPLITS_V2 = {   # spinopelvic-seg schema: train_tokens / val_tokens + test
    "test_tokens": ["900", "901"],
    "folds": [
        {"train_tokens": ["2", "3"], "val_tokens": ["10", "11"]},
        {"train_tokens": ["10", "3"], "val_tokens": ["2"]},
        {"train_tokens": ["10", "2"], "val_tokens": ["3"]},
    ],
}
_SPLITS_V6 = {   # CTSpinoPelvic1K schema: fold/train/val
    "folds": [
        {"fold": 0, "train": ["2"], "val": ["10"]},
        {"fold": 1, "train": ["10"], "val": ["2"]},
    ],
}


def test_heldout_fold_map_v2_schema_token_to_validation_fold():
    m = build_heldout_fold_map(_SPLITS_V2)
    # a token's held-out fold = the fold whose val set holds it
    assert m["10"] == 0 and m["11"] == 0
    assert m["2"] == 1
    assert m["3"] == 2
    # test-only tokens are NOT in the map -> caller uses ensemble (they
    # were never trained on, so no leakage either way)
    assert "900" not in m and "901" not in m


def test_heldout_fold_map_v6_schema_and_fold_index():
    m = build_heldout_fold_map(_SPLITS_V6)
    assert m == {"10": 0, "2": 1}


def test_heldout_fold_never_returns_a_fold_that_trained_the_token():
    # The whole point: for every mapped token, it must be ABSENT from that
    # fold's train set (that fold's model never saw it).
    m = build_heldout_fold_map(_SPLITS_V2)
    for tok, fold in m.items():
        train = _SPLITS_V2["folds"][fold].get("train_tokens", [])
        assert tok not in train, (
            f"token {tok} held-out fold {fold} also TRAINED on it — leak"
        )


def test_heldout_fold_map_tokens_are_strings():
    m = build_heldout_fold_map({"folds": [{"fold": 0, "train": [],
                                           "val": [10, 11]}]})
    assert m == {"10": 0, "11": 0}   # int tokens normalized to str


def test_residual_ignore_keeps_partial_flag_true():
    # If (hypothetically) IGNORE survived, partial must stay True. The merge
    # function collapses IGNORE, so feed updated_record an array with IGNORE
    # directly to lock the flag's semantics independently.
    merged = np.array([5, IGNORE_LABEL], dtype=np.int16)
    out = updated_record(_rec(), "pelvis", merged)
    assert out["partial_annotation"] is True
    assert out["n_voxels_ignore"] == 1


# --------------------------------------------------------------------------- #
# load_propagated_map — only ACCEPTED real-GT pelves, model fallback otherwise
# --------------------------------------------------------------------------- #

def test_load_propagated_map_keeps_only_accepted(tmp_path):
    man = {"cases": [
        {"patient_token": "17", "pelvic": {"series_uid": "1.2.3",
         "placed": "/data/prop/1.2.3_pelvic_propagated.nii.gz"},
         "propagation": {"accept": 1}},
        {"patient_token": "18", "pelvic": {"series_uid": "4.5.6",
         "placed": "/data/prop/4.5.6_pelvic_propagated.nii.gz"},
         "propagation": {"accept": 0}},          # rejected -> excluded -> model
    ]}
    p = tmp_path / "placed_manifest_propagated.json"
    p.write_text(json.dumps(man))
    m = load_propagated_map(p)
    assert set(m) == {"17"}                      # only the accepted token
    assert m["17"]["spine_uid"] == "1.2.3"


def test_load_propagated_map_falls_back_to_dir_for_path(tmp_path):
    man = {"cases": [{"patient_token": "17",
            "pelvic": {"series_uid": "1.2.3", "placed": "/gone/missing.nii.gz"},
            "propagation": {"accept": 1}}]}
    p = tmp_path / "m.json"; p.write_text(json.dumps(man))
    pdir = tmp_path / "prop"; pdir.mkdir()
    (pdir / "1.2.3_pelvic_propagated.nii.gz").write_text("x")   # exists -> used
    m = load_propagated_map(p, propagated_dir=pdir)
    assert m["17"]["path"].endswith("1.2.3_pelvic_propagated.nii.gz")
    assert str(pdir) in m["17"]["path"]


def test_load_propagated_map_missing_manifest_is_empty():
    assert load_propagated_map(None) == {}
