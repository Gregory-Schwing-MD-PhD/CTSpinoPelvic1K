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
    intensity_refine_pseudo,
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
# intensity_refine_pseudo — pseudo gives CLASS, CT intensity gives SHAPE
# --------------------------------------------------------------------------- #

def test_intensity_class_from_pred_shape_from_ct():
    # pred marks one sacrum (7) voxel; CT shows a bone bar through it + one
    # stray bone voxel far away (a rib the model didn't predict).
    manual = np.zeros((1, 5, 5), dtype=np.int16)
    manual[0, 0, 0] = 3                       # a manual spine voxel
    pred = np.zeros((1, 5, 5), dtype=np.int16)
    pred[0, 2, 2] = 7
    ct = np.full((1, 5, 5), -1000.0, dtype=np.float32)
    ct[0, 2, 1] = ct[0, 2, 2] = ct[0, 2, 3] = 300.0   # bone bar near pred
    ct[0, 0, 4] = 300.0                                # stray bone, far away
    out = intensity_refine_pseudo(manual, pred, ct, hu_threshold=150,
                                  dilate_vox=1, fill_holes=False)
    assert out[0, 2, 1] == 7 and out[0, 2, 2] == 7 and out[0, 2, 3] == 7
    assert out[0, 0, 4] == 0          # unrelated bone NOT swept in
    assert out[0, 0, 0] == 3          # manual voxel untouched
    assert out[0, 1, 1] == 0          # non-bone stays background


def test_intensity_fill_holes_solidifies_marrow():
    manual = np.zeros((1, 5, 5), dtype=np.int16)
    pred = np.zeros((1, 5, 5), dtype=np.int16)
    pred[0, 2, 2] = 8
    ct = np.full((1, 5, 5), -1000.0, dtype=np.float32)
    for di in (-1, 0, 1):                      # bone ring, hollow centre
        for dj in (-1, 0, 1):
            if di or dj:
                ct[0, 2 + di, 2 + dj] = 300.0
    out = intensity_refine_pseudo(manual, pred, ct, hu_threshold=150,
                                  dilate_vox=2, fill_holes=True)
    assert out[0, 2, 2] == 8           # enclosed marrow filled in
    assert out[0, 1, 1] == 8


def test_intensity_never_overwrites_manual_even_if_bone_and_predicted():
    manual = np.zeros((1, 3, 3), dtype=np.int16)
    manual[0, 1, 1] = 5                        # manual lumbar voxel
    pred = np.zeros((1, 3, 3), dtype=np.int16)
    pred[0, 1, 1] = 7                          # model wrongly predicts here
    ct = np.full((1, 3, 3), 300.0, dtype=np.float32)   # all bone
    out = intensity_refine_pseudo(manual, pred, ct, hu_threshold=150,
                                  dilate_vox=1, fill_holes=False)
    assert out[0, 1, 1] == 5           # manual wins over bone + prediction


def test_intensity_no_prediction_fills_nothing():
    manual = np.array([[[0, IGNORE_LABEL, 4]]], dtype=np.int16)
    pred = np.zeros((1, 1, 3), dtype=np.int16)
    ct = np.full((1, 1, 3), 300.0, dtype=np.float32)   # bone everywhere
    out = intensity_refine_pseudo(manual, pred, ct, hu_threshold=150)
    assert out.ravel().tolist() == [0, 0, 4]   # IGNORE->0, manual kept, no fabrication


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
