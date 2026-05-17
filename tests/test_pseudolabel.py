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


def test_residual_ignore_keeps_partial_flag_true():
    # If (hypothetically) IGNORE survived, partial must stay True. The merge
    # function collapses IGNORE, so feed updated_record an array with IGNORE
    # directly to lock the flag's semantics independently.
    merged = np.array([5, IGNORE_LABEL], dtype=np.int16)
    out = updated_record(_rec(), "pelvis", merged)
    assert out["partial_annotation"] is True
    assert out["n_voxels_ignore"] == 1
