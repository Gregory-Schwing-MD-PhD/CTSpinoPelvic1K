"""
Unit tests for the intensity-refinement core (manual-calibrated threshold +
connected-component gating). Pure numpy/scipy; no nibabel / no inference.

Locks the behavior that matters:
  * the bone threshold is read off the MANUAL annotation (per-case calibration);
  * threshold artifacts / unrelated bone (ribs) are dropped by keeping only
    connected components that OVERLAP the model prediction;
  * marrow interiors are solidified (hole fill);
  * manual voxels are NEVER modified; the pseudo region is re-segmented and
    each kept voxel takes the nearest predicted class.
"""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from intensity_refine import (  # noqa: E402
    calibrate_threshold,
    refine_label,
)

SPINE = {1, 2, 3, 4, 5, 6}
PELVIS = {7, 8, 9}


# --------------------------------------------------------------------------- #
# calibrate_threshold
# --------------------------------------------------------------------------- #

def test_calibrate_reads_percentile_of_manual_hu():
    ct = np.array([[[100.0, 200.0, 300.0, -1000.0]]], dtype=np.float32)
    mask = np.array([[[True, True, True, False]]])          # bone = {100,200,300}
    assert calibrate_threshold(ct, mask, percentile=0, erode_iter=0) == 100.0
    assert calibrate_threshold(ct, mask, percentile=50, erode_iter=0) == 200.0


def test_calibrate_none_without_manual_bone():
    ct = np.zeros((1, 1, 4), dtype=np.float32)
    assert calibrate_threshold(ct, np.zeros((1, 1, 4), bool), erode_iter=0) is None


# --------------------------------------------------------------------------- #
# refine_label
# --------------------------------------------------------------------------- #

def test_refine_segments_pseudo_from_intensity_and_drops_stray_bone():
    label = np.zeros((1, 7, 7), dtype=np.int16)
    label[0, 1, 1] = 3                       # manual spine (calibration source)
    label[0, 1, 2] = 3
    label[0, 4, 4] = 7                       # model predicts sacrum here
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 1, 1] = 150.0; ct[0, 1, 2] = 250.0           # manual bone HU
    ct[0, 4, 3] = ct[0, 4, 4] = ct[0, 4, 5] = 300.0    # pseudo-region bone bar
    ct[0, 6, 0] = 300.0                                # stray rib — separate CC

    out, thr = refine_label(label, ct, SPINE, PELVIS,
                            percentile=10, erode_iter=0, fill_holes=False)
    assert out[0, 1, 1] == 3 and out[0, 1, 2] == 3     # manual untouched
    assert out[0, 4, 3] == 7 and out[0, 4, 4] == 7 and out[0, 4, 5] == 7
    assert out[0, 6, 0] == 0                            # stray bone NOT kept
    assert 150.0 <= thr <= 250.0                        # calibrated off manual


def test_refine_fills_marrow_when_prediction_overlaps_bone():
    label = np.zeros((1, 7, 7), dtype=np.int16)
    label[0, 0, 0] = 3                       # manual spine (calibration)
    for di in (-1, 0, 1):                    # model predicts the whole structure
        for dj in (-1, 0, 1):
            label[0, 4 + di, 4 + dj] = 8
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 300.0
    for di in (-1, 0, 1):                    # bone ring, hollow (marrow) centre
        for dj in (-1, 0, 1):
            if di or dj:
                ct[0, 4 + di, 4 + dj] = 300.0

    out, _ = refine_label(label, ct, SPINE, PELVIS,
                          percentile=50, erode_iter=0, fill_holes=True)
    assert out[0, 4, 4] == 8                 # enclosed marrow filled + labelled
    assert out[0, 4, 3] == 8


def test_refine_never_modifies_manual_even_if_bone_and_adjacent_pred():
    label = np.zeros((1, 3, 3), dtype=np.int16)
    label[0, 1, 1] = 5                        # manual lumbar
    label[0, 1, 0] = 7                        # pseudo prediction next to it
    ct = np.full((1, 3, 3), 300.0, dtype=np.float32)   # everything bone
    out, _ = refine_label(label, ct, SPINE, PELVIS,
                          percentile=50, erode_iter=0, fill_holes=False)
    assert out[0, 1, 1] == 5                  # manual wins over bone + neighbour


def test_refine_no_prediction_returns_unchanged():
    label = np.zeros((1, 1, 3), dtype=np.int16)
    label[0, 0, 2] = 3                        # manual only, no pseudo classes
    ct = np.full((1, 1, 3), 300.0, dtype=np.float32)
    out, thr = refine_label(label, ct, SPINE, PELVIS)
    assert thr is None
    assert out.tolist() == label.tolist()
