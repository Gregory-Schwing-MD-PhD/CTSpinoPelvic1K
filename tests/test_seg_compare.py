"""Unit tests for the seg_compare metric cores (Dice/volumes + surface dist)."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from seg_compare import dice_volumes, surface_distance  # noqa: E402


def test_dice_volumes_identical_is_one():
    model = np.zeros((1, 4, 4), dtype=np.int16)
    model[0, 1:3, 1:3] = 7
    fillable = np.ones((1, 4, 4), dtype=bool)
    out = dice_volumes(model, model.copy(), fillable, (7, 8, 9))
    assert out[7]["dice"] == 1.0
    assert out[7]["vol_model"] == 4 and out[7]["vol_intensity"] == 4
    assert out[7]["vol_ratio"] == 1.0
    # classes absent from both -> NaN dice, zero volumes
    assert out[8]["dice"] != out[8]["dice"]        # NaN
    assert out[8]["vol_model"] == 0


def test_dice_volumes_partial_overlap_and_fillable_mask():
    model = np.zeros((1, 1, 4), dtype=np.int16)
    intensity = np.zeros((1, 1, 4), dtype=np.int16)
    model[0, 0, 0] = model[0, 0, 1] = 7            # 2 voxels
    intensity[0, 0, 1] = intensity[0, 0, 2] = 7    # 2 voxels, overlap = 1
    fillable = np.ones((1, 1, 4), dtype=bool)
    out = dice_volumes(model, intensity, fillable, (7,))
    assert out[7]["dice"] == 2 * 1 / (2 + 2)       # = 0.5
    # masking: exclude the overlap voxel -> no intersection
    fillable[0, 0, 1] = False
    out2 = dice_volumes(model, intensity, fillable, (7,))
    assert out2[7]["dice"] == 0.0


def test_surface_distance_identical_is_zero_and_empty_is_nan():
    a = np.zeros((1, 6, 6), dtype=bool)
    a[0, 1:5, 1:5] = True
    assert surface_distance(a, a.copy()) == 0.0
    assert surface_distance(a, np.zeros_like(a)) != surface_distance(a, np.zeros_like(a))  # NaN


def test_surface_distance_grows_with_displacement():
    a = np.zeros((1, 9, 9), dtype=bool)
    a[0, 2:7, 2:7] = True
    b = np.zeros((1, 9, 9), dtype=bool)
    b[0, 3:8, 3:8] = True            # same-size block shifted by (1,1)
    d = surface_distance(a, b)
    assert 0.0 < d < 3.0             # a small, positive boundary distance
