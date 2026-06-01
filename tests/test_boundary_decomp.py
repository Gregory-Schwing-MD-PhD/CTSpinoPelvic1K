"""Unit tests for the boundary/interior Dice decomposition core."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from boundary_decomp import decompose_class  # noqa: E402


def test_perfect_match():
    gt = np.zeros((12, 12, 12), dtype=bool)
    gt[3:9, 3:9, 3:9] = True
    m = decompose_class(gt, gt.copy(), k=1)
    assert m["dice"] == 1.0
    assert m["n_err"] == 0


def test_surface_shift_is_boundary_error():
    # a 1-voxel shift puts ALL disagreement at the surface -> boundary, and
    # the tolerant Dice should jump to ~1.0.
    gt = np.zeros((16, 16, 16), dtype=bool)
    gt[4:12, 4:12, 4:12] = True
    pred = np.zeros((16, 16, 16), dtype=bool)
    pred[5:13, 4:12, 4:12] = True              # shifted by 1 along axis 0
    m = decompose_class(gt, pred, k=1)
    assert m["boundary_frac"] == 1.0           # all error within 1 vox of surface
    assert m["interior_err"] == 0
    assert m["dice_tolerant"] > m["dice"]
    assert m["dice_tolerant"] >= 0.99


def test_far_blob_is_interior_error():
    # a detached mislabel far from the GT surface = interior (fixable) error.
    gt = np.zeros((20, 20, 20), dtype=bool)
    gt[3:7, 3:7, 3:7] = True
    pred = gt.copy()
    pred[14:18, 14:18, 14:18] = True           # spurious far component
    m = decompose_class(gt, pred, k=1)
    assert m["interior_err"] > 0
    assert m["boundary_frac"] < 1.0
