"""Unit tests for the GT-free vertebra neighbour-mixing metrics."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from vertebra_topology_qc import vertebra_topology_metrics  # noqa: E402

EYE = np.eye(4)            # world axes == voxel axes; si_axis=2, "up" = +z


def _block(vol, lab, z0, z1):
    vol[1:4, 1:4, z0:z1] = lab


def test_clean_stack_has_no_mixing():
    v = np.zeros((6, 6, 30), dtype=np.int16)
    _block(v, 1, 24, 29)        # L1 superior (high z)
    _block(v, 2, 17, 22)        # L2
    _block(v, 3, 10, 15)        # L3 inferior, all separated
    m = vertebra_topology_metrics(v, EYE)
    assert m["n_vertebrae"] == 3
    assert m["off_main_frac"] == 0.0
    assert m["n_fragmented"] == 0
    assert m["n_order_inversions"] == 0
    assert m["n_nonadjacent_touch"] == 0
    assert m["mixing_flag"] == 0


def test_off_main_island_flags_mixing():
    v = np.zeros((6, 6, 30), dtype=np.int16)
    _block(v, 1, 24, 29)
    _block(v, 2, 17, 22)
    _block(v, 3, 10, 15)
    v[1, 1, 19] = 1             # a stray L1 voxel stranded inside L2's slab
    m = vertebra_topology_metrics(v, EYE)
    assert m["n_fragmented"] >= 1
    assert m["off_main_frac"] > 0.0
    assert m["mixing_flag"] == 1


def test_order_inversion_detected():
    v = np.zeros((6, 6, 30), dtype=np.int16)
    _block(v, 1, 24, 29)        # L1 high
    _block(v, 3, 17, 22)        # L3 ABOVE L2 -> swap
    _block(v, 2, 10, 15)        # L2 low
    m = vertebra_topology_metrics(v, EYE)
    assert m["n_order_inversions"] == 1
    assert m["mixing_flag"] == 1


def test_nonadjacent_touch_detected():
    v = np.zeros((6, 6, 30), dtype=np.int16)
    _block(v, 1, 18, 23)        # L1
    _block(v, 3, 13, 18)        # L3 touching L1 (no L2 between them)
    m = vertebra_topology_metrics(v, EYE)
    assert m["n_vertebrae"] == 2
    assert m["n_nonadjacent_touch"] == 1
    assert m["mixing_flag"] == 1


def test_orientation_sign_robustness():
    # Flip the SI axis sign in the affine: L1 now at LOW z but still superior.
    aff = np.eye(4)
    aff[2, 2] = -1.0
    v = np.zeros((6, 6, 30), dtype=np.int16)
    _block(v, 1, 1, 6)          # low z, but si_sign=-1 makes this "superior"
    _block(v, 2, 8, 13)
    _block(v, 3, 15, 20)
    m = vertebra_topology_metrics(v, aff)
    assert m["n_order_inversions"] == 0     # correctly ordered under flipped axis
