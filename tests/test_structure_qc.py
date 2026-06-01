"""Unit tests for structure_qc: L/R swap, vertebra gap, pelvis, duplication."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from structure_qc import structure_metrics  # noqa: E402

EYE = np.eye(4)        # world-x == voxel axis 0; +x = "Right"


def _place(vol, lab, x):
    """A small blob for `lab` centred near voxel-x = x."""
    vol[x:x + 3, 4:7, 4:7] = lab


def test_correct_laterality_no_swap():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    _place(v, 8, 2)        # left_hip at LOW x  (patient Left = -x side)
    _place(v, 7, 13)       # sacrum midline
    _place(v, 9, 24)       # right_hip at HIGH x (patient Right = +x)
    m = structure_metrics(v, EYE)
    assert m["lr_known"] == 1
    assert m["lr_swap"] == 0
    assert m["lr_same_side"] == 0
    assert m["struct_flag"] == 0


def test_swapped_hips_flagged():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    _place(v, 9, 2)        # right_hip mistakenly on the LOW-x (Left) side
    _place(v, 7, 13)
    _place(v, 8, 24)       # left_hip on the HIGH-x (Right) side
    m = structure_metrics(v, EYE)
    assert m["lr_swap"] == 1
    assert m["struct_flag"] == 1


def test_flip_lr_inverts_convention():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    _place(v, 8, 2)
    _place(v, 7, 13)
    _place(v, 9, 24)
    # under flipped convention the previously-correct layout reads as swapped
    assert structure_metrics(v, EYE, flip_lr=True)["lr_swap"] == 1


def test_vertebra_gap_detected():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    _place(v, 1, 2)        # L1
    _place(v, 2, 6)        # L2
    _place(v, 4, 14)       # L4  (L3 missing -> gap of 1)
    m = structure_metrics(v, EYE)
    assert m["vertebra_gap"] == 1
    assert m["struct_flag"] == 1


def test_pelvis_incomplete_flagged():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    _place(v, 7, 13)       # sacrum only, hips missing
    _place(v, 8, 2)
    m = structure_metrics(v, EYE)        # 2 of 3 pelvis structures
    assert m["pelvis_incomplete"] == 1
    assert m["lr_known"] == 0            # can't judge L/R without all three
    assert m["struct_flag"] == 1


def test_duplicated_structure_flagged():
    v = np.zeros((30, 12, 12), dtype=np.int16)
    v[2:5, 4:7, 4:7] = 3                 # L3 component A (27 vox)
    v[20:23, 4:7, 4:7] = 3              # L3 component B, comparable size -> real split
    m = structure_metrics(v, EYE, min_dup_vox=10)
    assert m["n_dup_classes"] == 1
    assert m["duplication_flag"] == 1


def test_small_fragment_does_not_flag_duplication():
    # a big structure with a tiny detached speck must NOT trip duplication
    # (2nd component is below the size ratio).
    v = np.zeros((30, 12, 12), dtype=np.int16)
    v[2:10, 2:10, 2:10] = 3              # large component (512 vox)
    v[20:21, 5:6, 5:6] = 3             # 1-voxel speck
    m = structure_metrics(v, EYE, min_dup_vox=10, dup_ratio=0.2)
    assert m["n_dup_classes"] == 0
    assert m["duplication_flag"] == 0
