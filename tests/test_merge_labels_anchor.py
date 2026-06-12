"""
test_merge_labels_anchor.py — the rib anchor (VerSe T12=19 -> class 11) is
retained from CTSpine1K GT by export_hf.merge_labels, without disturbing the
lumbar, sacrum-fill, or partial-mode IGNORE behaviour.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_hf import merge_labels, IGNORE_LABEL  # noqa: E402


def _w(arr):
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib.Nifti1Image(arr.astype(np.int16), np.eye(4)), f)
    return f


def _spine():
    shape = (8, 8, 8)
    sp = np.zeros(shape, np.int16)
    sp[0, 0, 0] = 19                       # T12 = anchor
    sp[1, 0, 0] = 20                       # L1
    sp[2, 0, 0] = 21
    sp[3, 0, 0] = 22
    sp[4, 0, 0] = 23
    sp[5, 0, 0] = 24                       # L5
    sp[6, 0, 0] = 26                       # sacrum (from spine)
    return shape, sp


def test_fused_retains_anchor_and_preserves_regions():
    shape, sp = _spine()
    pe = np.zeros(shape, np.int16)
    pe[6, 1, 0] = 1                        # pelvic sacrum -> 7
    pe[7, 2, 0] = 2                        # left hip -> 8
    pe[7, 3, 0] = 3                        # right hip -> 9
    r = merge_labels(_w(sp), _w(pe), shape)

    assert r[0, 0, 0] == 11               # VerSe 19 (T12) -> last_rib_vertebra
    assert [int(r[i, 0, 0]) for i in range(1, 6)] == [1, 2, 3, 4, 5]  # L1-L5
    assert r[6, 0, 0] == 7                # sacrum-from-spine fill (was bg)
    assert (r[6, 1, 0], r[7, 2, 0], r[7, 3, 0]) == (7, 8, 9)          # pelvic kept


def test_partial_spine_only_anchor_and_ignore():
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    assert r[0, 0, 0] == 11               # anchor retained
    assert r[7, 7, 7] == IGNORE_LABEL     # untraced region -> IGNORE, not bg


def test_anchor_not_overwritten_by_sacrum_fill():
    # the anchor sits above the lumbar column; sacrum-fill must never touch it
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    assert r[0, 0, 0] == 11
    assert int((r == 11).sum()) == 1      # exactly the one anchor voxel
