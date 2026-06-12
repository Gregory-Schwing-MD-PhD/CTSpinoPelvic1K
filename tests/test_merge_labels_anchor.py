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


def test_cervical_and_thoracic_retained_from_gt():
    # the full vertebral column is no longer dropped: cervical (VerSe 1-7 -> 13-19)
    # and thoracic T1-T11 (VerSe 8-18 -> 20-30) are retained authoritatively, with
    # T12 still the anchor (VerSe 19 -> 11) and L1-L6 unchanged (VerSe 20-25 -> 1-6).
    shape = (10, 4, 4)
    sp = np.zeros(shape, np.int16)
    sp[0, 0, 0] = 1            # C1   -> 13
    sp[1, 0, 0] = 7            # C7   -> 19
    sp[2, 0, 0] = 8            # T1   -> 20
    sp[3, 0, 0] = 18           # T11  -> 30
    sp[4, 0, 0] = 19           # T12  -> 11 (anchor, unchanged)
    sp[5, 0, 0] = 20           # L1   -> 1
    sp[6, 0, 0] = 25           # L6   -> 6
    sp[7, 0, 0] = 26           # sacrum -> 7
    r = merge_labels(_w(sp), None, shape)
    assert r[0, 0, 0] == 13    # C1
    assert r[1, 0, 0] == 19    # C7
    assert r[2, 0, 0] == 20    # T1
    assert r[3, 0, 0] == 30    # T11
    assert r[4, 0, 0] == 11    # T12 -> anchor (not the per-level slot 31)
    assert r[5, 0, 0] == 1     # L1
    assert r[6, 0, 0] == 6     # L6
    assert r[7, 0, 0] == 7     # sacrum


def test_rib_overlay_paints_class_12_when_present():
    # v3 forward-compat: a student rib mask paints class 12; absent -> unchanged
    shape, sp = _spine()
    rib = np.zeros(shape, np.int16)
    rib[0, 1, 0] = 1                      # one rib voxel beside the anchor
    r_no = merge_labels(_w(sp), None, shape)               # v1/v2: no rib_path
    r_yes = merge_labels(_w(sp), None, shape, rib_path=_w(rib))
    assert int((r_no == 12).sum()) == 0   # dormant by default
    assert r_yes[0, 1, 0] == 12 and int((r_yes == 12).sum()) == 1
    assert r_yes[0, 0, 0] == 11           # anchor untouched
