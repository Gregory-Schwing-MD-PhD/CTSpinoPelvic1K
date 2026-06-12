"""Unit tests for pelvis_opposing_qc — pose-invariant pseudo-vs-GT pelvis check.

A correct pseudo pelvis (same rigid bone) must NOT flag against the patient's GT
pelvis; a left/right-hip swap MUST flag, since the descriptors are pose-invariant
but laterality is not."""
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

nib = pytest.importorskip("nibabel")
import pelvis_opposing_qc as M  # noqa: E402


def _mk(path, swap=False):
    """Identity-affine label vol: sacrum mid, hips on -X (left) / +X (right)."""
    v = np.zeros((40, 40, 40), np.int16)
    v[18:22, 10:30, 10:30] = 7                         # sacrum
    left, right = slice(4, 12), slice(28, 36)          # -X side / +X side
    v[(right if swap else left), 12:28, 12:28] = 8     # left_hip
    v[(left if swap else right), 12:28, 12:28] = 9     # right_hip
    nib.save(nib.Nifti1Image(v, np.eye(4)), str(path))


def test_identical_pelvis_does_not_flag(tmp_path):
    _mk(tmp_path / "gt.nii.gz")
    _mk(tmp_path / "good.nii.gz")
    r = M._eval_pair("good", tmp_path / "good.nii.gz", tmp_path / "gt.nii.gz", 15.0)
    assert r["flag"] == 0 and r["flags"] == ""
    assert r["lr_swap_vs_gt"] == 0


def test_lr_swap_is_caught(tmp_path):
    _mk(tmp_path / "gt.nii.gz")
    _mk(tmp_path / "swapped.nii.gz", swap=True)
    r = M._eval_pair("swap", tmp_path / "swapped.nii.gz", tmp_path / "gt.nii.gz", 15.0)
    assert r["lr_swap_vs_gt"] == 1 and "LR_SWAP" in r["flags"]


def test_pose_invariance_rotation_does_not_flag(tmp_path):
    """Rotating the GT pelvis 90deg in-plane (a pose change, no shape change)
    must NOT register as a difference — that is the whole premise."""
    _mk(tmp_path / "gt.nii.gz")
    lbl, aff = M._load_pir_int(tmp_path / "gt.nii.gz")
    # rotate the volume about the I axis (swap+flip the P/R plane) -> new pose
    rot = np.rot90(lbl, k=1, axes=(0, 2)).copy()
    nib.save(nib.Nifti1Image(rot, np.eye(4)), str(tmp_path / "rot.nii.gz"))
    d_gt = M._descriptors(tmp_path / "gt.nii.gz")
    d_rot = M._descriptors(tmp_path / "rot.nii.gz")
    # principal-axis extents of the sacrum are pose-invariant -> match closely
    e_gt = sorted(d_gt["sacrum"]["ext"], reverse=True)
    e_rot = sorted(d_rot["sacrum"]["ext"], reverse=True)
    for a, b in zip(e_gt, e_rot):
        assert abs(M._pct(a, b)) < 5.0


def test_pairing_requires_both_configs(tmp_path):
    pseudo = [{"token": "A", "config": "spine_only", "label_file": "a.nii.gz"},
              {"token": "B", "config": "fused", "label_file": "b.nii.gz"}]
    gt = [{"token": "A", "config": "pelvic_native", "label_file": "a_gt.nii.gz"},
          {"token": "C", "config": "pelvic_native", "label_file": "c.nii.gz"}]
    ps = M._index_by_token(pseudo, {"spine_only"})
    g = M._index_by_token(gt, {"pelvic_native"})
    assert set(ps) == {"A"} and set(g) == {"A", "C"}
    assert sorted(set(ps) & set(g)) == ["A"]               # only A has both
