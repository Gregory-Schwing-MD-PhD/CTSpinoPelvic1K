"""
Unit tests for the intensity-refinement core (manual-calibrated threshold +
connected-component gating). Pure numpy/scipy; no nibabel / no inference.

Locks the behavior that matters:
  * the bone threshold is read off the MANUAL annotation (per-case calibration);
  * threshold artifacts / unrelated bone (ribs) are dropped by keeping only
    connected components that OVERLAP the model prediction;
  * marrow interiors are solidified (hole fill);
  * pseudo voxels are identified by diffing the ORIGINAL manual tree (v1)
    against the pseudo tree (v2), so a manual sacrum-from-spine (class 7 in a
    spine_only case) is NEVER re-segmented;
  * each kept voxel takes the nearest predicted class.
"""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from intensity_refine import (  # noqa: E402
    IGNORE_LABEL,
    calibrate_threshold,
    link_or_copy,
    refine_label,
)


def test_link_or_copy_hardlinks_single_inode(tmp_path):
    import os
    src = tmp_path / "hf_export" / "ct" / "0001_ct.nii.gz"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"CTDATA")
    dst = tmp_path / "hf_export_v2" / "ct" / "0001_ct.nii.gz"
    method = link_or_copy(src, dst)
    assert dst.read_bytes() == b"CTDATA"
    if method == "hardlink":                     # normal same-fs case
        assert os.stat(src).st_ino == os.stat(dst).st_ino   # one physical copy
    # copy=True forces a real, independent copy
    dst2 = tmp_path / "hf_export_v3" / "ct" / "0001_ct.nii.gz"
    assert link_or_copy(src, dst2, copy=True) == "copy"
    assert os.stat(src).st_ino != os.stat(dst2).st_ino
    assert dst2.read_bytes() == b"CTDATA"


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
# refine_label  (v1 = manual, v2 = pseudo-labelled)
# --------------------------------------------------------------------------- #

def test_refine_clip_erases_oversegmentation_and_never_grows():
    # DEFAULT mode. Model over-predicts a sacrum bar; only part is real bone,
    # and there's real bone the model did NOT predict. clip keeps predicted∩bone
    # and never adds the un-predicted bone.
    v1 = np.full((1, 7, 7), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 3                                        # manual spine (calibration)
    v2 = np.zeros((1, 7, 7), dtype=np.int16)
    v2[0, 0, 0] = 3
    v2[0, 2, 1] = v2[0, 2, 2] = v2[0, 2, 3] = v2[0, 2, 4] = 7   # predicted bar (4 vox)
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 200.0                                   # manual bone HU
    ct[0, 2, 1] = ct[0, 2, 2] = ct[0, 2, 3] = 300.0       # bone (predicted)
    ct[0, 2, 4] = 50.0                                    # soft tissue (over-seg)
    ct[0, 3, 0] = 300.0                                   # real bone, NOT predicted

    out, _ = refine_label(v1, v2, ct, percentile=50, erode_iter=0,
                          fill_holes=False)                # default mode="clip"
    assert out[0, 2, 1] == 7 and out[0, 2, 2] == 7 and out[0, 2, 3] == 7
    assert out[0, 2, 4] == 0          # over-segmentation (soft tissue) erased
    assert out[0, 3, 0] == 0          # un-predicted bone NOT added (no growth)
    assert out[0, 0, 0] == 3          # manual untouched


def test_refine_resegment_grows_and_drops_stray_bone():
    v1 = np.full((1, 7, 7), IGNORE_LABEL, dtype=np.int16)   # un-annotated region
    v1[0, 1, 1] = 3; v1[0, 1, 2] = 3                        # manual spine
    v2 = np.zeros((1, 7, 7), dtype=np.int16)
    v2[0, 1, 1] = 3; v2[0, 1, 2] = 3                        # manual preserved
    v2[0, 4, 4] = 7                                         # model predicts sacrum
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 1, 1] = 150.0; ct[0, 1, 2] = 250.0               # manual bone HU
    ct[0, 4, 3] = ct[0, 4, 4] = ct[0, 4, 5] = 300.0        # pseudo-region bone bar
    ct[0, 6, 0] = 300.0                                    # stray rib — separate CC

    out, thr = refine_label(v1, v2, ct, mode="resegment", percentile=10,
                            erode_iter=0, fill_holes=False)
    assert out[0, 1, 1] == 3 and out[0, 1, 2] == 3         # manual untouched
    # grows along the bone bar beyond the single predicted voxel:
    assert out[0, 4, 3] == 7 and out[0, 4, 4] == 7 and out[0, 4, 5] == 7
    assert out[0, 6, 0] == 0                                # stray bone NOT kept
    assert 150.0 <= thr <= 250.0                            # calibrated off manual


def test_refine_fills_marrow_when_prediction_overlaps_bone():
    v1 = np.full((1, 7, 7), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 3                                         # manual (calibration)
    v2 = np.zeros((1, 7, 7), dtype=np.int16)
    v2[0, 0, 0] = 3
    for di in (-1, 0, 1):                                  # model predicts the
        for dj in (-1, 0, 1):                              # whole 3x3 structure
            v2[0, 4 + di, 4 + dj] = 8
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 300.0
    for di in (-1, 0, 1):                                  # bone ring, hollow centre
        for dj in (-1, 0, 1):
            if di or dj:
                ct[0, 4 + di, 4 + dj] = 300.0

    out, _ = refine_label(v1, v2, ct, percentile=50, erode_iter=0,
                          fill_holes=True)
    assert out[0, 4, 4] == 8                               # marrow filled + labelled
    assert out[0, 4, 3] == 8


def test_refine_never_modifies_manual_even_if_bone_and_adjacent_pred():
    v1 = np.full((1, 3, 3), IGNORE_LABEL, dtype=np.int16)
    v1[0, 1, 1] = 5                                        # manual lumbar
    v2 = np.zeros((1, 3, 3), dtype=np.int16)
    v2[0, 1, 1] = 5
    v2[0, 1, 0] = 7                                        # pseudo prediction next to it
    ct = np.full((1, 3, 3), 300.0, dtype=np.float32)       # everything bone
    out, _ = refine_label(v1, v2, ct, percentile=50, erode_iter=0,
                          fill_holes=False)
    assert out[0, 1, 1] == 5                               # manual wins over bone


def test_refine_preserves_manual_sacrum_from_spine():
    # spine_only case where the manual spine annotation includes a sacrum voxel
    # (class 7 from VerSe id 26). Class-based partitioning would treat it as
    # pseudo and re-segment it; diffing v1 vs v2 keeps it manual.
    v1 = np.full((1, 5, 5), IGNORE_LABEL, dtype=np.int16)
    v1[0, 1, 1] = 4                                        # manual L4
    v1[0, 2, 1] = 7                                        # manual sacrum-from-spine
    v2 = np.zeros((1, 5, 5), dtype=np.int16)
    v2[0, 1, 1] = 4; v2[0, 2, 1] = 7                       # manual preserved by pseudolabel
    v2[0, 4, 4] = 8                                        # pseudo hip fill
    ct = np.full((1, 5, 5), -1000.0, dtype=np.float32)
    ct[0, 1, 1] = ct[0, 2, 1] = 250.0                      # manual bone HU
    ct[0, 4, 4] = 300.0                                    # pseudo-region bone

    out, _ = refine_label(v1, v2, ct, percentile=10, erode_iter=0,
                          fill_holes=False)
    assert out[0, 2, 1] == 7                               # manual sacrum NOT touched
    assert out[0, 1, 1] == 4
    assert out[0, 4, 4] == 8                               # pseudo region refined


def test_refine_no_prediction_returns_unchanged():
    v1 = np.full((1, 1, 3), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 2] = 3                                        # manual only
    v2 = np.zeros((1, 1, 3), dtype=np.int16)
    v2[0, 0, 2] = 3                                        # no pseudo fill
    ct = np.full((1, 1, 3), 300.0, dtype=np.float32)
    out, thr = refine_label(v1, v2, ct)
    assert thr is None
    assert out.tolist() == v2.tolist()
