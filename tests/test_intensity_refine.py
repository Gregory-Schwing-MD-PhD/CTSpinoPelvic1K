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
    compete_relabel,
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


def test_refine_clip_with_grow_picks_up_adjacent_bone_but_not_far_bone():
    # clip + bounded grow: bone adjacent to the prediction IS labelled,
    # bone farther than grow_iters away is NOT.
    v1 = np.full((1, 7, 7), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 3                                      # manual spine (calibration)
    v2 = np.zeros((1, 7, 7), dtype=np.int16)
    v2[0, 0, 0] = 3
    v2[0, 2, 2] = 7                                      # single predicted voxel
    ct = np.full((1, 7, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 200.0
    ct[0, 2, 2] = 300.0                                  # predicted is bone
    ct[0, 2, 3] = 300.0                                  # adjacent bone, 1 step
    ct[0, 5, 5] = 300.0                                  # far bone, > grow_iters

    out, _ = refine_label(v1, v2, ct, percentile=50, erode_iter=0,
                          fill_holes=False, grow_iters=2)
    assert out[0, 2, 2] == 7                # predicted bone kept
    assert out[0, 2, 3] == 7                # adjacent bone picked up by grow
    assert out[0, 5, 5] == 0                # far bone NOT reached
    assert out[0, 0, 0] == 3                # manual untouched


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


# --------------------------------------------------------------------------- #
# compete mode  (per-component reclaim + fused-component flagging)
# --------------------------------------------------------------------------- #

def test_compete_fixes_cross_disc_bleed():
    # Two vertebrae = two SEPARATE bone components split by a non-bone disc gap.
    # Component B (class 4) has a class-3 voxel that bled across from A onto B's
    # bone. compete reclaims B wholesale to its dominant class 4 -> bleed fixed;
    # the non-bone gap is dropped.
    v1 = np.full((1, 5, 9), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 5                                        # manual (calibration)
    v2 = np.zeros((1, 5, 9), dtype=np.int16)
    v2[0, 0, 0] = 5
    v2[0, 2, 1] = v2[0, 2, 2] = 3                          # component A (L3)
    v2[0, 2, 4] = 3                                        # <- bleed onto B's bone
    v2[0, 2, 5] = v2[0, 2, 6] = 4                          # component B (L4)
    ct = np.full((1, 5, 9), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 200.0
    ct[0, 2, 1] = ct[0, 2, 2] = 300.0                      # A is bone
    ct[0, 2, 3] = 50.0                                     # disc gap (non-bone)
    ct[0, 2, 4] = ct[0, 2, 5] = ct[0, 2, 6] = 300.0        # B is bone (contiguous)

    out, _ = refine_label(v1, v2, ct, mode="compete", percentile=50,
                          erode_iter=0, fill_holes=False)
    assert out[0, 2, 1] == 3 and out[0, 2, 2] == 3         # A stays L3
    assert out[0, 2, 4] == 4                               # bleed reclaimed to L4
    assert out[0, 2, 5] == 4 and out[0, 2, 6] == 4
    assert out[0, 2, 3] == 0                               # disc gap dropped
    assert out[0, 0, 0] == 5                               # manual untouched


def test_compete_flags_fused_component_and_keeps_model_boundary():
    # One bone component shared ~50/50 by two classes = touching/fused bone.
    # compete must NOT force one label: it keeps the model's per-voxel boundary
    # and records a review flag. (min_bleed_vox=0 disables the small-bleed
    # escape so purity_tol governs.)
    v1 = np.full((1, 5, 7), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 5
    v2 = np.zeros((1, 5, 7), dtype=np.int16)
    v2[0, 0, 0] = 5
    v2[0, 2, 1] = v2[0, 2, 2] = 5                          # half the blob = L5
    v2[0, 2, 3] = v2[0, 2, 4] = 7                          # other half = sacrum
    ct = np.full((1, 5, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 200.0
    ct[0, 2, 1] = ct[0, 2, 2] = ct[0, 2, 3] = ct[0, 2, 4] = 300.0   # one bone blob

    flags = []
    out, _ = refine_label(v1, v2, ct, mode="compete", percentile=50,
                          erode_iter=0, fill_holes=False, min_bleed_vox=0,
                          purity_tol=0.15, flags_out=flags)
    assert out[0, 2, 1] == 5 and out[0, 2, 4] == 7         # model boundary kept
    assert len(flags) == 1
    assert set(flags[0]["classes"]) == {5, 7}             # both classes recorded


def test_compete_fills_enclosed_marrow_and_erases_external_overseg():
    # A predicted bone ring (class 8) with a sub-threshold marrow centre, plus an
    # external predicted soft-tissue voxel (over-seg). compete must fill the
    # enclosed marrow to the dominant class and erase the non-enclosed over-seg.
    v1 = np.full((1, 5, 6), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 5                                       # manual (calibration)
    v2 = np.zeros((1, 5, 6), dtype=np.int16)
    v2[0, 0, 0] = 5
    ct = np.full((1, 5, 6), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 300.0
    for i in (1, 2, 3):                                   # 3x3 ring, hollow centre
        for j in (1, 2, 3):
            v2[0, i, j] = 8
            ct[0, i, j] = 300.0 if (i, j) != (2, 2) else 50.0   # centre = marrow
    v2[0, 0, 5] = 8                                       # external over-seg
    ct[0, 0, 5] = 50.0                                    # ... in soft tissue

    out, _ = refine_label(v1, v2, ct, mode="compete", percentile=50,
                          erode_iter=0, fill_holes=True)
    assert out[0, 1, 2] == 8                              # ring labelled
    assert out[0, 2, 2] == 8                              # enclosed marrow filled
    assert out[0, 0, 5] == 0                              # external over-seg erased
    assert out[0, 0, 0] == 5                              # manual untouched


def test_bone_floor_separates_structures_for_compete():
    # Two bone cores bridged by a soft-tissue voxel (a disc-like gap). The manual
    # HU calibrates LOW (fatty marrow), so without a floor the bridge is included
    # -> one multi-class component -> flagged, no relabel. A 150 HU floor excludes
    # the bridge -> two single-class components -> separated & reclaimed.
    v1 = np.full((1, 3, 7), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 0] = 5
    v2 = np.zeros((1, 3, 7), dtype=np.int16)
    v2[0, 0, 0] = 5
    v2[0, 1, 1] = v2[0, 1, 2] = 3                  # core A (L3)
    v2[0, 1, 3] = 3                                # bridge (predicted)
    v2[0, 1, 4] = v2[0, 1, 5] = 4                  # core B (L4)
    ct = np.full((1, 3, 7), -1000.0, dtype=np.float32)
    ct[0, 0, 0] = 100.0                            # manual -> low calibration
    ct[0, 1, 1] = ct[0, 1, 2] = 400.0              # A is dense bone
    ct[0, 1, 3] = 100.0                            # soft-tissue bridge (disc)
    ct[0, 1, 4] = ct[0, 1, 5] = 400.0              # B is dense bone

    f0 = []
    refine_label(v1, v2, ct, mode="compete", percentile=50, erode_iter=0,
                 fill_holes=False, min_bleed_vox=0, purity_tol=0.15, flags_out=f0)
    assert len(f0) == 1                            # bridged into one fused blob

    f1 = []
    out1, thr1 = refine_label(v1, v2, ct, mode="compete", percentile=50,
                              erode_iter=0, fill_holes=False, min_bleed_vox=0,
                              purity_tol=0.15, bone_floor=150.0, flags_out=f1)
    assert thr1 == 150.0
    assert f1 == []                                # separated -> both confident
    assert out1[0, 1, 3] == 0                      # disc-bridge dropped
    assert out1[0, 1, 1] == 3 and out1[0, 1, 4] == 4


def test_compete_does_not_swallow_unpredicted_bone_at_grow0():
    # Hip (class 8) fused to UNpredicted femur bone in one component. With the
    # default grow_iters=0, compete only reclaims the predicted voxels -> the
    # femur (never predicted) is left as background, not absorbed into the hip.
    pred = np.zeros((1, 1, 6), dtype=np.int16)
    pred[0, 0, 0] = pred[0, 0, 1] = 8                      # predicted hip
    bone = np.ones((1, 1, 6), dtype=bool)                 # one long bone component
    out, flags = compete_relabel(pred, bone, grow_iters=0, fill_holes=False)
    assert out[0, 0, 0] == 8 and out[0, 0, 1] == 8         # hip reclaimed
    assert out[0, 0, 4] == 0 and out[0, 0, 5] == 0         # femur NOT swallowed
    assert flags == []                                     # single-class -> no flag


def test_refine_no_prediction_returns_unchanged():
    v1 = np.full((1, 1, 3), IGNORE_LABEL, dtype=np.int16)
    v1[0, 0, 2] = 3                                        # manual only
    v2 = np.zeros((1, 1, 3), dtype=np.int16)
    v2[0, 0, 2] = 3                                        # no pseudo fill
    ct = np.full((1, 1, 3), 300.0, dtype=np.float32)
    out, thr = refine_label(v1, v2, ct)
    assert thr is None
    assert out.tolist() == v2.tolist()
