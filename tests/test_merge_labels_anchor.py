"""
test_merge_labels_anchor.py — export_hf.merge_labels emits VerSe-NATIVE ids.

The spine column passes through VERBATIM (no remap — remapping the thoracic into the
rib id range was the v3 bug), the pelvis goes to fixed ids above 28 (S1 29, hips 30/31),
CTPelvic1K's L5 is dropped, and partial-mode IGNORE is preserved. The key regression
guard: no spine label may ever land in the rib id range (34-57).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_hf import merge_labels, IGNORE_LABEL  # noqa: E402
import label_scheme as LS  # noqa: E402


def _w(arr):
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    nib.save(nib.Nifti1Image(arr.astype(np.int16), np.eye(4)), f)
    return f


def _spine():
    """A VerSe spine column (the 0003-style ids: T4 … L6, plus sacrum + T13)."""
    shape = (10, 8, 8)
    sp = np.zeros(shape, np.int16)
    sp[0, 0, 0] = 11   # T4  (VerSe)
    sp[1, 0, 0] = 19   # T12
    sp[2, 0, 0] = 20   # L1
    sp[3, 0, 0] = 24   # L5
    sp[4, 0, 0] = 25   # L6 (supernumerary)
    sp[5, 0, 0] = 26   # sacrum (from spine)
    sp[6, 0, 0] = 28   # T13 (supernumerary)
    return shape, sp


def test_spine_passthrough_verbatim():
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    assert r[0, 0, 0] == 11   # T4 stays T4 — VerSe verbatim, NOT remapped
    assert r[1, 0, 0] == 19   # T12
    assert r[2, 0, 0] == 20   # L1
    assert r[3, 0, 0] == 24   # L5
    assert r[4, 0, 0] == 25   # L6
    assert r[6, 0, 0] == 28   # T13


def test_thoracic_never_in_rib_range():
    """THE v3 bug: thoracic vertebrae were remapped into the rib id range and read as ribs."""
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    rib_ids = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))  # 34..57
    for v in np.unique(r):
        if int(v) in (0, IGNORE_LABEL):
            continue
        assert int(v) not in rib_ids, f"label {v} fell into the rib range (34-57) — the v3 bug!"
        assert int(v) <= 28, f"spine label {v} must be a VerSe id (1-28)"


def test_pelvis_to_fixed_ids_above_verse():
    shape, sp = _spine()
    pe = np.zeros(shape, np.int16)
    pe[7, 1, 0] = 1   # CTPelvic1K sacrum -> 26 (shared with VerSe)
    pe[7, 2, 0] = 2   # left hip  -> 30
    pe[7, 3, 0] = 3   # right hip -> 31
    pe[7, 4, 0] = 4   # CTPelvic1K L5 -> DROPPED (spine already provides L1-L6)
    r = merge_labels(_w(sp), _w(pe), shape)
    assert r[7, 2, 0] == 30
    assert r[7, 3, 0] == 31
    assert r[7, 1, 0] == LS.SACRUM_ID         # 26
    assert r[7, 4, 0] == 0                     # L5 dropped -> background (fused mode)


def test_sacrum_fill_from_spine_when_pelvic_absent():
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    assert r[5, 0, 0] == LS.SACRUM_ID         # sacrum-from-spine fill (26)


def test_partial_spine_only_ignore():
    shape, sp = _spine()
    r = merge_labels(_w(sp), None, shape)
    assert r[9, 7, 7] == IGNORE_LABEL         # untraced region -> IGNORE, not background
