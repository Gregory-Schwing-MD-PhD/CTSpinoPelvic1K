"""Unit tests for the GT-free off-bone label-leak metric."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bone_leak_qc import bone_leak_metrics  # noqa: E402


def test_solid_bone_has_no_leak():
    lab = np.zeros((1, 6, 6), dtype=np.int16)
    lab[0, 1:4, 1:4] = 3
    ct = np.full((1, 6, 6), -1000.0, dtype=np.float32)
    ct[0, 1:4, 1:4] = 400.0                       # all bone
    m = bone_leak_metrics(lab, ct)
    assert m["off_bone_frac"] == 0.0
    assert m["leak_flag"] == 0


def test_enclosed_marrow_is_not_leak():
    # bone ring with a low-HU centre (fatty marrow) must NOT count as leak.
    lab = np.zeros((1, 6, 6), dtype=np.int16)
    ct = np.full((1, 6, 6), -1000.0, dtype=np.float32)
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            lab[0, i, j] = 4
            ct[0, i, j] = 400.0 if (i, j) != (2, 2) else 30.0   # centre = marrow
    m = bone_leak_metrics(lab, ct)
    assert m["off_bone_frac"] == 0.0              # marrow enclosed -> not leak


def test_exposed_softtissue_label_is_leak():
    # a labelled protrusion sitting in soft tissue (not enclosed) IS a leak.
    lab = np.zeros((1, 4, 8), dtype=np.int16)
    ct = np.full((1, 4, 8), 400.0, dtype=np.float32)
    lab[0, 1, 1:4] = 5                            # bone part (HU 400)
    lab[0, 1, 4] = 5                              # protrusion...
    ct[0, 1, 4] = 40.0                            # ...into soft tissue (leak)
    m = bone_leak_metrics(lab, ct)
    assert m["leak_vox"] == 1
    assert m["off_bone_frac"] > 0.0
    assert m["worst_class"] == "L5"


def test_background_air_leak_counted():
    lab = np.zeros((1, 4, 8), dtype=np.int16)
    ct = np.full((1, 4, 8), 400.0, dtype=np.float32)
    lab[0, 1, 1:4] = 7
    lab[0, 1, 4] = 7
    ct[0, 1, 4] = -800.0                          # bled into AIR
    m = bone_leak_metrics(lab, ct)
    assert m["bg_leak_vox"] == 1
    assert m["bg_leak_frac"] > 0.0
