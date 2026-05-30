"""Unit tests for refine_review's pure change-classification logic."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from refine_review import classify_change, _slice_changes  # noqa: E402


def test_classify_change_categories():
    before = np.array([[[3, 3, 0, 4]]], dtype=np.int16)
    after = np.array([[[3, 0, 5, 7]]], dtype=np.int16)
    cat = classify_change(before, after)
    assert cat[0, 0, 0] == 0          # unchanged (3 -> 3)
    assert cat[0, 0, 1] == 1          # removed  (3 -> 0)
    assert cat[0, 0, 2] == 2          # added    (0 -> 5)
    assert cat[0, 0, 3] == 3          # relabeled (4 -> 7)


def test_slice_changes_orders_busiest_first():
    cat = np.zeros((1, 3, 4), dtype=np.uint8)
    cat[0, 1, 0] = 3                  # axis-1 index 1: 1 change
    cat[0, 2, 0] = cat[0, 2, 1] = 1   # axis-1 index 2: 2 changes
    order = _slice_changes(cat, axis=1)
    assert list(order) == [2, 1]      # busiest slice first; empty slice 0 excluded
