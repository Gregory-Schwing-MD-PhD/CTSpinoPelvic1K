"""Unit tests for the crop/paste-back helpers (review-crop round trip)."""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from export_review_crops import foreground_bbox, paste_back, crop_dirname  # noqa: E402


def test_foreground_bbox_tight_with_pad():
    v = np.zeros((20, 20, 20), dtype=np.int16)
    v[8:11, 8:11, 8:11] = 5
    bb = foreground_bbox(v, pad=2)
    assert bb == (slice(6, 13), slice(6, 13), slice(6, 13))


def test_foreground_bbox_clipped_to_array():
    v = np.zeros((5, 5, 5), dtype=np.int16)
    v[0, 0, 0] = 3
    bb = foreground_bbox(v, pad=10)
    assert bb == (slice(0, 5), slice(0, 5), slice(0, 5))   # clipped, no negatives


def test_empty_label_has_no_bbox():
    assert foreground_bbox(np.zeros((4, 4, 4)), pad=1) is None


def test_crop_paste_roundtrip():
    # crop the bbox, "edit" the crop, paste back -> only the box changes.
    full = np.zeros((12, 12, 12), dtype=np.int16)
    full[4:8, 4:8, 4:8] = 4
    bb = foreground_bbox(full, pad=1)
    origin = [s.start for s in bb]
    crop = full[bb].copy()
    crop[crop == 4] = 7                       # reviewer relabels 4 -> 7
    merged = paste_back(full, crop, origin)
    assert (merged[4:8, 4:8, 4:8] == 7).all()
    assert merged[0, 0, 0] == 0               # outside the box untouched
    assert merged.shape == full.shape


def test_crop_dirname_is_filesystem_safe():
    assert crop_dirname("CTC-101/8", "pelvic_native") == "CTC-101_8__pelvic_native"
