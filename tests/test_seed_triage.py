"""Seeding triage: crops_index restricts the queue to the flagged worklist and
attaches each case's review-crop info."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "review_service", _ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from store import LocalBackend, ReviewStore, init_cases_from_manifest  # noqa: E402


def _recs():
    return [
        {"token": "1", "config": "spine_only", "ct_file": "ct/1.nii.gz",
         "label_file": "labels/1.nii.gz"},
        {"token": "2", "config": "pelvic_native", "ct_file": "ct/2.nii.gz",
         "label_file": "labels/2.nii.gz"},
        {"token": "3", "config": "fused", "ct_file": "ct/3.nii.gz",
         "label_file": "labels/3.nii.gz"},
    ]


def test_crops_index_triages_and_attaches_crop(tmp_path):
    store = ReviewStore(LocalBackend(tmp_path))
    crops_index = {"labels/1.nii.gz": {"ct_crop": "crops/a/ct.nii.gz",
                                       "seg_crop": "crops/a/seg.nii.gz",
                                       "origin": [4, 5, 6]}}
    n = init_cases_from_manifest(store, _recs(), crops_index=crops_index)
    assert n == 1                                  # only the flagged (token 1)
    cases = store.list_cases()
    assert [c["token"] for c in cases] == ["1"]
    assert cases[0]["crop"]["origin"] == [4, 5, 6]
    assert cases[0]["crop"]["ct_crop"] == "crops/a/ct.nii.gz"


def test_no_index_seeds_all_scoped(tmp_path):
    store = ReviewStore(LocalBackend(tmp_path))
    n = init_cases_from_manifest(store, _recs())   # no crops_index
    assert n == 2                                  # both scoped; fused excluded
    assert all("crop" not in c for c in store.list_cases())
