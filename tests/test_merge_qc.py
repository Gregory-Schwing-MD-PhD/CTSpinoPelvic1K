"""Unit tests for merge_qc.build_master (pure join + ranking)."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from merge_qc import build_master, recalibrate, _pct  # noqa: E402


def test_recalibrate_flags_relative_to_baseline():
    # baseline (radiologist) off_bone_frac ~0.01-0.03; p95 ~0.03.
    baseline = [{"token": str(i), "config": "c", "off_bone_frac": str(0.01 + i * 0.001),
                 "off_main_frac": "0.0"} for i in range(20)]
    master = [
        {"token": "hi", "config": "c", "off_bone_frac": "0.06", "off_main_frac": "0.0",
         "struct_flag": 0, "mixing_flag": 0, "leak_flag": 1},   # above baseline p95
        {"token": "lo", "config": "c", "off_bone_frac": "0.015", "off_main_frac": "0.0",
         "struct_flag": 0, "mixing_flag": 0, "leak_flag": 1},   # within baseline
    ]
    thr = recalibrate(master, baseline, pct=95)
    by = {r["token"]: r for r in master}
    assert by["hi"]["leak_flag"] == 1                 # genuinely leakier than gold
    assert by["lo"]["leak_flag"] == 0                 # within gold range -> cleared
    assert by["lo"]["needs_review"] == 0
    assert thr["off_bone_frac"] < 0.06


def test_pct_basic():
    assert _pct([1, 2, 3, 4, 5], 50) == 3
    assert _pct([], 95) == float("inf")


def test_union_join_and_needs_review():
    sources = {
        "mixing": [{"token": "1", "config": "spine_only", "mixing_flag": "1",
                    "off_main_frac": "0.02"},
                   {"token": "2", "config": "spine_only", "mixing_flag": "0",
                    "off_main_frac": "0.0"}],
        "leak": [{"token": "2", "config": "spine_only", "leak_flag": "1",
                  "off_bone_frac": "0.05"}],
        "structure": [{"token": "3", "config": "pelvic_native", "struct_flag": "1",
                       "lr_swap": "1"}],
    }
    master = build_master(sources)
    by = {(r["token"], r["config"]): r for r in master}
    assert len(master) == 3                       # union of all tokens
    assert by[("1", "spine_only")]["needs_review"] == 1
    assert by[("1", "spine_only")]["leak_flag"] == 0      # not in leak source
    assert by[("2", "spine_only")]["n_flags"] == 1        # leak only
    assert by[("3", "pelvic_native")]["struct_flag"] == 1


def test_multiflag_case_ranks_first():
    sources = {
        "mixing": [{"token": "9", "config": "c", "mixing_flag": "1",
                    "off_main_frac": "0.01"},
                   {"token": "1", "config": "c", "mixing_flag": "0"}],
        "leak": [{"token": "9", "config": "c", "leak_flag": "1",
                  "off_bone_frac": "0.2"}],
    }
    master = build_master(sources)
    assert master[0]["token"] == "9"              # 2 flags -> top of list
    assert master[0]["n_flags"] == 2
    assert master[-1]["needs_review"] == 0


def test_clean_case_not_flagged():
    sources = {"mixing": [{"token": "5", "config": "c", "mixing_flag": "0"}]}
    master = build_master(sources)
    assert master[0]["needs_review"] == 0
    assert master[0]["n_flags"] == 0
