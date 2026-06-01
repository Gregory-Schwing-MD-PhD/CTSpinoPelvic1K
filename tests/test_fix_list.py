"""Unit tests for reviewtool fix-list helpers (row filtering + flag hints)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "reviewtool", _ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cli  # noqa: E402  (reviewtool/cli.py)


def test_fixlist_keeps_only_flagged_in_order():
    rows = [
        {"token": "9", "config": "c", "needs_review": "1", "mixing_flag": "1"},
        {"token": "5", "config": "c", "needs_review": "0", "mixing_flag": "0"},
        {"token": "3", "config": "c", "needs_review": "1", "leak_flag": "1"},
    ]
    out = cli._fixlist_rows(rows)
    assert [r["token"] for r in out] == ["9", "3"]     # order preserved, clean dropped
    assert len(cli._fixlist_rows(rows, only_flagged=False)) == 3


def test_flag_hint_names_the_defects():
    h = cli._flag_hint({"mixing_flag": "1", "off_main_frac": "0.03",
                        "struct_flag": "1", "lr_swap": "1"})
    assert "MIXING" in h
    assert "L/R HIP SWAP" in h


def test_flag_hint_leak_only():
    h = cli._flag_hint({"leak_flag": "1", "off_bone_frac": "0.12"})
    assert "OFF-BONE" in h and "MIXING" not in h
