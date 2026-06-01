"""Unit test for eval_vs_manual.classes_for_config (the scoped-vs-fused logic)."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_vs_manual import classes_for_config, ALL_CLASSES  # noqa: E402


def test_scoped_configs_use_manual_region_only():
    assert classes_for_config("spine_only") == (1, 2, 3, 4, 5, 6)
    assert classes_for_config("pelvic_native") == (7, 8, 9)


def test_fused_is_opt_in_full_scan():
    assert classes_for_config("fused") is None                     # off by default
    assert classes_for_config("fused", include_fused=True) == ALL_CLASSES
    assert classes_for_config("other", include_fused=True) is None
