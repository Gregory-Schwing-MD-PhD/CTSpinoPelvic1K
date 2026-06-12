"""Unit test for eval_vs_manual.classes_for_config (the scoped-vs-fused logic)."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_vs_manual import (classes_for_config, ALL_CLASSES,     # noqa: E402
                            confusion_counts, _cname)


def test_scoped_configs_use_manual_region_only():
    assert classes_for_config("spine_only") == (1, 2, 3, 4, 5, 6)
    assert classes_for_config("pelvic_native") == (7, 8, 9)


def test_fused_is_opt_in_full_scan():
    assert classes_for_config("fused") is None                     # off by default
    assert classes_for_config("fused", include_fused=True) == ALL_CLASSES
    assert classes_for_config("other", include_fused=True) is None


def test_confusion_counts_pinpoints_lr_bleed():
    import numpy as np
    v1 = np.zeros((4, 4, 4), np.int16)
    pred = np.zeros((4, 4, 4), np.int16)
    v1[0] = 7; v1[1] = 8; v1[2] = 9            # true sacrum / left / right hip
    pred[0] = 7                                 # sacrum perfect
    pred[1] = 8                                 # left_hip perfect
    pred[2, 0] = 9; pred[2, 1] = 8             # 1/4 of true right_hip -> left_hip bleed
    conf = confusion_counts(pred, v1, [7, 8, 9])
    assert conf[7][7] == 16 and sum(conf[7]) == 16        # sacrum all-correct
    assert conf[9][9] == 4 and conf[9][8] == 4            # the 8<->9 L/R bleed
    assert conf[9][0] == 8                                 # rest under-segmented to bg


def test_confusion_counts_absent_gt_class_is_empty():
    import numpy as np
    v1 = np.zeros((3, 3, 3), np.int16); v1[0] = 7
    pred = np.zeros((3, 3, 3), np.int16); pred[0] = 7
    conf = confusion_counts(pred, v1, [7, 8, 9])
    assert sum(conf[8]) == 0 and sum(conf[9]) == 0        # no true 8/9 voxels
    assert len(conf[7]) == 10


def test_cname():
    assert _cname(0) == "bg" and _cname(7) == "sacrum" and _cname(8) == "left_hip"
