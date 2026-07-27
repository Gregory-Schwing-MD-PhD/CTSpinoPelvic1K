"""
Unit tests for scripts/auto_accept_spine.py — the ACCEPT/REJECT decision function.

Drives the PURE rule (`classify` / `is_auto_acceptable`) on synthetic QC-message lists shaped exactly
like scripts/review_anatomy_qc.spine_extend_qc output. No network, no --apply, no HF: only the core
rule that decides passthrough vs human.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "scripts", _ROOT / "review_service"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import auto_accept_spine as AAS       # noqa: E402


def test_fov_only_notes_accept():
    # ok == True and every line is an OK summary or an FOV-truncation advisory -> passthrough
    ok = True
    msgs = [
        "note L1 sits at/above L2 but one is FOV-truncated (clipped by the scan edge) -> advisory, "
        "not blocking",
        "note added T8 is clipped by the edge of the scan (FOV-truncated) -> size / one-piece "
        "checks skipped",
        "OK spine additions ['T8']: consecutive, ascending, connected",
    ]
    assert AAS.is_auto_acceptable(ok, msgs) is True


def test_structure_integrity_fov_note_accepts():
    ok = True
    msgs = [
        "note left_hip is split but clipped by the top/bottom of the scan (FOV-truncated) -> "
        "advisory, not blocking",
        "OK spine additions [none]: consecutive, ascending, connected",
    ]
    assert AAS.is_auto_acceptable(ok, msgs) is True


def test_pure_ok_lines_accept():
    assert AAS.is_auto_acceptable(True, ["OK spine additions [none]: consecutive, ascending, connected"]) is True


def test_blocking_X_rejects():
    # a hard "X" gate failure -> human, regardless of anything else
    ok = False
    msgs = ["X order: T7 sits at/above T8 -> a vertebra is mis-placed or mis-numbered",
            "note L1 is FOV-truncated (clipped by the scan edge)"]
    assert AAS.is_auto_acceptable(ok, msgs) is False


def test_order_violation_rejects():
    ok = False
    msgs = ["X order: L2 sits at/above L1 -> a vertebra is mis-placed or mis-numbered "
            "(numbering must ascend down the column)",
            "OK spine additions [none]: consecutive, ascending, connected"]
    assert AAS.is_auto_acceptable(ok, msgs) is False


def test_gap_rejects_even_if_ok_flag_true():
    # defensive: an "X" line must reject even if the ok flag were (wrongly) True
    assert AAS.is_auto_acceptable(True, ["X gap: T7 (id 15) is missing in the column -> label it"]) is False


def test_non_fov_note_rejects():
    # a note that is NOT about the scan edge implies real work -> human
    ok = True
    msgs = ["note: last full rib is between T11 and T12 — ambiguous, not blocking",
            "OK spine additions [none]: consecutive, ascending, connected"]
    assert AAS.is_auto_acceptable(ok, msgs) is False


def test_no_vertebrae_info_line_rejects():
    # "(no spine vertebrae in view)" is neither OK nor an FOV note -> conservative reject
    assert AAS.is_auto_acceptable(True, ["(no spine vertebrae in view)"]) is False


def test_classify_returns_reason():
    ok, reason = AAS.classify(False, ["X gap: T7 missing"])
    assert ok is False and "blocking" in reason.lower()
    ok, reason = AAS.classify(True, ["OK spine additions [none]: consecutive, ascending, connected"])
    assert ok is True
