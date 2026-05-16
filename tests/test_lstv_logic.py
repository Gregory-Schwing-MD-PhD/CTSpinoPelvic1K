"""
test_lstv_logic.py — value-logic tests for _lstv_derived_fields() in
scripts/place_fused_masks.py.

Scope: lstv_agreement / lstv_confusion_zone / lstv_class VALUE behavior
only. Schema presence and JSON type-safety are covered by
tests/test_manifest.py and are intentionally not re-tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# place_fused_masks.py is a standalone script, not an installed package.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from place_fused_masks import _lstv_derived_fields  # noqa: E402


# (spine_label, pelvic_label, expected_agreement, expected_confusion)
_AGREEMENT_CASES = [
    ("LUMBARIZATION", "SACRALIZATION", False, True),
    ("SACRALIZATION", "LUMBARIZATION", False, True),
    ("NORMAL",        "NORMAL",        True,  False),
    ("LUMBARIZATION", "LUMBARIZATION", True,  False),
    ("SACRALIZATION", "SACRALIZATION", True,  False),
    ("SEMI_SACRAL",   "SACRALIZATION", False, True),
    ("UNKNOWN",       "SACRALIZATION", None,  False),
    ("SACRALIZATION", "UNKNOWN",       None,  False),
    ("NORMAL",        "SACRALIZATION", False, True),
    ("NORMAL",        "LUMBARIZATION", False, True),
    ("",              "",              None,  False),
    ("NORMAL",        "",              None,  False),
]


@pytest.mark.parametrize(
    "spine,pelvic,exp_agreement,exp_confusion",
    _AGREEMENT_CASES,
    ids=[f"{s or 'EMPTY'}|{p or 'EMPTY'}" for s, p, _, _ in _AGREEMENT_CASES],
)
def test_agreement_and_confusion(spine, pelvic, exp_agreement, exp_confusion):
    out = _lstv_derived_fields(spine, pelvic)

    # `is` comparison: agreement must be the singletons True/False/None,
    # never a truthy/falsy stand-in.
    assert out["lstv_agreement"] is exp_agreement, (
        f"({spine!r},{pelvic!r}) agreement={out['lstv_agreement']!r} "
        f"expected {exp_agreement!r}"
    )
    assert out["lstv_confusion_zone"] is exp_confusion, (
        f"({spine!r},{pelvic!r}) confusion={out['lstv_confusion_zone']!r} "
        f"expected {exp_confusion!r}"
    )


# (spine_label, pelvic_label, expected_lstv_class)
_CLASS_CASES = [
    # pelvic uninformative -> falls back to spine label class
    ("LUMBARIZATION", "UNKNOWN", 1),
    ("SEMI_SACRAL",   "UNKNOWN", 2),
    ("SACRALIZATION", "UNKNOWN", 3),
    ("NORMAL",        "NORMAL",  0),
    # pelvic priority: pelvic SACRALIZATION (3) wins over spine LUMBARIZATION (1)
    ("LUMBARIZATION", "SACRALIZATION", 3),
]


@pytest.mark.parametrize(
    "spine,pelvic,exp_class",
    _CLASS_CASES,
    ids=[f"{s}|{p}->{c}" for s, p, c in _CLASS_CASES],
)
def test_lstv_class(spine, pelvic, exp_class):
    out = _lstv_derived_fields(spine, pelvic)
    assert out["lstv_class"] == exp_class, (
        f"({spine!r},{pelvic!r}) lstv_class={out['lstv_class']} "
        f"expected {exp_class}"
    )


def test_pelvic_priority_overrides_spine_class():
    """Explicit single-case guard for the pelvic-priority convention."""
    out = _lstv_derived_fields("LUMBARIZATION", "SACRALIZATION")
    assert out["lstv_class"] == 3
    assert out["lstv_vertebral"] == "LUMBARIZATION"
    assert out["lstv_pelvic"] == "SACRALIZATION"
