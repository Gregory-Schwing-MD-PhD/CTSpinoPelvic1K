"""test_label_scheme.py — the canonical VerSe-native label scheme is collision-proof
and keeps the spine VerSe-verbatim. This is THE contract every producer imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import label_scheme as LS  # noqa: E402


def test_no_collisions():
    LS.verify()  # raises on any duplicate id
    d = LS.label_dict()
    ids = [v for k, v in d.items() if k != "background"]
    assert len(ids) == len(set(ids)), "two structures share a label id"


def test_spine_is_verse_verbatim():
    d = LS.label_dict()
    expect = {"C1": 1, "C7": 7, "T1": 8, "T12": 19, "L1": 20, "L5": 24, "L6": 25,
              "sacrum": 26, "coccyx": 27, "T13": 28}
    for k, v in expect.items():
        assert d[k] == v, f"{k} should be VerSe {v}, got {d[k]}"
    assert all(LS.VERSE_SPINE[v] == v for v in range(1, 29)), "spine map is not identity"


def test_new_structures_above_28():
    d = LS.label_dict()
    for nm in ["S1", "left_hip", "right_hip", "femur_left", "femur_right",
               "rib_left_1", "rib_right_12", "iliolumbar_left", "iliac_vena_right"]:
        assert d[nm] >= 29, f"{nm}={d[nm]} must be appended above the VerSe range"


def test_specific_ids():
    d = LS.label_dict()
    assert d["S1"] == 29
    assert (d["left_hip"], d["right_hip"]) == (30, 31)
    assert (d["femur_left"], d["femur_right"]) == (32, 33)
    assert (d["rib_left_1"], d["rib_left_12"]) == (34, 45)
    assert (d["rib_right_1"], d["rib_right_12"]) == (46, 57)
    assert d["ignore"] == 255


def test_pelvic_remap_drops_l5():
    assert LS.PELVIC_REMAP == {1: 26, 2: 30, 3: 31}  # CTPelvic1K's 4th class (L5) dropped


def test_no_spine_label_in_rib_range():
    """Regression guard for the v3 bug: no vertebra id may overlap the rib block."""
    d = LS.label_dict()
    rib_ids = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))  # 34..57
    spine = [d[k] for k in (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                            + [f"T{n}" for n in range(1, 13)]
                            + [f"L{n}" for n in range(1, 7)] + ["sacrum", "coccyx", "T13"])]
    assert not (set(spine) & rib_ids)
