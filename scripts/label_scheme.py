"""label_scheme.py — THE single source of truth for CTSpinoPelvic1K label ids.

VerSe-native: the spine keeps its VerSe ids VERBATIM (no remap — that was the v3 bug),
and every structure NOT in VerSe gets a fixed, reserved id ABOVE the VerSe range, so no
two structures can ever share an id.

    spine   (VerSe, from CTSpine1K) : 1–7 C1–C7 · 8–19 T1–T12 · 20–25 L1–L6 · 26 sacrum
                                       · 27 coccyx · 28 T13           ← passed through AS-IS
    pelvis  (CTPelvic1K + TS femurs): 26 sacrum [shared] · 29 left_hip · 30 right_hip
                                       · 31 femur_left · 32 femur_right
    ribs    (RibSeg, numbered off GT thoracic): 33–44 rib_left_1..12 · 45–56 rib_right_1..12
    soft    (v4 overlay tasks)      : 57/58 iliolumbar · 59–64 nerve roots · 65/66 psoas
                                       · 67–72 great vessels
    ignore                          : 255

Import this EVERYWHERE (export_hf, build_v3, dataset.json, ostk, docs generators). Never
define ids anywhere else. `verify()` (run in tests) guarantees no collisions.
"""
from __future__ import annotations

from typing import Dict

IGNORE_LABEL = 255

# ── spine: VerSe verbatim (NO remap) ─────────────────────────────────────────
# VerSe-2020 numbering. CTSpine1K uses exactly this, so the spine mask passes through.
VERSE_SPINE: Dict[int, int] = {v: v for v in range(1, 29)}     # 1..28 -> identity
SACRUM_ID = 26                                                 # VerSe sacrum (below S1)
S1_ID = 29                                                     # S1 body — carved from sacrum top
                                                               # (needed for spinopelvic angles)

# ── pelvis: CTPelvic1K 4-class (1 sacrum, 2 left_hip, 3 right_hip, 4 L5) ──────
# Sacrum folds into the VerSe sacrum (26); CTPelvic1K's L5 (4) is DROPPED (the spine
# already provides L1–L6 at 20–25). S1 + hips + femurs get fixed ids above VerSe.
PELVIC_REMAP: Dict[int, int] = {1: SACRUM_ID, 2: 30, 3: 31}    # 4 -> dropped
FEMUR_LEFT, FEMUR_RIGHT = 32, 33

# ── ribs: numbered off the GT thoracic, fixed block above the femurs ─────────
RIB_LEFT_OFFSET, RIB_RIGHT_OFFSET = 33, 45                     # rib_*_N -> OFFSET+N (34-45, 46-57)

# ── soft-tissue overlays (v4) ────────────────────────────────────────────────
SOFT_TISSUE = {
    "iliolumbar_left": 58, "iliolumbar_right": 59,
    "nerve_L4_left": 60, "nerve_L4_right": 61, "nerve_L5_left": 62,
    "nerve_L5_right": 63, "nerve_S1_left": 64, "nerve_S1_right": 65,
    "psoas_left": 66, "psoas_right": 67,
    "aorta": 68, "inferior_vena_cava": 69, "iliac_artery_left": 70,
    "iliac_artery_right": 71, "iliac_vena_left": 72, "iliac_vena_right": 73,
}

_VERSE_NAMES = (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                + [f"T{n}" for n in range(1, 13)]              # T1..T12 -> 8..19
                + ["L1", "L2", "L3", "L4", "L5", "L6"])        # L1..L6 -> 20..25


def label_dict() -> Dict[str, int]:
    """Full {name: id} legend (background..ignore) — the ONE map for dataset.json + docs."""
    d: Dict[str, int] = {"background": 0}
    for i, nm in enumerate(_VERSE_NAMES, start=1):             # 1..25
        d[nm] = i
    d["sacrum"] = SACRUM_ID                                    # 26
    d["coccyx"] = 27
    d["T13"] = 28
    d["S1"] = S1_ID                                            # 29 (carved from sacrum top)
    d["left_hip"] = 30
    d["right_hip"] = 31
    d["femur_left"] = FEMUR_LEFT                               # 32
    d["femur_right"] = FEMUR_RIGHT                             # 33
    for n in range(1, 13):
        d[f"rib_left_{n}"] = RIB_LEFT_OFFSET + n               # 34..45
    for n in range(1, 13):
        d[f"rib_right_{n}"] = RIB_RIGHT_OFFSET + n             # 46..57
    d.update(SOFT_TISSUE)                                      # 58..73
    d["ignore"] = IGNORE_LABEL                                 # 255
    return d


def rib_id(side: str, number: int) -> int:
    return (RIB_LEFT_OFFSET if side == "left" else RIB_RIGHT_OFFSET) + number


def verify() -> None:
    """Assert the scheme is collision-proof + VerSe-faithful (run in tests / at import)."""
    d = label_dict()
    ids = [v for k, v in d.items() if k != "background"]
    assert len(ids) == len(set(ids)), "DUPLICATE label id — collision in label_scheme!"
    # spine is VerSe verbatim
    for v, out in VERSE_SPINE.items():
        assert v == out, f"spine id {v} is remapped to {out} — must be VerSe-native"
    # every non-spine structure sits at/above the sacrum, never inside the vertebra range
    for nm in list(SOFT_TISSUE) + ["left_hip", "right_hip", "femur_left", "femur_right",
                                   "rib_left_1", "rib_right_12"]:
        assert d[nm] >= 26, f"{nm}={d[nm]} collides with the VerSe vertebra range (1–25)"
    # ribs don't overlap femurs/pelvis
    assert RIB_LEFT_OFFSET + 1 > FEMUR_RIGHT, "ribs overlap femurs"


verify()
