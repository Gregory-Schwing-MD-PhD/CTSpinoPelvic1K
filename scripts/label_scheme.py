"""label_scheme.py — THE single source of truth for CTSpinoPelvic1K label ids.

VerSe-native: the spine keeps its VerSe ids VERBATIM (no remap — that was the v3 bug),
and every structure NOT in VerSe gets a fixed id ABOVE the VerSe range, so no two
structures can ever share an id. The space is contiguous, 0..68; the two VerSe ids
27 (coccyx) and 28 (T13) are kept and no released record uses them.

    spine   (VerSe, from CTSpine1K) : 1–7 C1–C7 · 8–19 T1–T12 · 20–25 L1–L6 · 26 sacrum
                                       · 27 coccyx · 28 T13           ← passed through AS-IS
    pelvis  (CTPelvic1K + TS femurs): 26 sacrum [shared] · 29 S1 · 30 left_hip · 31 right_hip
                                       · 32 femur_left · 33 femur_right
    ribs    (numbered off GT thoracic): 34–46 rib_left_1..13 · 47–59 rib_right_1..13
                                       (rib 13 = the true rib of a T13; empty so far)
    lumbar ribs                     : 60 left · 61 right (a rudimentary rib on a lumbar-type L1)
    hardware                        : 62 generic · 63 cage · 64 screw_rod · 65 plate
                                       · 66 arthroplasty · 67 si_screw · 68 osteosynthesis

The scheme is BONE AND HARDWARE ONLY. There is no soft-tissue class and no sentinel.

HISTORY. v9 (2026-09-07) closed the never-populated 58..73 gap and dropped the 255
sentinel (lumbar ribs 74/75 -> 58/59, hardware 76..82 -> 60..66; OLD_TO_NEW_V9). v10 adds a
thirteenth rib per side IN SEQUENCE after rib 12, which moves the right ribs up by one and
the lumbar ribs and hardware up by two (OLD_TO_NEW_V10). scripts/renumber_labels.py applies
a remap to every volume; volumes from earlier versions carry the earlier ids.

Import this EVERYWHERE (export_hf, build_v3, dataset.json, ostk, docs generators). Never
define ids anywhere else. `verify()` (run in tests) guarantees no collisions.
"""
from __future__ import annotations

from typing import Dict

# nnU-Net's ignore label for TRAINING volumes only. It is not part of the release scheme
# and appears in no released volume.
IGNORE_LABEL = 255

# ── spine: VerSe verbatim (NO remap) ─────────────────────────────────────────
# VerSe-2020 numbering. CTSpine1K uses exactly this, so the spine mask passes through.
VERSE_SPINE: Dict[int, int] = {v: v for v in range(1, 29)}     # 1..28 -> identity
SACRUM_ID = 26                                                 # VerSe sacrum (below S1)
S1_ID = 29                                                     # S1 body — carved from sacrum top
                                                               # (needed for spinopelvic angles)

# ── pelvis: CTPelvic1K 4-class (1 sacrum, 2 left_hip, 3 right_hip, 4 lumbar spine) ──
# Sacrum folds into the VerSe sacrum (26); CTPelvic1K's lumbar class (4) is DROPPED (the
# spine already provides L1–L6 at 20–25). S1 + hips + femurs get fixed ids above VerSe.
PELVIC_REMAP: Dict[int, int] = {1: SACRUM_ID, 2: 30, 3: 31}    # 4 -> dropped
FEMUR_LEFT, FEMUR_RIGHT = 32, 33

# ── ribs: numbered off the GT thoracic, thirteen per side ────────────────────
# rib_*_N -> OFFSET+N. Rib 13 is the true rib of a thirteenth thoracic vertebra (VerSe 28),
# a phenotype distinct from a lumbar rib (Du Plessis 2018; Poolman 2023). It sits in
# sequence after rib 12 so the rib block reads as one series per side.
N_RIBS = 13
RIB_LEFT_OFFSET, RIB_RIGHT_OFFSET = 33, 46                     # 34-46 left, 47-59 right
RIB13_LEFT, RIB13_RIGHT = RIB_LEFT_OFFSET + 13, RIB_RIGHT_OFFSET + 13   # 46, 59
# a rudimentary rib on a LUMBAR-type L1 gets its own id, directly above the ribs
LUMBAR_RIB_LEFT, LUMBAR_RIB_RIGHT = 60, 61

# ── surgical hardware ────────────────────────────────────────────────────────
# Instrumentation is not bone and not any anatomical class, but it is not nothing either:
# a cage bridging a disc space fuses two vertebrae into one connected object for any
# segmenter, and dense metal in the interspace is exactly what makes an iatrogenic fusion
# look like a congenital transitional vertebra to a distance measurement. Labelling it
# keeps that distinction recoverable.
#
# A BLOCK, not a single id. Cage / screw / rod / plate are different objects with
# different consequences, and a lone generic class cannot be subdivided later without
# rewriting every label that used it. 62 is the generic call when the subtype is not
# being distinguished; 63-65 keep that decision open. Subtypes are named where they are
# identifiable: collapsing `cage` into generic `hardware` later is a one-line merge,
# whereas splitting a generic label back into subtypes means revisiting every case.
HARDWARE = 62                       # instrumentation, subtype not distinguished
HARDWARE_CAGE = 63                  # interbody cage / spacer
HARDWARE_SCREW_ROD = 64             # pedicle screws and rods
HARDWARE_PLATE = 65                 # plates and other fixation
# The cohort held none of the four above and forced three more: a femoral stem is long
# and thin and a shape rule would call it a rod, but it replaces a joint where a screw holds
# parts of one bone together. Osteosynthesis and arthroplasty are different objects.
HARDWARE_ARTHROPLASTY = 66          # joint replacement (hip in this cohort)
HARDWARE_SI_SCREW = 67              # sacroiliac screw fixation
HARDWARE_OSTEOSYNTHESIS = 68        # fracture fixation within one bone
MAX_ID = HARDWARE_OSTEOSYNTHESIS    # the highest identifier in the scheme

# Remaps between published versions (see HISTORY). Applied by scripts/renumber_labels.py.
OLD_TO_NEW_V9: Dict[int, int] = {74: 58, 75: 59, 76: 60, 77: 61, 78: 62, 79: 63,
                                 80: 64, 81: 65, 82: 66}
OLD_TO_NEW_V10: Dict[int, int] = {**{46 + n: 47 + n for n in range(12)},   # right ribs 1..12
                                  58: 60, 59: 61,                             # lumbar ribs
                                  60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68}

_VERSE_NAMES = (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                + [f"T{n}" for n in range(1, 13)]              # T1..T12 -> 8..19
                + ["L1", "L2", "L3", "L4", "L5", "L6"])        # L1..L6 -> 20..25


def label_dict() -> Dict[str, int]:
    """Full {name: id} legend (background..hardware) — the ONE map for dataset.json + docs."""
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
    for n in range(1, N_RIBS + 1):
        d[f"rib_left_{n}"] = RIB_LEFT_OFFSET + n               # 34..46
    for n in range(1, N_RIBS + 1):
        d[f"rib_right_{n}"] = RIB_RIGHT_OFFSET + n             # 47..59
    # a rib articulating with a LUMBAR-type vertebra is its own class, NOT forced to be
    # "rib 12" and NOT a thirteenth rib
    d["rib_left_lumbar"] = LUMBAR_RIB_LEFT                     # 60
    d["rib_right_lumbar"] = LUMBAR_RIB_RIGHT                   # 61
    d["hardware"] = HARDWARE                                   # 62 — subtype not distinguished
    d["hardware_cage"] = HARDWARE_CAGE                         # 63 — interbody cage
    d["hardware_screw_rod"] = HARDWARE_SCREW_ROD               # 64 — screws / rods
    d["hardware_plate"] = HARDWARE_PLATE                       # 65 — plates
    d["hardware_arthroplasty"] = HARDWARE_ARTHROPLASTY       # 66
    d["hardware_si_screw"] = HARDWARE_SI_SCREW               # 67
    d["hardware_osteosynthesis"] = HARDWARE_OSTEOSYNTHESIS   # 68
    return d


def rib_id(side: str, number: int) -> int:
    return (RIB_LEFT_OFFSET if side == "left" else RIB_RIGHT_OFFSET) + number


def verify() -> None:
    """Assert the scheme is collision-proof, contiguous and VerSe-faithful (run in tests)."""
    d = label_dict()
    ids = [v for k, v in d.items() if k != "background"]
    assert len(ids) == len(set(ids)), "DUPLICATE label id — collision in label_scheme!"
    # spine is VerSe verbatim
    for v, out in VERSE_SPINE.items():
        assert v == out, f"spine id {v} is remapped to {out} — must be VerSe-native"
    # every non-spine structure sits at/above the sacrum, never inside the vertebra range
    for nm in ["left_hip", "right_hip", "femur_left", "femur_right",
               "rib_left_1", "rib_right_13", "rib_left_lumbar", "rib_right_lumbar"]:
        assert d[nm] >= 26, f"{nm}={d[nm]} collides with the VerSe vertebra range (1–25)"
    # the space is contiguous: every id from 0 to MAX_ID is assigned exactly once
    assert sorted(d.values()) == list(range(0, MAX_ID + 1)), "gap or overlap in the id space"
    # ribs don't overlap femurs/pelvis; right ribs follow left; lumbar ribs and hardware follow
    assert RIB_LEFT_OFFSET + 1 > FEMUR_RIGHT, "ribs overlap femurs"
    assert RIB_RIGHT_OFFSET == RIB_LEFT_OFFSET + N_RIBS, "right ribs must follow rib_left_13"
    assert LUMBAR_RIB_LEFT == RIB_RIGHT_OFFSET + N_RIBS + 1, "lumbar ribs must follow rib_right_13"
    assert HARDWARE == LUMBAR_RIB_RIGHT + 1, "hardware must follow the lumbar ribs"
    hw = (HARDWARE, HARDWARE_CAGE, HARDWARE_SCREW_ROD, HARDWARE_PLATE,
          HARDWARE_ARTHROPLASTY, HARDWARE_SI_SCREW, HARDWARE_OSTEOSYNTHESIS)
    assert len(set(hw)) == len(hw), "duplicate hardware id"
    assert IGNORE_LABEL not in d.values(), "the training ignore label is not a release class"
    assert set(OLD_TO_NEW_V10.values()) <= set(d.values()), "remap targets must be scheme ids"
    assert len(set(OLD_TO_NEW_V10.values())) == len(OLD_TO_NEW_V10), "remap must be injective"


verify()
