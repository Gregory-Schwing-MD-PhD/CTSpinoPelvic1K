"""label_scheme.py — THE single source of truth for CTSpinoPelvic1K label ids.

VerSe-native: the spine keeps its VerSe ids VERBATIM (no remap — that was the v3 bug),
and every structure NOT in VerSe gets a fixed, reserved id ABOVE the VerSe range, so no
two structures can ever share an id.

    spine   (VerSe, from CTSpine1K) : 1–7 C1–C7 · 8–19 T1–T12 · 20–25 L1–L6 · 26 sacrum
                                       · 27 coccyx · 28 T13           ← passed through AS-IS
    pelvis  (CTPelvic1K + TS femurs): 26 sacrum [shared] · 29 S1 · 30 left_hip · 31 right_hip
                                       · 32 femur_left · 33 femur_right
    ribs    (numbered off GT thoracic): 34–45 rib_left_1..12 · 46–57 rib_right_1..12
    58–73                           : RETIRED, unassigned (see RETIRED_IDS)
    lumbar ribs                     : 74 left · 75 right
    hardware                        : 76 generic · 77 cage · 78 screw_rod · 79 plate
                                       · 80 arthroplasty · 81 si_screw · 82 osteosynthesis
    ignore                          : 255

The scheme is BONE AND HARDWARE ONLY. There is no soft-tissue class.

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
# a rib on a LUMBAR vertebra (13th-rib / LSTV) gets its own id, above the retired 58-73 gap
LUMBAR_RIB_LEFT, LUMBAR_RIB_RIGHT = 74, 75

# ── surgical hardware ────────────────────────────────────────────────────────
# Instrumentation is not bone and not any anatomical class, but it is not nothing either:
# a cage bridging a disc space fuses two vertebrae into one connected object for any
# segmenter, and dense metal in the interspace is exactly what makes an iatrogenic fusion
# look like a congenital transitional vertebra to a distance measurement. Labelling it
# keeps that distinction recoverable; `ignore` would erase it.
#
# A RESERVED BLOCK, not a single id. Cage / screw / rod / plate are different objects with
# different consequences, and a lone generic class cannot be subdivided later without
# rewriting every label that used it. 76 is the generic call to use when the subtype is not
# being distinguished; 77-79 are reserved so that decision stays open.
# Subtypes are named where they are identifiable. A reader who can see it is a cage should
# say so: collapsing `cage` into generic `hardware` later is a one-line merge, whereas
# splitting a generic label back into subtypes means revisiting every case that used it.
# 76 stays available for instrumentation whose subtype is unclear or not being recorded.
HARDWARE = 76                       # instrumentation, subtype not distinguished
HARDWARE_CAGE = 77                  # interbody cage / spacer
HARDWARE_SCREW_ROD = 78             # pedicle screws and rods
HARDWARE_PLATE = 79                 # plates and other fixation
# The cohort held none of the four above and forced three more (v6): a femoral stem is long
# and thin and a shape rule would call it a rod, but it replaces a joint where a screw holds
# parts of one bone together. Osteosynthesis and arthroplasty are different objects.
HARDWARE_ARTHROPLASTY = 80          # joint replacement (hip in this cohort)
HARDWARE_SI_SCREW = 81              # sacroiliac screw fixation
HARDWARE_OSTEOSYNTHESIS = 82        # fracture fixation within one bone

# ── 58..73: RETIRED, deliberately unassigned ─────────────────────────────────
# These ids once reserved a soft-tissue overlay block (iliolumbar ligaments, nerve roots,
# psoas, great vessels). No released volume ever carried any of them -- confirmed by a
# census of all 802 v7 labels -- and the dataset is a bone dataset. The block is removed
# rather than left declared-but-empty, because three files had drifted into three
# different name lists for the same ids, which is what an unused block invites.
#
# The ids stay UNASSIGNED. Do not renumber the lumbar ribs (74/75) or hardware (76..82)
# down into the gap: those ids are in published volumes and a gap in an integer label
# space costs nothing. If a soft-tissue layer is ever released it takes fresh ids above
# the hardware block, with its own release note.
RETIRED_IDS = range(58, 74)

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
    # 58..73 intentionally absent: see RETIRED_IDS above.
    # a rib articulating with a LUMBAR vertebra (13th-rib / LSTV phenotype) is its own class, NOT
    # forced to be "rib 12" -- it keeps 74/75, the ids it has in every published volume.
    d["rib_left_lumbar"] = LUMBAR_RIB_LEFT                     # 74
    d["rib_right_lumbar"] = LUMBAR_RIB_RIGHT                   # 75
    d["hardware"] = HARDWARE                                   # 76 — subtype not distinguished
    d["hardware_cage"] = HARDWARE_CAGE                         # 77 — interbody cage
    d["hardware_screw_rod"] = HARDWARE_SCREW_ROD               # 78 — screws / rods
    d["hardware_plate"] = HARDWARE_PLATE                       # 79 — plates
    d["hardware_arthroplasty"] = HARDWARE_ARTHROPLASTY       # 80
    d["hardware_si_screw"] = HARDWARE_SI_SCREW               # 81
    d["hardware_osteosynthesis"] = HARDWARE_OSTEOSYNTHESIS   # 82
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
    for nm in ["left_hip", "right_hip", "femur_left", "femur_right",
               "rib_left_1", "rib_right_12", "rib_left_lumbar", "rib_right_lumbar"]:
        assert d[nm] >= 26, f"{nm}={d[nm]} collides with the VerSe vertebra range (1–25)"
    # the retired soft-tissue block must stay empty: reusing those ids would silently
    # collide with any third-party tooling still carrying the old names for them
    assert not (set(d.values()) & set(RETIRED_IDS)), "an id in the retired 58..73 block is assigned"
    # ribs don't overlap femurs/pelvis
    assert RIB_LEFT_OFFSET + 1 > FEMUR_RIGHT, "ribs overlap femurs"
    # hardware sits above every anatomical class and clear of its own reserved block
    assert HARDWARE > LUMBAR_RIB_RIGHT, "hardware collides with the lumbar-rib block"
    hw = (HARDWARE, HARDWARE_CAGE, HARDWARE_SCREW_ROD, HARDWARE_PLATE,
          HARDWARE_ARTHROPLASTY, HARDWARE_SI_SCREW, HARDWARE_OSTEOSYNTHESIS)
    assert len(set(hw)) == len(hw), "duplicate hardware id"
    assert all(h > LUMBAR_RIB_RIGHT for h in hw), "hardware collides with an anatomy block"


verify()
