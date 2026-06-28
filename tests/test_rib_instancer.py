"""
test_rib_instancer.py — phantom + invariant tests for the rib numbering contract.

Runnable with no data and no pytest:  python tests/test_rib_instancer.py
Builds synthetic rib cages (ribs = horizontal slabs at a regular pitch, with a medial
head near a numbered spine), perturbs them, and asserts that:
  * rib_invariants flags the perturbation we injected (order / coherence / gap), and
  * rib_instancer numbers the union mask correctly-by-construction (no dup / no spurious
    gap / no mis-order), merging split pieces, handling partial FOV, and clamping +
    flagging bone that extrapolates past rib 12 instead of inventing a rib 13.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import label_scheme as LS          # noqa: E402
import rib_invariants as INV       # noqa: E402
import rib_instancer as INST       # noqa: E402

SHAPE = (40, 60, 110)
SX = 20                            # spine x
PITCH = 7
ZTOP = 95                         # rib 1 z
DZ = 2
AFF = np.eye(4)


def zof(n):
    return ZTOP - (n - 1) * PITCH


def blank():
    return np.zeros(SHAPE, np.int16)


def put_vertebra(vol, k, value=None):
    z = zof(k)
    vol[SX - 2:SX + 2, 28:32, z - 2:z + 3] = k if value is None else value


def put_rib(vol, n, side, value, length=10, z=None, xgap=0, dz=DZ):
    """Place a rib slab with a midline gap (left/right are separate bones, never touch
    at the spine). The head is the medial end, ~3 vox off the spine; xgap pushes a piece
    further lateral (to make a detached anterior fragment)."""
    z = zof(n) if z is None else z
    if side == "left":
        x1 = max(1, SX - 3 - xgap); x0 = max(0, x1 - length)
    else:
        x0 = min(SHAPE[0] - 1, SX + 3 + xgap); x1 = min(SHAPE[0], x0 + length)
    vol[x0:x1, 28:34, z - dz:z + dz + 1] = value


def numbered_label(levels_l, levels_r):
    v = blank()
    for n in levels_l:
        put_rib(v, n, "left", LS.RIB_LEFT_OFFSET + n)
    for n in levels_r:
        put_rib(v, n, "right", LS.RIB_RIGHT_OFFSET + n)
    return v


def rib_binary(levels_l, levels_r):
    b = blank()
    for n in levels_l:
        put_rib(b, n, "left", 1)
    for n in levels_r:
        put_rib(b, n, "right", 1)
    return b > 0


def thoracic(levels):
    t = blank()
    for k in levels:
        put_vertebra(t, k)
    return t


def present(out, side):
    st = INV.rib_stats(out, AFF)
    return sorted(st[side])


# ── tests ─────────────────────────────────────────────────────────────────────
def t_invariant_correct():
    ok, v = INV.check_rib_invariants(numbered_label(range(1, 13), range(1, 13)), AFF)
    assert ok and not v, v
    return "correct cage -> invariants OK"


def t_invariant_gap_is_soft():
    ok, v = INV.check_rib_invariants(numbered_label([1, 2, 4, 5], [1, 2, 3, 4, 5]), AFF)
    assert ok, "gap must be SOFT (hard_ok stays True)"
    assert any(x["type"] == "gap" and x["number"] == 3 for x in v), v
    return "interior gap -> soft 'gap' flag, hard_ok True"


def t_invariant_order_is_hard():
    v = blank()
    put_rib(v, 5, "left", LS.RIB_LEFT_OFFSET + 5, z=zof(7))   # rib 5 placed where 7 belongs
    put_rib(v, 7, "left", LS.RIB_LEFT_OFFSET + 7, z=zof(5))   # rib 7 placed where 5 belongs
    ok, viol = INV.check_rib_invariants(v, AFF)
    assert not ok and any(x["type"] == "order" for x in viol), viol
    return "swapped ribs -> HARD order violation"


def t_invariant_coherence_is_hard():
    v = numbered_label(range(1, 13), range(1, 13))
    put_rib(v, 6, "left", LS.RIB_LEFT_OFFSET + 6, z=zof(1))   # rib-6 id also far up at rib-1 height
    ok, viol = INV.check_rib_invariants(v, AFF)
    assert not ok and any(x["type"] == "coherence" and x["number"] == 6 for x in viol), viol
    return "scattered rib id -> HARD coherence violation"


def t_instancer_recovers_full():
    out, rep = INST.instance_ribs(rib_binary(range(1, 13), range(1, 13)),
                                  thoracic(range(1, 13)), AFF)
    ok, v = INV.check_rib_invariants(out, AFF)
    assert ok, (v, rep)
    assert present(out, "left") == list(range(1, 13)), present(out, "left")
    assert present(out, "right") == list(range(1, 13)), present(out, "right")
    return "full cage -> instancer numbers 1..12 both sides, invariants OK"


def t_instancer_merges_split_rib():
    b = rib_binary(range(1, 13), range(1, 13))
    # add a DETACHED lateral piece of left rib 6 (anterior fragment, same height)
    frag = blank(); put_rib(frag, 6, "left", 1, length=6, xgap=14)
    b = b | (frag > 0)
    out, rep = INST.instance_ribs(b, thoracic(range(1, 13)), AFF)
    ok, v = INV.check_rib_invariants(out, AFF)
    assert ok, (v, rep)
    assert present(out, "left") == list(range(1, 13)), present(out, "left")   # still 12, not 13
    return "split rib 6 (2 pieces) -> merged into one rib 6, no dup/gap"


def t_instancer_partial_fov():
    out, rep = INST.instance_ribs(rib_binary(range(6, 13), range(6, 13)),
                                  thoracic(range(6, 13)), AFF)
    ok, v = INV.check_rib_invariants(out, AFF)
    assert ok, (v, rep)
    assert present(out, "left") == list(range(6, 13)), present(out, "left")   # anchored to 6..12
    return "partial FOV (T6-T12) -> ribs numbered 6..12, not 1..7"


def t_instancer_clamps_bottom_fragment():
    b = rib_binary(range(1, 13), range(1, 13))
    frag = blank(); put_rib(frag, 1, "left", 1, length=8, z=zof(12) - PITCH)  # one pitch below rib 12
    b = b | (frag > 0)
    out, rep = INST.instance_ribs(b, thoracic(range(1, 13)), AFF)
    ids = set(int(x) for x in np.unique(out)) - {0}
    assert all(LS.RIB_LEFT_OFFSET < i <= LS.RIB_RIGHT_OFFSET + 12 for i in ids), ids
    assert max((n for n in present(out, "left")), default=0) <= 12
    assert rep.get("out_of_range"), "fragment past rib 12 must be flagged, not made rib 13"
    return "below-rib-12 fragment -> clamped to 12 + flagged (no rib 13 invented)"


TESTS = [t_invariant_correct, t_invariant_gap_is_soft, t_invariant_order_is_hard,
         t_invariant_coherence_is_hard, t_instancer_recovers_full,
         t_instancer_merges_split_rib, t_instancer_partial_fov,
         t_instancer_clamps_bottom_fragment]


def main():
    fails = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"  PASS  {t.__name__}: {msg}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:                      # noqa: BLE001
            fails += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}  ({len(TESTS)} tests)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
