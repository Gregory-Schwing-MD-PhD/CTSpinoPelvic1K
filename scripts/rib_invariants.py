"""
rib_invariants.py — the anatomical CONTRACT for rib numbering, checkable on any label
volume. One source of truth shared by the build (gate), the instancer (self-check), and
the test suite. Pure: (label array, affine) -> verdict.

Per side, a correct rib labelling must satisfy:
  ORDER     numbers increase as world-Z decreases (rib 1 cranial -> rib 12 caudal).   [hard]
  COHERENT  each rib id is one tight z-band, not voxels scattered across the cage      [hard]
            (a rib that absorbed a far fragment = the duplicate-merge bug).
  CONTIGUOUS no number missing between the lowest and highest present rib.             [soft]
  SYMMETRIC the left and right present-number sets agree within a tolerance.           [soft]

HARD violations (ORDER, COHERENT) must never occur in a correctly-built case — the
instancer is constructed so they can't, and the tests assert it. SOFT violations
(CONTIGUOUS, SYMMETRIC) can reflect real anatomy / partial FOV, so they flag a case for
review rather than failing it.

Ids come from scripts/label_scheme.py: rib_left_N = RIB_LEFT_OFFSET+N (34..45),
rib_right_N = RIB_RIGHT_OFFSET+N (46..57), N in 1..12.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nibabel.affines import apply_affine
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402

HARD_TYPES = {"order", "coherence"}
SPREAD_FACTOR = 1.8        # a rib id wider than this * pitch (mm) in z = absorbed a far blob
SYMMETRY_TOL = 2           # > this many one-sided ribs (total) -> asymmetry flag


def rib_stats(lab: np.ndarray, affine) -> Dict[str, Dict[int, dict]]:
    """{side: {number: {z, zmin, zmax, ncomp, vox}}} for every present rib (world mm)."""
    out: Dict[str, Dict[int, dict]] = {"left": {}, "right": {}}
    st = ndimage.generate_binary_structure(3, 3)
    for side, off in (("left", LS.RIB_LEFT_OFFSET), ("right", LS.RIB_RIGHT_OFFSET)):
        for n in range(1, 13):
            m = lab == off + n
            if not m.any():
                continue
            ijk = np.argwhere(m)
            zw = apply_affine(affine, ijk)[:, 2]
            _, ncomp = ndimage.label(m, structure=st)
            out[side][n] = {"z": float(zw.mean()), "zmin": float(zw.min()),
                            "zmax": float(zw.max()), "ncomp": int(ncomp),
                            "vox": int(m.sum())}
    return out


def estimate_pitch(stats: Dict[str, Dict[int, dict]]) -> float:
    """Median |Δz| between consecutively-numbered ribs (mm). Falls back to 20 mm."""
    gaps = []
    for side in ("left", "right"):
        nums = sorted(stats[side])
        for a, b in zip(nums, nums[1:]):
            if b - a == 1:
                gaps.append(abs(stats[side][a]["z"] - stats[side][b]["z"]))
    return float(np.median(gaps)) if gaps else 20.0


def check_rib_invariants(lab: np.ndarray, affine, *,
                         spread_factor: float = SPREAD_FACTOR,
                         symmetry_tol: int = SYMMETRY_TOL
                         ) -> Tuple[bool, List[dict]]:
    """Return (hard_ok, violations). hard_ok is False iff any HARD (order/coherence)
    violation exists. `violations` lists ALL findings (hard + soft) with a severity."""
    stats = rib_stats(lab, affine)
    pitch = estimate_pitch(stats)
    viol: List[dict] = []

    for side in ("left", "right"):
        nums = sorted(stats[side])
        if not nums:
            continue
        # ORDER (hard): higher number must sit more caudal (lower world Z).
        for a, b in zip(nums, nums[1:]):
            if stats[side][b]["z"] >= stats[side][a]["z"]:
                viol.append({"type": "order", "severity": "hard", "side": side,
                             "between": [a, b], "za": round(stats[side][a]["z"], 1),
                             "zb": round(stats[side][b]["z"], 1)})
        # COHERENCE (hard): a rib id must be one tight z-band.
        for n in nums:
            spread = stats[side][n]["zmax"] - stats[side][n]["zmin"]
            if spread > spread_factor * pitch:
                viol.append({"type": "coherence", "severity": "hard", "side": side,
                             "number": n, "spread_mm": round(spread, 1),
                             "limit_mm": round(spread_factor * pitch, 1)})
        # CONTIGUOUS (soft): no interior missing number.
        for n in range(min(nums), max(nums) + 1):
            if n not in stats[side]:
                viol.append({"type": "gap", "severity": "soft", "side": side, "number": n})

    # SYMMETRIC (soft): present-number sets should roughly match L vs R.
    left, right = set(stats["left"]), set(stats["right"])
    only_l, only_r = sorted(left - right), sorted(right - left)
    if len(only_l) + len(only_r) > symmetry_tol:
        viol.append({"type": "asymmetry", "severity": "soft",
                     "left_only": only_l, "right_only": only_r})

    hard_ok = not any(v["severity"] == "hard" for v in viol)
    return hard_ok, viol


def summarize(violations: List[dict]) -> str:
    """One-line human summary of a violation list (for logs / the worklist)."""
    if not violations:
        return "OK"
    by = {}
    for v in violations:
        by.setdefault(v["type"], 0)
        by[v["type"]] += 1
    return ", ".join(f"{k}x{n}" for k, n in sorted(by.items()))


if __name__ == "__main__":
    import argparse
    import nibabel as nib
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", type=Path)
    a = ap.parse_args()
    img = nib.load(str(a.label))
    ok, v = check_rib_invariants(np.asanyarray(img.dataobj), img.affine)
    print(("HARD-OK" if ok else "HARD-FAIL"), "|", summarize(v))
    for x in v:
        print("  ", x)
    raise SystemExit(0 if ok else 1)
