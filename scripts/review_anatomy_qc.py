"""
review_anatomy_qc.py -> fast, VerSe-native anatomical QC for student review, run on
every Save so the annotator gets immediate PASS / "here's what's wrong" feedback.

Two checks (pick per cohort):

  spine  -> pelvic-native pseudo-spine review:
           * NO class mixing      (each vertebra label is one piece, not scattered)
           * vertebrae ASCENDING  (id up => more caudal => lower in the volume)
           * vertebrae CONTIGUOUS (no missing level inside the labelled run)

  ribs   -> rib-correction review:
           * every numbered rib CONTACTS its thoracic vertebra (rib N touches T-N)
           * every thoracic vertebra has a LEFT and RIGHT rib

Everything is integer/connected-component bookkeeping + one tiny cropped distance
transform per rib, so it returns in well under a second on a full label volume.

Ids come from scripts/label_scheme.py: vertebrae 1-28 (T-N = 7+N, so T1=8..T12=19),
sacrum 26; rib_left_N = RIB_LEFT_OFFSET+N (34..45), rib_right_N = RIB_RIGHT_OFFSET+N
(46..57).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from nibabel.affines import apply_affine
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS          # canonical VerSe-native ids

CONTACT_MM = 3.0       # a rib within this distance of its vertebra counts as "touching"
MIX_FRAC = 0.15        # a 2nd connected piece >= this fraction of a class = class mixing
MIX_MIN_VOX = 50       # ...and at least this many voxels (ignore tiny spurs)
GAP_MM_MISLABEL = 25.0  # rib in 2 pieces >= this far apart = two structures share a number
                        # (mislabel -> must fix); a smaller gap is one broken rib (advisory)


def _id2name() -> dict:
    return {v: k for k, v in LS.label_dict().items()}


def _rib_id(side: str, n: int) -> int:
    return (LS.RIB_LEFT_OFFSET if side == "left" else LS.RIB_RIGHT_OFFSET) + n


def _centroid_z(lab: np.ndarray, ids, affine) -> dict:
    """World (RAS) Z of each present id's centroid (cranio-caudal position)."""
    out = {}
    for v in ids:
        m = lab == v
        if m.any():
            ijk = np.array(np.nonzero(m)).mean(axis=1)
            out[v] = float(apply_affine(affine, ijk)[2])
    return out


def _gap_mm(rib_mask: np.ndarray, vert_mask: np.ndarray, spacing) -> float:
    """Min surface distance (mm) from a rib to a vertebra, via a cropped EDT."""
    both = rib_mask | vert_mask
    loc = ndimage.find_objects(both.astype(np.int8))
    if not loc or loc[0] is None:
        return float("inf")
    sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in loc[0])
    edt = ndimage.distance_transform_edt(~vert_mask[sl], sampling=spacing)
    sub = rib_mask[sl]
    return float(edt[sub].min()) if sub.any() else float("inf")


def spine_sanity(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Class-mixing + ascending + contiguous check over the vertebral column (1-26)."""
    names = _id2name()
    present = [v for v in range(1, 27) if (lab == v).any()]
    if not present:
        return True, ["(no spine vertebrae in view)"]
    ok, msgs = True, []
    st = ndimage.generate_binary_structure(3, 3)

    # 1) class mixing -> a vertebra label split into a big second blob
    for v in present:
        m = lab == v
        n = int(m.sum())
        cc, k = ndimage.label(m, structure=st)
        if k > 1:
            sizes = np.sort(np.bincount(cc.ravel())[1:])
            if sizes[-2] >= MIX_FRAC * n and sizes[-2] >= MIX_MIN_VOX:
                ok = False
                msgs.append(f"X {names.get(v, v)} (id {v}) is split into {k} pieces "
                            f"({100 * sizes[-2] / n:.0f}% in a stray blob) -> "
                            f"merge it / relabel the stray voxels")

    # 2) contiguity -> no missing level inside the labelled run. L6 (id 25) is an
    # anatomical variant (only present with lumbarization); normal spines go L5 (24)
    # straight to sacrum (26), so its absence is NOT a gap.
    OPTIONAL = {25}                # L6 (lumbar 20-25); present only with lumbarization
    lo, hi = min(present), max(present)
    for v in range(lo, hi + 1):
        if v not in present and v not in OPTIONAL:
            ok = False
            msgs.append(f"X gap: {names.get(v, v)} (id {v}) is missing between "
                        f"{names.get(lo, lo)} and {names.get(hi, hi)} -> label it")

    # 3) ascending order -> higher id must sit more caudal (lower world Z)
    zc = _centroid_z(lab, present, affine)
    ordered = sorted(present)
    for a, b in zip(ordered, ordered[1:]):
        if a in zc and b in zc and zc[b] >= zc[a]:
            ok = False
            msgs.append(f"X order: {names.get(b, b)} (id {b}) sits at/above "
                        f"{names.get(a, a)} (id {a}) -> vertebrae out of sequence "
                        f"(check the labels were not swapped)")

    if ok:
        msgs.append(f"OK spine: {names.get(lo, lo)} -> {names.get(hi, hi)} "
                    f"contiguous, ascending, no class mixing")
    return ok, msgs


def rib_contact(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Every rib touches its vertebra; every thoracic vertebra has L+R ribs."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    thoracic = [n for n in range(1, 13) if (lab == 7 + n).any()]      # T-N present (8..19)
    ribs_present = [(s, n) for s in ("left", "right") for n in range(1, 13)
                    if (lab == _rib_id(s, n)).any()]
    if not thoracic and not ribs_present:
        return True, ["(no ribs / thoracic vertebrae in view)"]
    ok, msgs = True, []

    # 1) each numbered rib contacts its own thoracic vertebra
    for side, n in ribs_present:
        rm = lab == _rib_id(side, n)
        vm = lab == (7 + n)
        if not vm.any():
            ok = False
            msgs.append(f"X {side} rib {n} present but T{n} (id {7 + n}) not labelled -> "
                        f"check the rib number")
            continue
        gap = _gap_mm(rm, vm, spacing)
        if gap > CONTACT_MM:
            ok = False
            msgs.append(f"X {side} rib {n} not touching T{n} (gap {gap:.1f} mm) -> "
                        f"extend the rib head to the vertebra")

    # 2) each thoracic vertebra has a left and right rib
    names = _id2name()
    for n in thoracic:
        for side in ("left", "right"):
            if not (lab == _rib_id(side, n)).any():
                ok = False
                msgs.append(f"X T{n} (id {7 + n}) has no {side.upper()} rib -> add it "
                            f"(or, if it's only partly in the scan, leave it)")

    if ok:
        msgs.append("OK ribs: every rib contacts its vertebra; every thoracic level "
                    "has a left + right rib")
    return ok, msgs


def rib_numbering(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Rib NUMBERING is clean (the v4 rib-correction task): per side the rib numbers form a
    consecutive run (no GAP) and each number is a single piece (no DUPLICATE -- e.g. a
    hyperplastic transverse process or stray sharing rib 12). Mirrors qc_v4_ribs' dup/gap
    flags, and -- unlike rib_contact -- does NOT require the thoracic vertebrae to be labelled
    (they usually aren't in a lumbosacral FOV, so contact can't be the gate here)."""
    st = ndimage.generate_binary_structure(3, 3)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    ok, msgs, any_rib = True, [], False
    for side in ("left", "right"):
        present = [n for n in range(1, 13) if (lab == _rib_id(side, n)).any()]
        if not present:
            continue
        any_rib = True
        for n in present:                                   # one number split into pieces?
            rid = _rib_id(side, n)
            idx = np.argwhere(lab == rid)
            lo = idx.min(0); hi = idx.max(0) + 1
            cc, k = ndimage.label(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == rid, structure=st)
            sizes = np.bincount(cc.ravel())[1:]
            big = [j for j in range(len(sizes)) if sizes[j] >= MIX_MIN_VOX]
            if len(big) < 2:
                continue
            order = sorted(big, key=lambda j: sizes[j], reverse=True)
            a = cc == order[0] + 1
            b = cc == order[1] + 1
            gap = float(ndimage.distance_transform_edt(~a, sampling=spacing)[b].min())
            ok = False                                      # any split rib needs a human decision
            if gap >= GAP_MM_MISLABEL:                      # far apart -> almost certainly 2 ribs
                msgs.append(f"X {side} rib {n}: two pieces {gap:.0f} mm apart -> two structures "
                            f"share rib {n}; relabel the wrong piece to its correct number "
                            f"(or delete it).")
            else:                                           # close -> could be a break OR 2 ribs
                msgs.append(f"X {side} rib {n}: two pieces {gap:.0f} mm apart -> if ONE broken "
                            f"rib, weld the pieces; if TWO different ribs, relabel the wrong one "
                            f"(a small gap can still be a mislabel near the spine).")
        gaps = [n for n in range(min(present), max(present) + 1) if n not in present]
        if gaps:                                            # GAP: a rib NUMBER is missing
            ok = False
            msgs.append(f"X {side} rib numbers have a gap at {gaps} -> a rib level is missing; "
                        f"label it, or renumber so the present ribs are consecutive")
    if not any_rib:
        return True, ["(no ribs in view)"]
    if ok:
        msgs.append("OK ribs: numbers consecutive per side, one piece each")
    return ok, msgs


def check_label(check: str, lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Run the requested check(s) and return (ok, messages) WITHOUT printing.
    The server-side review gate uses this; the CLI uses report() (which prints)."""
    blocks = []
    if check in ("spine", "both"):
        blocks.append(spine_sanity(lab, affine))
    if check in ("ribs", "both"):
        blocks.append(rib_numbering(lab, affine))
    ok = all(o for o, _ in blocks)
    msgs = [m for _, ms in blocks for m in ms]
    return ok, msgs


def report(check: str, lab: np.ndarray, affine) -> bool:
    """Run the requested check(s) and print a PASS/FAIL block. Returns overall ok."""
    blocks = []
    if check in ("spine", "both"):
        blocks.append(("SPINE", *spine_sanity(lab, affine)))
    if check in ("ribs", "both"):
        blocks.append(("RIBS", *rib_numbering(lab, affine)))   # v4 task: dup/gap, not contact
    allok = all(ok for _, ok, _ in blocks)
    for name, ok, msgs in blocks:
        print(f"  [{name}] {'PASS' if ok else 'FAIL'}")
        for line in msgs:
            print(f"    {line}")
    print("  ===> " + ("ALL CHECKS PASS -> Save once more if you edited, then quit to submit."
                       if allok else
                       "fix the 'X' items above, then Save again to re-check."))
    return allok


if __name__ == "__main__":
    import argparse
    import nibabel as nib
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", type=Path, help="a label NIfTI to check")
    ap.add_argument("--check", choices=("spine", "ribs", "both"), default="both")
    a = ap.parse_args()
    img = nib.load(str(a.label))
    raise SystemExit(0 if report(a.check, np.asanyarray(img.dataobj), img.affine) else 1)
