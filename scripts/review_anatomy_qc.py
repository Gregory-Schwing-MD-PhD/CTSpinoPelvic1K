"""
review_anatomy_qc.py -> fast, VerSe-native anatomical QC for student review, run on
every Save so the annotator gets immediate PASS / "here's what's wrong" feedback.

Two checks (pick per cohort):

  spine  -> pelvic-native pseudo-spine review:
           * NO class mixing      (each vertebra label is one piece, not scattered)
           * vertebrae ASCENDING  (id up => more caudal => lower in the volume)
           * vertebrae CONTIGUOUS (no missing level inside the labelled run)

  ribs   -> rib-correction review, three tests in priority order:
           1. ONE PIECE   each rib number is a single connected component per side
                          (no duplicate/split -- the primary v4 correction target)
           2. CONTIGUOUS  the rib numbers per side form a consecutive run 12,11,10,...
                          (no missing level inside the labelled range)
           3. ANCHOR      each rib N is incident on its own thoracic vertebra T-N
                          (the "12th rib sits on T12" landmark) -- only where T-N is
                          labelled (thoracic GT is FOV-limited, so absent vertebrae are
                          skipped, not failed); catches off-by-one numbering.

Everything is integer/connected-component bookkeeping + one tiny cropped distance
transform per rib, so it returns in well under a second on a full label volume.

Ids come from scripts/label_scheme.py: vertebrae 1-28 (T-N = 7+N, so T1=8..T12=19),
sacrum 26; rib_left_N = RIB_LEFT_OFFSET+N (34..45), rib_right_N = RIB_RIGHT_OFFSET+N
(46..57).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
ANCHOR_MM = 10.0        # rib N is "incident on" T-N if within this of the vertebra body.
                        # Loose on purpose: thoracic bodies are ~20 mm apart, so 10 mm cleanly
                        # separates the right vertebra (touching, ~0-5 mm) from an off-by-one
                        # neighbour (~15-25 mm), without false-flagging a small seg gap.
MIX_LABEL_MIN_VOX = 250  # in the "one bone -> one label" check, a 2nd label on the same eroded
                         # rib component counts as a real mix only above this (ignore sliver
                         # boundary voxels between genuinely-adjacent ribs).
GAP_DETACH_MM = 15.0     # a rib whose head is farther than this from its vertebra is DETACHED (the
                         # in-view gap distribution is sharply bimodal: ~85% <=3 mm touching, tail
                         # >30 mm detached, empty valley between -> 15 mm sits safely in the valley).
HEAD_FOV_MARGIN_MM = 15.0  # if a rib's HEAD (superior extent) is within this of the top of the scan,
                         # its costovertebral head exits the FOV -> unfixable -> advisory, never blocks.
HEAD_BAND_MM = 20.0      # look for the rib's vertebra within +/- this of the rib head's level.
SPINE_EDIT_TOL_VOX = 5000   # a rib reviewer may not touch the spine/pelvis. Tolerance absorbs
                            # trivial boundary noise from a re-save; a genuine renumbering shifts
                            # ~1e6 voxels (measured: 1,281,398 on 742__fused), so this cannot mask one.
HALO_MAX_VOX = 2000      # A "rib" this small that is FUSED TO AN ADJACENT RIB NUMBER (N+-1) is not a
                         # rib at all -- it is HALO: the residue of the OLD label clinging to the
                         # surface of the rib the annotator renumbered. Measured on every such case:
                         # 12/12 fragments under 2000 voxels touch rib N+-1 and sit >100 mm from the
                         # spine, while every real detached rib is >=2000 voxels (clean gap: the halo
                         # group tops out at 1538, the next real rib is 2008). Post-processing engulfs
                         # these deterministically (scripts/postprocess_halo.py), so QC must NOT make a
                         # student re-open a case to erase a 13-voxel dot.
SPINE_AT_LEVEL_MIN = 1000  # >= this many vertebra voxels at the head level = a vertebra IS segmented
                         # there (a detached rib must be connected); fewer = the vertebra is MISSING
                         # but in view -> it should be ADDED (numbered up from the bottom), NOT a rib
                         # defect. Cleanly separates real cases (present ~50k-150k vs missing 0-2).
                         # (number-agnostic: distance to ANY vertebra, no dependence on numbering.)
STRUCT_MIN_VOX = 3000        # ignore a barely-present (partial-FOV) bone in the integrity check
STRUCT_DOMINANT_FRAC = 0.85  # a solid bone (vertebra/sacrum/hip/femur) must be >= this fraction
                             # ONE connected piece; below it the label is split across the bone
                             # (the classic "half the hip is a different class" student mislabel).
MIN_VERT_VOX = 6000     # a thoracic vertebra must have >= this many voxels to anchor a rib.
                        # Not just specks: a vertebra only PARTIALLY in the FOV (a ~1-2k-voxel
                        # sliver at the top of the scan) also must not anchor a full rib, or the
                        # rib gets falsely flagged "not incident on T-N" (the rib is far from the
                        # sliver). A real, mostly-in-view vertebra body is tens of thousands of
                        # voxels, so 6000 keeps those while dropping edge slivers and specks.


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


def spine_extend_qc(lab: np.ndarray, affine,
                    given: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
    """QC for the SPINE-EXTENSION task (students ADD missing thoracic vertebrae, numbering upward).
    Gates on what the student controls:
      * CONTIGUOUS numbering  — no missing level in the run (a skipped / mis-counted vertebra)
      * ASCENDING order       — each vertebra caudal to the one above (an off-target / swapped add)
      * each ADDED vertebra is a single clean blob, of plausible size, TOUCHING the column
        (a floating blob out in the ribs/soft-tissue = 'far off target')
    A pre-existing split in an EXISTING (radiologist) vertebra is EXEMPTED — the student cannot fix it
    on this task, so it must never block; it is a spine_fix-queue item, not a rejection. `given` (the
    v4 base) identifies the additions; without it (client-side) only contiguity + order run."""
    names = _id2name()
    present = [v for v in range(1, 27) if (lab == v).any()]
    if not present:
        return True, ["(no spine vertebrae in view)"]
    ok, msgs = True, []
    st = ndimage.generate_binary_structure(3, 3)

    added = set()
    if given is not None and given.shape == lab.shape:
        added = {int(v) for v in np.unique(lab[(lab >= 8) & (lab <= 19) & (given == 0)])}

    # 1) contiguity — no missing level inside the run (L6 optional)
    lo, hi = min(present), max(present)
    for v in range(lo, hi + 1):
        if v not in present and v != 25:
            ok = False
            msgs.append(f"X gap: {names.get(v, v)} (id {v}) is missing in the column -> "
                        f"label it (number consecutively, no skipped levels)")

    # 2) ascending order — a mis-placed / mis-numbered vertebra breaks the sequence
    zc = _centroid_z(lab, present, affine)
    for a, b in zip(sorted(present), sorted(present)[1:]):
        if a in zc and b in zc and zc[b] >= zc[a]:
            ok = False
            msgs.append(f"X order: {names.get(b, b)} sits at/above {names.get(a, a)} -> a vertebra is "
                        f"mis-placed or mis-numbered (numbering must ascend down the column)")

    # 3) each ADDED vertebra: clean blob, plausible size, connected to the column
    ADDED_MIN_VOX = 800                              # a speck, not a (possibly FOV-truncated) vertebra
    for v in sorted(added):
        m = lab == v; nvox = int(m.sum())
        if nvox < ADDED_MIN_VOX:
            ok = False
            msgs.append(f"X added {names.get(v, v)} is only {nvox} voxels -> too small to be a vertebra")
            continue
        cc, k = ndimage.label(m, structure=st)
        if k > 1 and np.sort(np.bincount(cc.ravel())[1:])[-2] >= MIX_MIN_VOX:
            ok = False
            msgs.append(f"X added {names.get(v, v)} is in {k} pieces -> one vertebra must be ONE blob")
        others = (lab >= 1) & (lab <= 26) & (~m)
        if others.any() and not (ndimage.binary_dilation(m, iterations=2) & others).any():
            ok = False
            msgs.append(f"X added {names.get(v, v)} is floating (not touching the spine) -> it must sit "
                        f"in the column, adjacent to the vertebra below it")

    if ok:
        msgs.append(f"OK spine additions {sorted(names.get(v, v) for v in added) or '[none]'}: "
                    f"consecutive, ascending, connected")
    return ok, msgs


def rib_contact(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Every rib touches its vertebra; every thoracic vertebra has L+R ribs."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    thoracic = [n for n in range(1, 13)
                if (lab == 7 + n).sum() >= MIN_VERT_VOX]              # T-N really in FOV (8..19)
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
    # ONE pass over the volume for every label's bounding box, then work on tiny per-rib crops.
    # (The old per-id `lab == rid` / `argwhere` scanned the full 512^3 ~24x -> seconds/minutes;
    # find_objects is a single scan -> ~10-50x faster, and the per-rib crops are negligible.)
    objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))

    def _bbox(rid):
        return objs[rid - 1] if 0 <= rid - 1 < len(objs) else None

    ok, msgs, any_rib = True, [], False
    for side in ("left", "right"):
        present = [n for n in range(1, 13) if _bbox(_rib_id(side, n)) is not None]
        if not present:
            continue
        any_rib = True
        for n in present:                                   # one number split into pieces?
            rid = _rib_id(side, n)
            cc, k = ndimage.label(lab[_bbox(rid)] == rid, structure=st)
            sizes = np.bincount(cc.ravel())[1:]
            big = [j for j in range(len(sizes)) if sizes[j] >= MIX_MIN_VOX]
            if len(big) < 2:
                continue
            order = sorted(big, key=lambda j: sizes[j], reverse=True)
            # cheap min-distance (mm) between the two largest pieces: subsample each to <=300
            # voxels and take the min pairwise gap. Avoids a full EDT on far-apart dup crops
            # (which dominated the runtime); we only need it for the mislabel-vs-break hint.
            ca = np.argwhere(cc == order[0] + 1)
            cb = np.argwhere(cc == order[1] + 1)
            ca = ca[:: max(1, len(ca) // 300)]
            cb = cb[:: max(1, len(cb) // 300)]
            dd = (ca[:, None, :] - cb[None, :, :]) * spacing
            gap = float(np.sqrt((dd ** 2).sum(-1)).min())
            ok = False                                      # any split rib needs a human decision
            # Usually one rib the algorithm split (even far apart -> a rib is a long arc), so the
            # default fix is to CONNECT the pieces; relabel/delete are the exceptions.
            msgs.append(f"X {side} rib {n}: in 2 pieces ({gap:.0f} mm apart) -> usually ONE rib "
                        f"split: CONNECT the pieces. Only relabel a piece that is a DIFFERENT "
                        f"rib, or delete a piece that is not a rib.")
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


def rib_anchor(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """ANATOMICAL ANCHOR: each rib N must be incident on its OWN thoracic vertebra T-N
    (the classic 'the 12th rib sits on T12' landmark). Validates the numbering is anchored
    to real anatomy, not just internally consecutive -- so it catches an off-by-one run
    (e.g. a rib labelled 12 that actually sits on T11).

    Only checked where T-N is labelled: thoracic GT is FOV-limited (a lumbosacral scan
    usually has no thoracic vertebrae), so ribs whose vertebra is out of view are SKIPPED,
    never failed. A rib that is >ANCHOR_MM from its own T-N but touches a different labelled
    vertebra T-M gets a concrete 'renumber to M' hint."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    # T-N counts as "in the FOV" only if it's a real vertebra, not a stray speckle of its id
    # (a lower rib often enters the scan antero-laterally with its vertebra out of view).
    thoracic = [n for n in range(1, 13) if (lab == 7 + n).sum() >= MIN_VERT_VOX]
    if not thoracic:
        return True, ["(no thoracic vertebra in the FOV -> rib anchor skipped)"]

    # Cheap min surface distance via subsampled point clouds (a full EDT per rib-vertebra pair
    # blows up when a mislabelled rib is far from the vertebra -> a volume-spanning crop). We
    # only need the distance for the anchor decision, so subsample each mask to <=250 voxels.
    def _pts(mask, cap=250):
        p = np.argwhere(mask)
        return p[:: max(1, len(p) // cap)] if len(p) else p

    def _d(a, b):
        if len(a) == 0 or len(b) == 0:
            return float("inf")
        dd = (a[:, None, :] - b[None, :, :]) * spacing
        return float(np.sqrt((dd ** 2).sum(-1)).min())

    vpts = {t: _pts(lab == (7 + t)) for t in thoracic}   # vertebra points once, reused per rib
    ok, msgs, checked = True, [], 0
    for side in ("left", "right"):
        for n in range(1, 13):
            rm = lab == _rib_id(side, n)
            if not rm.any() or n not in thoracic:       # rib absent, or its T-N out of view
                continue
            checked += 1
            rp = _pts(rm)
            gap_self = _d(rp, vpts[n])
            if gap_self <= ANCHOR_MM:
                continue                                 # sits on its own vertebra -> good
            ok = False
            m, gm = min(((t, _d(rp, vpts[t])) for t in thoracic),
                        key=lambda x: x[1])              # nearest labelled thoracic vertebra
            if gm <= ANCHOR_MM and m != n:
                msgs.append(f"X {side} rib {n} sits on T{m} (gap {gm:.0f} mm), not T{n} "
                            f"(gap {gap_self:.0f} mm) -> renumber this rib to {m}")
            else:
                msgs.append(f"X {side} rib {n} not incident on T{n} (gap {gap_self:.0f} mm) -> "
                            f"check the rib number / extend the head to the vertebra")
    if checked == 0:
        return True, ["(no rib has its thoracic vertebra in view -> rib anchor skipped)"]
    if ok:
        msgs.append(f"OK rib anchor: each in-view rib sits on its own thoracic vertebra "
                    f"({checked} checked)")
    return ok, msgs


def _subpts(mask, cap=300):
    p = np.argwhere(mask)
    return p[:: max(1, len(p) // cap)] if len(p) else p


def _mindist(a, b, spacing):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    dd = (a[:, None, :] - b[None, :, :]) * spacing
    return float(np.sqrt((dd ** 2).sum(-1)).min())


def rib_label_mixing(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """ONE connected rib BONE must carry ONE rib number. Flags a single physical rib whose voxels
    are split across 2+ labels (a common student mislabel). The rib mask is eroded by 1 first so
    two genuinely-adjacent ribs that only touch at the spine SEPARATE and are not flagged; only a
    bone that stays a single component after erosion yet holds 2+ substantial labels is a mix."""
    lo_id, hi_id = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
    ribs = (lab >= lo_id) & (lab <= hi_id)
    if not ribs.any():
        return True, ["(no ribs in view)"]
    idx = np.argwhere(ribs); lo = idx.min(0); hi = idx.max(0) + 1
    sub = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    st = ndimage.generate_binary_structure(3, 3)
    er = ndimage.binary_erosion((sub >= lo_id) & (sub <= hi_id), structure=st, iterations=1)
    cc, k = ndimage.label(er, structure=st)
    names = _id2name(); ok = True; msgs = []
    for c in range(1, k + 1):
        cnt = np.bincount(sub[cc == c].ravel())
        big = [i for i in range(lo_id, hi_id + 1) if i < len(cnt) and cnt[i] >= MIX_LABEL_MIN_VOX]
        if len(big) >= 2:
            ok = False
            msgs.append("X one rib bone carries multiple labels: "
                        + ", ".join(names.get(i, str(i)) for i in big)
                        + " -> a single rib must be ONE number; relabel the whole bone to one")
    if ok:
        msgs.append("OK: each connected rib bone is a single label")
    return ok, msgs


def rib_vertebra_match(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Rib N must articulate with vertebra T-N (T1..T12 = ids 8..19, so rib N <-> id 7+N).

    WHY THIS EXISTS. Adjudication only catches DISAGREEMENT. If BOTH annotators shift the rib
    numbering the same way, they agree with each other and the error sails through unadjudicated --
    agreement is not correctness. This is the objective test that catches it, because once the
    radiologist's spine is restored the rib's number is DETERMINED by the vertebra it touches; it
    stops being an opinion.

    Only ribs that actually REACH a vertebra are judged (gap <= GAP_DETACH_MM). A detached or
    FOV-truncated rib has no articulation to read, so it is skipped rather than guessed at."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    lo_id, hi_id = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
    # Read the nearest vertebra including the LUMBAR levels (T1..L6 = 8..25). A rib whose head sits
    # on a LUMBAR vertebra is the transitional / 13th-rib (LSTV) phenotype -- the dataset's target --
    # so it must be RECOGNISED, not skipped; skipping it made the shift it causes read as an error.
    vspan = (lab >= 8) & (lab <= 25)
    ribs = (lab >= lo_id) & (lab <= hi_id)
    if not vspan.any() or not ribs.any():
        return True, ["(no thoracolumbar vertebrae or ribs in view -> rib<->vertebra match skipped)"]
    idx = np.argwhere(vspan | ribs); lo = idx.min(0); hi = idx.max(0) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    f = 2
    sub = lab[sl][::f, ::f, ::f]
    vv = (sub >= 8) & (sub <= 25)
    if not vv.any():
        return True, ["(no thoracolumbar vertebrae in the crop -> skipped)"]
    d, ind = ndimage.distance_transform_edt(~vv, sampling=spacing * f, return_indices=True)
    names = _id2name(); ok, msgs = True, []
    # nearest vertebra to each rib's medial-most (head) voxel
    reads = {}
    for rid in range(lo_id, hi_id + 1):
        m = (sub == rid)
        if not m.any():
            continue
        dd = np.where(m, d, np.inf)
        # the MEDIAL-MOST rib voxel (closest to the spine) = the head. Read the vertebra by NEAREST
        # NEIGHBOUR from that point -- the rib does not have to be directly incident.
        p = np.unravel_index(np.argmin(dd), dd.shape)
        gap = float(dd[p])
        if not np.isfinite(gap):
            continue
        v = int(sub[ind[0][p], ind[1][p], ind[2][p]])     # nearest vertebra to the rib head
        if 8 <= v <= 25:
            reads[rid] = (v, gap)
    for side, off in (("left", LS.RIB_LEFT_OFFSET), ("right", LS.RIB_RIGHT_OFFSET)):
        side_reads = {rid: vg for rid, vg in reads.items() if off < rid <= off + 12}
        # TRANSITIONAL: a rib on this side articulates with a LUMBAR vertebra (id >= 20). That is a
        # lumbar rib -- the 13th-rib / LSTV phenotype this dataset exists to catalogue. The rib-N<->
        # T-N mapping is then legitimately shifted (every thoracic rib reads one level 'low'), which
        # is ANATOMY, not a mislabel. Keep the numbering, flag for LSTV re-read, never call it an error.
        lumbar_ribs = [rid for rid, (v, g) in side_reads.items() if v >= 20 and g <= GAP_DETACH_MM]
        if lumbar_ribs:
            msgs.append(f"note {side}: {', '.join(str(names.get(r, r)) for r in sorted(lumbar_ribs))} "
                        f"articulate(s) with a LUMBAR vertebra -> transitional (13th-rib / LSTV) "
                        f"phenotype; rib numbering kept as-is, flag for LSTV re-read (not an error)")
            continue
        for rid, (v, gap) in sorted(side_reads.items()):
            if not (8 <= v <= 19):                        # only a thoracic articulation fixes T-N
                continue
            n = rid - off
            expect = v - 7                                # vertebra T-N  ->  rib N
            if n == expect:
                continue
            if gap <= GAP_DETACH_MM:
                # head REACHES a thoracic vertebra and no transitional rib explains the shift
                ok = False
                msgs.append(f"X {names.get(rid, rid)} is MISNUMBERED: its head articulates with "
                            f"{names.get(v, v)}, so it must be {side} rib {expect}, not {n}")
            else:
                # head is detached / FOV-truncated -> nearest-neighbour is the best read but less
                # certain, so report it without blocking.
                msgs.append(f"note {names.get(rid, rid)}: nearest vertebra to its head is "
                            f"{names.get(v, v)} ({gap:.0f} mm away) -> likely should be {side} rib "
                            f"{expect}, not {n}; head is detached/truncated so not blocking")
    if ok:
        msgs.append("OK: every rib that reaches the spine matches its vertebra (rib N on T-N)")
    return ok, msgs


def _adjacent_rib_ids(rid: int) -> set:
    """The rib numbers immediately above/below `rid` on the SAME side (N-1, N+1)."""
    off = LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET
    n = rid - off
    return {off + m for m in (n - 1, n + 1) if 1 <= m <= 12}


def is_halo_speck(lab: np.ndarray, rid: int, bbox) -> bool:
    """A HALO speck: a small rib-N fragment FUSED TO rib N+-1 -- i.e. the residue of the OLD label
    left clinging to the surface of the rib the annotator renumbered. Two independent signals must
    BOTH hold (size AND fused-to-an-adjacent-number), so a small-but-real rib is never exempted.
    Post-processing engulfs these, so they must not block a student."""
    pad = tuple(slice(max(0, bbox[i].start - 3), min(lab.shape[i], bbox[i].stop + 3))
                for i in range(3))
    sub = lab[pad]
    m = (sub == rid)
    if int(m.sum()) >= HALO_MAX_VOX:                     # big enough to be a real rib -> not halo
        return False
    dil = ndimage.binary_dilation(m, iterations=2)
    touching = {int(v) for v in np.unique(sub[dil & (sub > 0) & (sub != rid)])}
    # Fused to rib N+-1 AND that neighbour is itself a REAL rib. A halo clings to a real rib; two
    # small fragments merely touching each other are NOT halo (and must never engulf each other).
    return any(int((lab == nb).sum()) >= HALO_MAX_VOX
               for nb in (touching & _adjacent_rib_ids(rid)))


def rib_spine_gap(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Each rib should approximate its vertebra. A rib that sits FULLY INSIDE the scan (its bounding
    box touches NO face of the volume) but whose head is > GAP_DETACH_MM from the nearest vertebra is
    DETACHED and fixable -> BLOCK. A rib clipped by ANY scan face exits the FOV (its head may be
    unsegmentable) -> advisory, never blocks. Number-agnostic: distance to ANY vertebra, so it does
    not depend on rib/vertebra numbers being right (unlike the advisory anchor test)."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    lo_id, hi_id = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
    spine = (lab >= 1) & (lab <= LS.S1_ID)
    ribs = (lab >= lo_id) & (lab <= hi_id)
    if not spine.any() or not ribs.any():
        return True, ["(no spine/ribs in view -> rib-spine gap skipped)"]
    # crop to the bbox of spine+ribs, downsample 2x, ONE distance transform (mm to nearest spine
    # voxel), then ONE ndimage.minimum pass to get each rib's min distance (~2 mm precision, fine
    # for a 15 mm threshold). No per-rib argwhere -> fast.
    idx = np.argwhere(spine | ribs); lo = idx.min(0); hi = idx.max(0) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    f = 2
    subd = lab[sl][::f, ::f, ::f]
    edt = ndimage.distance_transform_edt(~((subd >= 1) & (subd <= LS.S1_ID)), sampling=spacing * f)
    rib_ids = list(range(lo_id, hi_id + 1))
    mins = ndimage.minimum(edt, labels=subd, index=rib_ids)     # min EDT within each rib label
    objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))
    shape = lab.shape
    # superior-inferior axis + which end is "up": a rib whose HEAD reaches the top-of-scan edge is
    # entering from OUTSIDE the FOV, so its head + vertebra are truncated and it can't be connected.
    si = int(np.argmax(np.abs(affine[:3, :3][2, :])))
    sup_face = (shape[si] - 1) if affine[2, si] >= 0 else 0
    margin = int(round(HEAD_FOV_MARGIN_MM / spacing[si]))
    band = int(round(HEAD_BAND_MM / spacing[si]))
    names = _id2name(); ok, msgs = True, []
    for rid, g in zip(rib_ids, mins):
        if g is None or not np.isfinite(g):                      # rib absent (or too thin at 2x)
            continue
        if g <= GAP_DETACH_MM:                                   # head reaches the spine -> fine
            continue
        o = objs[rid - 1] if rid - 1 < len(objs) else None
        if o is None:
            continue
        # 1. rib literally clipped by any scan face -> exits FOV
        if any(o[i].start == 0 or o[i].stop == shape[i] for i in range(3)):
            msgs.append(f"note {names.get(rid, rid)}: {g:.0f} mm but the rib is clipped by the scan "
                        f"edge (exits FOV) -> advisory, not blocking")
            continue
        # 2. rib HEAD (superior extent) near the top of the scan -> it enters from outside the FOV,
        #    so its costovertebral head + vertebra are truncated -> unfixable -> advisory.
        rib_sup = (o[si].stop - 1) if affine[2, si] >= 0 else o[si].start
        if abs(rib_sup - sup_face) <= margin:
            msgs.append(f"note {names.get(rid, rid)}: {g:.0f} mm but its head enters from the top of "
                        f"the scan (exits FOV) -> advisory, not blocking")
            continue
        # 3. rib head is DEEP in the FOV: is a vertebra segmented at its level?
        b0 = max(0, rib_sup - band); b1 = min(shape[si], rib_sup + band + 1)
        ss = [slice(None)] * 3; ss[si] = slice(b0, b1)
        sp_at_level = int(spine[tuple(ss)].sum())
        if sp_at_level >= SPINE_AT_LEVEL_MIN:
            # 3a. HALO: a small fragment fused to rib N+-1 is the residue of the OLD label on the rib
            #     the annotator renumbered -- post-processing engulfs it, so never block a student.
            if is_halo_speck(lab, rid, o):
                msgs.append(f"note {names.get(rid, rid)}: halo speck ({g:.0f} mm out) fused to the "
                            f"neighbouring rib -> auto-removed in post-processing, not blocking")
                continue
            ok = False                                           # vertebra IS there -> connect it
            msgs.append(f"X {names.get(rid, rid)}: DETACHED — {g:.0f} mm from its vertebra (which is "
                        f"segmented at this level, in view) -> connect the rib head to it, or DELETE "
                        f"the piece if it is not really a rib")
        else:
            msgs.append(f"note {names.get(rid, rid)}: no vertebra segmented at this rib's level though "
                        f"it is IN the FOV -> the VERTEBRA should be ADDED (number up from the bottom); "
                        f"not a rib defect, not blocking")
    if ok:
        msgs.append("OK: every in-view rib reaches its (segmented) vertebra")
    return ok, msgs


def structure_integrity(lab: np.ndarray, affine) -> Tuple[bool, List[str]]:
    """Each SOLID bone (vertebra / sacrum / S1 / hip / femur, ids 1..33) must be a single dominant
    connected piece. Flags a bone whose label is split so the largest piece is < STRUCT_DOMINANT_
    FRAC of it -- the classic 'half the hip is a different class' student mislabel (in the observed
    case both hips were ~65/35 two-piece). Ribs (34-57) have their own one-piece check."""
    st = ndimage.generate_binary_structure(3, 3)
    objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))
    # array axis most aligned with the superior-inferior (world Z) direction: a bone clipped by the
    # TOP or BOTTOM of the scan along this axis is FOV-truncated and may fragment for real -> advisory.
    R = np.asarray(affine)[:3, :3]
    si_axis = int(np.argmax(np.abs(R[2, :])))
    names = _id2name(); ok, msgs = True, []
    for sid in range(1, 34):                            # 1..33 = vertebrae, sacrum, S1, hips, femurs
        if sid - 1 >= len(objs) or objs[sid - 1] is None:
            continue
        sl = objs[sid - 1]
        m = lab[sl] == sid
        tot = int(m.sum())
        if tot < STRUCT_MIN_VOX:
            continue
        cc, k = ndimage.label(m, structure=st)
        if k < 2:
            continue
        mx = int(np.bincount(cc.ravel())[1:].max())
        if mx < STRUCT_DOMINANT_FRAC * tot:
            # A bone clipped by the top/bottom slice of the scan (superior-inferior FOV edge) can be
            # split by truncation and is UNFIXABLE -> advisory, never blocks. A fully-in-FOV bone that
            # fragments is the real 'half the hip is a different class' mislabel -> hard block.
            truncated = (sl[si_axis].start == 0) or (sl[si_axis].stop == lab.shape[si_axis])
            if truncated:
                msgs.append(f"note {names.get(sid, sid)} is split but clipped by the top/bottom of "
                            f"the scan (FOV-truncated) -> advisory, not blocking")
                continue
            ok = False
            msgs.append(f"X {names.get(sid, sid)} is split into pieces (largest {100 * mx // tot}% "
                        f"of the bone) -> one bone must be ONE class; relabel the stray part")
    if ok:
        msgs.append("OK: each spine/pelvis bone is a single class")
    return ok, msgs


def _vertebra_label_mixing(lab: np.ndarray) -> bool:
    """Does ONE connected vertebral bone carry TWO vertebra labels? That is the radiologist marking
    an AMBIGUOUS / transitional level on purpose -- a body deliberately split half-L3/half-L4 -- and
    it is the one and only case where an annotator may resolve the numbering. (structure_integrity
    cannot see this: each half is still a single connected piece of its own label.)"""
    spine = (lab >= 1) & (lab <= LS.S1_ID)
    if not spine.any():
        return False
    st = ndimage.generate_binary_structure(3, 3)
    er = ndimage.binary_erosion(spine, iterations=1)      # erode so touching bodies separate
    cc, k = ndimage.label(er, structure=st)
    for i in range(1, k + 1):
        vals = lab[cc == i]
        big = [int(v) for v in np.unique(vals)
               if v > 0 and int((vals == v).sum()) >= MIX_LABEL_MIN_VOX]
        if len(big) >= 2:                                 # one bone, two levels -> ambiguous body
            return True
    return False


def spine_untouched(lab: np.ndarray, given: np.ndarray) -> Tuple[bool, List[str]]:
    """A RIB reviewer must not alter the SPINE (or hips/femurs). The radiologist's vertebra labels
    are the ground truth of this dataset -- a rib annotator who renumbers them (e.g. inserting an L6
    and shifting every level) silently overwrites expert GT, and it cascades: shift the vertebrae and
    the rib numbering shifts with them, manufacturing a pile of fake 'rib disagreements'.

    `given` is the exact label the annotator was handed. Any non-rib voxel that differs is a REJECT.
    Compares ids 1..33 (vertebrae, sacrum, S1, hips, femurs); ribs (34-57) are theirs to edit."""
    if given is None or given.shape != lab.shape:
        return True, ["(starting label unavailable -> spine-untouched check skipped)"]
    names = _id2name()
    nonrib_before = (given >= 1) & (given <= 33)
    nonrib_after = (lab >= 1) & (lab <= 33)
    n = int(((given != lab) & (nonrib_before | nonrib_after)).sum())
    if n <= SPINE_EDIT_TOL_VOX:
        return True, ["OK: the spine/pelvis labels are untouched"]

    # A spine edit is permitted in EXACTLY ONE situation: the radiologist deliberately left a body
    # split across two levels (half-L3/half-L4) to mark an ambiguous/transitional level. Resolving
    # THAT is legitimate -- and it may legitimately change the level set too. Anything else is the
    # annotator overwriting expert ground truth.
    if _vertebra_label_mixing(given):
        return True, [f"OK: spine edited ({n:,} voxels) — the GIVEN spine had one bone carrying TWO "
                      f"vertebra labels (an ambiguous half-L3/half-L4 body); resolving that is the "
                      f"one permitted spine edit"]

    before = {int(v) for v in np.unique(given[nonrib_before])}
    after = {int(v) for v in np.unique(lab[nonrib_after])}
    add, rem = after - before, before - after
    msgs = [f"X the SPINE was ALTERED ({n:,} voxels) but the spine you were GIVEN was unambiguous — "
            f"the radiologist's vertebra labels are the ground truth; edit ONLY the ribs (34-57)"]
    if add:
        msgs.append(f"X   ADDED: {sorted(names.get(v, v) for v in add)}  -> you RE-NUMBERED the spine")
    if rem:
        msgs.append(f"X   DELETED: {sorted(names.get(v, v) for v in rem)}")
    msgs.append("X   the only permitted spine edit is resolving a body the radiologist split across "
                "TWO levels (half-L3/half-L4). If you think the enumeration is wrong (e.g. an L6 / "
                "transitional level), flag it for re-read — do not relabel it yourself.")
    return False, msgs


def check_label(check: str, lab: np.ndarray, affine,
                gating_only: bool = False,
                given: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
    """Run the requested check(s) and return (ok, messages) WITHOUT printing.
    The server-side review gate uses this; the CLI uses report() (which prints).
    `gating_only=True` skips the advisory checks (incl. the slow rib->spine EDT) when only the
    pass/fail verdict matters (e.g. auto-adjudication's auto-finalize decision)."""
    gating, advisory = [], []
    if check == "spine_extend":
        # SPINE-EXTENSION task: GATE on the additions (contiguous numbering, ascending order, each
        # added vertebra a clean connected blob). Pre-existing splits in the radiologist's vertebrae
        # are exempted -- they can't be fixed here and must not block. `given` marks the additions.
        return spine_extend_qc(lab, affine, given)
    if check in ("spine", "both"):
        gating.append(spine_sanity(lab, affine))
    if check in ("ribs", "both"):
        # GATES are only the FOV-SAFE class-purity checks: a rib/bone truncated by the FOV can
        # never trip these (truncation shrinks a bone or splits ONE rib number -> it can't make one
        # bone carry TWO labels), so they never create an impossible block.
        gating.append(rib_label_mixing(lab, affine))    # GATE: one connected rib bone -> one label
        gating.append(rib_spine_gap(lab, affine))        # GATE: a FULLY-IN-VIEW rib detached from its
        #   vertebra (>15 mm, fixable) blocks; a rib clipped by the FOV is exempted inside the check.
        if not gating_only:
            # rib_numbering (split / missing number) is FOV-AMBIGUOUS: an FOV-clipped rib exits and
            # re-enters the scan as 2 pieces, or is entirely out of view -> UNFIXABLE. Advisory only
            # so it never blocks; still printed so a genuinely-fixable split gets connected.
            advisory.append(rib_numbering(lab, affine))
            advisory.append(rib_anchor(lab, affine))     # ADVISORY only: rib N incident on T-N is
            #   unreliable (rib heads often unsegmented; thoracic vertebra labels can be off), so it
            #   informs but never blocks the submit.
    if check in ("spine", "both"):
        gating.append(structure_integrity(lab, affine))  # GATE: spine reviewer edits the spine
    elif check == "ribs":
        # structure_integrity inspects ONLY spine/pelvis bones (ids 1-33) -- and on a rib submission
        # the server FORCE-RESTORES every one of them to v4 (service._normalize_spine). So it can only
        # ever fire on a PRE-EXISTING v4 pseudo defect (a vertebra the pipeline split into pieces),
        # which the rib annotator cannot fix and whose edit is wiped anyway. Gating on it is an
        # IMPOSSIBLE block. Advisory only -- record the v4 defect, never block the student's ribs.
        if not gating_only:
            advisory.append(structure_integrity(lab, affine))
    if check == "ribs" and not gating_only:
        # ADVISORY, not a gate: rib N<->T-N is DETERMINED by the vertebra only when the spine is
        # NORMAL. Transitional anatomy -- a rib on L1 (13th-rib / LSTV), the dataset's TARGET
        # phenotype -- legitimately shifts every rib one level, so gating here BLOCKS exactly the
        # cases we exist to keep (and rib heads are often unsegmented, adding false shifts). It
        # informs + flags LSTV; genuine 'both annotators shifted the same way' errors surface in
        # adjudication and the stump-rib detector for manual read, not by blocking the student.
        advisory.append(rib_vertebra_match(lab, affine))
    if check == "ribs" and given is not None:
        # The server now FORCE-RESTORES the spine on every submission (service._normalize_spine), so a
        # spine edit can no longer reach the store. Kept as a safety net -- if it ever fires, the
        # normalization did not run and something upstream is broken.
        gating.append(spine_untouched(lab, given))
    ok = all(o for o, _ in gating)                      # advisory checks NEVER affect pass/fail
    # Advisory checks legitimately emit "X ..." lines (a split rib number that's usually an FOV clip;
    # on a rib review, a split v4 vertebra the student can't fix). Those are NOT blockers, so downgrade
    # their "X" to "note:" in the merged output -- the server reject reason (which keeps only lines
    # starting with "X") then reports ONLY genuine gating failures, not advisory noise the student
    # would chase in vain.
    def _adv(m):
        return ("note:" + m[1:]) if m.startswith("X") else m
    msgs = [m for _, ms in gating for m in ms] + [_adv(m) for _, ms in advisory for m in ms]
    return ok, msgs


def report(check: str, lab: np.ndarray, affine) -> bool:
    """Run the requested check(s) and print a PASS/FAIL block. Returns overall ok."""
    blocks = []   # (name, ok, msgs, gating?)
    if check == "spine_extend":
        # spine-EXTENSION task: validate the additions (contiguous, ascending, connected). Client has
        # no `given`, so per-added-vertebra checks are the server's job; contiguity + order run here.
        # Pre-existing vertebra splits are NOT gated (that's why we don't call spine_sanity here).
        blocks.append(("SPINE EXTENSION", *spine_extend_qc(lab, affine), True))
        allok = all(ok for _, ok, _, gate in blocks if gate)
        for name, ok, msgs, gate in blocks:
            print(f"  [{name}] {'PASS' if ok else 'FAIL'}")
            for m in msgs:
                print(f"    {m}")
        print("  ===> ALL CHECKS PASS -> Save once more if you edited, then quit to submit."
              if allok else "  ===> fix the 'X' items above, then Save again to re-check.")
        return allok
    if check in ("spine", "both"):
        blocks.append(("SPINE", *spine_sanity(lab, affine), True))
    if check in ("ribs", "both"):
        blocks.append(("RIB LABEL MIXING", *rib_label_mixing(lab, affine), True))  # GATE (FOV-safe)
        blocks.append(("RIB NUMBERING (advisory)", *rib_numbering(lab, affine), False))  # FOV-clip
        blocks.append(("RIB-SPINE GAP", *rib_spine_gap(lab, affine), True))  # GATE: in-view detached rib
        blocks.append(("RIB ANCHOR (advisory)", *rib_anchor(lab, affine), False))  # informs only
    if check in ("spine", "both"):
        blocks.append(("STRUCTURE INTEGRITY", *structure_integrity(lab, affine), True))  # gates
    elif check == "ribs":
        # advisory on rib reviews: the spine is force-restored to v4, so a split vertebra is a v4
        # defect the annotator cannot fix -> never block them (matches the server gate).
        blocks.append(("STRUCTURE INTEGRITY (advisory)", *structure_integrity(lab, affine), False))
    allok = all(ok for _, ok, _, gate in blocks if gate)     # advisory anchor never blocks
    for name, ok, msgs, gate in blocks:
        tag = "PASS" if ok else ("FAIL" if gate else "note")
        print(f"  [{name}] {tag}")
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
