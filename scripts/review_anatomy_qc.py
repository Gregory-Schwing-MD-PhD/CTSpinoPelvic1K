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
ANCHOR_MM = 10.0        # rib N is "incident on" T-N if within this of the vertebra body.
                        # Loose on purpose: thoracic bodies are ~20 mm apart, so 10 mm cleanly
                        # separates the right vertebra (touching, ~0-5 mm) from an off-by-one
                        # neighbour (~15-25 mm), without false-flagging a small seg gap.
MIX_LABEL_MIN_VOX = 250  # in the "one bone -> one label" check, a 2nd label on the same eroded
                         # rib component counts as a real mix only above this (ignore sliver
                         # boundary voxels between genuinely-adjacent ribs).
GAP_DETACH_MM = 15.0     # a rib FULLY IN VIEW (bbox touches no scan face) whose head is farther than
                         # this from the nearest vertebra is DETACHED and fixable -> BLOCK. Calibrated
                         # on real in-view ribs: gap-to-spine is sharply bimodal, ~85% at <=3 mm
                         # (head touching, incl. the ~3-4 mm costovertebral joint space) and a tail at
                         # >30 mm (detached), with a wide empty valley between -> 15 mm sits safely in
                         # the valley, above normal joint space + minor head under-segmentation.
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
    names = _id2name(); ok, msgs = True, []
    for rid, g in zip(rib_ids, mins):
        if g is None or not np.isfinite(g):                      # rib absent (or too thin at 2x)
            continue
        if g <= GAP_DETACH_MM:                                   # head reaches the spine -> fine
            continue
        o = objs[rid - 1] if rid - 1 < len(objs) else None
        exits_fov = o is not None and any(o[i].start == 0 or o[i].stop == shape[i] for i in range(3))
        if exits_fov:
            msgs.append(f"note {names.get(rid, rid)}: {g:.0f} mm from the spine but the rib is clipped "
                        f"by the scan edge (exits FOV) -> advisory, not blocking")
        else:
            ok = False
            msgs.append(f"X {names.get(rid, rid)}: DETACHED — {g:.0f} mm from the spine and fully in "
                        f"view -> connect the rib head to its vertebra")
    if ok:
        msgs.append("OK: every in-view rib reaches its vertebra")
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


def check_label(check: str, lab: np.ndarray, affine,
                gating_only: bool = False) -> Tuple[bool, List[str]]:
    """Run the requested check(s) and return (ok, messages) WITHOUT printing.
    The server-side review gate uses this; the CLI uses report() (which prints).
    `gating_only=True` skips the advisory checks (incl. the slow rib->spine EDT) when only the
    pass/fail verdict matters (e.g. auto-adjudication's auto-finalize decision)."""
    gating, advisory = [], []
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
    if check in ("spine", "ribs", "both"):
        gating.append(structure_integrity(lab, affine))  # GATE: each spine/pelvis bone one class
    ok = all(o for o, _ in gating)                      # anchor does NOT affect pass/fail
    msgs = [m for _, ms in gating for m in ms] + [m for _, ms in advisory for m in ms]
    return ok, msgs


def report(check: str, lab: np.ndarray, affine) -> bool:
    """Run the requested check(s) and print a PASS/FAIL block. Returns overall ok."""
    blocks = []   # (name, ok, msgs, gating?)
    if check in ("spine", "both"):
        blocks.append(("SPINE", *spine_sanity(lab, affine), True))
    if check in ("ribs", "both"):
        blocks.append(("RIB LABEL MIXING", *rib_label_mixing(lab, affine), True))  # GATE (FOV-safe)
        blocks.append(("RIB NUMBERING (advisory)", *rib_numbering(lab, affine), False))  # FOV-clip
        blocks.append(("RIB-SPINE GAP", *rib_spine_gap(lab, affine), True))  # GATE: in-view detached rib
        blocks.append(("RIB ANCHOR (advisory)", *rib_anchor(lab, affine), False))  # informs only
    if check in ("spine", "ribs", "both"):
        blocks.append(("STRUCTURE INTEGRITY", *structure_integrity(lab, affine), True))  # gates
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
