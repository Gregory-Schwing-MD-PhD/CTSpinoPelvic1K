"""
rib_instancer.py — correct-by-construction rib numbering.

Replaces the three independent heuristics (overlap vote / TS-offset / count) with ONE
ordered assignment per side, so duplicates and mis-ordering are impossible by
construction and gaps only appear where a rib is genuinely absent:

  1. components   : connected components of the union rib mask (>= min_voxels).
  2. side         : each component is L/R by its world-X centroid vs the spine.
  3. group->ribs  : components are grouped into PHYSICAL ribs by their costovertebral
                    HEAD height (the medial end, near the spine), with a gap threshold of
                    ~0.6*pitch — so split pieces of one rib merge, adjacent ribs don't.
  4. number       : ribs are ordered cranio-caudally and numbered PITCH-AWARE
                    (number step = round(Δz / pitch), >=1) — consecutive where ribs are
                    consecutive, a real gap only where the spacing shows a missing rib.
  5. anchor       : the absolute offset is pinned by the GT thoracic vertebrae a rib
                    articulates (rib touching dilated T-k => rib k). Disagreeing anchors
                    are reported as a conflict (never silently guessed).

`thoracic` is the v3 thoracic already remapped to rib numbers (VerSe 8..19 -> 1..12), the
same array build_v4_ribs feeds relabel_ribs. Returns (numbered int16 vol in 34..57, report).
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

GROUP_FRAC = 0.6           # head-z within this * pitch => same physical rib
HEAD_QUANTILE = 0.25       # most-medial fraction of a component = its head
DILATION = 4               # voxels, costovertebral articulation tolerance for anchoring


def _rib_id(side: str, n: int) -> int:
    return (LS.RIB_LEFT_OFFSET if side == "left" else LS.RIB_RIGHT_OFFSET) + int(n)


def _centroid_world(mask: np.ndarray, affine) -> np.ndarray:
    return apply_affine(affine, np.argwhere(mask).mean(axis=0))


def _pitch_mm(vcent: Dict[int, np.ndarray]) -> float:
    zs = sorted((float(c[2]), k) for k, c in vcent.items())
    if len(zs) >= 2 and zs[-1][1] != zs[0][1]:
        p = abs((zs[-1][0] - zs[0][0]) / (zs[-1][1] - zs[0][1]))
        if p > 1e-3:
            return p
    return 20.0


def instance_ribs(rib_binary: np.ndarray, thoracic: np.ndarray, affine, *,
                  min_voxels: int = 150) -> Tuple[np.ndarray, dict]:
    out = np.zeros(rib_binary.shape, np.int16)
    report: dict = {"sides": {}, "conflicts": [], "out_of_range": []}

    vlabs = [int(v) for v in np.unique(thoracic) if v > 0]
    if not vlabs:
        report["error"] = "no thoracic anchors"
        return out, report
    vcent = {v: _centroid_world(thoracic == v, affine) for v in vlabs}
    spine_x = float(np.mean([vcent[v][0] for v in vlabs]))
    right_is_plus = True                     # RAS+: +X -> patient Right
    pitch = _pitch_mm(vcent)

    # connected components of the whole union mask, size-filtered
    st = ndimage.generate_binary_structure(3, 3)
    labeled, n = ndimage.label(rib_binary > 0, structure=st)
    if n == 0:
        return out, report
    sizes = np.bincount(labeled.ravel())
    comps = [c for c in range(1, n + 1) if sizes[c] >= min_voxels]

    # per-component geometry: side + costovertebral head z (medial end), in world mm
    geom: Dict[int, dict] = {}
    for c in comps:
        ijk = np.argwhere(labeled == c)
        w = apply_affine(affine, ijk)
        x, z = w[:, 0], w[:, 2]
        medial = np.abs(x - spine_x)
        thr = np.quantile(medial, HEAD_QUANTILE)
        head_z = float(z[medial <= thr].mean())
        side = "right" if (float(x.mean()) - spine_x > 0) == right_is_plus else "left"
        geom[c] = {"side": side, "head_z": head_z}

    # dilated thoracic (per-vertebra) for the anchor test
    vdil = _dilate_vertebrae(thoracic, vlabs, DILATION)

    for side in ("left", "right"):
        sc = [c for c in comps if geom[c]["side"] == side]
        if not sc:
            continue
        # 3. group components into physical ribs by head-z (cranial first)
        sc.sort(key=lambda c: -geom[c]["head_z"])
        groups: List[List[int]] = []
        for c in sc:
            hz = geom[c]["head_z"]
            if groups and abs(hz - np.mean([geom[g]["head_z"] for g in groups[-1]])) <= GROUP_FRAC * pitch:
                groups[-1].append(c)
            else:
                groups.append([c])
        gz = [float(np.mean([geom[g]["head_z"] for g in grp])) for grp in groups]

        # 4. pitch-aware consecutive relative numbering (top group = 0)
        rel = [0]
        for i in range(1, len(groups)):
            rel.append(rel[-1] + max(1, int(round((gz[i - 1] - gz[i]) / pitch))))

        # 5. anchor: which groups articulate a thoracic vertebra -> its true number
        anchors: List[Tuple[int, int]] = []   # (group_index, true_number)
        for gi, grp in enumerate(groups):
            gmask = np.isin(labeled, grp)
            best_k, best_ov = None, 0
            for k in vlabs:
                sl, dm = vdil[k]
                ov = int(np.count_nonzero(gmask[sl] & dm))
                if ov > best_ov:
                    best_ov, best_k = ov, k
            if best_k is not None:
                anchors.append((gi, best_k))

        offsets = [k - rel[gi] for gi, k in anchors]
        if offsets:
            offset = int(round(np.median(offsets)))
            if len(set(offsets)) > 1:
                report["conflicts"].append({"side": side, "offsets": sorted(set(offsets))})
        else:
            # no anchored rib this side: fall back to nearest vertebra for the top group
            top_k = min(vlabs, key=lambda v: abs(vcent[v][2] - gz[0]))
            offset = top_k - rel[0]

        # 6. paint each physical rib with its number
        numbers = []
        for gi, grp in enumerate(groups):
            num = rel[gi] + offset
            if not (1 <= num <= 12):
                report["out_of_range"].append({"side": side, "number": int(num),
                                               "vox": int(sum(int(sizes[g]) for g in grp))})
                num = int(min(12, max(1, num)))
            numbers.append(num)
            rid = _rib_id(side, num)
            for g in grp:
                out[labeled == g] = rid
        report["sides"][side] = {"n_groups": len(groups), "n_components": len(sc),
                                 "n_anchored": len(anchors), "numbers": numbers}
    return out, report


def _dilate_vertebrae(vert: np.ndarray, vlabs, radius: int):
    """{k: (slices, dilated_submask)} — per-vertebra dilation in a padded bbox (cheap)."""
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (zz * zz + yy * yy + xx * xx) <= r * r
    objs = ndimage.find_objects(vert.astype(np.int32))
    pad = r + 2
    out = {}
    for k in vlabs:
        loc = objs[k - 1]
        if loc is None:
            continue
        sl = tuple(slice(max(0, s.start - pad), min(d, s.stop + pad))
                   for s, d in zip(loc, vert.shape))
        out[k] = (sl, ndimage.binary_dilation(vert[sl] == k, structure=ball))
    return out
