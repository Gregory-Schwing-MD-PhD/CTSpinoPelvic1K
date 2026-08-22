"""scripts/extract_transition_morphometrics.py — measure the transitional anatomy, per case.

WHAT THIS IS FOR. A transitional vertebra is a COUNTING problem wearing a morphology
disguise, and counting up from the sacrum begs the question when the sacrum is the thing
that changed. So every measure here is chosen to be independent of the vertebra labels
wherever it can be -- properties of the bone rather than of the numbering:

  SACRAL FORAMINA. Four pairs is a normal sacrum, five means it absorbed L5. This is the
  one measure that settles sacralization without trusting any vertebra label at all.
  Counted by sliding a THIN coronal slab: the foramina are oblique through-canals,
  so 3D hole-filling never finds them and any thick slab fills them in from the bone in
  front of one end and behind the other.

  NON-RIB-BEARING VERTEBRAE. Five is normal. Four means either a lumbar rib on L1 or a
  sacralized L5 -- the classic ambiguity -- which is exactly why it is plotted AGAINST the
  foramina count rather than alone.

  CASTELLVI PROXIES. The lowest lumbar's lateral span, and how close it comes to the
  sacrum/ilium. Castellvi grades on whether an enlarged transverse process approaches (II)
  or fuses with (III/IV) the sacral ala, so span and gap together carry most of the grade.
  Per side, because the asymmetry IS the phenotype.

  DISC RATIO. A sacralized L5 has a rudimentary L5-S1 disc. Measured against the disc
  above it, so the ratio is dimensionless and comparable across patients.

  THORACOLUMBAR END. Rib 12 length against rib 11 -- a hypoplastic 12th rib is the
  thoracolumbar counterpart, and the two ends of the spine co-occur.

Lengths use the first principal axis of the point cloud, not a bounding-box diagonal: a
rib is a curved oblique object and its box says more about its obliquity than its length.

    python scripts/extract_transition_morphometrics.py --labels data/v5_final \
        --manifest data/hf_export_v4/manifest.json --workers 24 --out morphometrics
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))
import label_scheme as LS                                          # noqa: E402

THORACIC_BASE = 7
SACRUM, S1, HIP_L, HIP_R = 26, 29, 30, 31
LUMBAR_IDS = list(range(20, 26))
THORACIC_IDS = list(range(8, 20))
RIB_L, RIB_R = LS.RIB_LEFT_OFFSET, LS.RIB_RIGHT_OFFSET
LUM_RIB_L, LUM_RIB_R = LS.LUMBAR_RIB_LEFT, LS.LUMBAR_RIB_RIGHT
ANCHOR_MM = 15.0
MIN_VERT_VOX = 6000


def _pts(mask, cap=400):
    p = np.argwhere(mask)
    return p[:: max(1, len(p) // cap)] if len(p) else p


def _mindist(a, b, spacing):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = (a[:, None, :] - b[None, :, :]) * spacing
    return float(np.sqrt((d ** 2).sum(-1)).min())


def _pca_length(mask, spacing):
    """Extent along the structure's own long axis, in mm."""
    p = _pts(mask, cap=600)
    if len(p) < 3:
        return 0.0
    q = (p * spacing).astype(float)
    q = q - q.mean(0)
    v = np.linalg.svd(q, full_matrices=False)[2][0]
    t = q @ v
    return float(t.max() - t.min())


def count_foramina(lab, spacing):
    """Foramina per side, by sliding a THIN slab through the sacrum front-to-back.

    A half-depth projection does not work, and the smoke test said so: counts came back
    0/0 and 4/3 where a normal sacrum should give 4/4. The sacral foramina are oblique
    canals, so any thick slab has bone in front of one end and behind the other, and the
    projection fills the hole in. Only a slab thinner than the obliquity keeps the canal
    open in projection.

    So: slide a ~12mm slab, count enclosed holes of plausible foramen area in each, and
    take the MAX over slabs per side -- the slab that happens to align with the canals is
    the one that sees them, and no other slab can invent extras of the right size.

    Returns (left, right, median area, best slab offset in mm) -- the offset is kept so a
    suspicious count can be re-rendered at the depth that produced it.
    """
    sac = np.isin(lab, [SACRUM, S1])
    if sac.sum() < 5000:
        return None, None, 0.0, None
    ys = np.nonzero(sac.any(axis=(0, 2)))[0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    px_mm2 = float(spacing[0] * spacing[2])
    mid = float(np.average(np.arange(lab.shape[0]), weights=sac.sum(axis=(1, 2))))

    slab = max(2, int(round(8.0 / max(spacing[1], 1e-6))))
    step = max(1, slab // 3)
    # a foramen at the lateral margin opens outward, so fill_holes never sees it as
    # enclosed. Closing the projection first turns that notch into a hole; the hole is
    # still subtracted from the ORIGINAL mask, so closing cannot invent bone.
    se = np.ones((max(3, int(round(4.0 / spacing[0]))) | 1,
                  max(3, int(round(4.0 / spacing[2]))) | 1), bool)
    best = (0, 0, 0.0, None)
    for yy in range(y0, max(y0 + 1, y1 - slab), step):
        proj = sac[:, yy:yy + slab].max(axis=1)
        if proj.sum() < 200:
            continue
        closed = ndimage.binary_closing(proj, structure=se)
        holes = ndimage.binary_fill_holes(closed) & ~proj
        lbl, n = ndimage.label(holes)
        if n == 0:
            continue
        left = right = 0
        areas = []
        for i in range(1, n + 1):
            m = lbl == i
            area = float(m.sum()) * px_mm2
            if not (8.0 <= area <= 700.0):
                continue
            areas.append(area)
            if float(np.argwhere(m)[:, 0].mean()) > mid:
                right += 1
            else:
                left += 1
        if (left + right) > (best[0] + best[1]):
            best = (left, right, float(np.median(areas)) if areas else 0.0,
                    round((yy - y0) * spacing[1], 1))
    return best


def _nearest_vert(mask, verts, sp):
    rp = _pts(mask)
    best, bid = float("inf"), None
    for vid, vp in verts.items():
        d = _mindist(rp, vp, sp)
        if d < best:
            best, bid = d, vid
    return bid, best


def one(path: str) -> dict:
    stem = Path(path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        img = nib.as_closest_canonical(nib.load(path))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": f"{type(exc).__name__}: {exc}"}

    # ---- sacrum ------------------------------------------------------------
    fl, fr, farea, fdepth = count_foramina(lab, sp)
    r["foramina_left"], r["foramina_right"] = fl, fr
    r["foramina_total"] = (fl + fr) if fl is not None else None
    r["foramina_max_side"] = (max(fl, fr) if fl is not None else None)
    r["foramen_area_mm2"] = round(farea, 1)
    r["foramen_best_slab_mm"] = fdepth
    sac = np.isin(lab, [SACRUM, S1])
    if sac.any():
        zs = np.nonzero(sac.any(axis=(0, 1)))[0]
        xs = np.nonzero(sac.any(axis=(1, 2)))[0]
        r["sacrum_height_mm"] = round(len(zs) * sp[2], 1)
        r["sacrum_width_mm"] = round(len(xs) * sp[0], 1)
        r["sacrum_aspect"] = round(len(zs) * sp[2] / max(1e-6, len(xs) * sp[0]), 3)

    # ---- which vertebrae exist ---------------------------------------------
    verts = {}
    for vid in THORACIC_IDS + LUMBAR_IDS:
        m = lab == vid
        if m.sum() >= MIN_VERT_VOX:
            verts[vid] = _pts(m)
    r["n_vertebrae_labelled"] = len(verts)
    r["has_l6_label"] = int(25 in verts)

    # ---- ribs --------------------------------------------------------------
    ribs = {}
    for base, side in ((RIB_L, "left"), (RIB_R, "right")):
        for n in range(1, 13):
            m = lab == base + n
            if m.any():
                ribs[(side, n)] = m
    r["n_ribs_left"] = sum(1 for s, _ in ribs if s == "left")
    r["n_ribs_right"] = sum(1 for s, _ in ribs if s == "right")
    r["has_lumbar_rib_left"] = int((lab == LUM_RIB_L).any())
    r["has_lumbar_rib_right"] = int((lab == LUM_RIB_R).any())
    r["has_lumbar_rib"] = int(r["has_lumbar_rib_left"] or r["has_lumbar_rib_right"])
    ll = 0.0
    for cid in (LUM_RIB_L, LUM_RIB_R):
        m = lab == cid
        if m.any():
            ll = max(ll, _pca_length(m, sp))
    r["lumbar_rib_len_mm"] = round(ll, 1) if ll else None

    bearing = set()
    for m in list(ribs.values()) + [lab == LUM_RIB_L, lab == LUM_RIB_R]:
        if not m.any():
            continue
        bid, gap = _nearest_vert(m, verts, sp)
        if bid is not None and gap <= ANCHOR_MM:
            bearing.add(bid)
    r["n_rib_bearing"] = len(bearing)
    if bearing:
        low = max(bearing)
        r["lowest_rib_bearing_id"] = low
        r["lowest_rib_bearing"] = (f"T{low - THORACIC_BASE}" if low < 20
                                   else f"L{low - 19}")
        r["n_non_rib_bearing"] = sum(1 for v in verts if v > low)

    # ---- lumbosacral transition -------------------------------------------
    lums = sorted(v for v in verts if v in LUMBAR_IDS)
    if lums and sac.any():
        low = lums[-1]
        m = lab == low
        w = m.sum(axis=(1, 2))
        vmid = float(np.average(np.arange(lab.shape[0]), weights=w))
        mx = np.nonzero(m.any(axis=(1, 2)))[0]
        r["lowest_lumbar"] = f"L{low - 19}"
        r["ll_span_left_mm"] = round((vmid - float(mx.min())) * sp[0], 1)
        r["ll_span_right_mm"] = round((float(mx.max()) - vmid) * sp[0], 1)
        r["ll_span_asym_mm"] = round(abs(r["ll_span_left_mm"] - r["ll_span_right_mm"]), 1)
        r["ll_span_total_mm"] = round(r["ll_span_left_mm"] + r["ll_span_right_mm"], 1)

        # THE SACRUM AND THE ILIUM ARE DIFFERENT TARGETS AND MUST NOT BE POOLED.
        # This measured the distance to sacrum, S1, AND both hip bones together, and the
        # iliac crest sits beside the L5 transverse process in everybody -- so the
        # "gap to the sacrum" was usually the gap to the ilium, and read a median of
        # 4.7 mm in NORMAL cases. A normal transverse process is 15-25 mm from the ala;
        # at 4.7 the whole cohort would be Castellvi II.
        #
        # Castellvi grades on approach to, articulation with, and fusion to the SACRAL
        # ALA. Contact with the ilium is a different finding with a different meaning, so
        # both are measured and neither is allowed to stand in for the other.
        sac_only = np.isin(lab, [SACRUM, S1])
        ilium_only = np.isin(lab, [HIP_L, HIP_R])
        tp_sac = _pts(sac_only, cap=700)
        tp_ili = _pts(ilium_only, cap=700)
        ax = np.arange(lab.shape[0])
        latL = int(vmid - 0.45 * (vmid - float(mx.min())))
        latR = int(vmid + 0.45 * (float(mx.max()) - vmid))
        for nm, sel in (("left", ax < latL), ("right", ax > latR)):
            mm = m & sel[:, None, None]
            if not mm.any():
                r[f"tp_gap_{nm}_mm"] = None
                continue
            pts_tp = _pts(mm)
            gs = _mindist(pts_tp, tp_sac, sp) if len(tp_sac) else float("inf")
            gi = _mindist(pts_tp, tp_ili, sp) if len(tp_ili) else float("inf")
            # tp_gap_* keeps its name and now means what it always claimed to: the gap
            # to the sacrum, which is the quantity Castellvi is defined on
            r[f"tp_gap_{nm}_mm"] = round(gs, 1) if np.isfinite(gs) else None
            r[f"tp_gap_ilium_{nm}_mm"] = round(gi, 1) if np.isfinite(gi) else None
        gl, gr = r.get("tp_gap_left_mm"), r.get("tp_gap_right_mm")
        if gl is not None and gr is not None:
            r["tp_gap_min_mm"] = min(gl, gr)
            r["tp_gap_asym_mm"] = round(abs(gl - gr), 1)

        if (low - 1) in verts:
            # the same pooled-target contamination reached the disc measurement: the
            # lowest lumbar vertebra is often nearer the ilium than the sacrum, so this
            # could silently return a vertebra-to-ilium distance and call it a disc
            d_low = _mindist(_pts(m), tp_sac, sp)
            d_up = _mindist(_pts(m), _pts(lab == low - 1), sp)
            r["disc_low_mm"] = round(d_low, 1)
            r["disc_above_mm"] = round(d_up, 1)
            if d_up > 0:
                r["disc_ratio"] = round(d_low / d_up, 3)

    # ---- iliac crest height ------------------------------------------------
    hip = np.isin(lab, [HIP_L, HIP_R])
    if hip.any() and verts:
        crest_z = int(np.nonzero(hip.any(axis=(0, 1)))[0].max())
        at = None
        for vid in sorted(verts):
            zz = np.nonzero((lab == vid).any(axis=(0, 1)))[0]
            if len(zz) and zz.min() <= crest_z <= zz.max():
                at = vid
                break
        if at is not None:
            r["iliac_crest_at"] = (f"T{at - THORACIC_BASE}" if at < 20
                                   else f"L{at - 19}")
            r["iliac_crest_at_id"] = at

    # ---- thoracolumbar end -------------------------------------------------
    for side, base in (("left", RIB_L), ("right", RIB_R)):
        for n in (11, 12):
            m = lab == base + n
            r[f"rib{n}_len_{side}_mm"] = (round(_pca_length(m, sp), 1)
                                          if m.any() else None)
        a, b = r[f"rib12_len_{side}_mm"], r[f"rib11_len_{side}_mm"]
        r[f"rib12_11_ratio_{side}"] = round(a / b, 3) if (a and b) else None
    rr = [r[f"rib12_11_ratio_{s}"] for s in ("left", "right")
          if r.get(f"rib12_11_ratio_{s}") is not None]
    r["rib12_11_ratio_min"] = min(rr) if rr else None
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.labels).glob("*.nii.gz"))
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} labels from {a.labels}\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=1), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    # join the LSTV labels, so known cases are HIGHLIGHTED rather than re-derived
    lut = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            stem = str(rec.get("label_file", "")).split("/")[-1]
            stem = stem.replace("_label.nii.gz", "")
            if stem:
                lut[stem] = {k: rec.get(k) for k in
                             ("lstv_label", "lstv_class", "lstv_pelvic",
                              "lstv_vertebral", "castellvi_type", "lstv_agreement",
                              "has_l6", "sex", "age", "config")}
    for r in res:
        r.update(lut.get(r["case"], {}))

    ok = [r for r in res if "error" not in r]
    cols = sorted({k for r in ok for k in r}, key=lambda k: (k != "case", k))
    p = out / "transition_morphometrics.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    bad = [r for r in res if "error" in r]
    print(f"\n  wrote {p}  ({len(ok)} cases, {len(bad)} failed)")
    if bad:
        print(f"  failures: {[b['case'] for b in bad][:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
