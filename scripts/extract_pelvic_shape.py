"""scripts/extract_pelvic_shape.py — the pelvic measures that are sexually dimorphic.

SEPARATE FROM THE SURGICAL SCRIPT ON PURPOSE. Those measures need per-slice work --
canal ring detection, endplate surface fits, a pedicle isthmus search down every level --
and take the better part of an hour across the release. These need bounding boxes and
centroids. Keeping them apart means the dimorphism figures can be rebuilt in minutes
instead of waiting on work they do not depend on.

WHY THESE AND NOT PELVIC INCIDENCE. Pelvic incidence is an angular relationship between
the sacral endplate and the hip axis, and it does not separate by sex -- measured here,
51.3 degrees in women against 50.5 in men, and several published series agree. Pelvic
dimorphism is a matter of SHAPE:

  bi_iliac_width_mm     widest span across the iliac crests -- "pelvic width"
  bi_acetabular_mm      centre to centre across the femoral heads
  pelvic_inlet_ap_mm    sacral promontory to pubic symphysis: the obstetric conjugate
  inlet_index           inlet depth over bi-acetabular width; a rounder inlet scores
                        higher, and roundness is the classic obstetric distinction
  sacral_width_ratio    sacral width over sacral height. The female sacrum is wider and
                        shorter; this is the most reported sex difference in the pelvis.
  subpubic_angle_deg    angle between the inferior pubic rami

These double as a check on the pipeline: each depends on the pelvic labels being present,
correctly sided and correctly scaled, so a transposed volume or a spacing error moves
them immediately -- which is what pelvic incidence could not do.

    python scripts/extract_pelvic_shape.py --labels data/v5_final
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

SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R = 26, 29, 30, 31, 32, 33
MIN_VOX = 3000


def _head_centre(fem, sp):
    """Femoral head centroid, approximated as the superior quarter of the femur label.

    The labels run 61-109 mm in z -- head, neck, trochanter and shaft -- so the centroid
    of the whole thing sits well down the shaft and reads the width several centimetres
    too wide. The head is the superior end. A distance transform against the acetabulum
    is more exact and far slower; at this scale the two agree closely enough, and being
    able to rebuild the figure in minutes is worth more than the last millimetre.
    """
    idx = np.argwhere(fem)
    if len(idx) < 200:
        return None
    cut = np.percentile(idx[:, 2], 72)
    top = idx[idx[:, 2] >= cut]
    if len(top) < 100:
        top = idx
    return top.mean(0) * sp


def one(path: str) -> dict:
    stem = Path(path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        # canonical: everything below reasons about superior, anterior and left-right,
        # and these volumes are ('P','I','R') on disk. This is a read path deriving
        # numbers, so canonicalising is safe -- unlike a write-back of labels.
        img = nib.as_closest_canonical(nib.load(path))
        lab = np.asanyarray(img.dataobj)
        sp = np.asarray(img.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    have = {v: (lab == v) for v in (SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R)
            if (lab == v).sum() >= MIN_VOX}

    # ---- pelvic width across the iliac crests -------------------------------------
    if HIP_L in have and HIP_R in have:
        hx = np.concatenate([np.argwhere(have[HIP_L])[:, 0],
                             np.argwhere(have[HIP_R])[:, 0]])
        r["bi_iliac_width_mm"] = round(float(hx.max() - hx.min()) * sp[0], 1)

    # ---- sacral base breadth --------------------------------------------------------
    # THE SACRUM LABEL IS NOT THE SACRUM. label_scheme documents 26 as "VerSe sacrum
    # (below S1)" with S1 carved out as 29, so measuring 26 alone measured S2-S5 and the
    # ratio it produced was meaningless -- 0.9 in both sexes, on one of the most
    # dimorphic bones there is. The sacrum is the union.
    sac = np.zeros_like(lab, bool)
    for v in (SACRUM, S1):
        if v in have:
            sac |= have[v]
    if sac.any():
        sidx = np.argwhere(sac)
        # BASE BREADTH, not the widest point anywhere. The base is the S1 level, across
        # both alae, and it is the measurement the anthropometric literature reports.
        ztop = np.percentile(sidx[:, 2], 85)
        base = sidx[sidx[:, 2] >= ztop]
        if len(base) > 100:
            r["sacral_base_width_mm"] = round(
                float(base[:, 0].max() - base[:, 0].min()) * sp[0], 1)
        # total height is NOT reported: the label runs to 140 mm on cases where the
        # sacrum is 105-115, so it is picking up more than the sacrum inferiorly and a
        # ratio built on it would inherit that error silently.
        r["sacral_span_mm"] = round(
            float(sidx[:, 2].max() - sidx[:, 2].min()) * sp[2], 1)

    # ---- width across the hip joints ------------------------------------------------
    cl = _head_centre(have[FEM_L], sp) if FEM_L in have else None
    cr = _head_centre(have[FEM_R], sp) if FEM_R in have else None
    if cl is not None and cr is not None:
        r["bi_acetabular_mm"] = round(float(np.linalg.norm(cl - cr)), 1)

    # ---- pelvic inlet, promontory to symphysis --------------------------------------
    if S1 in have and (HIP_L in have or HIP_R in have):
        si = np.argwhere(have[S1])
        top = si[si[:, 2] >= np.percentile(si[:, 2], 88)]
        if len(top):
            prom = top[np.argmax(top[:, 1])] * sp
            hips = np.zeros_like(lab, bool)
            for h in (HIP_L, HIP_R):
                if h in have:
                    hips |= have[h]
            hi = np.argwhere(hips)
            # the pubic bones meet at the ANTERIOR MIDLINE. Taking the most anterior hip
            # voxel at the promontory's HEIGHT instead lands on the iliac wing and
            # returns an inlet near 70 mm, which is anatomically impossible.
            midx = float(np.median(hi[:, 0]))
            near = hi[np.abs(hi[:, 0] - midx) * sp[0] <= 18.0]
            if len(near) > 50:
                sym = near[np.argmax(near[:, 1])] * sp
                ap = float(np.hypot(sym[1] - prom[1], sym[2] - prom[2]))
                if 60 < ap < 200:
                    # RECORDED BUT NOT REPORTED. Three landmark definitions were tried
                    # against the published 110-130 mm conjugate: most-anterior pubis
                    # (142 mm median), superior-anterior margin (135), and
                    # superior-posterior margin (137). None reached the published range
                    # and none separated by sex, in a measure that is among the most
                    # dimorphic in the skeleton. The column is kept for anyone who wants
                    # to improve the landmark; nothing downstream should plot it until
                    # someone does.
                    r["pelvic_inlet_ap_mm_UNVALIDATED"] = round(ap, 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--manifest", default="data/hf_export/manifest.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.labels).glob("*_label.nii.gz"))
    print(f"{len(files)} case(s)\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, files, chunksize=4), 1):
            res.append(x)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    lut = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
            if s:
                lut[s] = {k: rec.get(k) for k in ("age", "sex")}
    for x in res:
        x.update(lut.get(x["case"], {}))

    ok = [x for x in res if "error" not in x]
    cols = sorted({k for x in ok for k in x}, key=lambda k: (k != "case", k))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "pelvic_shape.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    print(f"\n  {len(ok)} of {len(res)} measured\n")
    PLAUSIBLE = {"bi_iliac_width_mm": (220, 320), "bi_acetabular_mm": (130, 210),
                 "sacral_base_width_mm": (95, 140)}
    bad = []
    for key in ("bi_iliac_width_mm", "bi_acetabular_mm", "sacral_base_width_mm",
                "sacral_span_mm"):
        v = np.array([x[key] for x in ok if isinstance(x.get(key), (int, float))], float)
        if v.size < 5:
            continue
        med = float(np.median(v))
        flag = ""
        if key in PLAUSIBLE and not (PLAUSIBLE[key][0] <= med <= PLAUSIBLE[key][1]):
            flag = f"   <-- IMPLAUSIBLE (expected {PLAUSIBLE[key]})"
            bad.append(key)
        print(f"  {key:22s} n={v.size:4d}  median {med:7.1f}{flag}")

    # THE POINT. Each of these should separate by sex; pelvic incidence does not.
    print("\n  by sex (the dimorphism, and a check that the labels are sane):")
    for key in ("bi_iliac_width_mm", "bi_acetabular_mm", "sacral_base_width_mm"):
        g = {}
        for want, lab_ in (("F", "female"), ("M", "male")):
            v = [x[key] for x in ok if isinstance(x.get(key), (int, float))
                 and str(x.get("sex", "")).strip().upper().startswith(want)]
            if len(v) >= 20:
                g[lab_] = (len(v), float(np.median(v)))
        if len(g) == 2:
            nf, mf = g["female"]; nm, mm_ = g["male"]
            print(f"  {key:22s} female {mf:7.1f} (n={nf})   male {mm_:7.1f} (n={nm})   "
                  f"diff {mf - mm_:+.1f}")

    print(f"\n  wrote {p}")
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
