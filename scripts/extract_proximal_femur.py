"""scripts/extract_proximal_femur.py — the hip, which was only ever used as a landmark.

The femur labels have been carrying a fracture-risk panel this whole time and were being
used for one thing: the midpoint of the bicoxofemoral axis, for pelvic incidence. Hip
geometry predicts fracture INDEPENDENTLY of bone density, which makes it exactly the kind
of measure an opportunistic dataset should carry.

  femoral_head_diameter_mm   Diameter of a sphere fitted to the head. One of the stronger
                             single skeletal sex discriminants, and a body-size proxy that
                             is not part of the pelvis -- which the pelvic width panels
                             need and currently borrow from the vertebrae.

  neck_shaft_angle_deg       Angle between the femoral neck axis and the shaft axis.
                             Published normal is about 125-130 degrees; below 120 is coxa
                             vara and above 135 coxa valga. A wide neck-shaft angle raises
                             hip fracture risk.

  hip_axis_length_mm         Faulkner's measurement: along the neck axis from the greater
                             trochanter through to the inner pelvic brim. It predicts hip
                             fracture independently of bone mineral density AND of FRAX,
                             which is what makes it worth extracting rather than inferring.

  femoral_neck_hu            Trabecular attenuation of the femoral neck, by the same
                             eroded-core method used on the vertebrae. The proximal femur
                             is where the fracture that matters actually happens, so a
                             density measured there is closer to the outcome than one
                             measured in the spine.

WHAT IS NOT HERE. The cortical thickness index is measured 100 mm below the lesser
trochanter, and these femur labels span only 61-109 mm in total -- the level does not
exist in the data. Reporting it would mean measuring something else and calling it CTI.

    python scripts/extract_proximal_femur.py --labels data/v5_final --ct data/hf_export/ct
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

HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33
MIN_VOX = 5000


def _fit_sphere(pts):
    """Algebraic sphere fit. -> (centre, radius) in the same units as pts.

    Linear least squares on x^2+y^2+z^2 = 2ax+2by+2cz+d, which is exact for a perfect
    sphere and stable for a partial one -- and a femoral head in a CT label IS partial,
    since it is continuous with the neck.
    """
    if len(pts) < 100:
        return None, None
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = (pts ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    c = sol[:3]
    r2 = sol[3] + (c ** 2).sum()
    if r2 <= 0:
        return None, None
    return c, float(np.sqrt(r2))


def _fit_sphere_robust(pts, iters=4, keep_mm=3.5):
    """Sphere fit that throws away the points that are not on the head.

    WHY THE PLAIN FIT WAS NOT ENOUGH. The head used to be taken as the top 30% of the
    femur label by height. That region is not the head: it also contains the greater
    trochanter, which sits at much the same height, and a variable amount of neck. How
    much of each depends on how far down the shaft the scan reached -- these labels span
    61 to 109 mm -- so the same anatomy gave different answers in different fields of
    view. It showed up as male head diameters clustering near 35 mm, where 50 is the
    median, and as left-right differences of up to 21 mm in the same patient, which no
    hip has.

    The fix is the standard one in hip morphometry and in femoroacetabular-impingement
    work: fit, keep the points that lie on the fitted surface, refit. Trochanter and
    neck points are far from the head's sphere and drop out within two passes; the head
    is close to spherical, which is the assumption the whole measurement already rests
    on. Points are kept within KEEP_MM of the surface rather than by a quantile, so a
    fit that started badly cannot keep a fixed fraction of the wrong cloud.

    Returns (centre, radius, inlier_fraction, rms_mm) so the caller can refuse a fit that
    did not converge onto something spherical.
    """
    c, rad = _fit_sphere(pts)
    if rad is None:
        return None, None, 0.0, None

    # TRIM BY RANK FIRST, THEN BY DISTANCE. Keeping every point within a fixed band of
    # the current sphere only works if the current sphere is roughly right, and the
    # first fit need not be: on a cloud carrying head, neck and greater trochanter the
    # opening fit came out at 53 mm diameter against a true 48, and a band drawn around
    # THAT sphere keeps the points that put it there. Discarding the worst half by
    # residual instead cannot be captured that way, because the head is the majority of
    # the cloud and the majority is what survives a rank trim. This is least-trimmed-
    # squares, and the reason it is used wherever a fit has to survive contamination it
    # cannot identify in advance.
    sel = pts
    for frac in (0.7, 0.55, 0.5):
        d = np.abs(np.linalg.norm(pts - c, axis=1) - rad)
        k = max(100, int(frac * len(pts)))
        if k >= len(pts):
            continue
        sel = pts[np.argsort(d)[:k]]
        c2, r2 = _fit_sphere(sel)
        if r2 is None:
            break
        c, rad = c2, r2

    # now the band, to settle onto the surface rather than a fixed share of it
    for _ in range(iters):
        d = np.abs(np.linalg.norm(pts - c, axis=1) - rad)
        keep = d <= keep_mm
        if keep.sum() < 100:
            break
        sel = pts[keep]
        c2, r2 = _fit_sphere(sel)
        if r2 is None:
            break
        if abs(r2 - rad) < 1e-3:
            c, rad = c2, r2
            break
        c, rad = c2, r2
    resid = np.linalg.norm(sel - c, axis=1) - rad
    rms = float(np.sqrt((resid ** 2).mean())) if len(sel) else None
    return c, rad, float(len(sel) / max(1, len(pts))), rms


def _axis(pts):
    """Principal axis of a point cloud, as a unit vector."""
    q = pts - pts.mean(0)
    return np.linalg.svd(q, full_matrices=False)[2][0]


def one(args) -> dict:
    lab_p, ct_p = args
    stem = Path(lab_p).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        li = nib.as_closest_canonical(nib.load(lab_p))
        lab = np.asanyarray(li.dataobj)
        sp = np.asarray(li.header.get_zooms()[:3], float)
        ct = None
        if ct_p and Path(ct_p).exists():
            ci = nib.as_closest_canonical(nib.load(ct_p))
            if ci.shape == li.shape:
                ct = np.asanyarray(ci.dataobj).astype(np.float32)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    for fid, hid, side in ((FEM_L, HIP_L, "left"), (FEM_R, HIP_R, "right")):
        fem = lab == fid
        if fem.sum() < MIN_VOX:
            continue
        idx = np.argwhere(fem)
        pts = idx * sp

        # --- head: fit a sphere to the superomedial quarter -------------------------
        # The head is the superior end; fitting the whole label would drag the centre
        # down the shaft, which is the error that read bi-acetabular width 18 mm wide.
        # THE ACETABULAR SEED WAS TRIED AND MADE IT WORSE, WHICH IS WORTH RECORDING.
        # Selecting femur voxels near the hip bone looked strictly better: the acetabulum
        # wraps the head and nothing else, so those voxels are head surface by
        # construction. But it selects only the CAP of the head that is in contact with
        # the socket, and a least-trimmed-squares fit to a cap converges onto the cap's
        # own curvature rather than the sphere it came from. Median head diameter fell
        # from 47.1 mm to 42.1 -- women 39.6 against a published 43-45, men 45.8 against
        # 48-52 -- and the neck-shaft angle, which is computed from the head centre, went
        # with it from 125.2 to 136.4 degrees. The plain height percentile validates and
        # the clever seed does not, so the percentile stays.
        zc = np.percentile(idx[:, 2], 70)
        head_pts = pts[idx[:, 2] >= zc]

        # PLAIN ALGEBRAIC FIT. Two attempts to improve this both made it worse and the
        # numbers are recorded so nobody tries a third time without reading them.
        #
        #   acetabular seeding    selects only the CAP of the head in contact with the
        #                         socket; a fit to a cap follows the cap's curvature.
        #                         Median diameter 47.1 -> 42.1 mm.
        #   least-trimmed-squares trimming to the closest fraction keeps the densest
        #                         region, which is again the central cap, and shrinks the
        #                         sphere the same way. Median 47.1 -> 40.0, and the
        #                         residual gate rejected 150 cases outright.
        #
        # Both were principled and both were wrong, because the failure they were built
        # to fix -- contamination by the greater trochanter -- affects a minority of cases
        # while the cure moved the median for everyone. The plain fit validates against
        # published values (F 45.2, M 50.6 against F 43-45, M 48-52); the clever ones do
        # not. The known defect that remains is a small left tail from trochanter
        # contamination, which is visible in the distribution and named in the caption
        # rather than traded for a worse median.
        c, rad = _fit_sphere(head_pts)
        if rad is not None and 15.0 < rad < 35.0:
            r[f"femoral_head_diameter_{side}_mm"] = round(2 * rad, 2)

        # --- neck-shaft angle -------------------------------------------------------
        # neck: between the head centre and the trochanteric mass; shaft: the inferior
        # third of the label. Each gets its principal axis and the angle between them
        # is taken in the coronal plane, which is where the angle is defined.
        if c is not None:
            zlo = np.percentile(idx[:, 2], 15)
            zmid = np.percentile(idx[:, 2], 45)
            shaft = pts[idx[:, 2] <= zlo]
            neck = pts[(idx[:, 2] > zlo) & (idx[:, 2] <= zmid)]
            if len(shaft) > 200 and len(neck) > 200:
                a_sh = _axis(shaft)
                a_nk = c - neck.mean(0)
                # coronal projection: drop the anteroposterior component, since neck
                # anteversion is a different measurement and would inflate this one
                a_sh = np.array([a_sh[0], 0.0, a_sh[2]])
                a_nk = np.array([a_nk[0], 0.0, a_nk[2]])
                n1 = np.linalg.norm(a_sh)
                n2 = np.linalg.norm(a_nk)
                if n1 > 1e-6 and n2 > 1e-6:
                    cosang = float(np.dot(a_sh, a_nk) / (n1 * n2))
                    ang = np.degrees(np.arccos(np.clip(abs(cosang), -1, 1)))
                    nsa = 180.0 - ang if ang < 90 else ang
                    if 90 < nsa < 170:
                        r[f"neck_shaft_angle_{side}_deg"] = round(float(nsa), 1)

        # --- hip axis length: trochanter to the inner pelvic brim, along the neck ----
        hip = lab == hid
        if c is not None and hip.sum() > MIN_VOX:
            troch = pts[idx[:, 0] == (idx[:, 0].max() if side == "right" else idx[:, 0].min())]
            if len(troch):
                t = troch.mean(0)
                d = c - t
                nrm = np.linalg.norm(d)
                if nrm > 1e-6:
                    d = d / nrm
                    hp = np.argwhere(hip) * sp
                    # project the hip onto the neck axis and take the far extent: that is
                    # where the axis leaves the pelvis at the inner brim
                    proj = (hp - t) @ d
                    lat = np.linalg.norm((hp - t) - np.outer(proj, d), axis=1)
                    near = proj[lat <= 12.0]
                    if near.size > 50:
                        hal = float(np.percentile(near, 97))
                        if 70 < hal < 145:
                            r[f"hip_axis_length_{side}_mm"] = round(hal, 1)

        # --- femoral neck trabecular attenuation ------------------------------------
        if ct is not None:
            zlo = np.percentile(idx[:, 2], 45)
            zhi = np.percentile(idx[:, 2], 72)
            band = fem.copy()
            band[:, :, : int(zlo)] = False
            band[:, :, int(zhi) + 1:] = False
            if band.sum() > 400:
                # 3 mm in from every surface, in millimetres so slice thickness cannot
                # bias it -- the same rule as the vertebral measurement
                dist = ndimage.distance_transform_edt(band, sampling=sp)
                core = dist >= 3.0
                if core.sum() < 150:
                    core = dist >= 2.0
                if core.sum() >= 100:
                    v = ct[core]
                    lo, hi = np.percentile(v, [10, 90])
                    v = v[(v >= lo) & (v <= hi)]
                    if v.size >= 50:
                        r[f"femoral_neck_hu_{side}"] = round(float(v.mean()), 1)

    # a single value per case for the panels: the side that is better measured
    for stem_key in ("femoral_head_diameter", "neck_shaft_angle", "hip_axis_length",
                     "femoral_neck_hu"):
        suff = "_deg" if "angle" in stem_key else ("" if stem_key.endswith("hu") else "_mm")
        vals = [r.get(f"{stem_key}_{s}{suff}") for s in ("left", "right")]
        vals = [x for x in vals if isinstance(x, (int, float))]
        if vals:
            r[f"{stem_key}{suff}"] = round(float(np.mean(vals)), 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export/ct")
    ap.add_argument("--manifest", default="data/hf_export/manifest.json")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    labs = sorted(Path(a.labels).glob("*_label.nii.gz"))
    jobs = [(str(p), str(Path(a.ct) / p.name.replace("_label", "_ct"))) for p in labs]
    print(f"{len(jobs)} case(s)\n", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, jobs, chunksize=2), 1):
            res.append(x)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

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
    p = out / "proximal_femur.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    print(f"\n  {len(ok)} of {len(res)} measured\n")
    PLAUSIBLE = {
        "femoral_head_diameter_mm": (38, 58),      # adult, both sexes pooled
        "neck_shaft_angle_deg": (118, 140),        # published normal ~125-130
        "hip_axis_length_mm": (85, 125),           # Faulkner-style measurement
        "femoral_neck_hu": (60, 260),
    }
    bad = []
    for key, (lo, hi) in PLAUSIBLE.items():
        v = np.array([x[key] for x in ok if isinstance(x.get(key), (int, float))], float)
        if v.size < 20:
            print(f"  {key:26} n={v.size} (too few)")
            continue
        med = float(np.median(v))
        flag = "" if lo <= med <= hi else f"   <-- IMPLAUSIBLE (expected {lo}-{hi})"
        if flag:
            bad.append(key)
        print(f"  {key:26} n={v.size:4d}  median {med:7.1f}  "
              f"IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}{flag}")

    print("\n  by sex (femoral head diameter is a known discriminant):")
    for key in PLAUSIBLE:
        g = {}
        for want, lab_ in (("F", "female"), ("M", "male")):
            v = [x[key] for x in ok if isinstance(x.get(key), (int, float))
                 and str(x.get("sex", "")).strip().upper().startswith(want)]
            if len(v) >= 20:
                g[lab_] = (len(v), float(np.median(v)))
        if len(g) == 2:
            nf, mf = g["female"]; nm, mm_ = g["male"]
            print(f"  {key:26} female {mf:7.1f} (n={nf})  male {mm_:7.1f} (n={nm})  "
                  f"diff {mf - mm_:+.1f}")

    print(f"\n  wrote {p}")
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
