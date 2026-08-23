"""scripts/measure_sacral_ala.py — look for the fused process on the sacrum's side of the label.

THE PROBLEM THIS EXISTS FOR. Scored against the radiologist grades, the transverse-process
screen recovers Castellvi I and II at median leave-one-out rank 32 and III/IV at rank 305.
That is backwards from the obvious expectation, since III and IV are gross bony fusion, and
the cause is not biology. When a transverse process fuses to the sacral ala, the fused mass
is labelled SACRUM. The vertebra-side measurement then describes whatever free vertebra is
left over -- a short process beside a wide gap -- so a Castellvi III measures like an
ordinary case: tp_height 20.8 mm against 18.4 unlabelled, gap 7.9/8.2 against 8.5/8.8.

If the bone moved across the label boundary, the evidence is on the other side of it. That
is what this measures.

WHAT IT MEASURES, AND WHY THIS QUANTITY. A normal sacrum's highest point is at the midline:
the promontory, the anterosuperior lip of S1. The alae slope down and away from it. A
transverse process fused into an ala adds bone laterally and, critically, ABOVE where the
ala would otherwise reach -- because the process it fused with sat above the ala. So:

    ala_rise = (most cranial sacral voxel in the lateral third, one side)
             - (most cranial sacral voxel near the midline)

is at or below zero for an ordinary sacrum and rises with fused accessory bone. Per side,
because Castellvi a and b are unilateral and bilateral, so the ASYMMETRY is part of the
grade rather than a nuisance.

Two things this deliberately does not do. It does not use the vertebral count, for the same
reason the other screen does not: the count is already recorded and a screen that leans on
it cannot find what nobody flagged. And it does not fit anything -- there are 25 III/IV
records, which is enough to check a measurement's direction and nowhere near enough to
train on.

HONESTY ABOUT WHAT A POSITIVE MEANS. A high ala_rise says the sacrum carries bone above and
lateral to where a sacrum usually ends. That is what a fused transitional process looks
like in a label map. It is also what a large osteophyte, a fused iliolumbar ligament, or a
mis-assigned voxel at the sacroiliac joint looks like. Every case this ranks is a request
for a radiologist to look.

    python scripts/measure_sacral_ala.py --labels data/hf_export_v5/labels --workers 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

SACRUM, S1 = 26, 29
L1, L6 = 20, 25
MID_FRAC = 0.18      # half-width of the midline band, as a fraction of sacral half-width
LAT_FRAC = 0.55      # the lateral third starts here, as a fraction of sacral half-width
ANT_FRAC = 0.50      # keep only the anterior half in depth: see top_of()


def axes_from(affine):
    """-> (lateral, craniocaudal, cranial sign, left-is-low, anterior-is-low)."""
    codes = nib.aff2axcodes(affine)
    lat = next(i for i, c in enumerate(codes) if c in ("L", "R"))
    cc = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    up = +1 if codes[cc] == "S" else -1
    left_is_low = codes[lat] == "R"       # index increases toward the right => low = left
    ap = ({0, 1, 2} - {lat, cc}).pop()
    ant_is_low = codes[ap] == "P"         # increases toward posterior => low = anterior
    return lat, cc, up, left_is_low, ant_is_low


def one(path: Path):
    case = path.name.split("_")[0]
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        lat, cc, up, left_is_low, ant_is_low = axes_from(img.affine)

        # one bincount instead of a sacrum test plus six lumbar equality scans; np.unique
        # and repeated `lab == v` each walk 160 million voxels, and that was the whole cost
        # of the earlier version of this pass
        counts = np.bincount(lab.reshape(-1))
        present = set(int(v) for v in np.nonzero(counts)[0])
        n_sac = (int(counts[SACRUM]) if SACRUM < len(counts) else 0)
        n_s1 = (int(counts[S1]) if S1 < len(counts) else 0)
        if n_sac + n_s1 < 5000:
            return {"case": case, "error": "no sacrum"}
        sac = (lab == SACRUM) | (lab == S1)

        # crop once; everything after is small
        idx = np.nonzero(sac)
        sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
        s = sac[sl]
        del sac

        # move the lateral axis to 0 and the craniocaudal axis to 1 (views, no copies)
        a = np.moveaxis(s, (lat, cc), (0, 1))
        n_lat = a.shape[0]
        # the sacrum's own midline, from its mass rather than from the image centre
        prof = a.sum(axis=(1, 2)).astype(float)
        mid = float((prof * np.arange(n_lat)).sum() / max(prof.sum(), 1))
        half = max(mid, n_lat - 1 - mid)

        # THE TOP OF THE LATERAL SACRUM IS NOT THE ALA. It is the S1 SUPERIOR ARTICULAR
        # PROCESS, which projects cranially and posteriorly in everybody, so measuring the
        # topmost lateral voxel returns that process in every case and separates nothing:
        # over the full cohort it gave AUC 0.450 for Castellvi III/IV against ungraded,
        # which is below chance. Same family as the original tp_gap error, where the
        # nearest point of the lateral vertebra to the sacrum was the L5-S1 facet joint.
        #
        # The ala proper is ANTERIOR to those processes. Restricting each band to the
        # anterior fraction of the sacrum's own depth excludes them. `ant` is a boolean
        # over the anteroposterior axis, built from the affine rather than assumed.
        n_ap = a.shape[2]
        ap_rows = np.nonzero(a.any(axis=(0, 1)))[0]
        ap_lo, ap_hi = int(ap_rows.min()), int(ap_rows.max())
        depth_ap = ap_hi - ap_lo + 1
        if ant_is_low:
            ap_slice = slice(ap_lo, ap_lo + max(1, int(round(ANT_FRAC * depth_ap))))
        else:
            ap_slice = slice(ap_hi + 1 - max(1, int(round(ANT_FRAC * depth_ap))), ap_hi + 1)

        def top_of(lo, hi, anterior_only=True):
            """most cranial sacral index within a lateral band, or None."""
            lo, hi = max(0, int(lo)), min(n_lat, int(hi))
            if hi <= lo:
                return None
            band = a[lo:hi, :, ap_slice] if anterior_only else a[lo:hi]
            rows = np.nonzero(band.any(axis=(0, 2)))[0]
            if not len(rows):
                return None
            return int(rows.max()) if up > 0 else int(rows.min())

        t_mid = top_of(mid - MID_FRAC * half, mid + MID_FRAC * half + 1)
        t_lowside = top_of(0, mid - LAT_FRAC * half)                 # low index side
        t_highside = top_of(mid + LAT_FRAC * half, n_lat)            # high index side
        # the whole-depth version, kept so the articular-process failure remains visible
        w_mid = top_of(mid - MID_FRAC * half, mid + MID_FRAC * half + 1, False)
        w_low = top_of(0, mid - LAT_FRAC * half, False)
        w_high = top_of(mid + LAT_FRAC * half, n_lat, False)

        out = {"case": case, "error": "",
               "axcodes": "".join(nib.aff2axcodes(img.affine)),
               "sacrum_voxels": int(s.sum())}

        def rise(t):
            if t is None or t_mid is None:
                return None
            d = (t - t_mid) if up > 0 else (t_mid - t)
            return round(float(d) * zooms[cc], 1)

        r_low, r_high = rise(t_lowside), rise(t_highside)
        # name the sides anatomically, never by index
        if left_is_low:
            out["ala_rise_left_mm"], out["ala_rise_right_mm"] = r_low, r_high
        else:
            out["ala_rise_left_mm"], out["ala_rise_right_mm"] = r_high, r_low

        def wrise(t):
            if t is None or w_mid is None:
                return None
            d = (t - w_mid) if up > 0 else (w_mid - t)
            return round(float(d) * zooms[cc], 1)

        wl, wh = wrise(w_low), wrise(w_high)
        wv = [v for v in (wl, wh) if v is not None]
        out["ala_rise_wholedepth_max_mm"] = max(wv) if wv else None

        vals = [v for v in (out["ala_rise_left_mm"], out["ala_rise_right_mm"])
                if v is not None]
        out["ala_rise_max_mm"] = max(vals) if vals else None
        out["ala_rise_asym_mm"] = (round(abs(vals[0] - vals[1]), 1)
                                   if len(vals) == 2 else None)

        # context, not criterion
        lum = [v for v in range(L1, L6 + 1) if v in present]
        out["lowest_lumbar_label"] = max(lum) if lum else ""
        return out
    except Exception as e:                                            # noqa: BLE001
        return {"case": case, "error": f"{type(e).__name__}: {e}"}


FIELDS = ["case", "lowest_lumbar_label", "axcodes", "sacrum_voxels",
          "ala_rise_left_mm", "ala_rise_right_mm",
          "ala_rise_max_mm", "ala_rise_asym_mm", "ala_rise_wholedepth_max_mm", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="morphometrics/sacral_ala.csv")
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if a.limit:
        files = files[:a.limit]
    print(f"  {len(files)} volume(s), {a.workers} worker(s)", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"    {i}/{len(files)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r.get("error")]
    print(f"\n  wrote {a.out}: {len(ok)} measured, {len(rows) - len(ok)} failed", flush=True)
    v = np.array([r["ala_rise_max_mm"] for r in ok
                  if r.get("ala_rise_max_mm") is not None], float)
    if len(v):
        print(f"  ala_rise_max: median {np.median(v):.1f} mm, "
              f"p90 {np.percentile(v, 90):.1f}, p99 {np.percentile(v, 99):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
