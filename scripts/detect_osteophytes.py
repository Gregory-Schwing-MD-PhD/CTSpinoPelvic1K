"""scripts/detect_osteophytes.py — measure osteophytic change from vertebral body shape.

THE GATING QUESTION, ASKED FIRST. A shape measure over the label can only see an osteophyte
if the segmenter included it. Osteophytes are contiguous bone so it usually will, but
"usually" is not a basis for a measurement -- so every case reports `mask_captures_bone`:
the fraction of bone-density voxels in a shell just outside the vertebra label. A high
value means bone was left out of the mask and the shape measures are blind to it.

THREE MEASURES, none of which needs a shape model or a training set:

  ant_spike_mm      Per axial slice, the distance from the body centroid to the boundary as
                    a function of angle, and the outward residual from a fitted ellipse. An
                    osteophyte is a LOCALISED radial spike where a normal margin is smooth.
                    Restricted to the ANTERIOR half, which excludes the spinous and
                    transverse processes for free -- they are also spikes, but not
                    pathology -- and anterior is where osteophytes predominate anyway.

  endplate_flare    Maximum anterior extent in the top and bottom fifths of the body against
                    the middle fifth. Claw and traction osteophytes form at the endplate
                    CORNERS, so they flare the ends while a normal body stays roughly
                    uniform.

  ant_bridge_ratio  Minimum distance to the vertebra below measured ANTERIORLY against the
                    same distance measured CENTRALLY. A bridging osteophyte closes the
                    anterior gap while the disc space itself stays open, so the ratio falls
                    well below 1.

WHY THE THIRD ONE MATTERS BEYOND OSTEOPHYTES. Bridging is a third route to a false
transitional finding, alongside congenital fusion and surgical hardware: all three read as
"no gap" to a distance measure. The transitional morphometrics use exactly such a distance,
so this measure is a confounder check as much as a pathology detector.

Validate before trusting: 0167's L1 is a reader-confirmed positive.

    python scripts/detect_osteophytes.py --labels data/v5_final --ct data/hf_export_v4/ct \\
        --cases 0167 --verbose
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

THORACIC_BASE = 7
NAMES = {**{THORACIC_BASE + n: f"T{n}" for n in range(1, 13)},
         **{19 + n: f"L{n}" for n in range(1, 7)}}
BONE_HU = 200.0
SHELL_MM = 3.0
MIN_VOX = 3000


def _fit_ellipse_residual(pts_mm):
    """Outward residual (mm) of a 2D boundary from its best-fit ellipse.

    An ellipse rather than a circle because vertebral bodies are reniform and wider than
    deep; fitting a circle would report that ordinary shape as a spike everywhere.
    """
    if len(pts_mm) < 12:
        return 0.0, 0.0
    c = pts_mm.mean(0)
    q = pts_mm - c
    # ellipse via the second-moment matrix: scale each point into the frame where the
    # fitted ellipse is the unit circle, then the radius is the deviation
    cov = np.cov(q.T)
    try:
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-9)
        white = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    rad = np.linalg.norm(q @ white.T, axis=1)
    med = float(np.median(rad))
    if med <= 0:
        return 0.0, 0.0
    scale = float(np.median(np.linalg.norm(q, axis=1))) / med   # back to mm
    resid = (rad - med) * scale
    return float(resid.max()), float(np.percentile(resid, 95))


def one(args) -> dict:
    stem, lp, cp, verbose = args
    out = {"case": stem, "levels": []}
    try:
        li = nib.as_closest_canonical(nib.load(lp))
        lab = np.asanyarray(li.dataobj).astype(np.int16)
        sp = np.array(li.header.get_zooms()[:3], float)
        ct = None
        if cp:
            ci = nib.as_closest_canonical(nib.load(cp))
            ct = np.asanyarray(ci.dataobj)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": f"{type(exc).__name__}"}

    ids = [v for v in NAMES if (lab == v).sum() >= MIN_VOX]
    for vid in sorted(ids):
        m = lab == vid
        idx = np.argwhere(m)
        cen = idx.mean(0)
        rec = {"level": NAMES[vid], "id": int(vid)}

        # --- is the osteophyte even in the mask? bone density just outside it -----------
        if ct is not None:
            it = max(1, int(round(SHELL_MM / max(sp.min(), 1e-6))))
            shell = ndimage.binary_dilation(m, iterations=it) & ~m
            if shell.any():
                rec["shell_bone_frac"] = round(float((ct[shell] > BONE_HU).mean()), 3)

        # --- anterior radial spikiness --------------------------------------------------
        # +y is ANTERIOR in RAS; the anterior half excludes the posterior elements
        ant = m & (np.arange(lab.shape[1])[None, :, None] > cen[1])
        spikes, p95s = [], []
        zs = np.unique(idx[:, 2])
        for z in zs[:: max(1, len(zs) // 24)]:
            sl = ant[:, :, int(z)]
            if sl.sum() < 40:
                continue
            b = sl & ~ndimage.binary_erosion(sl)
            p = np.argwhere(b).astype(float)
            if len(p) < 12:
                continue
            p[:, 0] *= sp[0]
            p[:, 1] *= sp[1]
            mx, p95 = _fit_ellipse_residual(p)
            spikes.append(mx)
            p95s.append(p95)
        if spikes:
            rec["ant_spike_mm"] = round(float(np.max(spikes)), 2)
            rec["ant_spike_p95_mm"] = round(float(np.percentile(p95s, 95)), 2)

        # --- endplate flare -------------------------------------------------------------
        zlo, zhi = int(idx[:, 2].min()), int(idx[:, 2].max())
        span = max(1, zhi - zlo)
        def _reach(a, b):
            sub = ant[:, :, int(a):int(b) + 1]
            if not sub.any():
                return None
            yy = np.argwhere(sub)[:, 1]
            return float(yy.max() - cen[1]) * sp[1]
        top = _reach(zhi - span // 5, zhi)
        bot = _reach(zlo, zlo + span // 5)
        mid = _reach(zlo + 2 * span // 5, zlo + 3 * span // 5)
        if mid and mid > 0 and (top or bot):
            rec["endplate_flare"] = round(max(v for v in (top, bot) if v) / mid, 3)

        # --- anterior bridging to the level below ---------------------------------------
        below = vid + 1 if (vid + 1) in NAMES else None
        if below is not None and (lab == below).sum() >= MIN_VOX:
            mb = lab == below
            a_self = m & (np.arange(lab.shape[1])[None, :, None] > cen[1] + 5 / sp[1])
            a_below = mb & (np.arange(lab.shape[1])[None, :, None] > cen[1] + 5 / sp[1])
            def _mind(x, y):
                px, py = np.argwhere(x), np.argwhere(y)
                if not len(px) or not len(py):
                    return None
                px = px[:: max(1, len(px) // 300)] * sp
                py = py[:: max(1, len(py) // 300)] * sp
                d = np.sqrt((((px[:, None, :] - py[None, :, :]) ** 2).sum(-1)))
                return float(d.min())
            da, dc = _mind(a_self, a_below), _mind(m, mb)
            if da is not None and dc is not None and dc > 0:
                rec["ant_gap_mm"] = round(da, 2)
                rec["central_gap_mm"] = round(dc, 2)
                rec["ant_bridge_ratio"] = round(da / dc, 3)
        out["levels"].append(rec)

    vals = [l.get("ant_spike_mm") for l in out["levels"] if l.get("ant_spike_mm")]
    out["max_ant_spike_mm"] = round(max(vals), 2) if vals else None
    out["worst_level"] = (max(out["levels"], key=lambda l: l.get("ant_spike_mm") or 0)
                          .get("level") if vals else None)
    if not verbose:
        out.pop("levels", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="qc_osteophyte")
    a = ap.parse_args()

    labdir, ctdir = Path(a.labels), Path(a.ct)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    jobs = [(s, str(labdir / f"{s}_label.nii.gz"),
             str(ctdir / f"{s}_ct.nii.gz") if (ctdir / f"{s}_ct.nii.gz").exists() else "",
             a.verbose)
            for s in stems if (labdir / f"{s}_label.nii.gz").exists()]
    print(f"{len(jobs)} case(s)\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=1), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "osteophyte.json").write_text(json.dumps(res, indent=1))
    ok = [r for r in res if "error" not in r]
    if a.verbose:
        for r in ok:
            print(f"\n  {r['case']}   worst {r.get('worst_level')} "
                  f"{r.get('max_ant_spike_mm')} mm")
            print(f"    {'level':6s} {'spike':>7s} {'p95':>6s} {'flare':>6s} "
                  f"{'ant_gap':>8s} {'ctr_gap':>8s} {'ratio':>6s} {'shell':>6s}")
            for l in r.get("levels", []):
                print(f"    {l['level']:6s} {str(l.get('ant_spike_mm')):>7s} "
                      f"{str(l.get('ant_spike_p95_mm')):>6s} {str(l.get('endplate_flare')):>6s} "
                      f"{str(l.get('ant_gap_mm')):>8s} {str(l.get('central_gap_mm')):>8s} "
                      f"{str(l.get('ant_bridge_ratio')):>6s} {str(l.get('shell_bone_frac')):>6s}")
    with open(out / "osteophyte.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "max_ant_spike_mm", "worst_level"])
        for r in sorted(ok, key=lambda r: -(r.get("max_ant_spike_mm") or 0)):
            w.writerow([r["case"], r.get("max_ant_spike_mm"), r.get("worst_level")])
    print(f"\n  wrote {out}/osteophyte.csv and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
