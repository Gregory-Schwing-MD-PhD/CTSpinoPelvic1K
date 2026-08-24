"""scripts/extract_degenerative.py — degenerative disease, by published criteria.

Everything here is detectable from bone labels plus the CT they came from, and every
measure follows a criterion someone else defined. Nothing is invented for this dataset.

  disc_height_mm        Gap between adjacent vertebral BODIES at the midline. Disc space
                        narrowing is the oldest radiographic sign of degenerative disc
                        disease and the one every grading scheme starts from.

  vacuum_phenomenon     Gas within the disc space. On CT this is unambiguous -- gas sits
                        near -1000 HU where disc is +50 to +100 -- and it is essentially
                        PATHOGNOMONIC of disc degeneration. There is no normal variant
                        that puts air inside a disc. Detected as voxels below -150 HU in
                        the disc space, which is far enough from fat (-100) to be safe.

  bridging / DISH       Resnick and Niwayama's criteria: flowing ossification bridging
                        four or more CONTIGUOUS vertebrae, with disc height preserved and
                        no apophyseal or sacroiliac inflammatory change. Preserved disc
                        height is what separates DISH from degenerative bridging, and it
                        is measurable here, so the two are distinguished rather than
                        conflated. A bridge is detected as bony continuity across a disc
                        space that the disc height says is NOT collapsed.

  osteophyte_index      Endplate width over mid-body width. A healthy vertebral body is
                        slightly waisted -- narrower in the middle than at its endplates.
                        Osteophytes form at the endplate RIM, so they raise this ratio.
                        This is the geometric expression of what a radiologist calls
                        endplate spurring.

WHY NO FACET ARTHROPATHY. The facet joints need the articular surfaces separated from
the rest of the posterior arch, and whole-vertebra labels do not carry that boundary.
Reporting a facet score off these labels would be inventing a measurement, so it is
absent rather than approximated.

    python scripts/extract_degenerative.py --labels data/v5_final --ct data/hf_export/ct

WITHHELD: DISH. Four detectors were built and none survives its own check.

  1.  0.5% against a published 3.8-27%. Too strict.
  2. 48.9%. The bridge box still contained vertebral body.
  3. 48.5%. The disc-space boundary used the MEDIAN of the per-column extremes, so
     half the columns still had body inside the box.
  4. 23.3%, which is IN RANGE -- and still wrong, for a reason the headline number
     cannot show.

WHY THE FOURTH ONE FAILS ANYWAY. Prevalence by decade runs 27.2%, 19.6%, 15.6% from
the fifties to the seventies and beyond. DISH accumulates; it does not resolve. Every
published series has prevalence rising steeply with age, and a detector that finds
less of it in older patients is detecting something else that happens to be commoner
in the young.

The obvious candidate was bone density -- a HU-threshold detector would find less
"ossification" in an osteoporotic spine, and this cohort's density does fall with age.
That is not the explanation, or not the whole one: the point-biserial correlation with
L1 trabecular attenuation is only r = 0.095, and the inverse age gradient survives
inside EVERY density tertile (18.4/14.1/8.8, 31.1/24.7/25.0, 28.3/21.7/22.2). Whatever
the detector is measuring, it is not ossification and it is not density.

The sex ratio, for what it is worth, points the right way: 27.2% in men against 19.1%
in women, where DISH is male-predominant. One correct gradient out of two is what a
partly-confounded measure looks like, not what a working one looks like.

WHAT WOULD FIX IT, for whoever picks this up. The anterior ossification is a bridge of
BONE spanning the disc space, and this pipeline only ever asks whether attenuation is
high in a box. It cannot distinguish a bridge from an osteophyte pair that nearly
touches, and near-touching pairs are commoner in the young because the discs are
taller. That is a plausible source of the inverted gradient and it is testable:
require a connected path of bone from one vertebral body to the next through the box,
rather than a count of bright voxels in it. Connectivity is the criterion Resnick's
definition actually states.

The bridge_* and max_contiguous_bridges columns are still written to the CSV so the
next attempt has something to compare against. dish_resnick is NOT plotted anywhere
and must not be until the age gradient points the right way.
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

LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}
S1 = 29
PAIRS = [(20, 21, "L1L2"), (21, 22, "L2L3"), (22, 23, "L3L4"),
         (23, 24, "L4L5"), (24, S1, "L5S1")]
GAS_HU = -150.0            # gas is near -1000; fat bottoms out near -100
MIN_VOX = 3000


def _canal_front(mask):
    zs = np.nonzero(mask.any(axis=(0, 1)))[0]
    if len(zs) < 5:
        return None
    fronts = []
    for z in range(int(np.percentile(zs, 30)), int(np.percentile(zs, 70)) + 1):
        sl = mask[:, :, z]
        if sl.sum() < 60:
            continue
        hole = ndimage.binary_fill_holes(sl) & ~sl
        if not hole.any():
            continue
        cc, n = ndimage.label(hole)
        if n == 0:
            continue
        sizes = ndimage.sum(hole, cc, range(1, n + 1))
        big = cc == (int(np.argmax(sizes)) + 1)
        if big.sum() < 20:
            continue
        fronts.append(int(np.nonzero(big.any(axis=0))[0].max()))
    return float(np.median(fronts)) if fronts else None


def body_of(mask):
    """The vertebral body: everything anterior to the canal's front wall."""
    front = _canal_front(mask)
    if front is None:
        ys = np.nonzero(mask.any(axis=(0, 2)))[0]
        if len(ys) < 4:
            return None
        front = ys.min() + 0.45 * (ys.max() - ys.min())
    out = np.zeros_like(mask)
    out[:, int(np.ceil(front)):, :] = mask[:, int(np.ceil(front)):, :]
    return out if out.sum() >= 300 else None


def one(args) -> dict:
    lab_p, ct_p = args
    stem = Path(lab_p).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        li = nib.as_closest_canonical(nib.load(lab_p))
        ci = nib.as_closest_canonical(nib.load(ct_p))
        if li.shape != ci.shape:
            return {"case": stem, "error": "shape mismatch"}
        lab = np.asanyarray(li.dataobj)
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
        sp = np.asarray(li.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    bodies = {}
    for vid in list(LUMBAR) + [S1]:
        m = lab == vid
        if m.sum() >= MIN_VOX:
            b = body_of(m)
            if b is not None:
                bodies[vid] = b

    # ---- osteophyte flare: endplate width against mid-body width -------------------
    for vid, name in LUMBAR.items():
        b = bodies.get(vid)
        if b is None:
            continue
        idx = np.argwhere(b)
        zs = idx[:, 2]
        def width_between(p0, p1):
            sel = idx[(zs >= np.percentile(zs, p0)) & (zs <= np.percentile(zs, p1))]
            if len(sel) < 60:
                return None
            return float(np.percentile(sel[:, 0], 99) - np.percentile(sel[:, 0], 1)) * sp[0]
        rim = width_between(80, 100)          # the endplate rim, where spurs grow
        waist = width_between(38, 62)         # mid-body, where a healthy body is narrowest
        if rim and waist and waist > 5:
            r[f"osteophyte_index_{name}"] = round(rim / waist, 3)

    # ---- per-level disc height, vacuum gas, and bony bridging -----------------------
    bridged = []
    for upper, lower, name in PAIRS:
        bu, bl = bodies.get(upper), bodies.get(lower)
        if bu is None or bl is None:
            bridged.append(None)
            continue
        iu, il = np.argwhere(bu), np.argwhere(bl)
        # a midline column through both bodies: the disc is measured where a radiologist
        # measures it, not at the rim where osteophytes distort the gap
        cx = float(np.median(np.concatenate([iu[:, 0], il[:, 0]])))
        cy = float(np.median(np.concatenate([iu[:, 1], il[:, 1]])))
        rx = max(2, int(round(8.0 / sp[0])))
        ry = max(2, int(round(8.0 / sp[1])))
        sel_u = iu[(np.abs(iu[:, 0] - cx) <= rx) & (np.abs(iu[:, 1] - cy) <= ry)]
        sel_l = il[(np.abs(il[:, 0] - cx) <= rx) & (np.abs(il[:, 1] - cy) <= ry)]
        if len(sel_u) < 40 or len(sel_l) < 40:
            bridged.append(None)
            continue
        # DISC HEIGHT, PER COLUMN, THEN THE MEDIAN.
        # Vertebral endplates are CONCAVE: the disc is thickest in the middle and the
        # rim projects toward it. Taking the lowest voxel of the upper body anywhere in
        # the column and the highest of the lower body measures RIM TO RIM -- the
        # narrowest part of the space -- which read 4 to 6 mm against a published 8 to 12.
        # Measuring each column separately and taking the median gives the height a
        # radiologist would read off a mid-sagittal slice.
        cols = {}
        for xx, yy, zz in sel_u:
            k = (xx, yy)
            cols.setdefault(k, [None, None])
            c0 = cols[k][0]
            cols[k][0] = zz if c0 is None else min(c0, zz)
        for xx, yy, zz in sel_l:
            k = (xx, yy)
            if k not in cols:
                continue
            c1 = cols[k][1]
            cols[k][1] = zz if c1 is None else max(c1, zz)
        gaps = [a - b for a, b in cols.values() if a is not None and b is not None]
        if len(gaps) < 15:
            bridged.append(None)
            continue
        gap_vox = float(np.median(gaps))
        if not (-2 < gap_vox < 60):
            bridged.append(None)
            continue
        h = max(0.0, gap_vox * sp[2])
        r[f"disc_height_{name}_mm"] = round(h, 1)
        # For the HEIGHT a median is right. For the BRIDGE BOX it is not: a median
        # boundary leaves half the columns with vertebral body inside the box, and that
        # body is bone, which is what the bridge test is looking for. The box needs
        # boundaries no column crosses.
        pairs = [(a, b) for a, b in cols.values() if a is not None and b is not None]
        z_low_of_upper = float(np.median([a for a, _ in pairs]))
        z_top_of_lower = z_low_of_upper - gap_vox
        z_safe_hi = float(min(a for a, _ in pairs))     # below every upper-body voxel
        z_safe_lo = float(max(b for _, b in pairs))     # above every lower-body voxel

        # gas inside that disc space -- pathognomonic for degeneration
        z0, z1 = int(np.floor(z_top_of_lower)) + 1, int(np.ceil(z_low_of_upper))
        if z1 > z0:
            box = ct[int(cx - rx):int(cx + rx) + 1,
                     int(cy - ry):int(cy + ry) + 1, z0:z1]
            if box.size > 30:
                frac = float((box < GAS_HU).mean())
                r[f"vacuum_frac_{name}"] = round(frac, 4)

        # A DISH BRIDGE LIES ANTERIOR TO THE VERTEBRAL BODY LINE.
        # Two attempts got this wrong in opposite directions. Asking whether two dilated
        # labels touch found any contact at all, including a posterior facet or a
        # segmentation seam, and reported 0.5%. Looking for bone 8-18 mm forward of the
        # body CENTRE was still inside the body footprint -- a lumbar body is about 30 mm
        # deep -- so it caught the normal endplate rim and reported 48.9%.
        #
        # Flowing ossification is visible precisely because it projects BEYOND the
        # anterior cortex. So find where the two bodies' anterior surfaces actually are
        # at this level, and look in the space in front of both of them.
        ay_u = float(np.percentile(sel_u[:, 1], 97)) if len(sel_u) else None
        ay_l = float(np.percentile(sel_l[:, 1], 97)) if len(sel_l) else None
        zb0, zb1 = int(np.floor(z_safe_lo)) + 1, int(np.ceil(z_safe_hi))
        if ay_u is not None and ay_l is not None and zb1 - zb0 >= 2 and h > 0:
            front_y = int(np.ceil(max(ay_u, ay_l)))            # the anterior body line
            depth = max(3, int(round(9.0 / sp[1])))
            fx0, fx1 = int(cx - rx), int(cx + rx) + 1
            front = ct[fx0:fx1, front_y:front_y + depth, zb0:zb1]
            if front.size > 20:
                bone = front > 200.0
                # a bridge must be CONTINUOUS across the disc: bone present on nearly
                # every axial slice of the space, not merely present somewhere in it
                per_slice = bone.any(axis=(0, 1))
                span = float(per_slice.mean()) if per_slice.size else 0.0
                dens = float(bone.mean())
                r[f"anterior_bone_{name}"] = round(dens, 4)
                is_bridge = bool(span >= 0.90 and dens >= 0.06)
                bridged.append(is_bridge)
                r[f"bridge_{name}"] = int(is_bridge)
            else:
                bridged.append(None)
        else:
            bridged.append(None)

    # Resnick: four or more CONTIGUOUS vertebrae bridged, i.e. three consecutive levels
    run = best = 0
    for b in bridged:
        run = run + 1 if b else 0
        best = max(best, run)
    r["max_contiguous_bridges"] = best
    r["dish_resnick"] = int(best >= 3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export/ct")
    ap.add_argument("--manifest", default="data/hf_export/manifest.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    labs = sorted(Path(a.labels).glob("*_label.nii.gz"))
    jobs = [(str(p), str(Path(a.ct) / p.name.replace("_label", "_ct"))) for p in labs
            if (Path(a.ct) / p.name.replace("_label", "_ct")).exists()]
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
    p = out / "degenerative.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    print(f"\n  {len(ok)} of {len(res)} measured\n")

    # published expectations, so a wrong measure announces itself
    print("  disc height by level (published lumbar disc 8-12 mm, tallest at L4-5):")
    for _, _, nm in PAIRS:
        v = np.array([x[f"disc_height_{nm}_mm"] for x in ok
                      if isinstance(x.get(f"disc_height_{nm}_mm"), (int, float))], float)
        if v.size > 50:
            print(f"    {nm:6} n={v.size:4d}  median {np.median(v):5.1f} mm")

    print("\n  vacuum phenomenon (gas in the disc; rises steeply with age):")
    for _, _, nm in PAIRS:
        v = np.array([x[f"vacuum_frac_{nm}"] for x in ok
                      if isinstance(x.get(f"vacuum_frac_{nm}"), (int, float))], float)
        if v.size > 50:
            pos = int((v > 0.02).sum())
            print(f"    {nm:6} n={v.size:4d}  {pos:4d} with gas ({100 * pos / v.size:.1f}%)")

    dish = [x for x in ok if x.get("dish_resnick")]
    withb = [x for x in ok if (x.get("max_contiguous_bridges") or 0) >= 1]
    print(f"\n  bridging: {len(withb)}/{len(ok)} have at least one bridged level "
          f"({100 * len(withb) / max(1, len(ok)):.1f}%)")
    print(f"  DISH by Resnick (>=4 contiguous vertebrae): {len(dish)}/{len(ok)} = "
          f"{100 * len(dish) / max(1, len(ok)):.1f}%   [CT series report 3.8-27%]")

    print("\n  osteophyte index (endplate width / mid-body width; 1.0 = no flare):")
    for nm in LUMBAR.values():
        v = np.array([x[f"osteophyte_index_{nm}"] for x in ok
                      if isinstance(x.get(f"osteophyte_index_{nm}"), (int, float))], float)
        if v.size > 50:
            print(f"    {nm:4} n={v.size:4d}  median {np.median(v):.3f}  "
                  f"p95 {np.percentile(v, 95):.3f}")

    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
