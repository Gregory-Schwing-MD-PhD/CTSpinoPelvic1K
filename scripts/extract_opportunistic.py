"""scripts/extract_opportunistic.py — what a CT taken for another reason still tells you.

Every scan here was acquired to look for colorectal polyps. It also contains, at no extra
dose and no extra cost, measurements that predict fracture, frailty and cardiovascular
events. That is the premise of opportunistic CT screening, and the foundational work for
it was done on exactly this kind of examination -- Pickhardt and colleagues established
the technique on CT colonography and later published normative values in more than
20,000 adults.

WHAT IS MEASURED, AND WHY EACH ONE

  l1_trabecular_hu     Mean attenuation of the trabecular core of L1. The single most
                       validated opportunistic measure: below about 110 HU is over 90%
                       specific for osteoporosis, above about 160 HU is roughly 90%
                       sensitive against it, and the population mean falls about 2.5 HU
                       per YEAR of age -- from ~226 HU under 30 to ~89 HU at 90.
                       (Pickhardt 2013 Ann Intern Med; Jang 2019 Radiology 291:360.)

  psoas_area_mm2       Cross-sectional psoas area at the mid-L3 body, and its mean
  psoas_hu             attenuation. Area indexes sarcopenia; attenuation indexes fat
                       infiltration of muscle, which carries risk independently of size.

  aortic_calc_*        Calcified plaque within the aorta label, by the conventional
                       130 HU threshold. A marker of cardiovascular risk that is simply
                       present in the image whether or not anyone looks.

THE TRABECULAR ROI IS THE WHOLE DIFFICULTY. Cortical bone runs 300-2000 HU and trabecular
bone 50-250, so a single cortical voxel in the region of interest drags the mean up and
turns an osteoporotic vertebra into a normal one. The published method places an elliptical
ROI by hand, inside the cortex, avoiding the basivertebral vein. Here the body is already
segmented, so the ROI can be derived: take the vertebral BODY (anterior to the spinal
canal, so the posterior elements are excluded), erode it by 3 mm to leave the cortex
behind, restrict to the middle of the body in height, and then trim the top and bottom
deciles of the remaining attenuation to remove the basivertebral vein and any residual
cortex. Every step is there because of a specific way the number goes wrong.

    python scripts/extract_opportunistic.py --labels data/v5_final --ct data/hf_export/ct
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
import label_scheme as LS                                          # noqa: E402

L1, L2, L3, L4 = 20, 21, 22, 23
MIN_VOX = 3000


def _canal_front(mask):
    """Anterior wall of the spinal canal in voxel units, or None."""
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


def trabecular_roi(vert_mask, sp):
    """The trabecular core of a vertebral body, with cortex and vein excluded.

    Erosion is in MILLIMETRES, not voxels: these scans differ in slice thickness, and a
    fixed voxel erosion would strip a different physical depth on each one, which would
    put a spacing-dependent bias straight into the attenuation.
    """
    front = _canal_front(vert_mask)
    body = np.zeros_like(vert_mask)
    if front is None:
        ys = np.nonzero(vert_mask.any(axis=(0, 2)))[0]
        if len(ys) < 4:
            return None
        front = ys.min() + 0.45 * (ys.max() - ys.min())
    f = int(np.ceil(front))
    body[:, f:, :] = vert_mask[:, f:, :]
    if body.sum() < 500:
        return None

    # 3 mm in from every surface leaves the cortical shell outside the ROI
    d = ndimage.distance_transform_edt(body, sampling=sp)
    core = d >= 3.0
    if core.sum() < 200:
        core = d >= 2.0
    if core.sum() < 100:
        return None

    # the middle of the body by height: the endplates are dense and are not trabecular
    zs = np.nonzero(core.any(axis=(0, 1)))[0]
    if len(zs) >= 5:
        lo, hi = np.percentile(zs, [25, 75])
        keep = np.zeros_like(core)
        keep[:, :, int(lo):int(hi) + 1] = core[:, :, int(lo):int(hi) + 1]
        if keep.sum() >= 100:
            core = keep
    return core


def one(args) -> dict:
    lab_p, ct_p = args
    stem = Path(lab_p).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        # CANONICAL, AND FOR A REASON. Everything below reasons about anterior and
        # superior -- where the canal is, which half is the body, the middle of the body
        # by height. These volumes are ('P','I','R'), so axis 1 runs INFERIOR and axis 2
        # runs RIGHT, and the first version of this cut the "body" along the wrong axis
        # entirely. The ROI it produced was an arbitrary slab carrying cortex and
        # endplate, which read 264 HU at L1 where a cohort this age should sit near 155.
        #
        # This is a READ path deriving numbers, not a write-back of labels, so
        # canonicalising is safe here -- unlike the renumbering pass, where it corrupted
        # a file. Both volumes go through it so they stay registered to each other.
        li = nib.as_closest_canonical(nib.load(lab_p))
        ci = nib.as_closest_canonical(nib.load(ct_p))
        if li.shape != ci.shape:
            return {"case": stem, "error": "shape mismatch"}
        lab = np.asanyarray(li.dataobj)
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
        sp = np.asarray(li.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    # ---- vertebral trabecular attenuation ----------------------------------------
    for vid, name in ((L1, "l1"), (L2, "l2"), (L3, "l3"), (L4, "l4")):
        m = lab == vid
        if m.sum() < MIN_VOX:
            continue
        core = trabecular_roi(m, sp)
        if core is None or core.sum() < 100:
            continue
        v = ct[core]
        # trim both tails: the top removes any cortex the erosion missed, the bottom
        # removes the basivertebral vein, which is the classic contaminant of this ROI
        lo, hi = np.percentile(v, [10, 90])
        v = v[(v >= lo) & (v <= hi)]
        if v.size < 50:
            continue
        r[f"{name}_trabecular_hu"] = round(float(v.mean()), 1)
        r[f"{name}_roi_voxels"] = int(core.sum())

    # Psoas area and aortic calcification once had blocks here that read soft-tissue ids
    # 60/61 and 66. The scheme is bone and hardware only and those ids are a retired gap
    # (label_scheme.RETIRED_IDS); a soft-tissue measure needs its own segmentation, not a
    # label lookup that can never hit.
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
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    # attach age and sex, which is the whole point: these measures are only interpretable
    # against them
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
    p = out / "opportunistic.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    print(f"\n  {len(ok)} of {len(res)} measured\n")

    def summarise(key, unit="", lo=None, hi=None):
        v = np.array([x[key] for x in ok if isinstance(x.get(key), (int, float))], float)
        if v.size < 5:
            print(f"  {key:22s} n={v.size} (too few)")
            return None
        med = float(np.median(v))
        flag = ""
        if lo is not None and not (lo <= med <= hi):
            flag = f"   <-- IMPLAUSIBLE (expected {lo}-{hi})"
        print(f"  {key:22s} n={v.size:4d}  median {med:7.1f}{unit}  "
              f"IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}{flag}")
        return med

    # L1 trabecular attenuation in a screening cohort with a median age near 60 should
    # land roughly between 120 and 200 HU on the published age curve. Outside that, the
    # ROI is picking up cortex or missing the body.
    summarise("l1_trabecular_hu", " HU", 110, 210)
    for k in ("l2_trabecular_hu", "l3_trabecular_hu", "l4_trabecular_hu"):
        summarise(k, " HU")
    summarise("psoas_area_mm2", " mm2", 800, 3500)
    summarise("psoas_hu", " HU", 25, 70)
    summarise("aortic_calc_frac", "")

    # the age relationship is the check that matters: attenuation must FALL with age, and
    # the published slope is about -2.5 HU per year
    xs = [(x.get("age"), x.get("l1_trabecular_hu")) for x in ok]
    xs = [(float(a_), float(b)) for a_, b in xs
          if a_ not in (None, "") and isinstance(b, (int, float))]
    if len(xs) > 50:
        ax = np.array([t[0] for t in xs]); ay = np.array([t[1] for t in xs])
        slope = float(np.polyfit(ax, ay, 1)[0])
        print(f"\n  L1 attenuation vs age: {slope:+.2f} HU per year "
              f"(published ~-2.5), n={len(xs)}")
        lowbone = int((ay < 110).sum())
        print(f"  below the 110 HU osteoporosis-specific threshold: "
              f"{lowbone}/{len(ay)} = {100 * lowbone / len(ay):.1f}%")

    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
