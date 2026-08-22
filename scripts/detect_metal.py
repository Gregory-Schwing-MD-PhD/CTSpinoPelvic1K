"""scripts/detect_metal.py — find and classify every metal implant in the corpus.

WHY, AND WHAT IT IS FOR. Two things this corpus cannot currently do:

  TEMPLATING. Implant sizing is done against real anatomy with a real implant in it, and
  the only examples of that are the cases that already carry hardware. Nobody knows which
  those are: the hardware classes (76-79) are declared and unpopulated, and the one case
  known to carry an interbody cage, 0068, was found by hand.

  POST-OPERATIVE SYNTHESIS. ostk.surgery.simulate_correction generates a post-op spine
  from a pre-op one. Synthetic post-op scans are only worth what they can be checked
  against, and a real instrumented spine is the check. A stratified list of them is the
  prerequisite for that comparison, not a nice-to-have after it.

HOW METAL IS FOUND, AND WHY NOT JUST A THRESHOLD. Metal sits far above bone in
attenuation -- cortical bone reaches perhaps 1900 HU on a generous scanner, surgical alloys
run past 3000 and clip at the reconstruction ceiling. So a high threshold finds it. But a
threshold alone also finds dense contrast in a vessel, a calcified plaque, and the
occasional very dense cortical rim, and it says nothing about what the metal IS.

Two things separate implants from those:

  STREAK. Beam hardening throws dark streaks off metal, so a genuine implant is
  surrounded by voxels well BELOW soft tissue that are not air -- a signature contrast and
  calcium do not produce. The dark halo is measured in a shell around each candidate and
  is the strongest single discriminator available without reading the image.

  SIZE AND SHAPE. A vascular clip is a few voxels. A pedicle screw is a centimetre-scale
  rod. An acetabular cup is a shell. Volume and elongation separate them, and both are
  cheap.

STRATIFICATION IS BY WHAT THE SURGERY WAS, which is what a templating study needs. Each
metal component is assigned to the labelled anatomy it sits in or beside, and the pattern
across components names the construct: hardware spanning several vertebrae posteriorly is
an instrumented fusion, hardware inside a disc space is an interbody device, hardware at
the femoral head with a matching acetabular component is an arthroplasty. Where the pattern
is not one of those, the case is reported as UNCLASSIFIED rather than forced into a
category -- an implant nobody can name is still an implant worth having on the list.

    python scripts/detect_metal.py --labels data/v5_final --ct data/hf_export/ct \\
        --workers 12 --out morphometrics
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

# Attenuation. Cortical bone tops out around 1900 HU; surgical alloys run well past 3000
# and are clipped by the reconstruction. 2200 sits above any bone this corpus contains and
# below any implant, and the margin is reported per case so the choice can be argued with.
METAL_HU = 2200.0
# A dark streak is below fat and above air: contrast and calcium do not cast one.
STREAK_HU = -250.0
AIR_HU = -800.0

MIN_VOX = 20                 # below this it is reconstruction noise, not an object
CLIP_VOX = 60                # below this it is a surgical clip or a staple, not an implant

VERT = {**{i: f"T{i - 7}" for i in range(8, 20)},
        **{i: f"L{i - 19}" for i in range(20, 26)}}
SACRUM, S1 = 26, 29
HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33


def _shell(mask, iters=3):
    """A shell just outside a mask, where the streak lives."""
    return ndimage.binary_dilation(mask, iterations=iters) & ~mask


def one(args) -> dict:
    lab_path, ct_path = args
    stem = Path(lab_path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        li = nib.load(lab_path)
        ci = nib.load(ct_path)
        lab = np.asanyarray(li.dataobj).astype(np.int16)
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
        sp = np.array(li.header.get_zooms()[:3], float)
    except Exception as exc:
        return {"case": stem, "error": type(exc).__name__}

    if ct.shape != lab.shape:
        return {"case": stem, "error": "shape_mismatch"}

    vox_mm3 = float(np.prod(sp))
    r["ct_max_hu"] = round(float(ct.max()), 1)

    hot = ct >= METAL_HU
    r["metal_voxels"] = int(hot.sum())
    if not hot.any():
        r["has_metal"] = 0
        r["construct"] = "none"
        return r

    cc, n = ndimage.label(hot)
    comps = []
    for i in range(1, n + 1):
        m = cc == i
        v = int(m.sum())
        if v < MIN_VOX:
            continue
        idx = np.argwhere(m)
        # streak: how dark does it get just outside, without being air
        sh = _shell(m)
        sv = ct[sh]
        dark = float(np.mean((sv < STREAK_HU) & (sv > AIR_HU))) if sv.size else 0.0
        ext = (idx.max(0) - idx.min(0) + 1) * sp
        elong = float(ext.max() / max(ext.min(), 1e-6))
        # which labelled structure is it in or beside
        near = lab[ndimage.binary_dilation(m, iterations=4)]
        near = near[near > 0]
        host = int(Counter(near.tolist()).most_common(1)[0][0]) if near.size else 0
        comps.append({"vox": v, "mm3": round(v * vox_mm3, 1), "dark": round(dark, 3),
                      "elong": round(elong, 2), "host": host,
                      "len_mm": round(float(ext.max()), 1)})

    comps = [c for c in comps if c["vox"] >= MIN_VOX]
    if not comps:
        r["has_metal"] = 0
        r["construct"] = "none"
        return r

    # An implant casts a streak. Without one, a bright blob is dense contrast, a calcified
    # plaque, or a very dense cortical rim -- all of which belong in the count of what was
    # found, and none of which belong in a templating series.
    implants = [c for c in comps if c["dark"] >= 0.15 and c["vox"] >= CLIP_VOX]
    small = [c for c in comps if c not in implants]

    r["has_metal"] = 1 if implants else 0
    r["n_metal_components"] = len(comps)
    r["n_implant_components"] = len(implants)
    r["n_small_dense_components"] = len(small)
    r["metal_mm3"] = round(sum(c["mm3"] for c in comps), 1)
    r["implant_mm3"] = round(sum(c["mm3"] for c in implants), 1)
    r["max_streak_fraction"] = round(max(c["dark"] for c in comps), 3)
    r["components"] = json.dumps(comps[:12])

    if not implants:
        r["construct"] = "dense_no_streak"
        return r

    hosts = [c["host"] for c in implants]
    verts = sorted({h for h in hosts if h in VERT})
    pelvic = [h for h in hosts if h in (HIP_L, HIP_R, FEM_L, FEM_R)]
    r["levels_involved"] = ",".join(VERT[v] for v in verts) if verts else ""
    r["n_levels_involved"] = len(verts)
    r["hosts"] = ",".join(str(h) for h in sorted(set(hosts)))

    # name the construct from the pattern, and refuse to name it when the pattern is not
    # one of these -- an implant nobody can classify is still an implant worth listing
    long_rods = [c for c in implants if c["len_mm"] >= 25 and c["elong"] >= 3.0]
    if pelvic and any(h in (FEM_L, FEM_R) for h in hosts):
        r["construct"] = ("hip_arthroplasty_bilateral"
                          if {FEM_L, FEM_R} <= set(hosts) else "hip_arthroplasty")
    elif len(verts) >= 2 and long_rods:
        r["construct"] = "spinal_instrumentation"
    elif len(verts) >= 1 and not long_rods:
        r["construct"] = "vertebral_implant"          # interbody device, cement, anchor
    elif pelvic:
        r["construct"] = "pelvic_implant"
    else:
        r["construct"] = "unclassified"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export/ct")
    ap.add_argument("--workers", type=int, default=12)
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

    cols = []
    for x in res:
        for k in x:
            if k not in cols:
                cols.append(k)
    outp = Path(a.out) / "metal.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(res)

    have = [x for x in res if x.get("has_metal")]
    print(f"\n  {len(have)} of {len(res)} case(s) carry an implant "
          f"({100 * len(have) / max(1, len(res)):.1f}%)\n")
    for k, v in Counter(x.get("construct", "?") for x in res).most_common():
        print(f"    {k:32s} {v:4d}")

    dense = [x for x in res if x.get("construct") == "dense_no_streak"]
    print(f"\n  {len(dense)} case(s) have bright voxels with no streak around them -- "
          f"contrast,")
    print("  calcified plaque or a dense cortical rim, NOT an implant. They are counted")
    print("  and named rather than dropped, because a threshold alone cannot tell them")
    print("  apart and the reader should know the difference was drawn.")
    print(f"\n  wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
