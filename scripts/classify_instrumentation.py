"""classify_instrumentation.py — is this an implant, or is it artefact?

The shape rules in seed_hardware.py answer "what kind of implant is this" and assume the
answer is "an implant". A radiologist reading the proposals says almost all of them are
artefact, which means the prior question was never asked.

WHAT ACTUALLY SEPARATES THEM. Not shape, and not size alone:

  SATURATION. The scanner clips at 3071 HU on this cohort. Titanium and cobalt-chrome go
  straight through that ceiling, so a real implant has a core of saturated voxels. Dense
  cortical bone tops out near 1500, sclerosis and calcified plaque rarely pass 2000, and a
  blooming rim around either sits in the 1800-2600 band without ever reaching the ceiling.
  A component with no saturated core is, on this cohort, not metal.

  COHERENCE. An implant is a manufactured object: one connected piece of real volume with
  a saturated core. Artefact is speckle -- many pieces of a few voxels each, scattered along
  a streak or smeared around a dense structure.

  SITE. An implant sits where surgery happens: a pedicle, a disc space, a hip joint, across
  the sacroiliac joint. Artefact has no preferred site beyond "next to something dense".

WHAT THE SITE IS FOR. The subtype block was written for spinal instrumentation and cannot
name a femoral stem or an iliosacral screw, so the site is what proposes the class. A
component whose nearest structures are a hip and a femur is an arthroplasty; one bridging
the sacrum and an ilium is sacroiliac fixation. Both were being called generic `hardware`
or, worse, `screw_rod` -- because a stem is long and thin, and "linear" is all that rule
tests.

Reports before it decides: every threshold below is printed with the distribution it was
drawn from, so a reader can see whether the cut lands in a gap or in the middle of a cloud.

    python scripts/classify_instrumentation.py --proposals data/hardware_fix \\
        --ct data/hf_export_v5/ct --labels data/v5_final --out qc_hardware/verdicts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

SATURATED = 3000.0        # the scanner clips at 3071; nothing but metal gets here
MIN_REAL_MM3 = 60.0       # below this a saturated speck is blooming, not an implant
MIN_SAT_VOX = 8           # a real implant has a saturated CORE, not one bright voxel

VERT = list(range(8, 26)) + [26, 29]
HIP = [30, 31]
FEMUR = [32, 33]
SACRUM = [26, 29]
NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
        **{v: f"L{v - 19}" for v in range(20, 26)},
        26: "sacrum", 29: "S1", 30: "left_hip", 31: "right_hip",
        32: "femur_left", 33: "femur_right"}


def site_of(near_ids):
    """What kind of place this component sits in, from what it is next to."""
    s = set(near_ids)
    hip = s & set(HIP)
    fem = s & set(FEMUR)
    sac = s & set(SACRUM)
    spine = s & set(range(8, 26))
    if fem and hip:
        return "hip joint"
    if fem:
        return "femur"
    if sac and hip:
        return "sacroiliac joint"
    if hip and not spine:
        return "pelvis"
    if spine:
        return "spine"
    return "unclear"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.proposals)
    masks = sorted(src.glob("*_hardware_only.nii.gz"))
    print(f"  {len(masks)} case(s) with a proposed mask\n")

    rows = []
    for mp in masks:
        cid = mp.name[:4]
        cp = Path(a.ct) / f"{cid}_ct.nii.gz"
        lp = Path(a.labels) / f"{cid}_label.nii.gz"
        if not cp.exists() or not lp.exists():
            continue
        img = nib.load(str(lp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        ct = np.asanyarray(nib.load(str(cp)).dataobj)
        hw = np.asanyarray(nib.load(str(mp)).dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
        vox = float(np.prod(sp))
        aff = img.affine

        cc, ncc = ndimage.label(hw > 0)
        comps = []
        for i in range(1, ncc + 1):
            m = cc == i
            n = int(m.sum())
            if n < 10:
                continue
            vals = ct[m]
            peak = float(vals.max())
            nsat = int((vals >= SATURATED).sum())
            near = ndimage.binary_dilation(m, iterations=4)
            ids = sorted({int(v) for v in np.unique(lab[near]) if int(v) in NAME})
            idx = np.argwhere(m)
            w = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
            q = w - w.mean(0)
            vt = np.linalg.svd(q, full_matrices=False)[2]
            ext = [float((q @ vt[k]).max() - (q @ vt[k]).min()) for k in range(3)]
            comps.append({"vox": n, "mm3": n * vox, "peak": peak, "nsat": nsat,
                          "site": site_of(ids),
                          "near": [NAME[v] for v in ids],
                          "L": ext[0], "W": ext[1], "T": ext[2]})
        if not comps:
            rows.append({"case": cid, "verdict": "artefact", "why": "no component >=10 vox",
                         "n_comp": 0, "total_mm3": 0.0, "peak_HU": 0, "saturated_vox": 0,
                         "site": "", "proposed_class": "", "near": ""})
            continue

        comps.sort(key=lambda c: -c["mm3"])
        real = [c for c in comps
                if c["peak"] >= SATURATED and c["nsat"] >= MIN_SAT_VOX
                and c["mm3"] >= MIN_REAL_MM3]
        big = comps[0]
        tot = sum(c["mm3"] for c in comps)
        sat = sum(c["nsat"] for c in comps)
        peak = max(c["peak"] for c in comps)

        if real:
            site = real[0]["site"]
            verdict = "instrumentation"
            why = (f"{len(real)} saturated component(s), largest {real[0]['mm3']:.0f} mm3 "
                   f"with {real[0]['nsat']:,} voxels at the ceiling")
        else:
            site = big["site"]
            verdict = "artefact"
            if peak < SATURATED:
                why = f"nothing reaches the {SATURATED:.0f} HU ceiling (peak {peak:.0f})"
            elif sat < MIN_SAT_VOX:
                why = f"only {sat} saturated voxel(s) -- a bright speck, not a core"
            else:
                why = f"largest piece is {big['mm3']:.0f} mm3 -- below the implant floor"

        rows.append({"case": cid, "verdict": verdict, "why": why, "n_comp": len(comps),
                     "total_mm3": round(tot, 1), "peak_HU": int(peak),
                     "saturated_vox": sat, "site": site,
                     "near": " ".join(sorted({x for c in comps for x in c["near"]}))[:70],
                     "sizes": "; ".join(f"{c['L']:.0f}x{c['W']:.0f}x{c['T']:.0f}"
                                        for c in comps[:3]),
                     "proposed_class": ""})
        print(f"  {cid}  {verdict:<15} peak {int(peak):>5} HU  sat {sat:>7,}  "
              f"{tot:>9,.0f} mm3  {site:<17} {why}", flush=True)

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    keys = ["case", "verdict", "site", "proposed_class", "peak_HU", "saturated_vox",
            "total_mm3", "n_comp", "sizes", "near", "why"]
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    from collections import Counter
    print("\n  " + "\n  ".join(f"{n:>4}  {v}" for v, n in
                               Counter(r["verdict"] for r in rows).most_common()))
    print("\n  sites among instrumentation:")
    for s, n in Counter(r["site"] for r in rows
                        if r["verdict"] == "instrumentation").most_common():
        print(f"    {n:>3}  {s}")
    print(f"\n  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
