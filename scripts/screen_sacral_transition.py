"""scripts/screen_sacral_transition.py — screen for sacralization / lumbarization by counting
sacral foramina, with the CT as the arbiter.

THE SCREEN. A sacrum has FOUR pairs of anterior foramina because it has five fused
segments. Assimilate L5 and it gains one: FIVE. Fail to fuse S1 and it loses one: THREE.
So a single count separates both directions of transition, and -- unlike counting vertebrae
up from the sacrum -- it does not depend on any vertebra label being right, which is
exactly the assumption that breaks when a transition is what you are looking for.

    3 pairs -> possible LUMBARIZATION      4 pairs -> normal      5 pairs -> possible SACRALIZATION

WHY THE CT AND NOT THE MASK. A mask-only version of this shipped and failed. On one
symmetric sacrum it returned 3 on the left and 7 on the right: bone masks close the small
S3/S4 canals, and the morphological closing used to recover the laterally-open ones
invents holes elsewhere. Geometry alone cannot tell a real canal from a dent in a label.

The CT can. A foramen is a hole THROUGH bone, so it is dark; a mask artefact sits over
bone, which is bright. So the mask proposes and the CT disposes:

    1. per coronal slab, fill the sacrum silhouette and subtract it -> CANDIDATE holes
    2. keep a candidate only if the CT inside it is genuinely not bone
    3. count what survives, per side

That inversion is what lets the size floor drop low enough to catch the small distal
canals without the noise: a false positive now has to be dark in the CT as well as
hole-shaped in the mask, and a closing artefact sitting on the sacral cortex is not.

CONFIDENCE, NOT JUST A COUNT. The two sides are independent measurements of the same
number, so their agreement is the natural confidence signal and is reported rather than
averaged away. A case where the sides disagree is one where the count should not be
trusted, and saying so is more useful than a tidy answer.

    python scripts/screen_sacral_transition.py --labels data/v5_final \\
        --ct data/hf_export_v4/ct --manifest data/hf_export_v4/manifest.json \\
        --workers 24 --out morphometrics
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

SACRUM, S1 = 26, 29

# A foramen is filled with fat, vessel and nerve; cortical bone is many hundreds of HU
# above any of them. The gap is wide, so the threshold does not need to be delicate --
# it needs to be clearly below bone and clearly above air.
BONE_HU = 200.0          # a candidate whose interior reaches this is bone, not a hole
MAX_DARK_HU = 150.0      # and its typical voxel must sit below this
MIN_AREA_MM2 = 6.0       # the distal canals are small; the CT check earns us this floor
MAX_AREA_MM2 = 800.0
SLAB_MM = 8.0            # thinner than the canals' obliquity, or projection closes them


def _label_slabs(sac, ct, spacing):
    """Yield (n_left, n_right, areas) per coronal slab, CT-verified."""
    ys = np.nonzero(sac.any(axis=(0, 2)))[0]
    if not len(ys):
        return
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    slab = max(2, int(round(SLAB_MM / max(spacing[1], 1e-6))))
    step = max(1, slab // 3)
    px_mm2 = float(spacing[0] * spacing[2])
    mid = float(np.average(np.arange(sac.shape[0]), weights=sac.sum(axis=(1, 2))))
    se = np.ones((max(3, int(round(4.0 / spacing[0]))) | 1,
                  max(3, int(round(4.0 / spacing[2]))) | 1), bool)

    for yy in range(y0, max(y0 + 1, y1 - slab), step):
        block = sac[:, yy:yy + slab]
        proj = block.max(axis=1)
        if proj.sum() < 200:
            continue
        # RECALL, then precision. A foramen at the lateral margin opens outward, so
        # fill_holes alone never sees it as enclosed and the count comes back one short --
        # which is exactly what the first run did, landing 7 of 12 cases on "3 pairs".
        # Closing turns that notch into a hole so it can be proposed at all; the CT check
        # below is what makes proposing aggressively safe, and holes are subtracted from
        # the ORIGINAL mask so closing can never manufacture bone.
        closed = ndimage.binary_closing(proj, structure=se)
        holes = ndimage.binary_fill_holes(closed) & ~proj
        if not holes.any():
            continue
        lbl, n = ndimage.label(holes)
        ctb = ct[:, yy:yy + slab]
        left = right = 0
        areas = []
        for i in range(1, n + 1):
            m = lbl == i
            area = float(m.sum()) * px_mm2
            if not (MIN_AREA_MM2 <= area <= MAX_AREA_MM2):
                continue
            # THE ARBITRATION: what does the CT say is inside this hole?
            # `m` is 2D (x, z) from the projection while ctb is 3D (x, slab, z), so the
            # footprint has to be gathered through the slab depth rather than indexed
            # directly -- and every voxel it gathers is one the mask left empty, which is
            # exactly the population the question is about.
            idx = np.argwhere(m)
            vals = ctb[idx[:, 0], :, idx[:, 1]].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if float(np.median(vals)) > MAX_DARK_HU or float(np.percentile(vals, 25)) > BONE_HU:
                continue                      # bright inside -> bone -> a mask artefact
            areas.append(area)
            if float(np.argwhere(m)[:, 0].mean()) > mid:
                right += 1
            else:
                left += 1
        if left or right:
            yield left, right, areas


def call_transition(pairs):
    """pairs -> screening verdict. Deliberately blunt: this is a screen, not a diagnosis."""
    if pairs is None:
        return "unscreenable"
    if pairs <= 2:
        return "unscreenable"
    if pairs == 3:
        return "possible_lumbarization"
    if pairs == 4:
        return "normal"
    return "possible_sacralization"


def one(args) -> dict:
    lab_path, ct_path = args
    stem = Path(lab_path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        limg = nib.as_closest_canonical(nib.load(lab_path))
        lab = np.asanyarray(limg.dataobj).astype(np.int16)
        sp = np.array(limg.header.get_zooms()[:3], float)
        cimg = nib.as_closest_canonical(nib.load(ct_path))
        ct = np.asanyarray(cimg.dataobj).astype(np.float32)
    except Exception as exc:                                       # noqa: BLE001
        return {"case": stem, "error": f"{type(exc).__name__}: {exc}"}

    if ct.shape != lab.shape:
        return {"case": stem, "error": f"shape mismatch ct{ct.shape} label{lab.shape}"}

    sac = np.isin(lab, [SACRUM, S1])
    if sac.sum() < 5000:
        r.update(foramina_pairs=None, screen="unscreenable",
                 note="no sacrum")
        return r

    best_l = best_r = 0
    areas_at_best = []
    for left, right, areas in _label_slabs(sac, ct, sp):
        if left + right > best_l + best_r:
            best_l, best_r, areas_at_best = left, right, areas

    # the two sides measure the same number twice; report both and their agreement
    pairs = max(best_l, best_r) if (best_l or best_r) else None
    r["foramina_left"] = best_l
    r["foramina_right"] = best_r
    r["foramina_pairs"] = pairs
    r["side_disagreement"] = abs(best_l - best_r)
    r["foramen_area_mm2"] = round(float(np.median(areas_at_best)), 1) if areas_at_best else None
    r["screen"] = call_transition(pairs)
    r["confident"] = int(r["side_disagreement"] <= 1 and r["screen"] != "unscreenable")

    zs = np.nonzero(sac.any(axis=(0, 1)))[0]
    xs = np.nonzero(sac.any(axis=(1, 2)))[0]
    r["sacrum_height_mm"] = round(len(zs) * sp[2], 1)
    r["sacrum_width_mm"] = round(len(xs) * sp[0], 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    lab_dir, ct_dir = Path(a.labels), Path(a.ct)
    if a.cases:
        stems = [c.strip() for c in a.cases.split(",") if c.strip()]
    else:
        stems = sorted(p.name.replace("_label.nii.gz", "")
                       for p in lab_dir.glob("*_label.nii.gz"))
    if a.limit:
        stems = stems[:a.limit]
    jobs = [(str(lab_dir / f"{s}_label.nii.gz"), str(ct_dir / f"{s}_ct.nii.gz"))
            for s in stems
            if (lab_dir / f"{s}_label.nii.gz").exists()
            and (ct_dir / f"{s}_ct.nii.gz").exists()]
    print(f"{len(jobs)} case(s) with both CT and label\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=1), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    lut = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
            if s:
                lut[s] = {"lstv_label": rec.get("lstv_label"),
                          "lstv_agreement": rec.get("lstv_agreement"),
                          "has_l6": rec.get("has_l6")}
    for r in res:
        r.update(lut.get(r["case"], {}))

    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in ok for k in r}, key=lambda k: (k != "case", k))
    p = out / "sacral_transition_screen.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    from collections import Counter
    tally = Counter(r["screen"] for r in ok)
    conf = Counter(r["screen"] for r in ok if r.get("confident"))
    print(f"\n  {len(ok)} screened ({len(bad)} failed)")
    for k in ("normal", "possible_sacralization", "possible_lumbarization", "unscreenable"):
        print(f"    {k:26s} {tally.get(k,0):4d}   (confident {conf.get(k,0)})")

    # does the screen agree with the label where a label exists? the honest check
    lab_rows = [r for r in ok if r.get("lstv_label")]
    if lab_rows:
        print("\n  screen vs manifest label:")
        x = Counter((str(r["lstv_label"]), r["screen"]) for r in lab_rows)
        for (lbl, scr), n in sorted(x.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {lbl:20s} -> {scr:26s} {n:4d}")
    if bad:
        print(f"\n  failures: {[b['case'] for b in bad][:6]}")
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
