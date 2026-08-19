"""scripts/sweep_foramina_params.py — can ANY parameter setting make the foramina count work?

THE QUESTION. A sacrum has four pairs of anterior foramina; assimilate L5 and it gains one,
fail to fuse S1 and it loses one. So the target is normals at 4, lumbarizations at 3,
sacralizations at 5. Two hand-tuned attempts missed badly -- sacralizations came back
3 to 6, lumbarizations 2 to 6, the two distributions sitting on top of each other.

Rather than keep adjusting thresholds one at a time and stopping at whichever setting
flatters the result, this maps the whole parameter space and reports what each setting
does to all three groups at once. If a separating setting exists it will show up; if the
groups overlap everywhere, that is a definitive negative and worth far more than another
round of tuning.

THE FOUR THINGS WORTH VARYING, and what each is trading:

    slab_mm      projection depth. Thin keeps oblique canals open; thick is less noisy
                 but fills them in from the bone in front and behind.
    close_mm     morphological closing radius. Recovers foramina that open at the lateral
                 margin, which fill_holes alone can never see as enclosed; too much and
                 it manufactures holes over cortex. 0 disables it.
    min_area     size floor. The distal S3/S4 canals are genuinely small, so a high floor
                 loses them and a low one admits noise.
    max_dark_hu  the CT arbitration. A real foramen is dark; a mask artefact sits over
                 bone and is bright. Set to a huge value to DISABLE the CT check entirely,
                 which is how the mask-only baseline gets measured on the same footing.

EFFICIENCY. Hole extraction is the expensive step and depends only on (slab_mm, close_mm),
while area and HU are pure filters on the result. So each case is read once, holes are
extracted once per geometry, and every (min_area, max_dark) pair is then counted for free.

OVERFITTING IS THE REAL RISK. Twelve cases and four knobs will always yield some setting
that agrees. So this runs on every LSTV-labelled case plus a large normal sample, reports
per-group distributions rather than a single score, and prints how many cases each
"winning" setting actually gets right -- a setting that separates group MEDIANS while
misclassifying half the individuals is not a screen.

    python scripts/sweep_foramina_params.py --labels data/v5_final --ct data/hf_export_v4/ct \\
        --manifest data/hf_export_v4/manifest.json --n-normal 60 --workers 24 --out morphometrics
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

SACRUM, S1 = 26, 29

SLAB_MM = (4.0, 8.0, 12.0)
CLOSE_MM = (0.0, 3.0, 6.0)
MIN_AREA = (4.0, 10.0, 25.0)
MAX_DARK = (100.0, 200.0, 1e9)          # 1e9 == CT check disabled (mask-only baseline)
MAX_AREA_MM2 = 800.0

TARGET = {"normal": 4, "LUMBARIZATION": 3, "SACRALIZATION": 5}


def extract_holes(sac, ct, spacing, slab_mm, close_mm):
    """[(area_mm2, median_hu, side)] over every slab, for one geometry setting."""
    ys = np.nonzero(sac.any(axis=(0, 2)))[0]
    if not len(ys):
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    slab = max(2, int(round(slab_mm / max(spacing[1], 1e-6))))
    step = max(1, slab // 3)
    px_mm2 = float(spacing[0] * spacing[2])
    mid = float(np.average(np.arange(sac.shape[0]), weights=sac.sum(axis=(1, 2))))
    se = None
    if close_mm > 0:
        se = np.ones((max(3, int(round(close_mm / spacing[0]))) | 1,
                      max(3, int(round(close_mm / spacing[2]))) | 1), bool)

    out = []
    for yy in range(y0, max(y0 + 1, y1 - slab), step):
        proj = sac[:, yy:yy + slab].max(axis=1)
        if proj.sum() < 200:
            continue
        base = ndimage.binary_closing(proj, structure=se) if se is not None else proj
        holes = ndimage.binary_fill_holes(base) & ~proj
        if not holes.any():
            continue
        lbl, n = ndimage.label(holes)
        ctb = ct[:, yy:yy + slab]
        slabholes = []
        for i in range(1, n + 1):
            m = lbl == i
            area = float(m.sum()) * px_mm2
            if area > MAX_AREA_MM2 or area < min(MIN_AREA):
                continue
            idx = np.argwhere(m)
            vals = ctb[idx[:, 0], :, idx[:, 1]].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            side = "right" if float(idx[:, 0].mean()) > mid else "left"
            slabholes.append((area, float(np.median(vals)), side))
        if slabholes:
            out.append(slabholes)
    return out


def count_for(slabs, min_area, max_dark):
    """Best slab's (left, right) under one filter setting."""
    best = (0, 0)
    for holes in slabs:
        l = r = 0
        for area, hu, side in holes:
            if area < min_area or hu > max_dark:
                continue
            if side == "right":
                r += 1
            else:
                l += 1
        if l + r > sum(best):
            best = (l, r)
    return best


def one(args) -> dict:
    lab_path, ct_path, label = args
    stem = Path(lab_path).name.replace("_label.nii.gz", "")
    try:
        limg = nib.as_closest_canonical(nib.load(lab_path))
        lab = np.asanyarray(limg.dataobj).astype(np.int16)
        sp = np.array(limg.header.get_zooms()[:3], float)
        cimg = nib.as_closest_canonical(nib.load(ct_path))
        ct = np.asanyarray(cimg.dataobj).astype(np.float32)
    except Exception as exc:                                       # noqa: BLE001
        return {"case": stem, "error": str(exc)[:120]}
    if ct.shape != lab.shape:
        return {"case": stem, "error": "shape mismatch"}
    sac = np.isin(lab, [SACRUM, S1])
    if sac.sum() < 5000:
        return {"case": stem, "error": "no sacrum"}

    rows = []
    for slab_mm in SLAB_MM:
        for close_mm in CLOSE_MM:
            slabs = extract_holes(sac, ct, sp, slab_mm, close_mm)
            for min_area in MIN_AREA:
                for max_dark in MAX_DARK:
                    l, r = count_for(slabs, min_area, max_dark)
                    rows.append({"case": stem, "label": label,
                                 "slab_mm": slab_mm, "close_mm": close_mm,
                                 "min_area": min_area, "max_dark": max_dark,
                                 "left": l, "right": r, "pairs": max(l, r),
                                 "disagree": abs(l - r)})
    return {"case": stem, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--n-normal", type=int, default=60)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    recs = json.load(open(a.manifest))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    lab_of = {}
    for rec in recs:
        s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
        if s:
            lab_of[s] = str(rec.get("lstv_label") or "normal")

    lab_dir, ct_dir = Path(a.labels), Path(a.ct)
    have = sorted(p.name.replace("_label.nii.gz", "") for p in lab_dir.glob("*_label.nii.gz"))
    lstv = [s for s in have if lab_of.get(s, "normal") != "normal"]
    normal = [s for s in have if lab_of.get(s, "normal") == "normal"][:a.n_normal]
    stems = lstv + normal
    jobs = [(str(lab_dir / f"{s}_label.nii.gz"), str(ct_dir / f"{s}_ct.nii.gz"),
             lab_of.get(s, "normal"))
            for s in stems if (ct_dir / f"{s}_ct.nii.gz").exists()]
    print(f"{len(jobs)} cases ({len(lstv)} LSTV-labelled, {len(normal)} normal sample)")
    print(f"{len(SLAB_MM)*len(CLOSE_MM)*len(MIN_AREA)*len(MAX_DARK)} settings each\n",
          flush=True)

    rows, bad = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(one, jobs, chunksize=1), 1):
            if "error" in res:
                bad.append(res)
            else:
                rows.extend(res["rows"])
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "foramina_param_sweep.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {p}  ({len(rows)} rows, {len(bad)} cases failed)")

    # ---- which setting, if any, puts each group on its target? -----------------
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["slab_mm"], r["close_mm"], r["min_area"], r["max_dark"])
        grp = r["label"] if r["label"] in TARGET else "SACRALIZATION" \
            if "SACRAL" in r["label"] else r["label"]
        by[key][grp].append(r["pairs"])

    scored = []
    for key, g in by.items():
        if not all(k in g for k in ("normal", "LUMBARIZATION", "SACRALIZATION")):
            continue
        med = {k: float(np.median(v)) for k, v in g.items()}
        # how many INDIVIDUAL cases land on their group's target -- the number that
        # matters, because separating medians while misclassifying half is not a screen
        hits = sum(1 for k, v in g.items() for x in v if x == TARGET[k])
        tot = sum(len(v) for v in g.values())
        med_err = sum(abs(med[k] - TARGET[k]) for k in TARGET)
        scored.append((med_err, -hits / tot, key, med, hits, tot))
    scored.sort()

    print("\n  BEST SETTINGS BY GROUP-MEDIAN ERROR")
    print("  (target: normal 4 · lumbarization 3 · sacralization 5)\n")
    print("  slab close minA maxHU |  norm  lumb  sacr  | med.err | per-case correct")
    for med_err, negacc, key, med, hits, tot in scored[:12]:
        s, c, mn, md = key
        md_s = "off" if md > 1e8 else f"{md:.0f}"
        print(f"  {s:4.0f} {c:5.0f} {mn:4.0f} {md_s:>5} |"
              f" {med['normal']:5.1f} {med['LUMBARIZATION']:5.1f} {med['SACRALIZATION']:5.1f} |"
              f" {med_err:7.1f} | {hits:4d}/{tot:<4d} {100*hits/tot:5.1f}%")

    print("\n  SPREAD AT THE BEST SETTING (is the count even stable?)")
    if scored:
        key = scored[0][2]
        for grp in ("normal", "LUMBARIZATION", "SACRALIZATION"):
            v = np.array(by[key][grp])
            print(f"    {grp:16s} n={len(v):3d}  median {np.median(v):.1f}  "
                  f"range {v.min():.0f}-{v.max():.0f}  "
                  f"distribution {dict(sorted(Counter(v.tolist()).items()))}")
    if bad:
        print(f"\n  failed: {[b['case'] for b in bad][:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
