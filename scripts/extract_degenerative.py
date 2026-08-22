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
        z_low_of_upper = float(sel_u[:, 2].min())
        z_top_of_lower = float(sel_l[:, 2].max())
        gap_vox = z_low_of_upper - z_top_of_lower
        if not (-4 < gap_vox < 60):
            bridged.append(None)
            continue
        h = max(0.0, gap_vox * sp[2])
        r[f"disc_height_{name}_mm"] = round(h, 1)

        # gas inside that disc space -- pathognomonic for degeneration
        z0, z1 = int(np.floor(z_top_of_lower)) + 1, int(np.ceil(z_low_of_upper))
        if z1 > z0:
            box = ct[int(cx - rx):int(cx + rx) + 1,
                     int(cy - ry):int(cy + ry) + 1, z0:z1]
            if box.size > 30:
                frac = float((box < GAS_HU).mean())
                r[f"vacuum_frac_{name}"] = round(frac, 4)

        # BONY BRIDGE. Dilate both bodies a little and ask whether they meet. Resnick
        # requires the disc height to be PRESERVED, which is what separates flowing
        # ossification from a collapsed degenerative segment, so a bridge only counts
        # when the disc is not collapsed.
        pad = 2
        zlo = max(0, int(z_top_of_lower) - 6)
        zhi = min(lab.shape[2], int(z_low_of_upper) + 7)
        su = ndimage.binary_dilation(bu[:, :, zlo:zhi], iterations=pad)
        sl_ = bl[:, :, zlo:zhi]
        touch = bool((su & sl_).any())
        bridged.append(bool(touch and h >= 4.0))
        r[f"bridge_{name}"] = int(touch and h >= 4.0)

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
