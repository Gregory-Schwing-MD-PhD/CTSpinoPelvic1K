"""scripts/extract_level_gradients.py — the lumbar spine changes shape as it descends.

Almost every dimension of a lumbar vertebra grows from L1 to L5, and it grows because the
load does: each level carries everything above it. That gradient is one of the most
reproducible findings in spinal morphometry, it is visible in any large series, and it
gives this dataset something a single-level measurement cannot -- a monotonic trend that
either reproduces or does not.

WHAT IS MEASURED, AND THE PUBLISHED GRADIENT IT SHOULD REPRODUCE

  endplate_width_mm     superior endplate, side to side. Published: ~41.8 mm at L1 rising
                        to ~50.7 mm at L5.
  body_height_mm        anterior body height. Published: ~29.9 to 34.5 mm, L1 to L5.
  canal_width_mm        transverse canal diameter. Published: ~22.0 mm at L1 to ~26.5 at
                        L5 -- the canal widens caudally even as the cord has ended.
  tp_span_mm            tip to tip across the transverse processes. Published: ~68 mm at
                        L1 to ~86 mm at L5, the steepest gradient of the four.
  wedge_ratio           anterior body height over posterior. Near 1 in a healthy body;
                        it falls when a body wedges, which is what a compression fracture
                        does and what makes this measure worth having in a screening
                        cohort.

EVERY ONE IS MEASURED ON THE BODY, NOT THE WHOLE VERTEBRA. The labels are whole
vertebrae, and the posterior elements ruin all of these: the spinous process roughly
triples the anteroposterior extent and spans more height than the posterior body wall.
The spinal canal supplies the boundary without a threshold -- its anterior wall IS the
posterior wall of the body.

    python scripts/extract_level_gradients.py --labels data/v5_final
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
MIN_VOX = 3000


def _canal(mask):
    """(anterior wall y, transverse width in voxels) of the canal, or (None, None)."""
    zs = np.nonzero(mask.any(axis=(0, 1)))[0]
    if len(zs) < 5:
        return None, None
    fronts, widths = [], []
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
        xs = np.nonzero(big.any(axis=1))[0]
        widths.append(int(xs.max() - xs.min() + 1))
    if not fronts:
        return None, None
    return float(np.median(fronts)), float(np.median(widths))


def one(path: str) -> dict:
    stem = Path(path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        # canonical for a read path: everything below reasons about anterior and superior
        img = nib.as_closest_canonical(nib.load(path))
        lab = np.asanyarray(img.dataobj)
        sp = np.asarray(img.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    for vid, name in LUMBAR.items():
        m = lab == vid
        if m.sum() < MIN_VOX:
            continue
        idx = np.argwhere(m)

        # TRANSVERSE PROCESS SPAN: TIP TO TIP, and the tips are the measurement.
        # Percentile-trimming this one was an over-correction -- it cut the very
        # structures being measured and dropped the span from 72.8 to 51.0 mm at L1
        # against a published 68. The right-hand skew that prompted the trim is not
        # noise either: an enlarged transverse process reaching toward the ala IS the
        # transitional phenotype this dataset exists to capture, so a long right tail
        # is the signal.
        #
        # A stray voxel is still excluded, by taking the largest connected component
        # rather than by trimming the extremes.
        _cc, _n = ndimage.label(m)
        if _n > 1:
            _sz = ndimage.sum(m, _cc, range(1, _n + 1))
            _big = np.argwhere(_cc == int(np.argmax(_sz)) + 1)
        else:
            _big = idx
        r[f"tp_span_{name}_mm"] = round(
            float(_big[:, 0].max() - _big[:, 0].min()) * sp[0], 1)

        front, cw = _canal(m)
        if cw is not None:
            r[f"canal_width_{name}_mm"] = round(cw * sp[0], 1)
        if front is None:
            continue

        body = np.zeros_like(m)
        f = int(np.ceil(front))
        body[:, f:, :] = m[:, f:, :]
        if body.sum() < 500:
            continue
        bidx = np.argwhere(body)

        # IS WHAT SURVIVED THE CUT ACTUALLY A VERTEBRAL BODY?
        #
        # Everything below -- endplate width, both body heights, the wedge ratio -- is
        # computed on this mask, and all of it is meaningless if the cut landed in the
        # wrong place. It does, occasionally: `front` comes from the largest filled hole
        # in an axial slice, and when that hole is a trabecular void near the front of
        # the body rather than the spinal canal, `front` is placed almost at the
        # anterior margin and the mask reduces to a thin anterior SLIVER.
        #
        # The sliver is what produced the wedge ratios of 0.27 to 0.54 that looked like
        # severe compression fractures. It tapers, so its tallest column at the back is
        # taller than at the front, and the ratio of the two comes out low -- the exact
        # signature of a wedge, on a vertebra with nothing wrong with it. Rendering the
        # mask is what settled it; two rounds of reasoning from the numbers alone got it
        # wrong, first blaming posterior-element contamination and then a fracture.
        #
        # A vertebral body is a substantial part of a vertebra's front-to-back extent,
        # not a rind on it. Anything much thinner is the cut having failed.
        vy = idx[:, 1]
        by = bidx[:, 1]
        depth_mm = (by.max() - by.min() + 1) * sp[1]
        frac = (by.max() - by.min() + 1) / max(1, (vy.max() - vy.min() + 1))
        if depth_mm < 12.0 or frac < 0.30:
            r[f"body_cut_failed_{name}"] = 1
            continue

        # SUPERIOR ENDPLATE WIDTH, ISOLATED FROM THE TRANSVERSE PROCESSES.
        # Cutting at the anterior wall of the canal separates body from posterior
        # elements at L1-L4, but not at L5: the L5 transverse processes arise so far
        # forward that they survive the cut, and the width came back 67.5 mm instead of
        # about 51. So measure per axial slice, erode to snap the narrow isthmus that
        # joins a process to the body, keep the largest remaining piece -- which is the
        # body -- and add the eroded margin back.
        ztop = int(np.percentile(bidx[:, 2], 80))
        zmax = int(bidx[:, 2].max())
        widths = []
        ER = 2
        for z in range(ztop, zmax + 1):
            sl = body[:, :, z]
            if sl.sum() < 40:
                continue
            def largest(mask):
                cc, n = ndimage.label(mask)
                if n == 0:
                    return None
                sizes = ndimage.sum(mask, cc, range(1, n + 1))
                return cc == (int(np.argmax(sizes)) + 1)

            # Erosion snaps the isthmus joining a transverse process to the body, but on
            # a body that is small or clipped by the field of view it can eat almost
            # everything -- that read L1 at 7.8 mm on one case. So if the eroded core
            # keeps less than a third of the slice, the erosion did harm rather than
            # good and the slice is measured as it is.
            core = ndimage.binary_erosion(sl, iterations=ER)
            big = largest(core) if core.any() else None
            if big is None or big.sum() < 0.35 * sl.sum():
                big = largest(sl)
                margin = 0
            else:
                margin = 2 * ER
            if big is None:
                continue
            xs = np.nonzero(big.any(axis=1))[0]
            if len(xs) < 3:
                continue
            # same robustness at the endplate: trim the extreme 2% of the coordinate so
            # a rim osteophyte cannot widen the body
            lo_x, hi_x = np.percentile(xs, [1, 99])
            widths.append((hi_x - lo_x + 1 + margin) * sp[0])
        if widths:
            r[f"endplate_width_{name}_mm"] = round(float(np.median(widths)), 1)

        # ANTERIOR AND POSTERIOR BODY HEIGHT, EACH AS THE TALLEST COLUMN IN ITS HALF.
        # Two earlier versions measured at the extreme anterior EDGE and read 6 to 12 mm
        # against a published 30. The tell was that the same code read the posterior
        # band correctly at 25 to 32: the posterior wall sits flat against the canal and
        # does not taper, so the method was never wrong, only where it pointed. The
        # front of a vertebral body is a rounded rim, and published anterior height is
        # measured at the anterior CORTEX -- the wall, which is the tallest part of its
        # half. A biconcave endplate makes the middle of the body shorter than either
        # wall, which is why the maximum and not the mean is the right statistic.
        bx = float(np.median(bidx[:, 0]))
        xlo = max(0, int(round(bx - 5.0 / sp[0])))
        xhi = int(round(bx + 5.0 / sp[0])) + 1
        slab = body[xlo:xhi]                       # mid-sagittal, 10 mm wide
        if slab.sum() >= 150:
            ys = np.nonzero(slab.any(axis=(0, 2)))[0]
            col_h = {}
            for y in ys:
                c = slab[:, y, :]
                if c.sum() < 3:
                    continue
                zc = np.nonzero(c.any(axis=0))[0]
                col_h[y] = (zc.max() - zc.min() + 1) * sp[2]
            if len(col_h) >= 6:
                ymid = (min(col_h) + max(col_h)) / 2.0
                ant = [v for y, v in col_h.items() if y > ymid]
                post = [v for y, v in col_h.items() if y <= ymid]
                if ant and post:
                    ha, hp = max(ant), max(post)

                    r[f"body_height_{name}_mm"] = round(float(ha), 1)
                    r[f"body_height_post_{name}_mm"] = round(float(hp), 1)
                    if hp > 1:
                        r[f"wedge_ratio_{name}"] = round(float(ha / hp), 3)
    _guard_wedge(r, list(LUMBAR.values()))
    return r


def _guard_wedge(r, levels):
    """Withhold a wedge ratio where the body mask was not a body.

    THE FAILURE. Everything upstream rests on the anterior wall of the canal correctly
    dividing body from posterior elements. When that detection fails, pedicles and
    articular processes stay in the mask and do two things at once: they are taller than
    the body, so the POSTERIOR maximum inflates, and they push the halfway point
    backwards so the "anterior" half lands on the biconcave middle of the body, which is
    its shortest part. Both errors drive the ratio down together and the result is
    indistinguishable from a severe compression fracture -- six vertebrae came out below
    0.55 and one at 0.273, an anterior height of 9.6 mm against a posterior 35.2.

    WHY THE OBVIOUS GUARD IS WORSE THAN NONE. The first attempt asked whether each half's
    tallest column sat near that half's outer wall, on the reasoning that the anterior
    cortex is the tallest part of its half. In a vertebra that is genuinely wedged the
    anterior wall is COLLAPSED and its tallest column is not at the wall -- so the guard
    rejected 29% of all levels and dropped Genant grade 1+ prevalence from 3.2% to 0.3%
    against a published 5-10%. It was anti-correlated with the finding it was protecting.

    WHAT ACTUALLY SEPARATES THEM. A compression fracture takes the ANTERIOR height; the
    posterior wall is what it is measured against precisely because it is spared. A
    contaminated mask inflates the POSTERIOR height instead. So the discriminator is the
    posterior height compared against the same patient's other levels -- a within-subject
    control, which needs no population range and survives whatever that patient's build
    happens to be. A level whose posterior height stands more than 25% above the median
    of its neighbours did not measure a vertebral body.
    """
    hp = {lv: r.get(f"body_height_post_{lv}_mm") for lv in levels}
    have = [v for v in hp.values() if v]
    if len(have) < 3:
        return r
    med = float(np.median(have))
    if med <= 0:
        return r
    for lv in levels:
        v = hp.get(lv)
        if v and v > 1.25 * med and f"wedge_ratio_{lv}" in r:
            r.pop(f"wedge_ratio_{lv}", None)
            r[f"wedge_rejected_{lv}"] = 1
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--manifest", default="data/hf_export/manifest.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.labels).glob("*_label.nii.gz"))
    print(f"{len(files)} case(s)\n", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, files, chunksize=4), 1):
            res.append(x)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", flush=True)

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
    p = out / "level_gradients.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    print(f"\n  {len(ok)} of {len(res)} measured\n")
    # THE GRADIENT IS THE CHECK. Each of these should rise monotonically L1 to L5; if one
    # does not, the measurement is wrong, because this trend is as reproducible as
    # anything in spinal anatomy.
    for stem, label, published in (
        ("endplate_width", "superior endplate width (mm)", "~41.8 -> ~50.7"),
        ("body_height", "anterior body height (mm)", "~29.9 -> ~34.5"),
        ("canal_width", "transverse canal width (mm)", "~22.0 -> ~26.5"),
        ("tp_span", "transverse process span (mm)", "~68 -> ~86"),
        ("wedge_ratio", "wedge ratio", "near 1.0 throughout"),
    ):
        meds = []
        for lv in ("L1", "L2", "L3", "L4", "L5"):
            key = f"{stem}_{lv}_mm" if stem != "wedge_ratio" else f"{stem}_{lv}"
            v = [x[key] for x in ok if isinstance(x.get(key), (int, float))]
            meds.append(float(np.median(v)) if len(v) >= 20 else float("nan"))
        good = [m for m in meds if m == m]
        mono = "rises" if len(good) > 2 and good[-1] > good[0] else "DOES NOT RISE"
        print(f"  {label:32s} " + "  ".join(f"{m:6.1f}" for m in meds)
              + f"   [{mono}; published {published}]")

    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
