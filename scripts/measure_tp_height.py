"""scripts/measure_tp_height.py — transverse-process height, and a label census, in one pass.

WHY THIS EXISTS SEPARATELY FROM extract_transition_morphometrics.py. That script computes
everything -- sacral foramina by sliding a thin slab, disc heights by column bundles, rib
lengths by principal axis. The Castellvi Type I criterion needs one number per side, and
re-running the whole battery to get it took two hours and had not finished.

WHERE THE TIME ACTUALLY WENT, measured rather than guessed. Loading one volume costs 3.5 to
4.7 s -- these are 512x512x600 label arrays, about 160 million voxels, gzipped. Then
`nib.as_closest_canonical` costs ANOTHER 4.2 to 4.5 s, because reorienting means rewriting
all 160 million voxels. Then `present = [v for v in range(20, 26) if (lab == v).any()]`
walks the whole array up to six more times. Seven such workers do not saturate sixteen
cores; they saturate memory bandwidth, which is why the cores looked idle at 14% while the
disk sat at 0.5%.

So this file does three things differently:

  * IT NEVER REORIENTS THE ARRAY. The rule about not reorienting is usually about not
    corrupting a label being written; here it is simply that reorientation is 4 seconds of
    pure memory traffic to obtain something we do not need. What the measurement needs is
    to know WHICH axis is left-right and which is craniocaudal, and the affine says so
    directly. On these volumes the axcodes are ('P','I','R'): axis 0 is anteroposterior,
    axis 1 is craniocaudal (increasing toward the feet), axis 2 is left-right. Reading them
    raw while assuming axis 0 is lateral reports a 51 mm process on the left and 9 mm on
    the right in nearly every case, so the axes are resolved from the affine and asserted,
    never assumed.
  * ONE np.unique REPLACES SIX EQUALITY SCANS to find which lumbar labels are present, and
    the same call yields the label census for free -- which is the other question that
    needed a pass over every volume, so the two are answered together rather than by two
    jobs competing for the same memory bus.
  * THE MASK IS CROPPED BEFORE ANY WORK. Once the lowest lumbar label is known, everything
    after that happens inside its bounding box, which is a few million voxels rather than
    160 million.

THE MEASUREMENT ITSELF is the craniocaudal extent of the LARGEST CONNECTED COMPONENT of a
12 mm slab at the lateral tip of the lowest lumbar vertebra. A slab's extent is set by its
two extreme voxels, so one detached speckle became the height -- case 0512 read 43.2 mm
against a true process of 16.0 mm, and entered the re-read queue at rank 3 on that basis.
The slab extent is kept as tp_height_slab_*, because the ratio of the two measures how
speckled the tip is.

    python scripts/measure_tp_height.py --labels data/hf_export_v5/labels --workers 6
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

L1, L6 = 20, 25
TIP_MM = 12.0
LATERAL_FRAC = 0.45


def axes_from(affine):
    """-> (lateral axis, craniocaudal axis). Read from the affine, never assumed.

    aff2axcodes gives one letter per ARRAY axis saying which anatomical direction that
    axis increases toward. 'L'/'R' is the left-right axis; 'S'/'I' is the craniocaudal
    one. Everything downstream indexes by these, so the array is never rewritten.
    """
    codes = nib.aff2axcodes(affine)
    lat = next(i for i, c in enumerate(codes) if c in ("L", "R"))
    cc = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    return lat, cc


def one(path: Path):
    case = path.name.split("_")[0]
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        lat, cc = axes_from(img.affine)
        other = ({0, 1, 2} - {lat, cc}).pop()

        present = [int(v) for v in np.unique(lab)]            # the only full-array pass
        out = {"case": case, "error": "", "axcodes": "".join(nib.aff2axcodes(img.affine)),
               "labels_present": " ".join(str(v) for v in present)}

        lum = [v for v in present if L1 <= v <= L6]
        if not lum:
            out["error"] = "no lumbar label"
            return out
        low = max(lum)
        out["lowest_lumbar_label"] = low

        m_full = lab == low
        del lab
        # crop to the vertebra: everything after this is a few million voxels
        idx = np.nonzero(m_full)
        sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
        m = m_full[sl]
        del m_full
        off_lat = sl[lat].start

        n_lat = m.shape[lat]
        along = np.moveaxis(m, lat, 0)                        # a view, not a copy
        cols = np.nonzero(along.any(axis=(1, 2)))[0]
        lo, hi = int(cols.min()), int(cols.max())
        vmid = 0.5 * (lo + hi)
        latL = int(vmid - LATERAL_FRAC * (vmid - lo))
        latR = int(vmid + LATERAL_FRAC * (hi - vmid))
        depth = max(1, int(round(TIP_MM / max(zooms[lat], 1e-6))))
        # after moveaxis the craniocaudal axis has shifted if it sat before lat
        cc_moved = cc if cc > lat else cc + 1
        red = tuple(i for i in range(3) if i != cc_moved)

        for nm, outward in (("left", -1), ("right", +1)):
            sel = np.zeros(n_lat, bool)
            if outward > 0:
                sel[latR + 1:] = True
            else:
                sel[:latL] = True
            if not sel.any():
                out[f"tp_height_{nm}_mm"] = ""
                continue
            band = along[sel]
            bcols = np.nonzero(band.any(axis=(1, 2)))[0]
            if not len(bcols):
                out[f"tp_height_{nm}_mm"] = ""
                continue
            edge = int(bcols.max()) if outward > 0 else int(bcols.min())
            keep = ((np.arange(len(band)) <= edge) & (np.arange(len(band)) >= edge - depth)
                    if outward > 0 else
                    (np.arange(len(band)) >= edge) & (np.arange(len(band)) <= edge + depth))
            tip = band[keep]
            if not tip.any():
                tip = band

            lt, n = ndimage.label(tip)
            if n > 1:
                sizes = ndimage.sum(tip, lt, range(1, n + 1))
                core = lt == (int(np.argmax(sizes)) + 1)
            else:
                core = tip
            zc = np.nonzero(core.any(axis=tuple(i for i in range(3) if i != cc_moved)))[0]
            zt = np.nonzero(tip.any(axis=tuple(i for i in range(3) if i != cc_moved)))[0]
            out[f"tp_height_{nm}_mm"] = round(float(zc.max() - zc.min() + 1) * zooms[cc], 1)
            out[f"tp_height_slab_{nm}_mm"] = round(float(zt.max() - zt.min() + 1) * zooms[cc], 1)
            out[f"tp_tip_components_{nm}"] = int(n)
        return out
    except Exception as e:                                            # noqa: BLE001
        return {"case": case, "error": f"{type(e).__name__}: {e}"}


FIELDS = ["case", "lowest_lumbar_label", "axcodes",
          "tp_height_left_mm", "tp_height_right_mm",
          "tp_height_slab_left_mm", "tp_height_slab_right_mm",
          "tp_tip_components_left", "tp_tip_components_right",
          "labels_present", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--scheme", default="data/hf_export_v5/dataset_labels.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="morphometrics/tp_height.csv")
    ap.add_argument("--census", default="morphometrics/label_census.csv")
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
    bad = [r for r in rows if r.get("error")]
    print(f"\n  wrote {a.out}: {len(ok)} measured, {len(bad)} failed", flush=True)
    for r in bad[:5]:
        print(f"    {r['case']}: {r['error']}")

    # --- the census, free from the same pass ---------------------------------------
    names = {}
    sp = Path(a.scheme)
    if sp.exists():
        names = {int(k): v for k, v in
                 json.loads(sp.read_text(encoding="utf-8"))["id_to_name"].items()}
    freq = Counter()
    for r in ok:
        freq.update(int(v) for v in r["labels_present"].split())
    with open(a.census, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "records", "pct"])
        for i in sorted(freq):
            w.writerow([i, names.get(i, "?"), freq[i],
                        round(100 * freq[i] / max(1, len(ok)), 2)])
    print(f"  wrote {a.census}")

    axc = Counter(r.get("axcodes", "") for r in ok)
    print(f"  axcodes seen: {dict(axc)}")

    def block(lo, hi, title):
        p = {i: freq[i] for i in range(lo, hi + 1) if freq.get(i)}
        print(f"  {title}: " + (", ".join(f"{names.get(i,i)}={p[i]}" for i in sorted(p))
                                if p else "none present"))

    block(25, 25, "L6")
    block(74, 75, "lumbar ribs")
    block(58, 73, "soft tissue")
    block(76, 79, "hardware")
    block(27, 29, "coccyx/T13/S1")

    if ok:
        h = np.array([max(float(r["tp_height_left_mm"] or 0),
                          float(r["tp_height_right_mm"] or 0)) for r in ok])
        h = h[h > 0]
        print(f"\n  tp_height_max: median {np.median(h):.1f} mm, "
              f"{100 * (h >= 19).mean():.1f}% at or above the 19 mm Type I criterion "
              f"(reference ~19.7% of an unselected population)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
