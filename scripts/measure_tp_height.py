"""scripts/measure_tp_height.py — transverse-process height alone, over the whole cohort.

WHY THIS EXISTS SEPARATELY FROM extract_transition_morphometrics.py. That script computes
everything -- sacral foramina by sliding a thin slab through the sacrum, disc heights by
column bundles, rib lengths by principal axis -- and on 802 volumes at 512x512x500 it runs
for hours and holds several gigabytes per worker. The Castellvi Type I criterion needs one
number per side, and re-running the whole battery to get it is the wrong trade.

This computes exactly that, and it computes it the corrected way: the craniocaudal extent
of the LARGEST CONNECTED COMPONENT of a 12 mm slab at the lateral tip of the lowest lumbar
vertebra, not the extent of the slab. A slab's extent is set by its two extreme voxels, so
a single detached speckle became the height -- case 0512 read 43.2 mm against a true
process of 16.0 mm. The slab extent is kept alongside as tp_height_slab_*, because the
ratio between the two is a usable measure of how speckled the tip is.

Memory is bounded deliberately: the label volume is read once, reduced immediately to the
lowest-lumbar boolean mask, and the full array dropped. Nothing here needs the CT.

    python scripts/measure_tp_height.py --labels data/hf_export_v5/labels --workers 4
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

L1, L6 = 20, 25
TIP_MM = 12.0
LATERAL_FRAC = 0.45


def one(path: Path):
    case = path.name.split("_")[0]
    try:
        # CANONICALISE TO ANALYSE, NEVER TO WRITE. On disk these are ('P','I','R');
        # as_closest_canonical gives RAS, where axis 0 is left-right and axis 2 is
        # craniocaudal. Reading them raw while assuming otherwise reports a 51 mm process
        # on the left and 9 mm on the right in nearly every case. Nothing is saved here.
        img = nib.as_closest_canonical(nib.load(str(path)))
        lab = np.asanyarray(img.dataobj)
        sp = [float(z) for z in img.header.get_zooms()[:3]]

        present = [v for v in range(L1, L6 + 1) if (lab == v).any()]
        if not present:
            return {"case": case, "error": "no lumbar label"}
        low = max(present)
        m = lab == low
        del lab

        mx = np.nonzero(m.any(axis=(1, 2)))[0]
        if not len(mx):
            return {"case": case, "error": "empty lowest lumbar"}
        vmid = 0.5 * (float(mx.min()) + float(mx.max()))
        ax = np.arange(m.shape[0])
        latL = int(vmid - LATERAL_FRAC * (vmid - float(mx.min())))
        latR = int(vmid + LATERAL_FRAC * (float(mx.max()) - vmid))

        out = {"case": case, "lowest_lumbar_label": low, "error": ""}
        for nm, sel, outward in (("left", ax < latL, -1), ("right", ax > latR, +1)):
            mm = m & sel[:, None, None]
            if not mm.any():
                out[f"tp_height_{nm}_mm"] = ""
                out[f"tp_height_slab_{nm}_mm"] = ""
                out[f"tp_tip_components_{nm}"] = ""
                continue
            cols = np.nonzero(mm.any(axis=(1, 2)))[0]
            edge = int(cols.max()) if outward > 0 else int(cols.min())
            depth = max(1, int(round(TIP_MM / max(sp[0], 1e-6))))
            keep = ((ax <= edge) & (ax >= edge - depth) if outward > 0
                    else (ax >= edge) & (ax <= edge + depth))
            tip = mm & keep[:, None, None]
            if not tip.any():
                tip = mm

            lt, n = ndimage.label(tip)
            if n > 1:
                sizes = ndimage.sum(tip, lt, range(1, n + 1))
                core = lt == (int(np.argmax(sizes)) + 1)
            else:
                core = tip
            zc = np.nonzero(core.any(axis=(0, 1)))[0]
            zt = np.nonzero(tip.any(axis=(0, 1)))[0]
            out[f"tp_height_{nm}_mm"] = round(float(zc.max() - zc.min() + 1) * sp[2], 1)
            out[f"tp_height_slab_{nm}_mm"] = round(float(zt.max() - zt.min() + 1) * sp[2], 1)
            out[f"tp_tip_components_{nm}"] = int(n)
        return out
    except Exception as e:                                            # noqa: BLE001
        return {"case": case, "error": f"{type(e).__name__}: {e}"}


FIELDS = ["case", "lowest_lumbar_label",
          "tp_height_left_mm", "tp_height_right_mm",
          "tp_height_slab_left_mm", "tp_height_slab_right_mm",
          "tp_tip_components_left", "tp_tip_components_right", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="morphometrics/tp_height.csv")
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

    bad = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    print(f"\n  wrote {a.out}: {len(ok)} measured, {len(bad)} failed", flush=True)
    if bad:
        for r in bad[:5]:
            print(f"    {r['case']}: {r['error']}")
    if ok:
        h = np.array([max(float(r["tp_height_left_mm"]), float(r["tp_height_right_mm"]))
                      for r in ok if r.get("tp_height_left_mm") != ""])
        print(f"  tp_height_max: median {np.median(h):.1f} mm, "
              f"{100 * (h >= 19).mean():.1f}% at or above the 19 mm Type I criterion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
