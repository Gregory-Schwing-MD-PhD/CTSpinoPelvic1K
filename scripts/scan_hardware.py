"""scripts/scan_hardware.py — find surgical instrumentation across the release.

WHY. Nothing in the manifest records hardware, so the instrumented population is unknown --
and it matters twice over. An IATROGENIC fusion is indistinguishable from a congenital one
to a distance measurement: a cage-bridged interspace reads as "no gap" exactly like a fused
transitional vertebra, so an unlabelled instrumented case can contribute a false positive
to the transitional-anatomy result. Separately, if LSTV predisposes to back pain and
surgery, instrumented cases may be ENRICHED for the phenotype, which would bias the same
result in the same direction. Both need the case list before they can be checked.

HOW. Metal runs far above bone: cortical bone tops out near 1500 HU while implants saturate
the scanner (3071 on the cases here). A threshold separates them cleanly, so the work is
not detection but LOCALISATION -- deciding what the bright thing is.

  region       the spine and sacrum masks DILATED by ~12mm. A cage sits in the disc SPACE,
               which is outside every vertebra mask, so searching inside the masks alone
               would miss exactly the implant type that causes the false-fusion problem.
  in_disc      a component that touches two different vertebra labels bridges an
               interspace -- cage-like, and the one that corrupts distance measures.
  in_bone      a component sitting mostly inside a single vertebra -- screw-like.
  elongation   PCA axis ratio; screws and rods are long and thin, cages are compact.

Reports, does not label. The subtype call (77 cage / 78 screw+rod / 79 plate) is left to a
reader, because a threshold cannot tell a cage from a plate lying against a body.

    python scripts/scan_hardware.py --labels data/v5_final --ct data/hf_export_v4/ct \\
        --workers 12 --out qc_hardware
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

THORACIC = list(range(8, 20))
LUMBAR = list(range(20, 26))
SACRAL = [26, 29]
SPINE_IDS = THORACIC + LUMBAR + SACRAL

METAL_HU = 1800.0        # well clear of cortical bone (~1500 max), well below saturation
DILATE_MM = 12.0         # enough to reach into a disc space from the adjacent bodies
MIN_VOX = 40             # below this is beam-hardening speckle, not an implant


def one(args) -> dict:
    stem, lp, cp = args
    r = {"case": stem}
    try:
        li = nib.as_closest_canonical(nib.load(lp))
        lab = np.asanyarray(li.dataobj).astype(np.int16)
        sp = np.array(li.header.get_zooms()[:3], float)
        ci = nib.as_closest_canonical(nib.load(cp))
        ct = np.asanyarray(ci.dataobj)                 # native dtype: half the memory
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": f"{type(exc).__name__}"}
    if ct.shape != lab.shape:
        return {"case": stem, "error": "shape mismatch"}

    spine = np.isin(lab, SPINE_IDS)
    if not spine.any():
        return {"case": stem, "n_hardware": 0, "note": "no spine labels"}

    # search a dilated shell: a cage lives in the disc space, outside every vertebra mask
    it = max(1, int(round(DILATE_MM / max(sp.min(), 1e-6))))
    region = ndimage.binary_dilation(spine, iterations=it)
    bright = (ct > METAL_HU) & region
    r["max_hu_in_region"] = int(ct[region].max()) if region.any() else None
    if not bright.any():
        r["n_hardware"] = 0
        return r

    cc, ncc = ndimage.label(bright)
    comps = []
    for i in range(1, ncc + 1):
        m = cc == i
        vox = int(m.sum())
        if vox < MIN_VOX:
            continue
        idx = np.argwhere(m)
        # which vertebra labels does this piece touch? two or more -> it bridges a space
        near = ndimage.binary_dilation(m, iterations=max(1, it // 4))
        touched = sorted({int(v) for v in np.unique(lab[near]) if v in SPINE_IDS})
        q = (idx * sp).astype(float)
        q -= q.mean(0)
        sv = np.linalg.svd(q, full_matrices=False)[1]
        elong = float(sv[0] / max(sv[1], 1e-6))
        comps.append({"voxels": vox, "mm3": round(vox * float(np.prod(sp)), 1),
                      "touches": touched, "n_touched": len(touched),
                      "elongation": round(elong, 2),
                      "bridges_interspace": len(touched) >= 2,
                      "z": float(idx[:, 2].mean())})
    comps.sort(key=lambda c: -c["voxels"])
    r["n_hardware"] = len(comps)
    r["hardware_vox"] = sum(c["voxels"] for c in comps)
    r["any_bridges"] = int(any(c["bridges_interspace"] for c in comps))
    r["max_elongation"] = max((c["elongation"] for c in comps), default=None)
    r["components"] = comps[:8]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="qc_hardware")
    a = ap.parse_args()

    labdir, ctdir = Path(a.labels), Path(a.ct)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    jobs = [(s, str(labdir / f"{s}_label.nii.gz"), str(ctdir / f"{s}_ct.nii.gz"))
            for s in stems
            if (labdir / f"{s}_label.nii.gz").exists() and (ctdir / f"{s}_ct.nii.gz").exists()]
    print(f"{len(jobs)} case(s), metal threshold {METAL_HU:.0f} HU\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, jobs, chunksize=1), 1):
            res.append(x)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    lstv = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
            if s:
                lstv[s] = str(rec.get("lstv_label") or "normal")

    hits = [r for r in res if r.get("n_hardware")]
    hits.sort(key=lambda r: -r.get("hardware_vox", 0))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "hardware_scan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "n_components", "voxels", "mm3", "bridges_interspace",
                    "max_elongation", "max_hu", "lstv_label"])
        for r in hits:
            mm3 = sum(c["mm3"] for c in r.get("components", []))
            w.writerow([r["case"], r["n_hardware"], r.get("hardware_vox"), round(mm3, 1),
                        r.get("any_bridges"), r.get("max_elongation"),
                        r.get("max_hu_in_region"), lstv.get(r["case"], "")])
    (out / "hardware_scan.json").write_text(json.dumps(hits, indent=1))

    print(f"\n  {len(hits)} case(s) with hardware of {len(res)} scanned")
    print(f"  {'case':7s} {'ncomp':>5s} {'mm3':>9s} {'bridge':>7s} {'elong':>6s}  lstv")
    for r in hits[:30]:
        mm3 = sum(c["mm3"] for c in r.get("components", []))
        print(f"  {r['case']:7s} {r['n_hardware']:5d} {mm3:9.0f} "
              f"{str(bool(r.get('any_bridges'))):>7s} {str(r.get('max_elongation')):>6s}"
              f"  {lstv.get(r['case'],'')}")

    # the confounder check this scan exists for
    if hits:
        hw = {r["case"] for r in hits}
        a_ = sum(1 for c in hw if lstv.get(c, "normal") != "normal")
        b_ = len(hw) - a_
        c_ = sum(1 for c, v in lstv.items() if c not in hw and v != "normal")
        d_ = len(lstv) - len(hw) - c_
        print(f"\n  hardware x LSTV:  hw+lstv {a_}  hw only {b_}  lstv only {c_}  neither {d_}")
        if b_ and c_:
            print(f"  odds ratio {(a_*d_)/max(1,(b_*c_)):.2f}  "
                  f"(>1 means instrumented cases are enriched for LSTV)")
    print(f"\n  wrote {out}/hardware_scan.csv and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
