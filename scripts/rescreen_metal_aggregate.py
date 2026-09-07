"""rescreen_metal_aggregate.py -- find constructs the per-component size floor threw away.

detect_metal.py tests each connected component against IMPLANT_MM3 = 300, a floor sized for
a pedicle screw as one solid blob. A thin anterior plate, a fragmented rod, a low-density
cage does not arrive as one blob: at 2200 HU it breaks into pieces of a few tens of mm3 each,
every piece fails the floor, and the case is called clean. Case 0816 is exactly this -- 12
components, largest 118 mm3, 663 mm3 in total, screws visibly in the L3-L5 bodies.

So this re-screens on the per-case AGGREGATE: total bone-hosted metal volume, which is the
quantity the floor was meant to express in the first place. It reports candidates; it does
not label anything and does not decide anything. A radiologist still reads the list.

    python rescreen_metal_aggregate.py --tree data/hf_export_osc \
        --out rescreen_metal.csv --workers 12
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# identical to detect_metal.py, so the only thing that changes is per-component vs per-case
METAL_HU = 2200.0
STREAK_HU = -250.0
AIR_HU = -800.0
MIN_VOX = 20
CLIP_VOX = 60
BONE_REACH_MM = 12.0
IMPLANT_MM3 = 300.0


def one(pair):
    lab_p, ct_p = pair
    stem = Path(lab_p).name.split("_")[0]
    try:
        li, ci = nib.load(lab_p), nib.load(ct_p)
        lab = np.asanyarray(li.dataobj).astype(np.int16)
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
        sp = np.array(li.header.get_zooms()[:3], float)
    except Exception as exc:
        return {"case": stem, "error": type(exc).__name__}

    if ct.shape != lab.shape:
        return {"case": stem, "error": "shape_mismatch"}

    vox_mm3 = float(np.prod(sp))
    hot = ct >= METAL_HU
    r = {"case": stem, "error": "", "ct_max_hu": round(float(ct.max()), 1),
         "metal_voxels": int(hot.sum()), "n_components": 0,
         "agg_mm3_hosted": 0.0, "max_component_mm3": 0.0,
         "hosts": "", "old_rule_implant": 0, "new_rule_candidate": 0}
    if not hot.any():
        return r

    cc, n = ndimage.label(hot)
    it = max(1, int(round(BONE_REACH_MM / float(min(sp)))))
    agg = 0.0
    biggest = 0.0
    hosts: Counter = Counter()
    old_hit = False
    n_comp = 0

    for i in range(1, n + 1):
        m = cc == i
        v = int(m.sum())
        if v < MIN_VOX:
            continue
        n_comp += 1
        mm3 = v * vox_mm3
        biggest = max(biggest, mm3)

        sh = ndimage.binary_dilation(m, iterations=2) & ~m
        sv = ct[sh]
        dark = float(np.mean((sv < STREAK_HU) & (sv > AIR_HU))) if sv.size else 0.0

        near = lab[ndimage.binary_dilation(m, iterations=it)]
        near = near[(near > 0) & (near != 255)]
        host = int(Counter(near.tolist()).most_common(1)[0][0]) if near.size else 0

        if dark >= 0.15 and v >= CLIP_VOX and host > 0 and mm3 >= IMPLANT_MM3:
            old_hit = True
        # aggregate: bone-hosted, streak-bearing metal, regardless of how it fragmented
        if host > 0 and dark >= 0.10:
            agg += mm3
            hosts[host] += v

    r["n_components"] = n_comp
    r["agg_mm3_hosted"] = round(agg, 1)
    r["max_component_mm3"] = round(biggest, 1)
    r["hosts"] = ";".join(f"{k}:{v}" for k, v in hosts.most_common(6))
    r["old_rule_implant"] = int(old_hit)
    r["new_rule_candidate"] = int(agg >= IMPLANT_MM3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    tree = Path(a.tree)
    pairs = []
    for lab in sorted((tree / "labels").glob("*_label.nii.gz")):
        ct = tree / "ct" / lab.name.replace("_label", "_ct")
        if ct.exists():
            pairs.append((str(lab), str(ct)))
    print(f"  {len(pairs)} pairs", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(one, pairs, chunksize=2), 1):
            rows.append(r)
            if n % 100 == 0:
                print(f"  {n}/{len(pairs)}", flush=True)

    cols = ["case", "error", "ct_max_hu", "metal_voxels", "n_components",
            "agg_mm3_hosted", "max_component_mm3", "hosts",
            "old_rule_implant", "new_rule_candidate"]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: -float(x.get("agg_mm3_hosted") or 0)):
            w.writerow(r)

    old = [r for r in rows if r.get("old_rule_implant")]
    new = [r for r in rows if r.get("new_rule_candidate")]
    gained = [r for r in new if not r.get("old_rule_implant")]
    lost = [r for r in old if not r.get("new_rule_candidate")]
    print(f"  old rule (per-component >=300mm3) : {len(old)} cases")
    print(f"  new rule (per-case aggregate)     : {len(new)} cases")
    print(f"  NEWLY flagged (need a read)       : {len(gained)}")
    for r in sorted(gained, key=lambda x: -x['agg_mm3_hosted']):
        print(f"     {r['case']}  agg={r['agg_mm3_hosted']:>8} mm3  "
              f"max_comp={r['max_component_mm3']:>7}  n={r['n_components']:>3}  hosts={r['hosts']}")
    print(f"  flagged before, not now           : {[r['case'] for r in lost]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
