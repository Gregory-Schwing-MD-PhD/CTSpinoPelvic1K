"""
scripts/lumbar_rib_reclass.py — treat a rib that articulates with a LUMBAR vertebra as its own class
(rib_left_lumbar=58, rib_right_lumbar=59) instead of forcing it to be "rib 12".

Core (reusable):
  rib_articulations(lab, affine) -> {rib_id: (vertebra_id, gap_mm)}   # head -> nearest vertebra, incl lumbar
  detect_lumbar_ribs(lab, affine) -> (left_ids, right_ids, arts)      # ribs whose head is on L1..L6
  remap_lumbar(lab, affine)      -> (out, moved)                      # 58/59 for lumbar ribs; MINIMAL
      (pulls the lumbar rib into its own class; does NOT renumber the other ribs -- the residual
       numbering is reported so Greg can decide whether a re-anchor is also wanted.)

CLI:
  python scripts/lumbar_rib_reclass.py --tally         # count lumbar-rib cases over cached v4 labels
  python scripts/lumbar_rib_reclass.py --prototype N   # show articulation + before/after for token N,
                                                       #   and save the remapped label for eyeballing
Run with HF_HUB_OFFLINE=1 to use ONLY the local cache (no network).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS            # noqa: E402
import review_anatomy_qc as RA       # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
RIB_LO, RIB_HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
LUM_LO, LUM_HI = 20, 25              # L1..L6
GAP_MM = 15.0                        # rib head must reach within this of the vertebra to count
LUMBAR_L, LUMBAR_R = 74, 75         # proposed new class ids -- AFTER the soft-tissue block (58-73),
                                    # so existing ids are unchanged (58/59 are iliolumbar, NOT free)
SCRATCH = Path(os.environ.get("SCRATCH", "."))


def rib_articulations(lab: np.ndarray, affine) -> dict:
    """rib_id -> (nearest vertebra id incl lumbar, gap_mm from the rib HEAD)."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    vspan = (lab >= 8) & (lab <= LUM_HI)                  # T1..L6
    ribs = (lab >= RIB_LO) & (lab <= RIB_HI)
    if not vspan.any() or not ribs.any():
        return {}
    idx = np.argwhere(vspan | ribs); lo = idx.min(0); hi = idx.max(0) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3)); f = 2
    sub = lab[sl][::f, ::f, ::f]
    vv = (sub >= 8) & (sub <= LUM_HI)
    if not vv.any():
        return {}
    d, ind = ndimage.distance_transform_edt(~vv, sampling=spacing * f, return_indices=True)
    out = {}
    for rid in range(RIB_LO, RIB_HI + 1):
        m = (sub == rid)
        if not m.any():
            continue
        dd = np.where(m, d, np.inf)
        p = np.unravel_index(np.argmin(dd), dd.shape)
        gap = float(dd[p])
        if not np.isfinite(gap):
            continue
        v = int(sub[ind[0][p], ind[1][p], ind[2][p]])
        out[rid] = (v, gap)
    return out


def detect_lumbar_ribs(lab: np.ndarray, affine):
    arts = rib_articulations(lab, affine)
    lum = {rid: (v, g) for rid, (v, g) in arts.items() if LUM_LO <= v <= LUM_HI and g <= GAP_MM}
    left = sorted(r for r in lum if r <= LS.RIB_LEFT_OFFSET + 12)
    right = sorted(r for r in lum if r > LS.RIB_LEFT_OFFSET + 12)
    return left, right, arts


def remap_lumbar(lab: np.ndarray, affine):
    """MINIMAL remap: the lumbar-articulating rib(s) -> their own class; nothing else renumbered."""
    left, right, arts = detect_lumbar_ribs(lab, affine)
    out = lab.copy()
    for rid in left:
        out[lab == rid] = LUMBAR_L
    for rid in right:
        out[lab == rid] = LUMBAR_R
    return out, left, right, arts


def _load(token, path, offline):
    p = hf_hub_download(DS, path, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"), revision="v4",
                        local_files_only=offline)
    img = nib.load(p)
    return np.asanyarray(img.dataobj), img.affine, img


def _names():
    return RA._id2name()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--prototype", metavar="TOKEN")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    names = _names()

    mp = hf_hub_download(DS, "manifest.json", repo_type="dataset",
                         token=os.environ.get("HF_TOKEN"), revision="v4",
                         local_files_only=offline)
    recs = json.load(open(mp)); recs = recs if isinstance(recs, list) else recs.get("records", [])
    by_token = {str(r.get("token")): (r.get("pseudo_label_file") or r.get("label_file")) for r in recs}

    if a.prototype:
        t = a.prototype
        lab, aff, img = _load(t, by_token[t], offline)
        arts = rib_articulations(lab, aff)
        print(f"=== case {t} : rib -> vertebra articulation (head nearest-neighbour) ===")
        for rid in sorted(arts):
            v, g = arts[rid]
            flag = "  <== LUMBAR RIB" if LUM_LO <= v <= LUM_HI and g <= GAP_MM else ""
            print(f"   {names.get(rid, rid):14s} -> {names.get(v, v):8s} ({g:4.0f} mm){flag}")
        out, left, right, _ = remap_lumbar(lab, aff)
        print(f"\nBEFORE rib ids: {sorted(int(x) for x in np.unique(lab) if RIB_LO<=x<=RIB_HI)}")
        print(f"AFTER  rib ids: {sorted(int(x) for x in np.unique(out) if RIB_LO<=x<=RIB_HI)}"
              f"  + lumbar {sorted(int(x) for x in np.unique(out) if x in (LUMBAR_L,LUMBAR_R))}")
        print(f"moved -> lumbar class: left={[names.get(r,r) for r in left]} "
              f"right={[names.get(r,r) for r in right]}")
        outp = SCRATCH / f"remap_{t}.nii.gz"
        nib.save(nib.Nifti1Image(out.astype(lab.dtype), aff, img.header), str(outp))
        print(f"\nsaved remapped label -> {outp}  (open in ITK-SNAP to eyeball)")
        return 0

    # --tally (default): scan cached v4 labels
    items = list(by_token.items())
    if a.limit:
        items = items[:a.limit]
    uni = bilat = 0; rows = []; scanned = miss = 0
    for t, p in items:
        try:
            lab, aff, _ = _load(t, p, offline)
        except Exception:
            miss += 1; continue
        scanned += 1
        left, right, _ = detect_lumbar_ribs(lab, aff)
        if not (left or right):
            continue
        both = bool(left) and bool(right)
        bilat += both; uni += (not both)
        rows.append({"token": t, "left": " ".join(names.get(r, r) for r in left),
                     "right": " ".join(names.get(r, r) for r in right),
                     "bilateral": both})
    with open("lumbar_ribs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "left", "right", "bilateral"])
        w.writeheader(); w.writerows(rows)
    print(f"scanned {scanned} cached v4 labels ({miss} not in cache -> skipped)")
    print(f"LUMBAR-RIB cases: {len(rows)}   bilateral: {bilat}   unilateral: {uni}   -> lumbar_ribs.csv")
    for r in rows[:20]:
        print(f"   {r['token']:8s} L=[{r['left']}] R=[{r['right']}]{'  (bilateral)' if r['bilateral'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
