"""
scripts/qc_v4_ribs_and_verts.py — the ONLY two questions Greg cares about, measured over ALL of v4:

  1. DO THE RIB PSEUDOLABELS PASS QC?  Run the real gating rib checks (rib_label_mixing +
     rib_spine_gap) on every case. structure_integrity / rib_vertebra_match are deliberately NOT
     used here -- they are advisory (the spine is force-restored; transitional anatomy shifts ribs),
     so they never define "does the rib label pass". Output: per-case pass/fail + reasons.

  2. ARE THERE VERTEBRAE IN THE FOV THAT AREN'T ANNOTATED?  Two signals, both FOV-safe:
       (a) INTERIOR GAP -- a vertebra id missing BETWEEN two labelled ones (e.g. T11, [T12], L1).
           It is spatially bracketed by annotated bone, so it is unambiguously in the FOV. HIGH conf.
       (b) RIB ABOVE THE SPINE -- a segmented rib whose head sits SUPERIOR to the topmost annotated
           vertebra. A rib can only be segmented if its level is imaged, so its vertebra IS in the
           FOV -- just not annotated. A rib clipped/out-of-view doesn't reach the spine, so it won't
           false-trigger. This is the "real examples of a vertebra in FOV, just not in the GT" case.

Output: counts (reviewed vs unreviewed) + two CSVs (rib_qc_fail.csv, missing_vertebra.csv).

  python scripts/qc_v4_ribs_and_verts.py [--limit N] [--workers 14]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
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
VERT_LO, VERT_HI = 8, 25             # T1..L6 = the contiguous thoracolumbar run
LUM_LO_ID, LUM_HI_ID = 20, 25        # L1..L6
RIB_LO, RIB_HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
ABOVE_MARGIN_MM = 20.0               # a rib head must clear the top vertebra by this to count "above"


def missing_vertebra(lab: np.ndarray, affine) -> tuple:
    """(interior_gap_names, ribs_above_spine_names, lumbar_rib_names). See module docstring.

    Fast + EDT-FREE: ONE np.unique + ONE ndimage.find_objects, then everything from bounding boxes.
    Each rib is mapped to a vertebra by SUPERIOR-INFERIOR LEVEL (the rib and its vertebra sit at the
    same axial height) rather than head-CONTACT -- which is both cheaper and MORE robust, because rib
    heads are frequently unsegmented (contact-based detection undercounts lumbar ribs badly).
      * gaps   = vertebra id missing between two labelled ones (bracketed -> definitely in FOV)
      * above  = rib whose head level is superior to the topmost annotated vertebra (in-FOV, unlabelled)
      * lumbar = rib whose head level falls on a LUMBAR vertebra (L1..L6) -> candidate own-class
    """
    names = RA._id2name()
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    ids = set(int(v) for v in np.unique(lab))                # ONE pass
    present = sorted(v for v in ids if VERT_LO <= v <= VERT_HI)
    gaps = []
    if present:
        gaps = [names.get(v, v) for v in range(present[0], present[-1] + 1) if v not in ids]

    above, lumbar = [], []
    rib_ids = sorted(v for v in ids if RIB_LO <= v <= RIB_HI)
    if present and rib_ids:
        objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))  # ONE pass
        R = np.asarray(affine)[:3, :3]
        si = int(np.argmax(np.abs(R[2, :])))
        sup = R[2, si] >= 0
        vranges = [(v, objs[v - 1][si].start, objs[v - 1][si].stop)
                   for v in present if objs[v - 1] is not None]
        if vranges:
            vtop = max(h for _, _, h in vranges) if sup else min(l for _, l, _ in vranges)
            margin = ABOVE_MARGIN_MM / spacing[si]
            for rid in rib_ids:
                sl = objs[rid - 1]
                if sl is None:
                    continue
                head_si = sl[si].stop - 1 if sup else sl[si].start    # head = superior end of the rib
                off = LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET
                tag = f"{'L' if off == LS.RIB_LEFT_OFFSET else 'R'}{rid - off}"
                if (head_si > vtop + margin) if sup else (head_si < vtop - margin):
                    above.append(tag)
                    continue                                         # above the spine -> not lumbar
                inside = [v for v, lo, hi in vranges if lo <= head_si < hi]
                v = inside[0] if inside else \
                    min(vranges, key=lambda x: abs((x[1] + x[2]) // 2 - head_si))[0]
                if LUM_LO_ID <= v <= LUM_HI_ID:
                    lumbar.append(tag)
    return gaps, above, lumbar


def _work(args):
    t, p, rev = args
    tok = os.environ["HF_TOKEN"]
    try:
        img = nib.load(hf_hub_download(DS, p, repo_type="dataset", token=tok, revision="v4"))
        lab = np.asanyarray(img.dataobj); aff = img.affine
        ok, msgs = RA.check_label("ribs", lab, aff, gating_only=True)   # gates = mixing + spine_gap
        reasons = [m for m in msgs if m.startswith("X")]
        gaps, above, lum = missing_vertebra(lab, aff)                   # gaps + above + lumbar (1 pass)
        return (t, rev, ok, reasons, gaps, above, lum)
    except Exception:                                          # noqa: BLE001
        return None


def main(argv=None) -> int:
    # Executor choice depends on the bottleneck:
    #  - ProcessPool (default): TRUE parallelism for the CPU-bound QC (find_objects/EDT hold or thrash
    #    the GIL, so threads barely scale). Correct when labels are CACHED (offline) -> no network in
    #    the workers, so no BrokenProcessPool from connection deaths.
    #  - ThreadPool (--threads): use when labels must be DOWNLOADED (I/O-bound) and the Windows spawn
    #    ProcessPool dies on network errors.
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", action="store_true", help="use threads (I/O-bound) not processes")
    a = ap.parse_args(argv)
    tok = os.environ["HF_TOKEN"]

    mp = hf_hub_download(DS, "manifest.json", repo_type="dataset", token=tok, revision="v4")
    recs = json.load(open(mp)); recs = recs if isinstance(recs, list) else recs.get("records", [])
    reviewed = set()
    try:
        wp = hf_hub_download(DS, "rib_worklist.json", repo_type="dataset", token=tok, revision="v4")
        wl = json.load(open(wp))
        reviewed = {str(t) for t in (wl.get("tokens") if isinstance(wl, dict) else wl)}
    except Exception:                                          # noqa: BLE001
        pass
    items = [(str(r.get("token")), r.get("pseudo_label_file") or r.get("label_file"),
              str(r.get("token")) in reviewed)
             for r in recs if (r.get("pseudo_label_file") or r.get("label_file"))]
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][:a.limit]
    Executor = ThreadPoolExecutor if a.threads else ProcessPoolExecutor
    print(f"scanning {len(items)} v4 cases ({len(reviewed)} reviewed) "
          f"[{a.workers} {'threads' if a.threads else 'procs'}]\n", flush=True)

    rib_fail, vert_rows, lum_rows = [], [], []
    n = tot_unrev = 0
    rib_fail_rev = rib_fail_unrev = 0
    gap_cases = above_cases = 0
    lum_uni = lum_bilat = 0
    done = 0
    with Executor(max_workers=a.workers) as ex:
        for r in ex.map(_work, items, chunksize=1 if a.threads else 4):
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(items)}", flush=True)
            if not r:
                continue
            t, rev, ok, reasons, gaps, above, lum = r
            n += 1
            if not rev:
                tot_unrev += 1
            if not ok:
                if rev: rib_fail_rev += 1
                else:   rib_fail_unrev += 1
                rib_fail.append({"token": t, "reviewed": rev,
                                 "reasons": " | ".join(reasons)})
            if gaps or above:
                if gaps: gap_cases += 1
                if above: above_cases += 1
                vert_rows.append({"token": t, "reviewed": rev,
                                  "interior_gaps": " ".join(str(g) for g in gaps),
                                  "n_gaps": len(gaps),
                                  "ribs_above_spine": " ".join(above),
                                  "n_levels_above": len({x[1:] for x in above})})
            if lum:
                left = [x for x in lum if x.startswith("L")]
                right = [x for x in lum if x.startswith("R")]
                both = bool(left) and bool(right)
                lum_bilat += both; lum_uni += (not both)
                lum_rows.append({"token": t, "reviewed": rev,
                                 "lumbar_ribs": " ".join(lum), "bilateral": both})

    with open("rib_qc_fail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "reasons"])
        w.writeheader(); w.writerows(rib_fail)
    with open("missing_vertebra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "interior_gaps", "n_gaps",
                                          "ribs_above_spine", "n_levels_above"])
        w.writeheader(); w.writerows(vert_rows)
    with open("lumbar_ribs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "lumbar_ribs", "bilateral"])
        w.writeheader(); w.writerows(lum_rows)

    print(f"\n===== v4 QC ({n} cases; {tot_unrev} unreviewed) =====")
    print("\n1) RIB PSEUDOLABELS vs QC (gates: label-mixing + spine-gap)  -> rib_qc_fail.csv")
    print(f"   FAIL: {len(rib_fail)}/{n} = {100*len(rib_fail)/max(1,n):.1f}%   "
          f"(reviewed {rib_fail_rev}, unreviewed {rib_fail_unrev})")
    print(f"   PASS: {n-len(rib_fail)}/{n}")
    print("\n2) VERTEBRA IN FOV BUT NOT ANNOTATED  -> missing_vertebra.csv")
    print(f"   interior GAP (bracketed, definitely in-FOV): {gap_cases} cases  <- high-confidence fixes")
    print(f"   rib ABOVE the annotated spine top:           {above_cases} cases")
    print("\n3) LUMBAR RIBS (rib head on L1..L6 -> candidate own-class)  -> lumbar_ribs.csv")
    print(f"   cases with >=1 lumbar rib: {len(lum_rows)}   bilateral: {lum_bilat}   unilateral: {lum_uni}")
    print("   NOTE: rib heads are often unsegmented, so this is a FLOOR (contact-based undercounts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
