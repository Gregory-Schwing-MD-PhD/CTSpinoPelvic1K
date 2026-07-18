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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS            # noqa: E402
import review_anatomy_qc as RA       # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
VERT_LO, VERT_HI = 8, 25             # T1..L6 = the contiguous thoracolumbar run
RIB_LO, RIB_HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
ABOVE_MARGIN_MM = 20.0               # a rib head must clear the top vertebra by this to count "above"


def missing_vertebra(lab: np.ndarray, affine) -> tuple:
    """(interior_gap_names, ribs_above_spine_names). See module docstring."""
    names = RA._id2name()
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    present = [v for v in range(VERT_LO, VERT_HI + 1) if (lab == v).any()]
    gaps = []
    if present:
        gaps = [names.get(v, v) for v in range(min(present), max(present) + 1) if v not in present]

    above = []
    vmask = (lab >= VERT_LO) & (lab <= VERT_HI)
    if vmask.any():
        R = np.asarray(affine)[:3, :3]
        si = int(np.argmax(np.abs(R[2, :])))
        sup = R[2, si] >= 0                                   # +si direction is superior?
        plane = [a for a in range(3) if a != si]
        vidx = np.argwhere(vmask)
        vtop = vidx[:, si].max() if sup else vidx[:, si].min()
        cx = vidx[:, plane[0]].mean(); cy = vidx[:, plane[1]].mean()   # spine column centre
        margin = ABOVE_MARGIN_MM / spacing[si]
        for side, off in (("L", LS.RIB_LEFT_OFFSET), ("R", LS.RIB_RIGHT_OFFSET)):
            for n in range(1, 13):
                m = (lab == off + n)
                if not m.any():
                    continue
                r = np.argwhere(m)
                # rib HEAD = the rib voxel closest (in-plane) to the spine column axis
                dx = r[:, plane[0]] - cx; dy = r[:, plane[1]] - cy
                head_si = r[np.argmin(dx * dx + dy * dy), si]
                is_above = (head_si > vtop + margin) if sup else (head_si < vtop - margin)
                if is_above:
                    above.append(f"{side}{n}")
    return gaps, above


def _work(args):
    t, p, rev = args
    tok = os.environ["HF_TOKEN"]
    try:
        img = nib.load(hf_hub_download(DS, p, repo_type="dataset", token=tok, revision="v4"))
        lab = np.asanyarray(img.dataobj); aff = img.affine
        ok, msgs = RA.check_label("ribs", lab, aff, gating_only=True)   # gates = mixing + spine_gap
        reasons = [m for m in msgs if m.startswith("X")]
        gaps, above = missing_vertebra(lab, aff)
        return (t, rev, ok, reasons, gaps, above)
    except Exception:                                          # noqa: BLE001
        return None


def main(argv=None) -> int:
    # ThreadPool, not ProcessPool: this job is DOWNLOAD-bound (802 labels off HF), and the numpy/
    # scipy work releases the GIL, so threads win -- and they avoid the Windows BrokenProcessPool
    # that spawn+native-EDT workers hit here.
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
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
    print(f"scanning {len(items)} v4 cases ({len(reviewed)} reviewed) [{a.workers} procs]\n", flush=True)

    rib_fail, vert_rows = [], []
    n = tot_unrev = 0
    rib_fail_rev = rib_fail_unrev = 0
    gap_cases = above_cases = 0
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, items):
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(items)}", flush=True)
            if not r:
                continue
            t, rev, ok, reasons, gaps, above = r
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

    with open("rib_qc_fail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "reasons"])
        w.writeheader(); w.writerows(rib_fail)
    with open("missing_vertebra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "interior_gaps", "n_gaps",
                                          "ribs_above_spine", "n_levels_above"])
        w.writeheader(); w.writerows(vert_rows)

    print(f"\n===== v4 QC ({n} cases; {tot_unrev} unreviewed) =====")
    print("\n1) RIB PSEUDOLABELS vs QC (gates: label-mixing + spine-gap)  -> rib_qc_fail.csv")
    print(f"   FAIL: {len(rib_fail)}/{n} = {100*len(rib_fail)/max(1,n):.1f}%   "
          f"(reviewed {rib_fail_rev}, unreviewed {rib_fail_unrev})")
    print(f"   PASS: {n-len(rib_fail)}/{n}")
    print("\n2) VERTEBRA IN FOV BUT NOT ANNOTATED  -> missing_vertebra.csv")
    print(f"   interior GAP (bracketed, definitely in-FOV): {gap_cases} cases  <- high-confidence fixes")
    print(f"   rib ABOVE the annotated spine top:           {above_cases} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
