"""
scripts/detect_stump_ribs.py — find thoracolumbar stump ribs + measure residual QC error, over the
WHOLE v4 dataset, so the decision "do students need to review more cases" is made on NUMBERS.

Two things QC alone cannot tell you:

  1. MISSED STUMP RIBS. The rib pipeline is Moller (binary rib foreground) grafted onto TS numbering,
     keeping only Moller voxels connected to a TS rib. A stump rib that Moller segments but TS never
     NUMBERS (it called the level L1 and left the little rib as background) has no TS rib to attach to
     and is DROPPED. QC is blind to it -- an absent rib flags nothing. So we detect stump ribs
     directly from the final label:
       * a rib whose head articulates with a LUMBAR vertebra (L1, id 20) -> transitional rib, OR
       * the lowest rib on a side is SHORT (<= STUMP_MM, Moller's 38 mm) -> stump rib.
     These are TLTV markers -- the phenotype the dataset exists to catalogue.

  2. RESIDUAL QC ERROR in the ~642 cases that were never student-reviewed (they passed only the OLD,
     weak triage). We run the CURRENT gates (mixing / structure / detached / rib<->vertebra) so you
     get the true error rate, split reviewed vs unreviewed -> a measurement, not a commitment to
     re-review 802.

Output: prevalence numbers + two candidate CSVs (stump-rib candidates, QC-fail candidates) = the
bounded student worklist, if any.

  python scripts/detect_stump_ribs.py [--limit N] [--workers 16]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
L1 = 20                               # lumbar L1 id
STUMP_MM = 38.0                       # Moller: a stump rib is <= 38 mm at the lowest thoracic level


def _rib_length_mm(mask: np.ndarray, spacing) -> float:
    """Cheap length proxy: max pairwise distance across a subsample of the rib's voxels (mm). A stump
    rib is <=38 mm; a full rib is ~150-250 mm, so max-extent separates them cleanly without Moller's
    full path algorithm."""
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    if len(idx) > 300:
        idx = idx[:: len(idx) // 300]
    pts = idx * np.asarray(spacing)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def analyze(lab: np.ndarray, affine, run_qc: bool = False):
    """Return (stump_ribs, qc_fails). stump_ribs = list of (rib_name, reason).
    The 4 QC gates each run a distance transform (CPU-bound, GIL-held) so they are OFF by default --
    the stump detection is the fast headline; add --qc for the residual-error measurement."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    names = RA._id2name()
    ribs_present = [r for r in range(LO, HI + 1) if (lab == r).any()]
    stumps = []

    # (a) rib articulating with L1 -> transitional rib (nearest-vertebra from the rib head)
    if (lab == L1).any() and ribs_present:
        thor_lum = (lab >= 8) & (lab <= 25)              # T1..L6 span
        idx = np.argwhere(thor_lum | np.isin(lab, ribs_present))
        lo = idx.min(0); hi = idx.max(0) + 1
        sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3)); f = 2
        sub = lab[sl][::f, ::f, ::f]
        vmask = (sub >= 8) & (sub <= 25)
        if vmask.any():
            d, ind = ndimage.distance_transform_edt(~vmask, sampling=spacing * f, return_indices=True)
            for r in ribs_present:
                m = (sub == r)
                if not m.any():
                    continue
                dd = np.where(m, d, np.inf)
                p = np.unravel_index(np.argmin(dd), dd.shape)
                v = int(sub[ind[0][p], ind[1][p], ind[2][p]])
                if v == L1:
                    stumps.append((names.get(r, r), "articulates_with_L1"))

    # (b) a SHORT lowest rib -> stump rib (Moller length rule)
    for side, off in (("left", LS.RIB_LEFT_OFFSET), ("right", LS.RIB_RIGHT_OFFSET)):
        side_ribs = [off + n for n in range(1, 13) if (lab == off + n).any()]
        if not side_ribs:
            continue
        lowest = max(side_ribs)                          # highest number = lowest rib
        L = _rib_length_mm(lab == lowest, spacing)
        if 0 < L <= STUMP_MM:
            stumps.append((names.get(lowest, lowest), f"short_{L:.0f}mm"))

    fails = []
    if run_qc:
        if not RA.rib_label_mixing(lab, affine)[0]:
            fails.append("mixing")
        if not RA.structure_integrity(lab, affine)[0]:
            fails.append("structure")
        if not RA.rib_spine_gap(lab, affine)[0]:
            fails.append("detached")
        # rib_vertebra_match (misnumbered) is DELIBERATELY excluded here: it runs against RAW TS
        # numbering, which is unreliable exactly at transitional/FOV levels, so on the pseudo it
        # over-flags. It is only valid on the corrected spine after adjudication.
    # de-dup stump reasons per rib
    seen = {}
    for rn, why in stumps:
        seen.setdefault(rn, why)
    return [(k, v) for k, v in seen.items()], fails


def _work(args):
    """Module-level worker (picklable for ProcessPoolExecutor -> TRUE parallelism, not GIL-bound
    threads). Each process re-imports the module; HF_TOKEN comes from the inherited environment."""
    t, p, rev, run_qc = args
    tok = os.environ["HF_TOKEN"]
    try:
        img = nib.load(hf_hub_download(DS, p, repo_type="dataset", token=tok, revision="v4"))
        stumps, fails = analyze(np.asanyarray(img.dataobj), img.affine, run_qc=run_qc)
        return (t, rev, stumps, fails)
    except Exception:                                    # noqa: BLE001
        return None


def main(argv=None) -> int:
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="sample only N cases (0 = all)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--qc", action="store_true", help="also run the (slower) residual-QC gates")
    a = ap.parse_args(argv)
    tok = os.environ["HF_TOKEN"]

    mp = hf_hub_download(DS, "manifest.json", repo_type="dataset", token=tok, revision="v4")
    recs = json.load(open(mp)); recs = recs if isinstance(recs, list) else recs.get("records", [])
    # which tokens were student-reviewed (the rib worklist)?
    reviewed = set()
    try:
        wp = hf_hub_download(DS, "rib_worklist.json", repo_type="dataset", token=tok, revision="v4")
        wl = json.load(open(wp)); reviewed = {str(t) for t in (wl.get("tokens") if isinstance(wl, dict) else wl)}
    except Exception:                                    # noqa: BLE001
        pass
    items = [(str(r.get("token")), r.get("pseudo_label_file") or r.get("label_file"),
              str(r.get("token")) in reviewed, a.qc)
             for r in recs if (r.get("pseudo_label_file") or r.get("label_file"))]
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][:a.limit]
    print(f"scanning {len(items)} v4 cases ({len(reviewed)} reviewed) "
          f"[qc={'on' if a.qc else 'off'}, {a.workers} processes]\n", flush=True)

    stump_rows, qc_rows = [], []
    n = 0
    n_l1 = n_short = 0                                    # split the two stump signals
    fail_by = Counter(); err_unrev = tot_unrev = 0
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, items, chunksize=4):
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(items)}", flush=True)
            if not r:
                continue
            t, rev, stumps, fails = r
            n += 1
            if not rev:
                tot_unrev += 1
            has_l1 = any(v == "articulates_with_L1" for _, v in stumps)
            has_short = any(v.startswith("short_") for _, v in stumps)
            if has_l1: n_l1 += 1
            if has_short: n_short += 1
            if stumps:
                stump_rows.append({"token": t, "reviewed": rev,
                                   "on_L1": has_l1, "short_rib": has_short,
                                   "detail": "; ".join(f"{k}({v})" for k, v in stumps)})
            if fails:
                for f in fails: fail_by[f] += 1
                if not rev:
                    err_unrev += 1
                    qc_rows.append({"token": t, "fails": " ".join(fails)})

    with open("stump_rib_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "on_L1", "short_rib", "detail"])
        w.writeheader(); w.writerows(stump_rows)
    with open("qc_fail_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "fails"]); w.writeheader(); w.writerows(qc_rows)

    l1_unrev = sum(1 for r in stump_rows if r["on_L1"] and not r["reviewed"])
    print(f"\n===== RESULTS ({n} cases) =====")
    print("\nTLTV / STUMP RIBS  -> stump_rib_candidates.csv")
    print(f"   rib articulates with L1 (HIGH-confidence TLTV): {n_l1}/{n} = {100*n_l1/n:.1f}%")
    print(f"      of which UNREVIEWED (a possibly-missed phenotype): {l1_unrev}")
    print(f"   lowest rib short <=38mm (noisy -- FOV clip inflates this): {n_short}/{n} = {100*n_short/n:.1f}%")
    if a.qc:
        print(f"\nRESIDUAL QC in UNREVIEWED cases (mixing/structure/detached; misnumbered excluded):")
        print(f"   {err_unrev}/{tot_unrev} = {100*err_unrev/max(1,tot_unrev):.1f}%   -> qc_fail_candidates.csv")
        print(f"   fail types: {dict(fail_by)}")
    print(f"\nBOUNDED STUDENT WORK: {l1_unrev} high-confidence TLTV to confirm"
          + (f" + {err_unrev} QC-fails in unreviewed cases" if a.qc else " (add --qc for the error count)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
