"""
vertebra_topology_qc.py — ground-truth-FREE quality metrics for vertebra labels.

Operationalizes the "mixing neighboring vertebrae classes" failure mode
(Wasserthal et al, Radiology:AI 2023, Fig 4) as objective numbers computed from
a single label map — no reference segmentation required. Because they are
intrinsic, the SAME metrics run on a radiologist tree and a pseudolabel tree,
so you can compare the distributions and quantify how much more of this failure
the pseudolabels carry.

Per case (vertebra classes L1..L6 = 1..6):
  off_main_frac        fraction of vertebra voxels NOT in their label's largest
                       connected component — islands of one label stranded
                       inside a neighbour (the literal class-mixing).
  n_fragmented         how many vertebra labels split into >1 component.
  n_order_inversions   adjacent present labels whose craniocaudal centroid order
                       disagrees with the label index (a level swap, e.g. L3
                       sitting superior to L2).
  adj_overlap_frac     mean craniocaudal extent OVERLAP of adjacent vertebrae,
                       as a fraction of the smaller one's extent (interdigitation).
  n_nonadjacent_touch  pairs of labels >=2 apart that physically touch (a level
                       merged or skipped, e.g. L1 abutting L3).
  mixing_flag          1 if any of: off_main_frac > --tol, an order inversion,
                       or a non-adjacent touch.

Usage
-----
  python scripts/vertebra_topology_qc.py --tree data/hf_export     --out data/qc_manual.csv
  python scripts/vertebra_topology_qc.py --tree data/hf_export_v2  --out data/qc_pseudo.csv \
      --compare data/qc_manual.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from intensity_refine import _load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.vertebra_topology_qc")

VERT_LABELS = (1, 2, 3, 4, 5, 6)        # L1..L6, craniocaudal (L1 most superior)


# ===========================================================================
# Pure core (unit-tested)
# ===========================================================================

def vertebra_topology_metrics(label, affine, *, vert_labels=VERT_LABELS,
                              tol: float = 0.005) -> Dict[str, float]:
    """Intrinsic neighbour-mixing / topology metrics for one vertebra label map.

    `affine` is the NIfTI voxel→world affine; it is used only to find the
    superior–inferior axis and its sign, so the metric is orientation-robust.
    Returns a flat dict (see module docstring). Pure aside from numpy/scipy.
    """
    import numpy as np
    from scipy.ndimage import (label as cc_label, generate_binary_structure,
                               binary_dilation, find_objects, center_of_mass)
    lab = np.asarray(label)
    affine = np.asarray(affine, dtype=np.float64)

    # Superior–inferior voxel axis + sign (which way is "up" in world S+).
    si_axis = int(np.argmax(np.abs(affine[2, :3])))
    si_sign = 1.0 if affine[2, si_axis] >= 0 else -1.0

    present = [int(c) for c in vert_labels if np.any(lab == c)]
    struct = generate_binary_structure(lab.ndim, lab.ndim)   # full (26-)connectivity

    # ── fragmentation / off-main islands ────────────────────────────────────
    total_vox = 0
    offmain_vox = 0
    n_fragmented = 0
    for c in present:
        m = lab == c
        cc, n = cc_label(m, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]        # drop background count
        tot = int(sizes.sum())
        total_vox += tot
        offmain_vox += tot - int(sizes.max())
        if n > 1:
            n_fragmented += 1
    off_main_frac = (offmain_vox / total_vox) if total_vox else 0.0

    # ── craniocaudal ordering + adjacent overlap (along si_axis) ────────────
    boxes = find_objects(lab.astype(np.int32))
    coms = center_of_mass(np.ones(lab.shape, dtype=np.uint8), lab,
                          present) if present else []
    coms = {c: com for c, com in zip(present, np.atleast_2d(coms))} if present else {}
    si_span: Dict[int, Tuple[float, float]] = {}
    si_pos: Dict[int, float] = {}
    for c in present:
        sl = boxes[c - 1]                          # find_objects is 1-indexed
        if sl is None:
            continue
        lo, hi = sl[si_axis].start, sl[si_axis].stop - 1
        si_span[c] = (float(lo), float(hi))
        si_pos[c] = si_sign * float(coms[c][si_axis])   # higher = more superior

    n_order_inversions = 0
    overlaps: List[float] = []
    ordered = [c for c in present if c in si_pos]
    for a, b in zip(ordered, ordered[1:]):
        # label index increases (a<b) => b should be more INFERIOR => si_pos[b] < si_pos[a]
        if si_pos[b] >= si_pos[a]:
            n_order_inversions += 1
        if b - a == 1 and a in si_span and b in si_span:    # adjacent levels
            la, ha = si_span[a]; lb, hb = si_span[b]
            ov = max(0.0, min(ha, hb) - max(la, lb))
            ext = min(ha - la, hb - lb) + 1.0
            overlaps.append(ov / ext if ext > 0 else 0.0)
    adj_overlap_frac = float(np.mean(overlaps)) if overlaps else 0.0

    # ── non-adjacent contact (labels >=2 apart that touch) ──────────────────
    n_nonadjacent_touch = 0
    nonadjacent_touch_vox = 0
    for ai in range(len(present)):
        for bi in range(ai + 1, len(present)):
            a, b = present[ai], present[bi]
            if b - a < 2:
                continue
            if a not in si_span or b not in si_span:
                continue
            tv = _touch_voxels(lab, a, b, boxes, struct)
            if tv > 0:
                n_nonadjacent_touch += 1
                nonadjacent_touch_vox += tv

    mixing_flag = int(off_main_frac > tol or n_order_inversions > 0
                      or n_nonadjacent_touch > 0)
    return {
        "n_vertebrae": len(present),
        "off_main_frac": round(off_main_frac, 6),
        "n_fragmented": n_fragmented,
        "n_order_inversions": n_order_inversions,
        "adj_overlap_frac": round(adj_overlap_frac, 6),
        "n_nonadjacent_touch": n_nonadjacent_touch,
        "nonadjacent_touch_vox": int(nonadjacent_touch_vox),
        "mixing_flag": mixing_flag,
    }


def _touch_voxels(lab, a, b, boxes, struct) -> int:
    """# voxels of label b adjacent to label a, computed on the cropped union
    bbox so it stays cheap on big volumes (0 if their bboxes don't even meet)."""
    import numpy as np
    from scipy.ndimage import binary_dilation
    sa, sb = boxes[a - 1], boxes[b - 1]
    if sa is None or sb is None:
        return 0
    sl = []
    for da, db in zip(sa, sb):
        lo = min(da.start, db.start) - 1
        hi = max(da.stop, db.stop) + 1
        lo = max(lo, 0)
        sl.append(slice(lo, hi))
    sub = lab[tuple(sl)]
    ma = sub == a
    if not ma.any():
        return 0
    grown = binary_dilation(ma, structure=struct)
    return int((grown & (sub == b)).sum())


# ===========================================================================
# Orchestrator
# ===========================================================================

_FIELDS = ["token", "config", "n_vertebrae", "off_main_frac", "n_fragmented",
           "n_order_inversions", "adj_overlap_frac", "n_nonadjacent_touch",
           "nonadjacent_touch_vox", "mixing_flag"]


def _qc_one(task: dict) -> Optional[dict]:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        img = nib.load(task["label_path"])
        lab = np.asarray(img.dataobj)
        m = vertebra_topology_metrics(lab, img.affine, tol=task["tol"])
        if m["n_vertebrae"] < 2:          # need >=2 levels to judge adjacency
            return None
        return {"token": tok, "config": task["config"], **m}
    except Exception as exc:              # noqa: BLE001
        log.warning("token=%s: QC failed (%s)", tok, exc)
        return None


def _summarize(rows: List[dict], name: str) -> dict:
    import numpy as np
    if not rows:
        return {"name": name, "n": 0}
    def pct(key, p):
        return float(np.percentile([r[key] for r in rows], p))
    n = len(rows)
    flagged = sum(r["mixing_flag"] for r in rows)
    return {
        "name": name, "n": n,
        "pct_flagged": round(100.0 * flagged / n, 1),
        "off_main_frac_mean": round(float(np.mean([r["off_main_frac"] for r in rows])), 6),
        "off_main_frac_p95": round(pct("off_main_frac", 95), 6),
        "order_inv_cases": sum(1 for r in rows if r["n_order_inversions"] > 0),
        "nonadj_touch_cases": sum(1 for r in rows if r["n_nonadjacent_touch"] > 0),
        "adj_overlap_p95": round(pct("adj_overlap_frac", 95), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path,
                    help="export tree with manifest.json + labels/")
    ap.add_argument("--out", required=True, type=Path, help="per-case CSV")
    ap.add_argument("--compare", type=Path, default=None,
                    help="another QC CSV to print side-by-side in the summary")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="off_main_frac above this trips mixing_flag (default .005)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = args.tree / "manifest.json"
    if not man.exists():
        log.error("no manifest.json in %s", args.tree)
        return 1
    records = [r for r in _load_manifest(man) if r.get("label_file")]
    if args.limit:
        records = records[:args.limit]
    tasks = [{"token": str(r.get("token")), "config": r.get("config"),
              "label_path": str(args.tree / r["label_file"]), "tol": args.tol}
             for r in records if (args.tree / r["label_file"]).exists()]
    log.info("vertebra_topology_qc: %d label maps, %d workers", len(tasks), args.workers)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_qc_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 100 == 0 or i == len(futs):
                log.info("  %d/%d processed (%d with >=2 vertebrae)",
                         i, len(futs), len(rows))

    rows.sort(key=lambda r: (r["mixing_flag"], r["off_main_frac"],
                             r["n_nonadjacent_touch"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    this = _summarize(rows, args.tree.name)
    log.info("=" * 64)
    log.info("QC SUMMARY")
    _log_summary(this)
    if args.compare and args.compare.exists():
        other_rows = list(csv.DictReader(open(args.compare)))
        for r in other_rows:                      # csv reads strings
            for k in ("off_main_frac", "adj_overlap_frac"):
                r[k] = float(r[k])
            for k in ("n_order_inversions", "n_nonadjacent_touch", "mixing_flag"):
                r[k] = int(r[k])
        log.info("-" * 64)
        _log_summary(_summarize(other_rows, args.compare.stem))
    log.info("=" * 64)
    log.info("wrote per-case CSV -> %s", args.out)
    return 0


def _log_summary(s: dict) -> None:
    if s.get("n", 0) == 0:
        log.info("  %s: no cases with >=2 vertebrae", s["name"]); return
    log.info("  %-22s n=%-4d  flagged=%.1f%%", s["name"], s["n"], s["pct_flagged"])
    log.info("      off_main_frac mean=%.4f p95=%.4f | order-inv cases=%d | "
             "non-adj-touch cases=%d | adj_overlap p95=%.3f",
             s["off_main_frac_mean"], s["off_main_frac_p95"],
             s["order_inv_cases"], s["nonadj_touch_cases"], s["adj_overlap_p95"])


if __name__ == "__main__":
    raise SystemExit(main())
