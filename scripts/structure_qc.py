"""
structure_qc.py — structure-level GT-free QC: presence, duplication, missing
levels, and LEFT/RIGHT hip swap. Label-only (+ the NIfTI affine for laterality),
so the SAME metric runs on the radiologist tree and the pseudolabels and is
baselined by --compare.

Per case (classes L1..L6=1..6, sacrum=7, left_hip=8, right_hip=9):
  vertebra_gap     missing levels within the present spine span (L1,L2,L4 -> 1).
  pelvis_incomplete  1 or 2 of {sacrum,left_hip,right_hip} present (a hemipelvis
                   or sacrum dropped) — 0 if none or all three present.
  n_dup_classes    classes split into >=2 LARGE components (a duplicated /
                   broken structure), and duplication_flag = (n_dup_classes>0).
  lr_swap          1 if, with sacrum + both hips present, the hip labelled
                   right_hip sits on the patient's LEFT of left_hip along the
                   world L-R axis (affine). lr_same_side = both hips on the same
                   side of the sacrum (a gross laterality error).
  struct_flag      1 if any of the above fire.

LATERALITY / ORIENTATION NOTE
-----------------------------
"Right" is taken as the +world-X end (NIfTI RAS+). This is VALIDATED by the
radiologist baseline: that tree should show lr_swap ~0%. If it instead shows
~100%, your export's L-R is flipped relative to RAS+ — rerun with
--flip_lr to invert the convention (the summary warns when manual swap >50%).

Usage
-----
  python scripts/structure_qc.py --tree data/hf_export     --out data/struct_manual.csv
  python scripts/structure_qc.py --tree data/hf_export_v2  --out data/struct_pseudo.csv \
      --compare data/struct_manual.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from intensity_refine import _load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.structure_qc")

SPINE = (1, 2, 3, 4, 5, 6)
SACRUM, LEFT_HIP, RIGHT_HIP = 7, 8, 9


def _world_x(affine, com):
    import numpy as np
    a = np.asarray(affine, dtype=np.float64)
    return float(a[0, 0] * com[0] + a[0, 1] * com[1] + a[0, 2] * com[2] + a[0, 3])


def structure_metrics(label, affine, *, min_dup_vox: int = 500,
                      dup_ratio: float = 0.2, flip_lr: bool = False) -> Dict[str, float]:
    """Structure-level QC for one label map (+ affine). See module docstring."""
    import numpy as np
    from scipy.ndimage import (label as cc_label, generate_binary_structure,
                               center_of_mass)
    lab = np.asarray(label)
    struct = generate_binary_structure(lab.ndim, lab.ndim)

    present = [c for c in (*SPINE, SACRUM, LEFT_HIP, RIGHT_HIP) if np.any(lab == c)]

    # ── duplication: a class split into >=2 SUBSTANTIAL components ───────────
    # A real duplicate/split has a second component that is both absolutely
    # large (>= min_dup_vox) AND a meaningful FRACTION of the largest
    # (>= dup_ratio). The fraction gate rejects small fragments/specks hanging
    # off an otherwise-fine structure, which a pure size cutoff over-flags.
    _NAME = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
             7: "sacrum", 8: "left_hip", 9: "right_hip"}
    n_dup = 0
    dup_classes: List[str] = []          # which structures are duplicated
    for c in present:
        cc, n = cc_label(lab == c, structure=struct)
        if n < 2:
            continue
        sizes = np.sort(np.bincount(cc.ravel())[1:])[::-1]   # descending
        largest, second = int(sizes[0]), int(sizes[1])
        if second >= min_dup_vox and second >= dup_ratio * largest:
            n_dup += 1
            dup_classes.append(_NAME.get(c, str(c)))

    # ── vertebra gap: missing levels inside the present spine span ───────────
    sp = [c for c in SPINE if c in present]
    vertebra_gap = (max(sp) - min(sp) + 1 - len(sp)) if len(sp) >= 2 else 0

    # ── pelvis completeness ──────────────────────────────────────────────────
    pelvis = [c for c in (SACRUM, LEFT_HIP, RIGHT_HIP) if c in present]
    pelvis_incomplete = int(0 < len(pelvis) < 3)

    # ── left/right hip swap (needs sacrum + both hips) ───────────────────────
    lr_known = int(all(c in present for c in (SACRUM, LEFT_HIP, RIGHT_HIP)))
    lr_swap = 0
    lr_same_side = 0
    lhx = shx = rhx = float("nan")
    if lr_known:
        coms = center_of_mass(np.ones(lab.shape, dtype=np.uint8), lab,
                              [LEFT_HIP, SACRUM, RIGHT_HIP])
        sgn = -1.0 if flip_lr else 1.0        # +world-x = patient Right (RAS+)
        lhx = sgn * _world_x(affine, coms[0])
        shx = sgn * _world_x(affine, coms[1])
        rhx = sgn * _world_x(affine, coms[2])
        # expected: left_hip more-left (smaller x) than right_hip
        lr_swap = int(rhx < lhx)
        # both hips on the same side of the sacrum midline = gross error
        lr_same_side = int((lhx - shx) * (rhx - shx) > 0)

    struct_flag = int(n_dup > 0 or vertebra_gap > 0 or pelvis_incomplete
                      or lr_swap or lr_same_side)
    return {
        "n_present": len(present),
        "vertebra_gap": int(vertebra_gap),
        "pelvis_incomplete": pelvis_incomplete,
        "n_dup_classes": n_dup,
        "dup_classes": ",".join(dup_classes),    # e.g. "L3,L4"
        "duplication_flag": int(n_dup > 0),
        "lr_known": lr_known,
        "lr_swap": lr_swap,
        "lr_same_side": lr_same_side,
        "left_hip_x": round(lhx, 1) if lr_known else "",
        "sacrum_x": round(shx, 1) if lr_known else "",
        "right_hip_x": round(rhx, 1) if lr_known else "",
        "struct_flag": struct_flag,
    }


# ===========================================================================
# Orchestrator
# ===========================================================================

_FIELDS = ["token", "config", "n_present", "vertebra_gap", "pelvis_incomplete",
           "n_dup_classes", "dup_classes", "duplication_flag", "lr_known", "lr_swap",
           "lr_same_side", "left_hip_x", "sacrum_x", "right_hip_x", "struct_flag"]


def _qc_one(task: dict) -> Optional[dict]:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        img = nib.load(task["label_path"])
        lab = np.asarray(img.dataobj)
        m = structure_metrics(lab, img.affine, dup_ratio=task["dup_ratio"],
                              flip_lr=task["flip_lr"])
        return {"token": tok, "config": task["config"], **m}
    except Exception as exc:                 # noqa: BLE001
        log.warning("token=%s: structure QC failed (%s)", tok, exc)
        return None


def _summarize(rows: List[dict], name: str) -> dict:
    lr = [r for r in rows if int(r["lr_known"])]
    n = len(rows)
    return {
        "name": name, "n": n, "n_lr": len(lr),
        "pct_flagged": round(100.0 * sum(int(r["struct_flag"]) for r in rows) / n, 1) if n else 0.0,
        "lr_swap_pct": round(100.0 * sum(int(r["lr_swap"]) for r in lr) / len(lr), 1) if lr else 0.0,
        "same_side": sum(int(r["lr_same_side"]) for r in lr),
        "vertebra_gap_cases": sum(1 for r in rows if int(r["vertebra_gap"]) > 0),
        "pelvis_incomplete_cases": sum(int(r["pelvis_incomplete"]) for r in rows),
        "dup_cases": sum(int(r["duplication_flag"]) for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--dup_ratio", type=float, default=0.2,
                    help="flag duplication only when the 2nd-largest component "
                         "is >= this fraction of the largest (default 0.2) — "
                         "rejects specks, keeps real splits.")
    ap.add_argument("--flip_lr", action="store_true",
                    help="invert the L-R convention (use if the radiologist "
                         "baseline shows lr_swap ~100%)")
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
              "label_path": str(args.tree / r["label_file"]),
              "dup_ratio": args.dup_ratio, "flip_lr": args.flip_lr}
             for r in records if (args.tree / r["label_file"]).exists()]
    log.info("structure_qc: %d label maps, %d workers", len(tasks), args.workers)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_qc_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 100 == 0 or i == len(futs):
                log.info("  %d/%d processed", i, len(futs))

    rows.sort(key=lambda r: (int(r["struct_flag"]), int(r["lr_swap"])), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    log.info("=" * 64)
    log.info("STRUCTURE QC SUMMARY")
    this = _summarize(rows, args.tree.name)
    _log_summary(this)
    if this["n_lr"] and this["lr_swap_pct"] > 50.0:
        log.warning("  >> lr_swap is %.0f%% here — your L-R is likely flipped vs "
                    "RAS+; rerun with --flip_lr.", this["lr_swap_pct"])
    if args.compare and args.compare.exists():
        other = list(csv.DictReader(open(args.compare)))
        log.info("-" * 64)
        _log_summary(_summarize(other, args.compare.stem))
    log.info("=" * 64)
    log.info("wrote per-case CSV -> %s", args.out)
    return 0


def _log_summary(s: dict) -> None:
    if s.get("n", 0) == 0:
        log.info("  %s: no cases", s["name"]); return
    log.info("  %-22s n=%-4d  flagged=%.1f%%", s["name"], s["n"], s["pct_flagged"])
    log.info("      lr_swap=%.1f%% (of %d w/ pelvis) | same-side=%d | "
             "vertebra-gap=%d | pelvis-incomplete=%d | duplicated=%d",
             s["lr_swap_pct"], s["n_lr"], s["same_side"], s["vertebra_gap_cases"],
             s["pelvis_incomplete_cases"], s["dup_cases"])


if __name__ == "__main__":
    raise SystemExit(main())
