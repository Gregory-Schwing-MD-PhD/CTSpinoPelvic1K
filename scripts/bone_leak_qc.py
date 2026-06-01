"""
bone_leak_qc.py — ground-truth-FREE check for label bled OFF the bone.

Flags the over-segmentation failure: voxels labelled as a bone class whose CT
attenuation is soft tissue / air AND that are NOT enclosed marrow — i.e. the
label spills past the cortical shell into background. Because it only needs the
label + its CT, the SAME metric runs on the radiologist tree and the pseudolabel
tree, so you can compare leak against the human baseline (--compare).

A low-HU labelled voxel is only a LEAK if it is exposed (not surrounded by
cortex): fatty marrow is low-HU too, so we keep only voxels that the structure's
own bone shell does not enclose (the _solid_fill enclosure test, same one
intensity_refine/clip use).

Per case (over bone classes 1..9):
  off_bone_frac   fraction of labelled voxels that are exposed sub-bone-HU
                  (bled off the bone) — the headline leak number.
  bg_leak_frac    of those, the fraction bled into AIR/background (CT < --bg_hu)
                  — the most egregious leaks.
  worst_class     the class with the highest leak fraction, and that fraction.
  leak_flag       1 if off_bone_frac > --tol.

Usage
-----
  python scripts/bone_leak_qc.py --tree data/hf_export     --out data/leak_manual.csv
  python scripts/bone_leak_qc.py --tree data/hf_export_v2  --out data/leak_pseudo.csv \
      --compare data/leak_manual.csv
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
from intensity_refine import _load_manifest, _solid_fill  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.bone_leak_qc")

BONE_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
CLASS_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
               7: "sacrum", 8: "left_hip", 9: "right_hip"}


# ===========================================================================
# Pure core (unit-tested)
# ===========================================================================

def bone_leak_metrics(label, ct, *, bone_hu: float = 150.0, bg_hu: float = -200.0,
                      fg_labels=BONE_CLASSES, tol: float = 0.02) -> Dict[str, float]:
    """Off-bone label-leak metrics for one (label, CT) pair. See module doc.

    leak_c   = (label==c) AND NOT solid_fill((label==c) & CT>=bone_hu)
               i.e. labelled voxels neither bone nor enclosed marrow.
    bg_leak  = leak voxels with CT < bg_hu (bled into air).
    """
    import numpy as np
    lab = np.asarray(label)
    ct = np.asarray(ct, dtype=np.float32)

    total_fg = 0
    total_leak = 0
    total_bg_leak = 0
    worst_class = 0
    worst_frac = 0.0
    for c in fg_labels:
        m = lab == c
        n = int(m.sum())
        if n == 0:
            continue
        bone = m & (ct >= bone_hu)
        solid = _solid_fill(bone)          # bone + enclosed marrow
        leak = m & ~solid                  # exposed sub-bone-HU label = leak
        lk = int(leak.sum())
        total_fg += n
        total_leak += lk
        total_bg_leak += int((leak & (ct < bg_hu)).sum())
        frac = lk / n if n else 0.0
        if frac > worst_frac:
            worst_frac, worst_class = frac, c

    off_bone_frac = (total_leak / total_fg) if total_fg else 0.0
    bg_leak_frac = (total_bg_leak / total_fg) if total_fg else 0.0
    return {
        "fg_vox": total_fg,
        "off_bone_frac": round(off_bone_frac, 6),
        "bg_leak_frac": round(bg_leak_frac, 6),
        "leak_vox": total_leak,
        "bg_leak_vox": total_bg_leak,
        "worst_class": CLASS_NAMES.get(worst_class, ""),
        "worst_class_frac": round(worst_frac, 6),
        "leak_flag": int(off_bone_frac > tol),
    }


# ===========================================================================
# Orchestrator
# ===========================================================================

_FIELDS = ["token", "config", "fg_vox", "off_bone_frac", "bg_leak_frac",
           "leak_vox", "bg_leak_vox", "worst_class", "worst_class_frac",
           "leak_flag"]


def _leak_one(task: dict) -> Optional[dict]:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        lab = np.asarray(nib.load(task["label_path"]).dataobj)
        ct = np.asarray(nib.load(task["ct_path"]).dataobj).astype(np.float32)
        if lab.shape[:3] != ct.shape[:3]:
            return None
        m = bone_leak_metrics(lab, ct, bone_hu=task["bone_hu"],
                              bg_hu=task["bg_hu"], tol=task["tol"])
        if m["fg_vox"] == 0:
            return None
        return {"token": tok, "config": task["config"], **m}
    except Exception as exc:                 # noqa: BLE001
        log.warning("token=%s: leak QC failed (%s)", tok, exc)
        return None


def _summarize(rows: List[dict], name: str) -> dict:
    import numpy as np
    if not rows:
        return {"name": name, "n": 0}
    of = [r["off_bone_frac"] for r in rows]
    return {
        "name": name, "n": len(rows),
        "pct_flagged": round(100.0 * sum(r["leak_flag"] for r in rows) / len(rows), 1),
        "off_bone_mean": round(float(np.mean(of)), 6),
        "off_bone_p95": round(float(np.percentile(of, 95)), 6),
        "bg_leak_mean": round(float(np.mean([r["bg_leak_frac"] for r in rows])), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--bone_hu", type=float, default=150.0,
                    help="HU at/above which a voxel counts as bone (default 150)")
    ap.add_argument("--bg_hu", type=float, default=-200.0,
                    help="leak voxels below this HU count as air/background leak")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="off_bone_frac above this trips leak_flag (default .02)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = args.tree / "manifest.json"
    if not man.exists():
        log.error("no manifest.json in %s", args.tree)
        return 1
    records = [r for r in _load_manifest(man)
               if r.get("label_file") and r.get("ct_file")]
    if args.limit:
        records = records[:args.limit]
    tasks = [{"token": str(r.get("token")), "config": r.get("config"),
              "label_path": str(args.tree / r["label_file"]),
              "ct_path": str(args.tree / r["ct_file"]),
              "bone_hu": args.bone_hu, "bg_hu": args.bg_hu, "tol": args.tol}
             for r in records
             if (args.tree / r["label_file"]).exists()
             and (args.tree / r["ct_file"]).exists()]
    log.info("bone_leak_qc: %d cases, %d workers (bone_hu=%.0f)",
             len(tasks), args.workers, args.bone_hu)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_leak_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 100 == 0 or i == len(futs):
                log.info("  %d/%d processed (%d kept)", i, len(futs), len(rows))

    rows.sort(key=lambda r: (r["leak_flag"], r["off_bone_frac"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    log.info("=" * 64)
    log.info("BONE-LEAK SUMMARY")
    _log_summary(_summarize(rows, args.tree.name))
    if args.compare and args.compare.exists():
        other = list(csv.DictReader(open(args.compare)))
        for r in other:
            r["off_bone_frac"] = float(r["off_bone_frac"])
            r["bg_leak_frac"] = float(r["bg_leak_frac"])
            r["leak_flag"] = int(r["leak_flag"])
        log.info("-" * 64)
        _log_summary(_summarize(other, args.compare.stem))
    log.info("=" * 64)
    log.info("wrote per-case CSV -> %s", args.out)
    return 0


def _log_summary(s: dict) -> None:
    if s.get("n", 0) == 0:
        log.info("  %s: no cases", s["name"]); return
    log.info("  %-22s n=%-4d  flagged=%.1f%%", s["name"], s["n"], s["pct_flagged"])
    log.info("      off_bone_frac mean=%.4f p95=%.4f | bg(air)_leak mean=%.4f",
             s["off_bone_mean"], s["off_bone_p95"], s["bg_leak_mean"])


if __name__ == "__main__":
    raise SystemExit(main())
