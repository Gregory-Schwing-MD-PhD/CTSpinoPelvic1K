"""
export_review_crops.py — make small CT+mask CROPS of the QC-flagged cases so
review (ITK-SNAP or a future web viewer) opens a few MB instead of the full
~200 MB volume. No HTTP-range tricks needed (and gzipped NIfTI can't be range-
read anyway): we just cut a padded bounding box server-side.

The crop is the bounding box of ALL labelled bone + padding — it drops the empty
air/table/FOV but KEEPS the whole spine+pelvis, so reviewers retain the
anatomical context they need to recount levels / judge L-R. The crop's NIfTI
affine is preserved (nibabel slicer), and a crop.json records the voxel offset
so `reviewtool fix-list --crops` can paste the correction back into the full-res
mask.

  python scripts/export_review_crops.py \
      --qc_csv data/qc_master.csv --tree data/hf_export_v2 \
      --out data/review_crops [--pad 8] [--limit N]

Pure crop/paste helpers (foreground_bbox, paste_back, crop_dirname) are
unit-tested and reused by reviewtool.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from intensity_refine import _load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.export_review_crops")


# ── pure helpers (unit-tested; shared with reviewtool) ──────────────────────

def crop_dirname(token, config) -> str:
    """Stable per-case crop folder name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{token}__{config}")


def foreground_bbox(label, pad: int = 8):
    """Voxel bbox (tuple of slices) of nonzero `label` grown by `pad`, clipped
    to the array. None if the label is empty."""
    import numpy as np
    coords = np.argwhere(np.asarray(label) > 0)
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(0) - pad, 0)
    hi = np.minimum(coords.max(0) + pad + 1, label.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def paste_back(full_label, crop_label, origin):
    """Return a copy of `full_label` with `crop_label` written in at voxel
    `origin` (the bbox starts). Used to fold an edited crop back to full-res."""
    import numpy as np
    out = np.asarray(full_label).copy()
    sl = tuple(slice(int(o), int(o) + int(s))
               for o, s in zip(origin, crop_label.shape))
    out[sl] = np.asarray(crop_label).astype(out.dtype)
    return out


# ── orchestrator ────────────────────────────────────────────────────────────

def _flagged_rows(qc_csv: Path) -> List[dict]:
    rows = list(csv.DictReader(open(qc_csv)))
    return [r for r in rows if str(r.get("needs_review", "")).strip() in ("1", "1.0")]


def _export_one(task: dict) -> dict:
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        lbl_img = nib.load(task["label_path"])
        lab = np.asarray(lbl_img.dataobj)
        bbox = foreground_bbox(lab, pad=task["pad"])
        if bbox is None:
            return {"token": tok, "status": "empty"}
        ct_img = nib.load(task["ct_path"])
        out_dir = Path(task["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # nibabel slicer keeps the affine correct for the crop.
        nib.save(ct_img.slicer[bbox], str(out_dir / "ct.nii.gz"))
        nib.save(lbl_img.slicer[bbox], str(out_dir / "seg.nii.gz"))
        (out_dir / "labels.txt").write_text(task["labels_txt"])
        (out_dir / "crop.json").write_text(json.dumps({
            "token": tok, "config": task["config"],
            "label_file": task["label_file"],
            "origin": [int(s.start) for s in bbox],
            "crop_shape": [int(s.stop - s.start) for s in bbox],
            "full_shape": [int(x) for x in lab.shape],
        }, indent=2))
        mb = sum(f.stat().st_size for f in out_dir.glob("*.nii.gz")) / 1e6
        return {"token": tok, "status": "ok", "mb": round(mb, 1)}
    except Exception as exc:                       # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc_csv", required=True, type=Path, help="merged QC worklist")
    ap.add_argument("--tree", required=True, type=Path, help="pseudo tree (ct/ + labels/)")
    ap.add_argument("--out", required=True, type=Path, help="crops output dir")
    ap.add_argument("--pad", type=int, default=8, help="voxel padding around bone bbox")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from review import labels_descriptor
    labels_txt = labels_descriptor.descriptor_text()

    rows = _flagged_rows(args.qc_csv)
    if args.limit:
        rows = rows[:args.limit]
    index = {(str(r.get("token")), str(r.get("config"))): r
             for r in _load_manifest(args.tree / "manifest.json")}

    tasks = []
    for r in rows:
        rec = index.get((str(r["token"]), str(r["config"])))
        if not rec or not rec.get("ct_file") or not rec.get("label_file"):
            continue
        ct, lbl = args.tree / rec["ct_file"], args.tree / rec["label_file"]
        if not ct.exists() or not lbl.exists():
            continue
        tasks.append({
            "token": str(r["token"]), "config": str(r["config"]),
            "ct_path": str(ct), "label_path": str(lbl),
            "label_file": rec["label_file"], "pad": args.pad,
            "labels_txt": labels_txt,
            "out_dir": str(args.out / crop_dirname(r["token"], r["config"])),
        })
    log.info("exporting %d flagged crops -> %s (pad=%d, %d workers)",
             len(tasks), args.out, args.pad, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_ok = n_fail = 0
    sizes = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_export_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "ok":
                n_ok += 1
                sizes.append(r["mb"])
            elif r["status"] == "fail":
                n_fail += 1
                log.warning("token=%s: crop failed (%s)", r["token"], r.get("error"))
            if i % 25 == 0 or i == len(futs):
                log.info("  %d/%d done", i, len(futs))

    avg = (sum(sizes) / len(sizes)) if sizes else 0.0
    log.info("=" * 60)
    log.info("wrote %d crops (%d failed) -> %s", n_ok, n_fail, args.out)
    log.info("  avg crop size: %.1f MB  (vs ~200 MB full volume)", avg)
    log.info("  review with: python -m reviewtool fix-list %s --tree %s --crops %s",
             args.qc_csv, args.tree, args.out)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
