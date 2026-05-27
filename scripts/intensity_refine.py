"""
intensity_refine.py — refine the PSEUDO-filled region of a v2 tree with
CT-intensity bone segmentation, keyed by the model's class prediction.

Runs AFTER pseudolabel.py. The pseudo-labelled tree it reads is PRESERVED
(the model's class masks stay the source of truth); this writes a NEW tree, so
the pipeline can continue from either. Only `spine_only` / `pelvic_native`
records are refined (in the pseudo-filled region only); `fused` and everything
else pass through verbatim.

It needs BOTH trees:
  --manual_from  the ORIGINAL staged/manual tree (pseudolabel.py's --hf_export)
  --in           the pseudo-labelled tree           (pseudolabel.py's --out)
The pseudo-filled voxels are identified EXACTLY by diffing the two — a voxel is
"pseudo" only where the manual tree was background/IGNORE and v2 now has a
class. This matters because a manual spine annotation can legitimately contain
a sacrum voxel (class 7 from VerSe id 26); keying off class alone would wrongly
re-segment that manual voxel. Manual voxels are NEVER modified.

IDEA
====
The pseudo-filled structures (vertebrae, sacrum, hips) are all BONE, which CT
separates cleanly by HU. The model is reliable at saying WHICH structure a
region is, but a CT threshold gives crisper bone boundaries. So, per scoped
case, in the pseudo region only:

  1. CALIBRATE a bone HU threshold from THIS scan's MANUAL annotation. The
     manual side is known bone in the same acquisition (same kVp/kernel), so
     its own HU distribution sets the threshold per-case — no magic global
     number. We read it off the manual mask's trabecular INTERIOR (eroded):
     cancellous centre HU is the LOW end of bone (cortical borders are higher),
     so a threshold near it keeps the whole bone instead of a hollow shell.

  2. THRESHOLD the CT to a bone mask, take CONNECTED COMPONENTS, and keep only
     components that OVERLAP the model's prediction. This drops threshold
     artifacts and unrelated bone the model didn't predict — ribs in spine
     crops, femurs in pelvic crops.

  3. Solidify marrow (per-slice 2D hole-fill along all 3 axes) and label each
     kept voxel with the NEAREST predicted class.

What intensity CANNOT do: separate two TOUCHING bones (L4/L5 facets, sacrum/
ilium at the SI joint) — they're one bright blob. There the boundary BETWEEN
labels still comes from the model (nearest-class); only the OUTER bone contour
comes from intensity.

Usage
-----
  python scripts/intensity_refine.py \
      --manual_from data/hf_export \
      --in          data/hf_export_v2 \
      --out         data/hf_export_v2_refined \
      [--percentile 10] [--erode_iter 1] [--no_fill_holes] \
      [--limit N] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.intensity_refine")

IGNORE_LABEL = 10
# config -> (manual region, pseudo-filled region) — for scoping + logging only.
SCOPE = {
    "spine_only":    ("spine",  "pelvis"),
    "pelvic_native": ("pelvis", "spine"),
}


# ===========================================================================
# Pure core  (unit-tested in tests/test_intensity_refine.py)
# ===========================================================================

def calibrate_threshold(ct, manual_mask, *, percentile: float = 10.0,
                        erode_iter: int = 1) -> Optional[float]:
    """Per-case bone HU threshold read off the MANUAL annotation.

    The manual mask is known bone in the same scan. We sample its trabecular
    INTERIOR (eroded, to exclude the high-HU cortical rim) and take a low
    percentile — the dim end of bone — so thresholding keeps the whole bone,
    not just the cortical shell. Falls back to the full mask if erosion empties
    it (thin structures). Returns None if there is no manual bone to learn from.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion
    manual_mask = np.asarray(manual_mask, dtype=bool)
    if not manual_mask.any():
        return None
    interior = (binary_erosion(manual_mask, iterations=int(erode_iter))
                if erode_iter else manual_mask)
    region = interior if interior.any() else manual_mask
    return float(np.percentile(np.asarray(ct, dtype=np.float32)[region],
                               percentile))


def _solid_fill(mask) -> "object":
    """Fill enclosed holes per 2D slice along ALL three axes, then union.

    More robust than 3D fill for bone: a vertebral body's marrow is 3D-connected
    to the exterior through foramina (3D fill leaves it hollow) but is enclosed
    within the axial cross-section, so per-slice filling solidifies it.
    """
    import numpy as np
    from scipy.ndimage import binary_fill_holes
    filled = np.asarray(mask, dtype=bool).copy()
    for axis in range(filled.ndim):
        slabs = np.moveaxis(filled, axis, 0)        # view; writes hit `filled`
        for i in range(slabs.shape[0]):
            slabs[i] = binary_fill_holes(slabs[i])
    return filled


def refine_label(v1_label, v2_label, ct, *,
                 percentile: float = 10.0, erode_iter: int = 1,
                 fill_holes: bool = True) -> Tuple["object", Optional[float]]:
    """Re-segment the pseudo region of one case from CT intensity.

    `v1_label` is the ORIGINAL manual label (foreground 1..9 = manual; 0 /
    IGNORE_LABEL = un-annotated). `v2_label` is the pseudo-labelled result.
    Manual voxels (v1 in 1..9) are never touched; the pseudo-filled voxels
    (where v1 was background/IGNORE and v2 has a class) are replaced by a
    calibrated, CC-gated, nearest-class intensity segmentation.

    Returns (refined_label, threshold_used).
    """
    import numpy as np
    from scipy.ndimage import label as cc_label, distance_transform_edt

    v1 = np.asarray(v1_label, dtype=np.int16)
    v2 = np.asarray(v2_label, dtype=np.int16)
    ct = np.asarray(ct, dtype=np.float32)
    out = v2.copy()

    manual_mask = (v1 >= 1) & (v1 <= 9)        # exact manual region (any class)
    fillable = ~manual_mask                    # v1 was 0 / IGNORE here
    pred_classmap = np.where(fillable & (v2 >= 1) & (v2 <= 9),
                             v2, 0).astype(np.int16)   # the model's pseudo fill
    pred_fg = pred_classmap > 0
    if not pred_fg.any():                      # nothing pseudo to refine
        return out, None
    thr = calibrate_threshold(ct, manual_mask, percentile=percentile,
                              erode_iter=erode_iter)
    if thr is None:                            # no manual bone to calibrate from
        return out, None

    bone = (ct >= thr) & fillable
    cc, n = cc_label(bone)
    out[pred_fg] = 0                            # clear the old model pseudo voxels
    if n:
        keep_ids = [int(v) for v in np.unique(cc[pred_fg]) if v != 0]
        kept = np.isin(cc, keep_ids) if keep_ids else np.zeros_like(bone)
        if fill_holes and kept.any():
            kept = _solid_fill(kept) & fillable
        if kept.any():
            idx = distance_transform_edt(~pred_fg, return_distances=False,
                                         return_indices=True)
            nearest = pred_classmap[tuple(idx)]
            out[kept] = nearest[kept].astype(np.int16)
    return out, thr


# ===========================================================================
# Orchestrator
# ===========================================================================

def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_tree", required=True, type=Path,
                    help="The pseudo-labelled tree (pseudolabel.py --out).")
    ap.add_argument("--manual_from", required=True, type=Path,
                    help="The ORIGINAL staged/manual tree (pseudolabel.py "
                         "--hf_export). Used to identify pseudo voxels exactly "
                         "and to calibrate the threshold from manual bone.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="Percentile of manual trabecular HU used as the bone "
                         "threshold (default 10).")
    ap.add_argument("--erode_iter", type=int, default=1,
                    help="Erode the manual mask this many voxels before "
                         "sampling HU, to exclude the cortical rim (default 1).")
    ap.add_argument("--no_fill_holes", dest="fill_holes", action="store_false",
                    help="Don't hole-fill marrow; leaves hollow structures.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only: log per-case scope, write nothing.")
    ap.set_defaults(fill_holes=True)
    args = ap.parse_args()

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(line_buffering=True)
        except Exception:
            pass

    import numpy as np
    import nibabel as nib

    src, man_src, out = args.in_tree, args.manual_from, args.out
    man_path = src / "manifest.json"
    if not man_path.exists():
        log.error("No manifest.json in %s", src)
        return 1
    records = _load_manifest(man_path)
    scoped = [r for r in records if r.get("config") in SCOPE
              and r.get("ct_file") and r.get("label_file")]
    log.info("intensity_refine: %d records, %d scoped; percentile=%.0f "
             "erode=%d fill_holes=%s", len(records), len(scoped),
             args.percentile, args.erode_iter, args.fill_holes)

    if args.dry_run:
        for i, r in enumerate(scoped[:args.limit or None], 1):
            man_reg, ps_reg = SCOPE[r["config"]]
            log.info("[%d/%d] token=%s cfg=%s  refine %s (manual %s calibrates)",
                     i, len(scoped), r.get("token"), r["config"], ps_reg, man_reg)
        log.info("DRY-RUN: nothing written.")
        return 0

    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("manifest.json", "manifest.csv", "splits_5fold.json",
                  "splits_summary.json", "dataset_interface.py", "README.md"):
        if (src / extra).exists():
            shutil.copy2(str(src / extra), str(out / extra))

    def _copy(rel):
        if rel and (src / rel).exists() and not (out / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src / rel), str(out / rel))

    n_refined = n_pass = n_skip = n_fail = 0
    todo = scoped[:args.limit] if args.limit else scoped
    refine_ids = {id(r) for r in todo}

    for rec in records:
        ct_rel, lbl_rel = rec.get("ct_file"), rec.get("label_file")
        if id(rec) not in refine_ids:
            _copy(ct_rel); _copy(lbl_rel); n_pass += 1
            continue
        man_reg, ps_reg = SCOPE[rec["config"]]
        try:
            v1_path = man_src / lbl_rel
            if not v1_path.exists():
                log.warning("token=%s: manual label %s missing — passthrough",
                            rec.get("token"), v1_path)
                _copy(ct_rel); _copy(lbl_rel); n_skip += 1
                continue
            ref = nib.load(str(src / lbl_rel))
            v2 = np.asarray(ref.dataobj).astype(np.int16)
            v1 = np.asarray(nib.load(str(v1_path)).dataobj).astype(np.int16)
            ct = np.asarray(nib.load(str(src / ct_rel)).dataobj).astype(np.float32)
            if not (v1.shape[:3] == v2.shape[:3] == ct.shape[:3]):
                log.warning("token=%s: grid mismatch (v1 %s / v2 %s / ct %s) — "
                            "passthrough", rec.get("token"), v1.shape, v2.shape,
                            ct.shape)
                _copy(ct_rel); _copy(lbl_rel); n_skip += 1
                continue
            refined, thr = refine_label(
                v1, v2, ct, percentile=args.percentile,
                erode_iter=args.erode_iter, fill_holes=args.fill_holes)
            _copy(ct_rel)
            nib.save(nib.Nifti1Image(refined, ref.affine, ref.header),
                     str(out / lbl_rel))
            n_refined += 1
            fg = int((refined > 0).sum())
            log.info("token=%s: refined %s  HU>=%.0f  (%d fg vox)  [%d/%d]",
                     rec.get("token"), ps_reg,
                     thr if thr is not None else float("nan"), fg,
                     n_refined, len(todo))
        except Exception as exc:                         # noqa: BLE001
            log.warning("token=%s: refine failed (%s) — passthrough",
                        rec.get("token"), exc)
            _copy(ct_rel); _copy(lbl_rel); n_fail += 1

    log.info("=" * 60)
    log.info("intensity-refined tree -> %s", out)
    log.info("  refined      : %d", n_refined)
    log.info("  passthrough  : %d", n_pass)
    log.info("  grid-skipped : %d", n_skip)
    log.info("  failed       : %d", n_fail)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
