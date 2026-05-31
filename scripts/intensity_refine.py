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
region is, but a CT threshold gives crisper bone boundaries. Per scoped case,
in the pseudo region only:

  1. CALIBRATE a bone HU threshold from THIS scan's MANUAL annotation. The
     manual side is known bone in the same acquisition (same kVp/kernel), so
     its own HU distribution sets the threshold per-case — no magic global
     number. We read it off the manual mask's trabecular INTERIOR (eroded):
     cancellous centre HU is the LOW end of bone (cortical borders are higher),
     so a threshold near it keeps the whole bone instead of a hollow shell.

  2. Apply it, per `--mode`:
     * clip (DEFAULT) — keep each predicted voxel ONLY where the CT is bone
       (>= threshold), plus the marrow enclosed within it; clipped to the
       prediction. This is SUBTRACTIVE: the result is always a subset of the
       model mask, so over-segmentation (the usual failure — the model bleeds
       past the bone) is erased, and unrelated bone (ribs/femurs) can't be
       added because it was never predicted. Nothing to gate, nothing to grow.
     * resegment — threshold to a bone mask, keep CONNECTED COMPONENTS that
       overlap the prediction, and nearest-class label them. This CAN grow into
       bone the model MISSED (under-segmentation), at the cost of needing the
       CC gate to keep ribs/femurs out.

  3. Solidify marrow (per-slice 2D hole-fill along all 3 axes).

What intensity CANNOT do: separate two TOUCHING bones (L4/L5 facets, sacrum/
ilium at the SI joint) — they're one bright blob. Class boundaries there come
from the model; only the OUTER bone contour comes from intensity.

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
import os
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
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    if not mask.any():
        return out
    # Crop to the mask's bounding box: holes are enclosed WITHIN the mask, so
    # per-slice fill on the bbox is identical to the full volume but avoids
    # looping over every (mostly empty) slice of a 512^3 CT.
    coords = np.argwhere(mask)
    sl = tuple(slice(int(lo), int(hi) + 1)
               for lo, hi in zip(coords.min(0), coords.max(0)))
    sub = out[sl]
    for axis in range(sub.ndim):
        slabs = np.moveaxis(sub, axis, 0)           # view; writes hit `out[sl]`
        for i in range(slabs.shape[0]):
            slabs[i] = binary_fill_holes(slabs[i])
    return out


def compete_relabel(pred_classmap, bone, *, purity_tol: float = 0.15,
                    min_bleed_vox: int = 50, grow_iters: int = 0,
                    fill_holes: bool = True) -> Tuple["object", list]:
    """Per-connected-bone-component label competition.

    Each bone connected-component the model predicted is reclaimed WHOLE to its
    dominant predicted class — fixing a neighbour's label that bled across a
    (non-bone) disc onto this bone — BUT only when the component is class-pure
    enough to be clearly ONE structure. A component substantially shared by >1
    predicted class is TOUCHING / FUSED bone (SI joint, L5-S1 fusion,
    hip+sacrum) that intensity cannot separate: there the model's own per-voxel
    boundary is kept and the component is FLAGGED for human review.

    A component counts as confident when the minority (non-dominant) predicted
    voxels are either <= `purity_tol` of the component's predicted voxels OR
    <= `min_bleed_vox` in absolute count (a tiny bleed). Growth past the
    predicted voxels is bounded (`grow_iters`, confined to the same component)
    so a target fused to UNpredicted bone — femur at the hip — can't be
    swallowed.

    Returns (region_label, flags): an int array with the reclaimed/kept class
    per bone voxel (0 elsewhere), and a list of dicts describing the multi-class
    (ambiguous) components that were left to the model and flagged.
    """
    import numpy as np
    from scipy.ndimage import (label as cc_label, generate_binary_structure,
                               binary_dilation, center_of_mass)
    pred_classmap = np.asarray(pred_classmap, dtype=np.int16)
    bone = np.asarray(bone, dtype=bool)
    out = np.zeros_like(pred_classmap)
    flags: list = []
    pred_fg = pred_classmap > 0
    pred_bone = pred_fg & bone                 # the voxels we reclaim (predicted ∩ bone)
    if not pred_bone.any():
        return out, flags

    cc, n = cc_label(bone, structure=generate_binary_structure(bone.ndim, 1))
    if n == 0:
        return out, flags

    # Per-component tally of predicted classes, over predicted-bone voxels only.
    # Vectorised: bincount on (component_id * ncls + class) — NO python loop over
    # components (there can be tens of thousands from thresholding the whole CT).
    comp_p = cc[pred_bone].astype(np.int64)
    cls_p = pred_classmap[pred_bone].astype(np.int64)
    ncls = int(cls_p.max()) + 1
    counts = np.bincount(comp_p * ncls + cls_p,
                         minlength=(n + 1) * ncls).reshape(n + 1, ncls)
    counts[0] = 0                              # background component
    total = counts.sum(axis=1)
    dom = counts.argmax(axis=1)                # dominant predicted class per component
    minority = total - counts[np.arange(n + 1), dom]
    minority_frac = np.where(total > 0, minority / np.maximum(total, 1), 0.0)
    has_pred = total > 0
    confident = has_pred & ((minority_frac <= purity_tol)
                            | (minority <= min_bleed_vox))
    ambiguous = has_pred & ~confident

    # Confident components -> reclaim their predicted-bone voxels to the dominant
    # class (a neighbour's cross-disc bleed flips to this structure). Growth is
    # bounded WITHIN the component so unpredicted bone (femur) can't be swallowed.
    conf_vox = confident[cc] & pred_bone
    if int(grow_iters) > 0:
        conf_vox = binary_dilation(conf_vox, iterations=int(grow_iters),
                                   mask=confident[cc] & bone)
    out[conf_vox] = dom[cc[conf_vox]].astype(np.int16)

    # Ambiguous (touching/fused) components -> keep the model's per-voxel boundary.
    amb_vox = ambiguous[cc] & pred_bone
    out[amb_vox] = pred_classmap[amb_vox].astype(np.int16)

    # Recover enclosed marrow per labelled class (sub-threshold interior the
    # cortical ring encloses), the SAME way clip/resegment do — and ONLY enclosed
    # holes, so external over-segmentation past the cortex stays erased.
    if fill_holes:
        for k in np.unique(out[out > 0]):
            newly = _solid_fill(out == int(k)) & (out == 0)
            out[newly] = int(k)

    # Flag the ambiguous components (few) for review; centroids in one pass.
    amb_ids = np.nonzero(ambiguous)[0]
    if amb_ids.size:
        coms = center_of_mass(np.ones(cc.shape, dtype=np.uint8), cc,
                              list(amb_ids.tolist()))
        coms = np.atleast_2d(coms)             # (k, ndim) for both 1 and many
        for cid, com in zip(amb_ids.tolist(), coms):
            col = counts[cid]
            flags.append({
                "dominant": int(dom[cid]),
                "classes": {int(c): int(col[c]) for c in np.nonzero(col)[0]},
                "minority_frac": round(float(minority_frac[cid]), 3),
                "n_pred_vox": int(total[cid]),
                "centroid": [int(round(x)) for x in com],
            })
    return out, flags


def refine_label(v1_label, v2_label, ct, *, mode: str = "clip",
                 percentile: float = 10.0, erode_iter: int = 1,
                 fill_holes: bool = True,
                 grow_iters: int = 0,
                 purity_tol: float = 0.15, min_bleed_vox: int = 50,
                 bone_floor: float = 0.0,
                 flags_out: Optional[list] = None
                 ) -> Tuple["object", Optional[float]]:
    """Re-segment the pseudo region of one case from CT intensity.

    `v1_label` is the ORIGINAL manual label (foreground 1..9 = manual; 0 /
    IGNORE_LABEL = un-annotated). `v2_label` is the pseudo-labelled result.
    Manual voxels (v1 in 1..9) are never touched.

    mode="clip" (default): keep predicted voxels that are bone (+ enclosed
    marrow). If `grow_iters > 0`, also geodesically grow the predicted-bone
    seed through CT-bone for that many voxels — picks up adjacent bone the
    model narrowly missed without runaway into ribs/femurs (the dilation can
    only travel through bone, and only `grow_iters` steps). With grow_iters=0
    it is pure clip: a subset of the model mask, no growth.
    mode="resegment": keep bone connected-components overlapping the prediction
    and nearest-class label them; UNBOUNDED grow via CC connectivity.
    mode="compete": per bone connected-component, reclaim the whole component to
    its dominant predicted class (fixes a neighbour's label that bled across a
    disc), EXCEPT multi-class (touching/fused) components, which keep the model
    boundary and are appended to `flags_out` for review. See `compete_relabel`.

    Returns (refined_label, threshold_used). When `flags_out` is a list and
    mode="compete", ambiguous-component flags are appended to it.
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
    # Optional HU floor on the calibrated threshold. Fatty marrow drags the
    # per-case trabecular percentile far below soft tissue (~0-60 HU), which
    # bridges every bone through the discs/joints into one component — fatal for
    # compete's per-structure separation. A floor (~150 HU) lifts the threshold
    # above disc/joint soft tissue so healthy gaps separate the bones; the
    # excluded marrow is recovered by the hole-fill step.
    if bone_floor:
        thr = max(thr, float(bone_floor))

    bone = (ct >= thr) & fillable
    out[pred_fg] = 0                            # clear the old model pseudo voxels
    if mode == "compete":
        region_lbl, flags = compete_relabel(
            pred_classmap, bone, purity_tol=purity_tol,
            min_bleed_vox=min_bleed_vox, grow_iters=int(grow_iters),
            fill_holes=fill_holes)
        sel = region_lbl > 0
        out[sel] = region_lbl[sel]
        if flags_out is not None:
            flags_out.extend(flags)
    elif mode == "clip" and int(grow_iters) > 0:
        # Bounded grow: dilate the predicted-bone seed THROUGH the bone mask
        # (geodesic dilation), for `grow_iters` voxels. Bone the model just
        # narrowly missed is picked up; bone disconnected from the seed (ribs,
        # femurs) is unreachable. Nearest-class assignment resolves overlaps.
        from scipy.ndimage import binary_dilation
        candidates = np.zeros(bone.shape, dtype=bool)
        for c in np.unique(pred_classmap[pred_fg]):
            seed = (pred_classmap == int(c)) & bone
            grown = binary_dilation(seed, iterations=int(grow_iters), mask=bone)
            if fill_holes:
                grown = _solid_fill(grown) & fillable
            candidates |= grown
        idx = distance_transform_edt(~pred_fg, return_distances=False,
                                     return_indices=True)
        nearest = pred_classmap[tuple(idx)]
        out[candidates] = nearest[candidates].astype(np.int16)
    elif mode == "clip":
        # Pure clip (grow_iters=0): keep predicted bone (+ enclosed marrow),
        # clipped to the prediction. Per-class masks are disjoint -> labels stay
        # the model's; no EDT needed.
        for c in np.unique(pred_classmap[pred_fg]):
            pc = pred_classmap == int(c)
            seg_c = pc & bone
            if fill_holes:
                seg_c = _solid_fill(seg_c) & pc
            out[seg_c] = int(c)
    else:  # "resegment" — CC-gated, can grow into bone the model missed
        cc, n = cc_label(bone)
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

def link_or_copy(src_path, dst_path, *, copy: bool = False) -> str:
    """Place src at dst with ONE physical copy of the data on disk.

    Hard-link first (same filesystem: zero extra bytes, and the result is a
    normal file to every tool incl. the HF uploader). Fall back to a relative
    symlink (cross-fs), then a full copy. `copy=True` forces a real copy.
    Returns the method used. Used for the big CT volumes so the v2 / refined
    export trees reference the single v1 CT store instead of duplicating 280 GB.
    """
    src_path, dst_path = str(src_path), str(dst_path)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src_path, dst_path)
        return "copy"
    try:
        os.link(src_path, dst_path)                       # hardlink: 0 extra bytes
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(os.path.relpath(src_path, os.path.dirname(dst_path)), dst_path)
        return "symlink"
    except OSError:
        shutil.copy2(src_path, dst_path)
        return "copy"


def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _refine_one(task: dict) -> dict:
    """ProcessPoolExecutor worker: load v1/v2/CT for one case, refine, save the
    label, hard-link the CT into the out tree. Returns a result dict."""
    import numpy as np
    import nibabel as nib
    tok = task["token"]
    try:
        v1_path = task["v1_path"]
        if not Path(v1_path).exists():
            return {"token": tok, "status": "skip_no_v1"}
        ref = nib.load(task["v2_path"])
        v2 = np.asarray(ref.dataobj).astype(np.int16)
        v1 = np.asarray(nib.load(v1_path).dataobj).astype(np.int16)
        ct = np.asarray(nib.load(task["ct_path"]).dataobj).astype(np.float32)
        if not (v1.shape[:3] == v2.shape[:3] == ct.shape[:3]):
            return {"token": tok, "status": "skip_shape"}
        flags: list = []
        refined, thr = refine_label(
            v1, v2, ct,
            mode=task["mode"], percentile=task["percentile"],
            erode_iter=task["erode_iter"], fill_holes=task["fill_holes"],
            grow_iters=task["grow_iters"],
            purity_tol=task["purity_tol"], min_bleed_vox=task["min_bleed_vox"],
            bone_floor=task["bone_floor"],
            flags_out=flags)
        out_lbl = Path(task["out_lbl"])
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(refined, ref.affine, ref.header), str(out_lbl))
        out_ct = Path(task["out_ct"])
        if not out_ct.exists():
            link_or_copy(task["ct_link_src"], out_ct, copy=task["copy_ct"])
        return {"token": tok, "status": "refined", "thr": thr,
                "fg": int((refined > 0).sum()),
                "flags": [{**f, "token": tok} for f in flags]}
    except Exception as exc:                                 # noqa: BLE001
        return {"token": tok, "status": "fail", "error": str(exc)}


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
    ap.add_argument("--mode", choices=("clip", "resegment", "compete"),
                    default="clip",
                    help="clip (default): subtractive, keep predicted voxels "
                         "that are bone (erases over-segmentation, never grows). "
                         "resegment: CC-gated, can grow into missed bone. "
                         "compete: per bone component, reclaim it to its dominant "
                         "class (fixes cross-disc bleed); multi-class (fused) "
                         "components keep the model boundary and are flagged.")
    ap.add_argument("--purity_tol", type=float, default=0.15,
                    help="compete: a bone component is confidently ONE structure "
                         "when its minority predicted classes are <= this "
                         "fraction (default 0.15); above it the component is "
                         "treated as touching/fused and flagged for review.")
    ap.add_argument("--min_bleed_vox", type=int, default=50,
                    help="compete: minority voxels <= this count are always "
                         "treated as a bleed and absorbed, regardless of "
                         "--purity_tol (default 50).")
    ap.add_argument("--bone_floor", type=float, default=0.0,
                    help="HU floor on the per-case calibrated threshold "
                         "(default 0 = off). Fatty marrow drags the calibrated "
                         "threshold below soft tissue, bridging all bones into "
                         "one component; ~150 lifts it above disc/joint tissue "
                         "so structures separate (needed for compete).")
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="Percentile of manual trabecular HU used as the bone "
                         "threshold (default 10).")
    ap.add_argument("--erode_iter", type=int, default=1,
                    help="Erode the manual mask this many voxels before "
                         "sampling HU, to exclude the cortical rim (default 1).")
    ap.add_argument("--no_fill_holes", dest="fill_holes", action="store_false",
                    help="Don't hole-fill marrow; leaves hollow structures.")
    ap.add_argument("--grow_iters", type=int, default=0,
                    help="In clip mode, geodesically dilate the predicted-bone "
                         "seed through CT-bone for this many voxels — picks up "
                         "adjacent bone the model narrowly missed without "
                         "runaway (ribs/femurs are unreachable through bone). "
                         "0 = pure clip (default); 3-5 = bounded grow.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 8) // 2),
                    help="Parallel worker processes for the per-case refine "
                         "(default = nproc/2).")
    ap.add_argument("--copy_ct", action="store_true",
                    help="Copy CT volumes instead of hard-linking them to the "
                         "v1 store (only needed if the trees are on different "
                         "filesystems). Default hard-links: one copy on disk.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-refine cases whose output label already exists "
                         "(default: SKIP them — resume-friendly). If you "
                         "changed any refinement param and want fresh output, "
                         "set this OR rm -rf the output dir first.")
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
    log.info("intensity_refine: %d records, %d scoped; mode=%s grow_iters=%d "
             "percentile=%.0f erode=%d fill_holes=%s workers=%d",
             len(records), len(scoped), args.mode, args.grow_iters,
             args.percentile, args.erode_iter, args.fill_holes, args.workers)

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

    def _copy(rel):                                       # for small label files
        if rel and (src / rel).exists() and not (out / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src / rel), str(out / rel))

    def _place_ct(rel):                                   # ONE physical CT copy
        if not rel or (out / rel).exists():
            return
        srcp = man_src / rel if (man_src / rel).exists() else src / rel
        if Path(srcp).exists():
            link_or_copy(srcp, out / rel, copy=args.copy_ct)

    import time
    n_refined = n_pass = n_skip = n_fail = 0
    todo = scoped[:args.limit] if args.limit else scoped
    refine_ids = {id(r) for r in todo}

    # Passthrough cases (fused / out-of-scope): place sequentially (cheap I/O).
    n_pt = len(records) - len(refine_ids)
    log.info("placing %d passthrough cases (fused / out-of-scope) ...", n_pt)
    t_pt = time.time()
    for rec in records:
        if id(rec) in refine_ids:
            continue
        _place_ct(rec.get("ct_file"))
        _copy(rec.get("label_file"))
        n_pass += 1
        if n_pass % 50 == 0 or n_pass == n_pt:
            log.info("  passthrough %d/%d  (%.0fs)", n_pass, n_pt,
                     time.time() - t_pt)

    # Scoped refines: skip cases already refined (resume) unless --overwrite.
    n_cached = 0
    to_refine = []
    for rec in todo:
        out_lbl = out / rec["label_file"]
        if out_lbl.exists() and not args.overwrite:
            _place_ct(rec.get("ct_file"))   # belt-and-braces: ensure CT linked
            n_cached += 1
        else:
            to_refine.append(rec)
    if n_cached:
        log.info("resume: %d cases already refined (skipping); %d to refine. "
                 "Use --overwrite to force a fresh refine.",
                 n_cached, len(to_refine))
    n_refined += n_cached

    # CPU-heavy work: parallel ProcessPoolExecutor on the to-do list only.
    tasks = [{
        "token": rec.get("token", "?"),
        "v1_path":    str(man_src / rec["label_file"]),
        "v2_path":    str(src     / rec["label_file"]),
        "ct_path":    str(src     / rec["ct_file"]),
        "out_lbl":    str(out     / rec["label_file"]),
        "out_ct":     str(out     / rec["ct_file"]),
        "ct_link_src": str(man_src / rec["ct_file"])
                       if (man_src / rec["ct_file"]).exists()
                       else str(src / rec["ct_file"]),
        "mode": args.mode, "percentile": args.percentile,
        "erode_iter": args.erode_iter, "fill_holes": args.fill_holes,
        "grow_iters": args.grow_iters, "copy_ct": args.copy_ct,
        "purity_tol": args.purity_tol, "min_bleed_vox": args.min_bleed_vox,
        "bone_floor": args.bone_floor,
    } for rec in to_refine]

    if not tasks:
        log.info("nothing to refine — all %d scoped cases already cached.",
                 n_cached)
    else:
        log.info("refining %d cases on %d worker(s) — first result usually "
                 "takes a few minutes (NFS warm-up + parallel CT loads) ...",
                 len(tasks), args.workers)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    review_flags: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_refine_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tok, status = r["token"], r["status"]
            if status == "refined":
                n_refined += 1
                review_flags.extend(r.get("flags", []))
            elif status == "fail":
                n_fail += 1
                log.warning("token=%s: refine failed (%s) — passthrough",
                            tok, r.get("error"))
                src_rec = next((rec for rec in todo
                                if str(rec.get("token", "?")) == str(tok)), None)
                if src_rec:
                    _place_ct(src_rec.get("ct_file"))
                    _copy(src_rec.get("label_file"))
            else:   # skip_no_v1 / skip_shape
                n_skip += 1
                log.warning("token=%s: %s — passthrough", tok, status)
                src_rec = next((rec for rec in todo
                                if str(rec.get("token", "?")) == str(tok)), None)
                if src_rec:
                    _place_ct(src_rec.get("ct_file"))
                    _copy(src_rec.get("label_file"))
            # log EVERY case completion (proof of life + continuous progress)
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1.0)
            eta_s = (len(futures) - i) / rate if rate > 0 else 0.0
            thr_s = ("%.0f" % r["thr"]) if r.get("thr") is not None else "—"
            log.info("  [%d/%d] %s token=%s HU>=%s fg=%d  "
                     "elapsed=%dm%02ds  rate=%.2f/s  ETA=%dm",
                     i, len(futures), status, tok, thr_s,
                     r.get("fg", 0),
                     int(elapsed) // 60, int(elapsed) % 60,
                     rate, int(eta_s) // 60)

    # compete mode: persist the ambiguous (touching/fused) components for review.
    if args.mode == "compete":
        flags_path = out / "review_flags.json"
        flags_path.write_text(json.dumps(review_flags, indent=2))
        n_cases_flagged = len({f["token"] for f in review_flags})
        log.info("compete: %d ambiguous (touching/fused) component(s) across "
                 "%d case(s) flagged for review -> %s",
                 len(review_flags), n_cases_flagged, flags_path)

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
