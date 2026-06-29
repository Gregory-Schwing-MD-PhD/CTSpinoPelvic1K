"""
build_v4_ribs.py — v4 = v3 + high-quality ribs, numbered into our VerSe-native scheme.

Pipeline per case (sharded like build_v3_totalseg, resumable):
  CT --(Möller binary rib nnU-Net)--> binary rib mask        [GPU]
  v3 thoracic (VerSe 8-19 = T1-T12) --remap--> rib-number anchors (T1..T12 -> 1..12)
  relabel_ribs(binary ribs, anchors) --> ribs numbered into 34-57 (label_scheme)  [CPU]
  overlay on v3 (clear old ribs 34-57, write new on background, never touch GT) -> v4

WHY this design (not Möller's run_all_steps): his published checkpoint is only a BINARY
rib segmenter (rib=1/bg=0, Dice 0.997); his per-rib numbering (run_all_steps) needs a
SPINEPS *subregion* mask (labels 41-49) we don't have. So we take his binary ribs and
do the numbering ourselves off our v3 thoracic via relabel_ribs (costovertebral vote) —
no SPINEPS, no TPTBox, just nnUNetv2_predict (already in the TS container) + relabel_ribs.
That also keeps the numbering FOV-invariant (anchored to T12, not the FOV edge).

WEIGHTS (one-time): download ribseg_model_weights.zip from Zenodo 10.5281/zenodo.14850928
and unzip into  $nnUNet_results/Dataset<ID>_<name>/<trainer>__<plans>__<config>/  (it
contains fold_*/, dataset.json, plans.json). Read the <ID>/<trainer>/<plans>/<config>
off the unzipped dataset.json + plans.json and pass via --dataset_id/--trainer/--plans/
--config (OPEN ITEM: those names live inside the zip, not in the paper/README).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS          # canonical VerSe-native ids
import relabel_ribs as RR          # costovertebral rib numbering

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.v4ribs")

# v3 thoracic (VerSe) -> rib-number anchor for relabel_ribs (which uses the vertebra
# LABEL as the rib number). T1..T12 = VerSe 8..19 -> 1..12. (T13/28 omitted: no slot in
# label_scheme's 1-12 per side; rare, would vote onto T12.)
THORACIC_TO_RIBNUM = {8 + i: 1 + i for i in range(12)}        # 8->1 … 19->12
RIB_IDS = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))   # 34..57


def _base(ct_path) -> str:
    """CT '<base>_ct.nii.gz' (or '<base>.nii.gz') -> '<base>'; label is '<base>_label.nii.gz'."""
    n = ct_path.name
    return n[:-len("_ct.nii.gz")] if n.endswith("_ct.nii.gz") else n[:-len(".nii.gz")]


def predict_ribs_for_shard(cts, work, model_folder, folds, checkpoint, device) -> Path:
    """Stage the shard's CTs and run Möller's binary rib nnU-Net over the folder via the
    nnU-Net v2 PYTHON API, pointing straight at the (flattened) model folder. Möller's
    zip is just one model dir (dataset.json + plans.json + fold_*/), NOT a Dataset<ID>
    hierarchy, so the predictor's folder init is the right entry point (no id/trainer/
    plans/config needed — it reads them from the folder). Returns the predictions dir."""
    in_dir, pred_dir = work / "in", work / "pred"
    in_dir.mkdir(parents=True, exist_ok=True); pred_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for ct in cts:
        cid = _base(ct)
        if pred_dir.joinpath(f"{cid}.nii.gz").exists():       # already predicted (resume)
            continue
        dst = in_dir / f"{cid}_0000.nii.gz"
        if not dst.exists():
            try:
                os.symlink(os.path.abspath(ct), dst)
            except OSError:
                import shutil; shutil.copy2(str(ct), str(dst))
        staged += 1
    if not staged:
        return pred_dir
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=False,          # mirror TTA off: ~8x faster, fine for bone
                                device=torch.device(device), verbose=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=tuple(int(f) for f in folds), checkpoint_name=checkpoint)
    log.info("Möller rib nnU-Net: predicting %d CT(s) (folds=%s, ckpt=%s)", staged, folds, checkpoint)
    predictor.predict_from_files(str(in_dir), str(pred_dir), save_probabilities=False, overwrite=False)
    return pred_dir


def recarve_s1_symmetric(lab: np.ndarray, affine) -> int:
    """Re-carve S1 (29) from the GT sacrum SYMMETRICALLY, in-place, with NO TS/rebuild.

    The v3 carve tilted left-right (PCA axis had a world-X component) -> asymmetric S1.
    Here we reconstitute the whole GT sacrum (26 ∪ 29), use the EXISTING v3 S1 as the
    seed for the cut HEIGHT, and cut along a sagittally-tilted axis with its left-right
    component zeroed -> the S1/S2 plane is level across the midline (symmetric). The
    sacrum's outer boundary stays GT. Returns the new S1 voxel count (0 if no S1/sacrum).
    """
    sac = (lab == LS.SACRUM_ID) | (lab == LS.S1_ID)
    seed = (lab == LS.S1_ID)
    if not seed.any() or not sac.any():
        return 0
    lab[lab == LS.S1_ID] = LS.SACRUM_ID                       # reconstitute whole sacrum
    sac_ijk = np.array(np.nonzero(sac)).T
    sac_w = nib.affines.apply_affine(affine, sac_ijk)
    center = sac_w.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((sac_w - center).T))
    axis = None
    for k in np.argsort(evals)[::-1]:                          # prefer a cranio-caudal axis
        if abs(evecs[2, k]) > 0.3:
            axis = evecs[:, k].copy(); break
    if axis is None:
        axis = evecs[:, int(np.argmax(evals))].copy()
    if axis[2] < 0:
        axis = -axis
    axis[0] = 0.0                                              # symmetry: no left-right roll
    n = np.linalg.norm(axis)
    if n == 0:
        return 0
    axis = axis / n
    sac_proj = (sac_w - center) @ axis
    seed_w = nib.affines.apply_affine(affine, np.array(np.nonzero(seed)).T)
    cut = float(np.percentile((seed_w - center) @ axis, 10))   # S1/S2 boundary from the old S1
    promote = sac_ijk[sac_proj >= cut]
    lab[promote[:, 0], promote[:, 1], promote[:, 2]] = LS.S1_ID
    return int(promote.shape[0])


def _rib_connection_qc(union_vox, labeled, kept, assigns, sizes, v4) -> dict:
    """Per-case rib↔vertebra connection QC, computed at union time (the only point
    where the dropped/orphaned rib bone is still observable). Reports how much of
    the union rib bone got numbered vs dropped, plus numbering GAPS (a missing
    number between present ones — the adjacent-rib bridging bug) and DUPLICATE ids
    (two components voting to one number — a merge). All cheap integer bookkeeping."""
    assigned_comps = list(assigns)
    unassigned_comps = [c for c in kept if c not in assigns]
    assigned_vox = int(sum(int(sizes[c]) for c in assigned_comps))
    unassigned_vox = int(sum(int(sizes[c]) for c in unassigned_comps))
    kept_vox = int(sum(int(sizes[c]) for c in kept))
    dup_ids = sorted(i for i, c in Counter(
        RR.rib_label_id(s, n) for (s, n) in assigns.values()).items() if c > 1)

    present = {int(x) for x in np.unique(v4)}
    thor = sorted(7 + n for n in range(1, 13) if (7 + n) in present)        # T1..T12 = 8..19
    left = sorted(i - LS.RIB_LEFT_OFFSET for i in range(LS.RIB_LEFT_OFFSET + 1,
                  LS.RIB_LEFT_OFFSET + 13) if i in present)
    right = sorted(i - LS.RIB_RIGHT_OFFSET for i in range(LS.RIB_RIGHT_OFFSET + 1,
                   LS.RIB_RIGHT_OFFSET + 13) if i in present)

    def _gaps(nums):
        return [n for n in range(min(nums), max(nums) + 1) if n not in nums] if nums else []

    return {
        "union_vox": int(union_vox), "kept_vox": kept_vox,
        "assigned_vox": assigned_vox, "unassigned_vox": unassigned_vox,
        "noise_vox": int(union_vox) - kept_vox,
        # fraction of union rib bone that did NOT become a numbered rib (didn't connect)
        "drop_frac": round((int(union_vox) - assigned_vox) / int(union_vox), 4) if union_vox else 0.0,
        "n_comp_kept": len(kept), "n_assigned": len(assigned_comps),
        "n_unassigned": len(unassigned_comps),
        "unassigned_sizes": sorted((int(sizes[c]) for c in unassigned_comps), reverse=True)[:10],
        "thoracic_levels": thor,
        "left_rib_nums": left, "right_rib_nums": right,
        "left_gaps": _gaps(left), "right_gaps": _gaps(right),   # missing number between present ones
        "duplicate_rib_ids": dup_ids,                          # two components -> one number (merge)
    }


# Tunable: below this a component is a speckle, not worth flagging for a number check.
MIN_RIB_VOX = 300


def _ts_component_numbers(labeled, kept, lab):
    """Dominant TS rib (side, number) per union component, read from the TS numbering
    already in v3 (ids 34-57) — or None if the component has no TS rib voxel (a
    Möller-only piece). TS numbers ribs consecutively but counts from the top of the
    FOV, so its per-rib numbers are right RELATIVE to each other; only the global offset
    is wrong (corrected in _ts_offset_assign)."""
    lo, hi = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12       # 34..57
    out = {}
    for c in kept:
        vals = lab[labeled == c]
        vals = vals[(vals >= lo) & (vals <= hi)]
        if vals.size == 0:
            out[c] = None
            continue
        u, cnt = np.unique(vals, return_counts=True)
        dom = int(u[cnt.argmax()])
        out[c] = (("left", dom - LS.RIB_LEFT_OFFSET) if dom <= LS.RIB_LEFT_OFFSET + 12
                  else ("right", dom - LS.RIB_RIGHT_OFFSET))
    return out


def _ts_offset_assign(labeled, kept, lab, assigns):
    """Reuse TS's own consecutive rib numbering (v3 34-57) for every component that has
    it, correcting ONLY TS's FOV offset using the anchored ribs — so TS fragments keep
    their (now-correct) number instead of being thrown away. Per side, offset =
    mode(true - TS) over anchored components that also carry a TS number; apply it to
    every other TS-having component. Returns ({comp:(side,number)}, offset_per_side)."""
    tsmap = _ts_component_numbers(labeled, kept, lab)
    votes = {"left": [], "right": []}
    for c, (side, truen) in assigns.items():
        ts = tsmap.get(c)
        if ts and ts[0] == side:
            votes[side].append(truen - ts[1])
    offset = {s: Counter(v).most_common(1)[0][0] for s, v in votes.items() if v}
    out = {}
    for c in kept:
        if c in assigns:
            continue
        ts = tsmap.get(c)
        if ts and ts[0] in offset:
            out[c] = (ts[0], int(min(12, max(1, ts[1] + offset[ts[0]]))))
    return out, offset


def _review_flags(sizes, extra):
    """Advisory (drops NOTHING): the EXTRAPOLATED ribs — components with no TS number
    AND no vertebra of their own, numbered purely by counting from the anchored ribs.
    Those are the only best-guess numbers, so flag the sizable ones for a quick check."""
    return [{"rib_id": RR.rib_label_id(side, num), "side": side, "number": int(num),
             "size_vox": int(sizes[c])}
            for c, (side, num) in extra.items() if int(sizes[c]) >= MIN_RIB_VOX]


def number_and_overlay(v3_label_path: Path, rib_pred_path: Path, out_path: Path,
                       rib_filter: bool = True) -> dict:
    """relabel_ribs the binary rib mask onto v3 thoracic, overlay onto v3 -> v4 label.
    Returns a per-case rib-connection QC dict (see _rib_connection_qc)."""
    v3 = nib.load(str(v3_label_path))
    lab = np.asanyarray(v3.dataobj).astype(np.int16)
    affine = v3.affine

    # rib-number anchors from v3 thoracic (relabel_ribs reads the vertebra label as the
    # rib number, so remap VerSe 8-19 -> 1-12).
    anchors = np.zeros_like(lab)
    for verse_id, ribnum in THORACIC_TO_RIBNUM.items():
        anchors[lab == verse_id] = ribnum
    if not anchors.any():
        log.warning("%s: no thoracic anchors (8-19) — ribs cannot be numbered", out_path.name)

    # UNION of the two rib segmentations: Möller's binary (clean, reaches the
    # costovertebral joint) ∪ the v3 TS ribs already in `lab` (catches ribs Möller
    # missed). Completeness comes from the union; the false-positive filter below
    # (windowed by connectivity to the spine) removes Möller's off-anatomy blobs.
    moller = np.asanyarray(nib.load(str(rib_pred_path)).dataobj) > 0
    ts_ribs = np.isin(lab, list(RIB_IDS))                    # v3 TS rib voxels (34-57)
    binary = (moller | ts_ribs).astype(np.uint8)
    union_vox = int(binary.sum())

    # canonical rib ids via relabel_ribs offsets (rib_left_N -> 33+N, rib_right_N -> 45+N)
    RR.LEFT_OFFSET, RR.RIGHT_OFFSET = LS.RIB_LEFT_OFFSET, LS.RIB_RIGHT_OFFSET
    labeled, kept = RR.label_and_filter_components(binary, min_voxels=150)
    sizes = np.bincount(labeled.ravel()) if kept else np.zeros(1, dtype=np.int64)
    rib_vol = np.zeros_like(lab)
    assigns: Dict[int, tuple] = {}
    n_overlap = n_tsoff = n_extrap = 0
    n_dropped_fp = 0; dropped_fp_vox = 0
    review_ribs: list = []
    if kept and anchors.any():
        dil = RR.dilate_vertebrae_local(anchors, dilation_radius=4, pad=10)
        # ---- false-positive filter: keep a component if TS-corroborated OR spine-anchored
        # --------------------------------------------------------------------------------
        # Möller's binary rib net hallucinates "rib" on dense abdominal structures (bowel,
        # vascular calcification, contrast); those blobs are Möller-only (no TS) AND float
        # anterior, off the spine. So keep a union COMPONENT if it EITHER overlaps a TS rib
        # (TS corroborates it is rib bone -- this keeps the upper ribs whose vertebra is
        # above the labelled FOV and therefore have no anchor to reach) OR reaches the
        # (generously) dilated thoracic spine (a real rib that TS happened to miss). Bowel
        # satisfies NEITHER -> dropped before numbering, so it can't be extrapolated/clamped
        # onto rib 12. We do NOT require BOTH: that would drop the FOV-edge upper TS ribs,
        # which are real but un-anchored. Equivalently, the gate only ever filters
        # Möller-only components (where the blobs live) -- any component containing a TS rib
        # voxel is always kept. NOTE: dilate_vertebrae_local returns
        # {label:(slices, submask)} -- OR the submasks into a full-volume mask (a bare
        # `dil > 0` is a dict>int TypeError).
        if rib_filter:
            dil_reach = RR.dilate_vertebrae_local(anchors, dilation_radius=8, pad=14)
            spine_dil = np.zeros(lab.shape, dtype=bool)
            for slc, submask in dil_reach.values():
                spine_dil[slc] |= submask
            touch_ts = set(int(c) for c in np.unique(labeled[ts_ribs])) - {0}
            touch_sp = set(int(c) for c in np.unique(labeled[spine_dil])) - {0}
            keepset = (touch_ts | touch_sp)      # TS-corroborated OR spine-anchored
            dropped = [c for c in kept if c not in keepset]
            n_dropped_fp = len(dropped)
            dropped_fp_vox = int(sum(int(sizes[c]) for c in dropped))
            kept = [c for c in kept if c in keepset]
        # ---------------------------------------------------------------------------
        assigns = RR.assign_ribs(labeled, kept, anchors, dil, affine)         # confident overlap vote
        n_overlap = len(assigns)
        # Number the FILTERED components (bowel / off-spine junk already dropped above).
        # (1) Reuse TS's own consecutive numbering for TS-having components, correcting only
        # its FOV offset via the anchored ribs (so TS fragments keep their now-correct
        # number); (2) any Möller-only piece that survived is counted cranio-caudally from
        # the anchored ribs (rare now that Möller-only components are dropped by the filter).
        tsoff, _offset = _ts_offset_assign(labeled, kept, lab, assigns)
        assigns.update(tsoff); n_tsoff = len(tsoff)
        extra = RR.extrapolate_ribs_by_count(labeled, kept, assigns, anchors, affine)
        assigns.update(extra); n_extrap = len(extra)
        review_ribs = _review_flags(sizes, extra)
        rib_vol = RR.build_output_volume(labeled, assigns).astype(np.int16)   # values in 34..57

    v4 = lab.copy()
    v4[np.isin(v4, list(RIB_IDS))] = 0                        # drop any prior (TS) ribs
    place = (rib_vol > 0) & (v4 == 0)                         # ribs only on background
    v4[place] = rib_vol[place]
    recarve_s1_symmetric(v4, affine)                         # fix the asymmetric v3 S1 carve
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v4, affine, v3.header), str(out_path))

    qc = _rib_connection_qc(union_vox, labeled, kept, assigns, sizes, v4)
    qc["rib_vox"] = int(place.sum())
    qc["n_overlap"], qc["n_tsoff"], qc["n_extrap"] = n_overlap, n_tsoff, n_extrap
    qc["n_dropped_fp"], qc["dropped_fp_vox"] = n_dropped_fp, dropped_fp_vox  # Möller off-anatomy blobs removed
    qc["review_ribs"] = review_ribs        # pure-guess extrapolated ribs -> optional check
    return qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3_dir", type=Path, required=True, help="v3 tree (ct/ + labels/)")
    ap.add_argument("--out_dir", type=Path, required=True, help="v4 tree to write")
    ap.add_argument("--model_folder", type=Path, required=True,
                    help="Möller model dir (unzipped ribseg_model_weights/ with dataset.json, plans.json, fold_*/)")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--folds", default="0", help="comma-separated folds, e.g. 0 or 0,1,2 (ensemble)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--no_rib_filter", action="store_true",
                    help="keep Möller off-anatomy blobs (disable the spine-attachment false-positive filter)")
    a = ap.parse_args()

    cts = sorted((a.v3_dir / "ct").glob("*.nii.gz"))
    if a.n_shards > 1:
        cts = [c for i, c in enumerate(cts) if i % a.n_shards == a.shard_id]
    if a.limit:
        cts = cts[:a.limit]
    (a.out_dir / "labels").mkdir(parents=True, exist_ok=True)
    done_dir = a.out_dir / "_v4ribs_done"; done_dir.mkdir(parents=True, exist_ok=True)
    work = a.out_dir / f"_v4ribs_work/shard{a.shard_id}"; work.mkdir(parents=True, exist_ok=True)

    todo = []
    for ct in cts:
        cid = _base(ct)
        out = a.out_dir / "labels" / f"{cid}_label.nii.gz"
        if not a.no_resume and (done_dir / f"{cid}.json").exists() and out.exists():
            continue
        if not (a.v3_dir / "labels" / f"{cid}_label.nii.gz").exists():
            log.warning("%s: no v3 label — skip", cid); continue
        todo.append(ct)
    log.info("shard %d/%d: %d to process (%d total in shard)", a.shard_id, a.n_shards, len(todo), len(cts))
    if not todo:
        return 0

    folds = [f.strip() for f in a.folds.split(",")]
    pred_dir = predict_ribs_for_shard(todo, work, a.model_folder, folds, a.checkpoint, a.device)

    n_ok = 0
    for ct in todo:
        cid = _base(ct)
        rib_pred = pred_dir / f"{cid}.nii.gz"
        if not rib_pred.exists():
            log.warning("%s: no rib prediction — skip", cid); continue
        try:
            qc = number_and_overlay(a.v3_dir / "labels" / f"{cid}_label.nii.gz", rib_pred,
                                    a.out_dir / "labels" / f"{cid}_label.nii.gz",
                                    rib_filter=not a.no_rib_filter)
        except Exception as exc:                              # one odd case must not kill the shard
            log.warning("%s: rib overlay failed (%s) — skip", cid, str(exc)[:140]); continue
        qc["ct"] = cid
        (done_dir / f"{cid}.json").write_text(json.dumps(qc))
        n_ok += 1
        log.info("%s: %d rib vox -> v4 | overlap=%d ts_offset=%d extrap=%d fp_drop=%d(%dvox) "
                 "gaps L%s R%s dup=%s review=%d", cid, qc["rib_vox"], qc["n_overlap"],
                 qc["n_tsoff"], qc["n_extrap"], qc["n_dropped_fp"], qc["dropped_fp_vox"],
                 qc["left_gaps"], qc["right_gaps"], qc["duplicate_rib_ids"], len(qc["review_ribs"]))
    log.info("shard %d/%d done: %d cases", a.shard_id, a.n_shards, n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
