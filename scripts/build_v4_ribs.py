"""
build_v4_ribs.py — v4 = v3 + high-quality ribs, in our VerSe-native scheme (ids 34-57).

Pipeline per case (sharded like build_v3_totalseg, resumable):
  CT --(Möller binary rib nnU-Net)--> binary rib mask                               [GPU]
  TRUST the v3 TS rib numbering (ids 34-57) verbatim; GRAFT on the Möller voxels that
  connect to a TS rib -- extending each rib medially toward the spine (costovertebral
  joint) and bridging TS fragmentation -- via the NEAREST TS rib number; DROP Möller
  components with no TS rib (bowel / vascular-calcification false positives)         [CPU]
  overlay on v3 (clear old ribs 34-57, write new on background, never touch GT) -> v4

WHY this design: TotalSegmentator already numbers ribs correctly per level, so we keep its
numbering and avoid any renumber step -- the old renumber (overlap-vote / FOV-offset /
extrapolate) was what collapsed FOV-limited cases (e.g. all ribs -> rib 12 when only T12 is an
anchor) and manufactured numbering gaps. Möller's binary net (Dice 0.997) is more complete
medially and better connected, so it is used ONLY to complete/repair the TS ribs, never to
(re)number them. Bowel false positives are excluded for free: a Möller blob that does not
connect to a TS rib is simply never grafted.

WEIGHTS (one-time): download ribseg_model_weights.zip from Zenodo 10.5281/zenodo.14850928 and
unzip into the Möller model dir (fold_*/, dataset.json, plans.json); pass it via --model_folder
(the nnU-Net Python API reads id/trainer/plans/config from the folder).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS          # canonical VerSe-native ids

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.v4ribs")

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


# Tunable: below this a component is a speckle, not worth flagging for a number check.
MIN_RIB_VOX = 300
# auto-drop a same-id "duplicate" piece when it's a small fraction of the main rib: such a
# spur/stray (or a small hyperplastic-TP nub) is not a real second rib. Larger second pieces
# (a true fragment, or a substantial TP / mis-numbered neighbour) are KEPT and flagged for
# manual review. Calibrated on sampled flags: real strays were <2% of the main rib; genuine
# second ribs / substantial TPs were >15%.
STRAY_ABS_VOX, STRAY_FRAC = 500, 0.10
# a same-id "dup" whose two pieces are >= this far apart is almost certainly two DIFFERENT
# structures sharing one number (a TS mislabel) -> mandatory review; a smaller gap is one rib
# broken into nearby pieces (a benign interrupted rib, often CT-bridgeable) -> advisory only.
GAP_MM_MISLABEL = 25.0


def _drop_same_id_strays(out: np.ndarray):
    """keep-largest-per-id: drop a same-id component when it is a small fraction of the main
    rib (a spur / small TP nub). Substantial second pieces are kept (they surface as a dup)."""
    st = ndimage.generate_binary_structure(3, 3)
    n = 0; vox = 0
    for rid in range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13):       # 34..57
        idx = np.argwhere(out == rid)
        if idx.size == 0:
            continue
        lo = idx.min(0); hi = idx.max(0) + 1
        sub = out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]                      # view into out
        cc, k = ndimage.label(sub == rid, structure=st)
        if k < 2:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        order = np.argsort(sizes)[::-1]
        thresh = max(STRAY_ABS_VOX, STRAY_FRAC * int(sizes[order[0]]))
        for j in order[1:]:
            if int(sizes[j]) < thresh:
                sub[cc == (j + 1)] = 0
                n += 1; vox += int(sizes[j])
    return n, vox


def _trust_ts_graft_moller(lab: np.ndarray, ts_mask: np.ndarray, moller: np.ndarray):
    """Trust TS's rib numbering; graft on the Möller voxels that connect to a TS rib.

    TS already numbers ribs per level (v3 ids 34-57) and its numbering is reliable, so we keep
    it verbatim -- no renumber, hence no FOV-offset collapse (the all-rib-12 failure). Möller's
    binary mask is more complete medially (it reaches the costovertebral joint) and better
    connected, so it is used ONLY to extend / repair the TS ribs: a Möller voxel is added
    (taking the number of the NEAREST TS rib voxel) iff its union component contains a TS rib.
    A Möller component with no TS rib (bowel / vascular calcification -- the classic false
    positive) is never grafted. This simultaneously bridges TS fragmentation (the two halves of
    a split rib reconnect through Möller bone -> one piece) and excludes the FP blobs, with no
    number-guessing anywhere. Returns (out_rib int16 in 34..57, stats dict)."""
    st = ndimage.generate_binary_structure(3, 3)
    out = np.where(ts_mask, lab, 0).astype(np.int16)             # TS ribs verbatim
    moller_extra = moller & ~ts_mask
    graft_vox = 0
    moller_only_vox = int(moller_extra.sum())
    if ts_mask.any() and moller_extra.any():
        union = ts_mask | moller
        labeled, _ = ndimage.label(union, structure=st)
        ts_comps = set(int(c) for c in np.unique(labeled[ts_mask])) - {0}
        in_ts_comp = (np.isin(labeled, list(ts_comps)) if ts_comps
                      else np.zeros_like(union, dtype=bool))
        graft = moller_extra & in_ts_comp
        moller_only_vox = int((moller_extra & ~in_ts_comp).sum())  # not grafted -> dropped (FP)
        if graft.any():
            # nearest-TS-rib number per grafted voxel, on a cropped EDT (cheap): EDT of ~ts
            # gives, for each voxel, the index of the closest TS voxel; read its rib number.
            box = ndimage.find_objects(union.astype(np.int8))[0]
            sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in box)
            _, ind = ndimage.distance_transform_edt(~ts_mask[sl], return_indices=True)
            sub = out[sl]; gsub = graft[sl]
            nearest = sub[tuple(ind)]
            sub[gsub] = nearest[gsub]
            out[sl] = sub
            graft_vox = int(gsub.sum())
    n_stray, stray_vox = _drop_same_id_strays(out)
    return out, {"graft_vox": graft_vox, "moller_only_vox": moller_only_vox,
                 "n_stray_dropped": n_stray, "stray_dropped_vox": stray_vox,
                 "n_ts_ribs": int(len({int(v) for v in np.unique(out) if v}))}


def _rib_qc_from_v4(v4: np.ndarray, affine, union_vox: int, stats: dict) -> dict:
    """Rib QC from the FINAL v4 label: per side the rib numbers present, GAPS (a missing number
    between present ones) and DUPLICATE ids (a number in 2+ pieces >= 50 vox, after the stray
    drop), plus graft / dropped-Möller bookkeeping. Keeps the keys qc_v4_ribs.py reads."""
    st = ndimage.generate_binary_structure(3, 3)
    present = {int(x) for x in np.unique(v4)}

    def _nums(off):
        return sorted(i - off for i in range(off + 1, off + 13) if i in present)

    def _gaps(ns):
        return [n for n in range(min(ns), max(ns) + 1) if n not in ns] if ns else []

    sp = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))     # voxel spacing (mm)
    left = _nums(LS.RIB_LEFT_OFFSET); right = _nums(LS.RIB_RIGHT_OFFSET)
    dup_break = []          # one rib in 2 NEARBY pieces -> advisory (often CT-bridgeable)
    dup_mislabel = []       # 2 pieces FAR apart         -> mandatory review (TS mislabel)
    for off in (LS.RIB_LEFT_OFFSET, LS.RIB_RIGHT_OFFSET):
        for n in range(1, 13):
            rid = off + n
            idx = np.argwhere(v4 == rid)
            if idx.size == 0:
                continue
            lo = idx.min(0); hi = idx.max(0) + 1
            cc, k = ndimage.label(v4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == rid, structure=st)
            sizes = np.bincount(cc.ravel())[1:]
            big = [j for j in range(len(sizes)) if sizes[j] >= 50]
            if len(big) < 2:
                continue
            order = sorted(big, key=lambda j: sizes[j], reverse=True)
            a = cc == order[0] + 1
            b = cc == order[1] + 1
            gap = float(ndimage.distance_transform_edt(~a, sampling=sp)[b].min())
            (dup_mislabel if gap >= GAP_MM_MISLABEL else dup_break).append(rid)
    dups = sorted(dup_break + dup_mislabel)
    thor = sorted(7 + n for n in range(1, 13) if (7 + n) in present)
    mo = stats["moller_only_vox"]
    return {
        "union_vox": int(union_vox), "graft_vox": stats["graft_vox"], "moller_only_vox": mo,
        "n_stray_dropped": stats["n_stray_dropped"], "stray_dropped_vox": stats["stray_dropped_vox"],
        "thoracic_levels": thor, "left_rib_nums": left, "right_rib_nums": right,
        "left_gaps": _gaps(left), "right_gaps": _gaps(right),
        "duplicate_rib_ids": dups,                 # all dups (back-compat)
        "dup_mislabel": sorted(dup_mislabel),      # pieces far apart -> mandatory review
        "dup_break": sorted(dup_break),            # nearby pieces -> advisory / CT-bridgeable
        # back-compat keys for qc_v4_ribs.py / the per-case log line:
        "n_overlap": stats["n_ts_ribs"], "n_tsoff": 0, "n_extrap": 0,
        "n_dropped_fp": 0, "dropped_fp_vox": mo,
        "drop_frac": round(mo / union_vox, 4) if union_vox else 0.0,
        "review_ribs": [],
    }


def number_and_overlay(v3_label_path: Path, rib_pred_path: Path, out_path: Path,
                       rib_filter: bool = True) -> dict:
    """Trust TS's per-level rib numbering (v3 ids 34-57) and graft on the Möller voxels that
    connect to a TS rib (extend each rib toward the spine + bridge TS fragmentation); drop
    Möller-only blobs (bowel FPs); overlay onto v3 -> v4. `rib_filter` is retained for the CLI;
    the Möller-only drop is intrinsic to the design. Returns a per-case rib QC dict."""
    v3 = nib.load(str(v3_label_path))
    lab = np.asanyarray(v3.dataobj).astype(np.int16)
    affine = v3.affine

    # Trust TS's per-level numbering (v3 ribs at 34-57); graft Möller's spine-bridge / repair.
    moller = np.asanyarray(nib.load(str(rib_pred_path)).dataobj) > 0
    ts_mask = np.isin(lab, list(RIB_IDS))                    # v3 TS ribs (34-57); numbering trusted
    union_vox = int((ts_mask | moller).sum())

    out_rib, stats = _trust_ts_graft_moller(lab, ts_mask, moller)

    v4 = lab.copy()
    v4[np.isin(v4, list(RIB_IDS))] = 0                        # drop the prior TS ribs
    place = (out_rib > 0) & (v4 == 0)                         # write ribs only on background
    v4[place] = out_rib[place]
    recarve_s1_symmetric(v4, affine)                         # fix the asymmetric v3 S1 carve
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v4, affine, v3.header), str(out_path))

    qc = _rib_qc_from_v4(v4, affine, union_vox, stats)
    qc["rib_vox"] = int(place.sum())
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
        log.info("%s: %d rib vox -> v4 | ts_ribs=%d graft=%dvox moller_drop=%dvox stray_drop=%d "
                 "gaps L%s R%s dup=%s", cid, qc["rib_vox"], qc["n_overlap"], qc["graft_vox"],
                 qc["moller_only_vox"], qc["n_stray_dropped"], qc["left_gaps"], qc["right_gaps"],
                 qc["duplicate_rib_ids"])
    log.info("shard %d/%d done: %d cases", a.shard_id, a.n_shards, n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
