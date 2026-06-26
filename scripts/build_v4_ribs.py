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
import logging
import os
import subprocess
import sys
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


def number_and_overlay(v3_label_path: Path, rib_pred_path: Path, out_path: Path) -> int:
    """relabel_ribs the binary rib mask onto v3 thoracic, overlay onto v3 -> v4 label."""
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

    binary = (np.asanyarray(nib.load(str(rib_pred_path)).dataobj) > 0).astype(np.uint8)

    # canonical rib ids via relabel_ribs offsets (rib_left_N -> 33+N, rib_right_N -> 45+N)
    RR.LEFT_OFFSET, RR.RIGHT_OFFSET = LS.RIB_LEFT_OFFSET, LS.RIB_RIGHT_OFFSET
    labeled, kept = RR.label_and_filter_components(binary, min_voxels=150)
    rib_vol = np.zeros_like(lab)
    if kept and anchors.any():
        dil = RR.dilate_vertebrae_local(anchors, dilation_radius=4, pad=10)
        assigns = RR.assign_ribs(labeled, kept, anchors, dil, affine)
        rib_vol = RR.build_output_volume(labeled, assigns).astype(np.int16)   # values in 34..57

    v4 = lab.copy()
    v4[np.isin(v4, list(RIB_IDS))] = 0                        # drop any prior (TS) ribs
    place = (rib_vol > 0) & (v4 == 0)                         # ribs only on background
    v4[place] = rib_vol[place]
    recarve_s1_symmetric(v4, affine)                         # fix the asymmetric v3 S1 carve
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v4, affine, v3.header), str(out_path))
    return int(place.sum())


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
            n = number_and_overlay(a.v3_dir / "labels" / f"{cid}_label.nii.gz", rib_pred,
                                   a.out_dir / "labels" / f"{cid}_label.nii.gz")
        except Exception as exc:                              # one odd case must not kill the shard
            log.warning("%s: rib overlay failed (%s) — skip", cid, str(exc)[:140]); continue
        (done_dir / f"{cid}.json").write_text(f'{{"ct":"{cid}","rib_vox":{n}}}')
        n_ok += 1
        log.info("%s: %d rib voxels numbered -> v4", cid, n)
    log.info("shard %d/%d done: %d cases", a.shard_id, a.n_shards, n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
