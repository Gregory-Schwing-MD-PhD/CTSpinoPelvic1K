#!/usr/bin/env python3
"""
spineps_ct_pipeline.py — run the full Hendrik-code CT stack over our CTs, so SPINEPS can be
benchmarked head-to-head against CTSpinoPelvic1K labels (and so we get his rib measurements).

Three stages, all inside docker/Dockerfile.spineps:

  1. SPINEPS (CT mode)      spineps sample -model_semantic ct -model_instance ct_instance
                            -model_labeling ct_labeling
                            -> vertebra INSTANCE mask (VerSe ids 1-25, same convention as ours)
                            -> spine SEMANTIC mask (subregion Locations 26-100)
                            -> centroid/POI json
     This is the new modality-specific CT release (v1.4.2 weights); the older SPINEPS was
     T2w/T1w/vibe only, so `-model_semantic t2w` on a CT is NOT the thing to benchmark.

  2. Möller binary rib nnU-Net (optional, --rib_model)
     Same weights the v4 build already uses (Zenodo 10.5281/zenodo.14850928, flattened model
     dir with dataset.json + plans.json + fold_*/). Batched per shard through one predictor.

  3. rib-segmentation (github.com/Hendrik-code/rib-segmentation)
     run_all_steps(rib_mask, vert_instance, vert_semantic) — instance rib assignment, the rib
     length algorithm, and the stump-rib morphology features. Consumes 1 + 2 verbatim, so
     these are HIS numbers on OUR cohort, not a reimplementation.

Everything downstream of SPINEPS runs in SPINEPS' own label space (TPTBox Locations /
Vertebra_Instance). Mapping into our label_scheme ids happens only in benchmark_spineps.py,
at scoring time — the predictions on disk stay native so they can be re-scored later.

Sharded + resumable, same contract as build_v4_ribs.py:

    python scripts/spineps_ct_pipeline.py \\
        --ct_dir  /data/hf_export_v5/ct \\
        --out_dir /data/spineps_bench \\
        --rib_model /workspace/models/moller_ribseg/ribseg_model_weights \\
        --shard_id 0 --n_shards 8

Outputs
    <out>/spineps/<case>_seg-vert_msk.nii.gz     vertebra instances (native SPINEPS ids)
    <out>/spineps/<case>_seg-spine_msk.nii.gz    subregion semantic
    <out>/spineps/<case>_ctd.json                centroids/POI (if produced)
    <out>/ribs/<case>_ribmask.nii.gz             Möller binary ribs
    <out>/ribs/<case>_rib-{inst,sem}_msk.nii.gz  ribs assigned to their vertebra
    <out>/rib_measurements_shard<k>.csv          one row per (case, vertebra, side)
    <out>/rib_features/<case>.json               full feature dicts (arrays included)
    <out>/_done/<case>.json                      resume marker + per-case timings
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.spineps")

# rib-segmentation is a plain module tree (no package install), so its clone root has to be
# importable. The container sets both, but keep the fallback for bare-metal runs.
_RIBSEG_DIR = os.environ.get("RIBSEG_REPO_DIR", "/opt/ribseg")
if Path(_RIBSEG_DIR).is_dir() and _RIBSEG_DIR not in sys.path:
    sys.path.insert(0, _RIBSEG_DIR)

# TPTBox subregion Locations that only ever appear in a SPINEPS *semantic* mask — used to tell
# the two output masks apart by content when the filenames don't say (see _classify_masks).
_SUBREG_LOCATIONS = set(range(41, 51))


def _base(ct_path: Path) -> str:
    """'<base>_ct.nii.gz' (or '<base>.nii.gz') -> '<base>'."""
    n = ct_path.name
    return n[:-len("_ct.nii.gz")] if n.endswith("_ct.nii.gz") else n[:-len(".nii.gz")]


def _bids_label(case_id: str) -> str:
    """BIDS entity labels are alphanumeric-only; SPINEPS' parser wants sub-<label>_ct."""
    return re.sub(r"[^A-Za-z0-9]", "", case_id) or "case"


# ── stage 1: SPINEPS ─────────────────────────────────────────────────────────
def run_spineps(ct: Path, case_id: str, work: Path, out_dir: Path, *, device: str,
                model_semantic: str, model_instance: str, model_labeling: str,
                n4: bool, nocrop: bool, extra: List[str], timeout: int,
                verbose: bool) -> Dict[str, Optional[Path]]:
    """Run `spineps sample` on one CT in an isolated BIDS-ish stage dir and harvest the masks.

    SPINEPS writes into a derivatives folder next to the input and names files by BIDS
    entities, so we give each case its own stage dir and glob the result rather than
    predicting filenames (upstream has renamed them before).
    """
    stage = work / "stage" / case_id
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / f"sub-{_bids_label(case_id)}_ct.nii.gz"
    try:
        os.symlink(os.path.abspath(ct), staged)
    except OSError:
        shutil.copy2(str(ct), str(staged))

    cmd = ["spineps", "sample", "-i", str(staged),
           "-model_semantic", model_semantic,
           "-model_instance", model_instance,
           "-ignore_bids_filter", "-ignore_inference_compatibility"]
    if model_labeling and model_labeling.lower() != "none":
        cmd += ["-model_labeling", model_labeling]
    else:
        cmd += ["-model_labeling", "none"]
    if not n4:
        cmd += ["-non4"]            # N4 bias correction is an MR step; pointless + slow on CT
    if nocrop:
        cmd += ["-nocrop"]
    if device == "cpu":
        cmd += ["-cpu"]
    if verbose:
        cmd += ["-verbose"]
    cmd += extra

    log.info("%s: spineps sample (%s)", case_id, " ".join(cmd[2:]))
    proc = subprocess.run(cmd, cwd=str(stage), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"spineps exited {proc.returncode}\n{tail}")

    return _harvest(stage, case_id, out_dir)


def _classify_masks(masks: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """(vertebra instance, spine semantic) out of whatever SPINEPS wrote.

    Filename first (`seg-vert` / `seg-spine`); if that misses, fall back to content: only the
    semantic mask carries vertebra-subregion Locations (41-50), while the instance mask is
    VerSe vertebra ids (1-25, plus 100+/200+ for IVD/endplate).
    """
    import nibabel as nib

    vert = next((m for m in masks if "seg-vert" in m.name), None)
    sem = next((m for m in masks if "seg-spine" in m.name or "subreg" in m.name), None)
    if vert is not None and sem is not None:
        return vert, sem

    for m in masks:
        if m in (vert, sem):
            continue
        try:
            u = set(np.unique(np.asanyarray(nib.load(str(m)).dataobj)).astype(int).tolist())
        except Exception:
            continue
        core = {x for x in u if 0 < x < 100}      # ignore IVD (100+X) / endplate (200+X)
        if u & _SUBREG_LOCATIONS:
            sem = sem or m
        elif core and max(core) <= 30:            # VerSe vertebra ids only -> instance mask
            vert = vert or m
    return vert, sem


def _harvest(stage: Path, case_id: str, out_dir: Path) -> Dict[str, Optional[Path]]:
    masks = sorted(p for p in stage.rglob("*.nii.gz")
                   if not p.name.endswith("_ct.nii.gz"))     # drop the staged input itself
    vert, sem = _classify_masks(masks)
    if vert is None or sem is None:
        raise RuntimeError(f"could not find SPINEPS masks in {stage} "
                           f"(saw: {[m.name for m in masks]})")

    dst = out_dir / "spineps"
    dst.mkdir(parents=True, exist_ok=True)
    out = {"vert": dst / f"{case_id}_seg-vert_msk.nii.gz",
           "sem": dst / f"{case_id}_seg-spine_msk.nii.gz",
           "ctd": None}
    shutil.copy2(str(vert), str(out["vert"]))
    shutil.copy2(str(sem), str(out["sem"]))
    ctds = sorted(stage.rglob("*ctd*.json")) or sorted(stage.rglob("*poi*.json"))
    if ctds:
        out["ctd"] = dst / f"{case_id}_ctd.json"
        shutil.copy2(str(ctds[0]), str(out["ctd"]))
    return out


# ── stage 2: Möller binary rib nnU-Net ───────────────────────────────────────
def predict_ribs_for_shard(cts: List[Path], work: Path, model_folder: Path, folds: List[str],
                           checkpoint: str, device: str) -> Path:
    """One predictor init for the whole shard (model load dominates per-case cost).

    Möller's zip is a FLATTENED nnU-Net model dir (dataset.json + plans.json + fold_*/), not a
    Dataset<ID> hierarchy — so initialize_from_trained_model_folder is the entry point, exactly
    as in build_v4_ribs.py. Keep the two in step if either is retuned.
    """
    in_dir, pred_dir = work / "rib_in", work / "rib_pred"
    in_dir.mkdir(parents=True, exist_ok=True); pred_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for ct in cts:
        cid = _base(ct)
        if (pred_dir / f"{cid}.nii.gz").exists():
            continue
        dst = in_dir / f"{cid}_0000.nii.gz"
        if not dst.exists():
            try:
                os.symlink(os.path.abspath(ct), dst)
            except OSError:
                shutil.copy2(str(ct), str(dst))
        staged += 1
    if not staged:
        return pred_dir

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=False,      # mirror TTA off: ~8x faster, fine for bone
                                device=torch.device(device), verbose=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=tuple(int(f) for f in folds), checkpoint_name=checkpoint)
    log.info("Möller rib nnU-Net: predicting %d CT(s) (folds=%s, ckpt=%s)", staged, folds, checkpoint)
    predictor.predict_from_files(str(in_dir), str(pred_dir), save_probabilities=False,
                                 overwrite=False)
    return pred_dir


# ── stage 3: rib assignment + length + stump features ────────────────────────
_SCALARS = (str, int, float, bool, np.integer, np.floating, np.bool_)


def _vert_name(v) -> str:
    try:
        from TPTBox import Vertebra_Instance
        return str(Vertebra_Instance(int(v)).name)
    except Exception:
        return str(v)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _SCALARS) or o is None:
        return o
    return None                                   # NII crops etc. — not serialised


def measure_ribs(case_id: str, rib_path: Path, vert_path: Path, sem_path: Path,
                 out_dir: Path, *, calc_orientation: bool, verbose: bool) -> List[Dict]:
    """Upstream run_all_steps() verbatim, then flatten to CSV rows + a features json.

    Returns one row per (vertebra, side): rib_length, the stump-rib flag `sr`, and whatever
    other scalars upstream emits — keys are collected dynamically so an upstream addition
    shows up in the CSV instead of being silently dropped.
    """
    from TPTBox import NII
    from run import run_all_steps                 # /opt/ribseg/run.py — his orchestration, unmodified

    rib = NII.load(str(rib_path), seg=True)
    vert = NII.load(str(vert_path), seg=True)
    sem = NII.load(str(sem_path), seg=True)

    if rib.shape != vert.shape:                   # nnU-Net writes in CT space, SPINEPS resamples back
        log.info("%s: resampling rib mask %s -> %s", case_id, rib.shape, vert.shape)
        rib = rib.resample_from_to(vert)

    results = run_all_steps(rib, vert, sem, poi=None, calc_orientation=calc_orientation,
                            verbose=verbose)

    rib_out = out_dir / "ribs"; rib_out.mkdir(parents=True, exist_ok=True)
    feat_out = out_dir / "rib_features"; feat_out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for d in results:
        row = {"case": case_id}
        for k, v in d.items():
            if k == "features":
                continue
            if isinstance(v, _SCALARS) or v is None:
                row[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, _SCALARS) for x in v):
                row[k] = ";".join(str(x) for x in v)
        row["vertebra_name"] = _vert_name(d.get("vertebra"))
        rows.append(row)

    (feat_out / f"{case_id}.json").write_text(json.dumps(
        [{k: _jsonable(v) for k, v in d.items()} for d in results], indent=1))
    return rows


def _write_rows(csv_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    lead = [f for f in ("case", "vertebra", "vertebra_name", "side", "rib_length", "sr") if f in seen]
    fields = lead + [f for f in fields if f not in lead]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct_dir", type=Path, required=True, help="dir of <case>_ct.nii.gz")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model_semantic", default="ct")
    ap.add_argument("--model_instance", default="ct_instance")
    ap.add_argument("--model_labeling", default="ct_labeling", help="'none' to skip C1..L6 naming")
    ap.add_argument("--n4", action="store_true", help="keep N4 bias correction (MR step; off for CT)")
    ap.add_argument("--nocrop", action="store_true")
    ap.add_argument("--spineps_extra", default="", help="extra flags passed straight to spineps")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per case for spineps")
    ap.add_argument("--rib_model", type=Path, default=None,
                    help="Möller model dir (dataset.json + plans.json + fold_*/); omit to skip ribs")
    ap.add_argument("--folds", default="0", help="comma-separated, e.g. 0 or 0,1,2 (ensemble)")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--calc_orientation", action="store_true",
                    help="also compute vertebra orientation (slower)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    cts = sorted(a.ct_dir.glob("*.nii.gz"))
    if a.n_shards > 1:
        cts = [c for i, c in enumerate(cts) if i % a.n_shards == a.shard_id]
    if a.limit:
        cts = cts[:a.limit]

    done_dir = a.out_dir / "_done"; done_dir.mkdir(parents=True, exist_ok=True)
    work = a.out_dir / f"_work/shard{a.shard_id}"; work.mkdir(parents=True, exist_ok=True)

    todo = [c for c in cts
            if a.no_resume or not (done_dir / f"{_base(c)}.json").exists()]
    log.info("shard %d/%d: %d to process (%d in shard, %d already done)",
             a.shard_id, a.n_shards, len(todo), len(cts), len(cts) - len(todo))
    if not todo:
        return 0

    # Ribs first: one predictor init for the shard, then per-case SPINEPS + measurement.
    pred_dir = None
    if a.rib_model is not None:
        if not (a.rib_model / "plans.json").exists():
            log.error("no nnU-Net model at %s (need dataset.json/plans.json/fold_*)", a.rib_model)
            return 1
        pred_dir = predict_ribs_for_shard(todo, work, a.rib_model,
                                          [f.strip() for f in a.folds.split(",")],
                                          a.checkpoint, a.device)

    extra = a.spineps_extra.split() if a.spineps_extra else []
    all_rows: List[Dict] = []
    csv_path = a.out_dir / f"rib_measurements_shard{a.shard_id}.csv"
    n_ok = n_fail = 0

    for ct in todo:
        cid = _base(ct)
        rec: Dict[str, object] = {"case": cid}
        t0 = time.time()
        try:
            paths = run_spineps(ct, cid, work, a.out_dir, device=a.device,
                                model_semantic=a.model_semantic, model_instance=a.model_instance,
                                model_labeling=a.model_labeling, n4=a.n4, nocrop=a.nocrop,
                                extra=extra, timeout=a.timeout, verbose=a.verbose)
            rec["spineps_sec"] = round(time.time() - t0, 1)
            rec["vert"] = paths["vert"].name
            rec["sem"] = paths["sem"].name
        except Exception as exc:                  # one odd case must not kill the shard
            log.warning("%s: SPINEPS failed (%s)", cid, str(exc)[:400])
            rec["error"] = f"spineps: {str(exc)[:400]}"
            (done_dir / f"{cid}.json").write_text(json.dumps(rec))
            n_fail += 1
            continue

        if pred_dir is not None:
            rib_pred = pred_dir / f"{cid}.nii.gz"
            if not rib_pred.exists():
                log.warning("%s: no rib prediction — ribs skipped", cid)
                rec["error"] = "no rib prediction"
            else:
                keep = a.out_dir / "ribs" / f"{cid}_ribmask.nii.gz"
                keep.parent.mkdir(parents=True, exist_ok=True)
                if not keep.exists():
                    shutil.copy2(str(rib_pred), str(keep))
                t1 = time.time()
                try:
                    rows = measure_ribs(cid, keep, paths["vert"], paths["sem"], a.out_dir,
                                        calc_orientation=a.calc_orientation, verbose=a.verbose)
                    all_rows.extend(rows)
                    rec["n_rib_rows"] = len(rows)
                    rec["ribs_sec"] = round(time.time() - t1, 1)
                    _write_rows(csv_path, all_rows)          # flush per case — shard can die anytime
                except Exception:
                    log.warning("%s: rib measurement failed\n%s", cid,
                                traceback.format_exc(limit=3))
                    rec["error"] = "rib measurement: " + traceback.format_exc(limit=1)[-300:]

        (done_dir / f"{cid}.json").write_text(json.dumps(rec))
        shutil.rmtree(work / "stage" / cid, ignore_errors=True)
        n_ok += 1
        log.info("%s: done in %.0fs (%s rib rows)", cid, time.time() - t0,
                 rec.get("n_rib_rows", "-"))

    _write_rows(csv_path, all_rows)
    log.info("shard %d/%d complete: %d ok, %d failed -> %s",
             a.shard_id, a.n_shards, n_ok, n_fail, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
