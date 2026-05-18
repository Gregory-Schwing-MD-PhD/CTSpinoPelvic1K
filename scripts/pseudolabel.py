"""
pseudolabel.py — complete the partially-annotated cases with OUT-OF-FOLD
model predictions, producing a FULL pseudo-labelled tree for a v2 release.

WHAT THIS DOES (and deliberately does NOT)
==========================================
Operates ONLY on an already-staged HF export tree (`make hf-stage` →
data/hf_export/: ct/, labels/, manifest.json, splits_5fold.json, ...). It
NEVER re-runs the export and NEVER trains.

Scope: `spine_only` and `pelvic_native` records — the partially-annotated
scans with ONE region manually labelled and the other absent. The MISSING
region is filled with a prediction from the spinopelvic-seg 5-fold model:

  spine_only     manual L1-L6      → pseudo-fill sacrum+hips
  pelvic_native  manual sacrum+hip → pseudo-fill L1-L6

`fused` records (both regions already manual) pass through UNCHANGED.
Finding a better high-fidelity CT for fused cases is a deferred follow-up
(see TODO at end).

OUT-OF-FOLD (no train→pseudo-label leakage)
===========================================
The scoped records WERE training data for the checkpoints. Pseudo-labelling
a case with a model that trained on it would leak. So per case we use ONLY
the single fold that HELD IT OUT: the fold whose `val` set contains the
case's patient token (that fold's model never saw it). This is read from
the SAME splits_5fold.json the model was trained on. A token absent from
every val set (test-only / never trained) safely falls back to the full
5-fold ensemble.

  nnUNetv2_predict ... -f <held-out fold>      # single fold, never trained
                                               #   on this token

HARD CONTRACT (mirrors place_fused_masks.py / export_hf.py)
A manual voxel is NEVER overwritten. Pseudo only fills voxels the manual
annotator left as background (0) or IGNORE_LABEL (10). prov_<region> for the
filled side flips null→"pseudo"; the manual side keeps prov="manual".

MODEL
=====
nnU-Net v2, Dataset803 (merged-label), downloaded from HuggingFace
(configs/pseudolabel_models.json → checkpoints.hf_repo_id). One model
predicts both spine and pelvis (9 merged classes). The model-output →
canonical-10-class mapping (incl. merged last_lumbar) is the externalized
`label_remap` — edit the JSON, never this file, if you retrain.

Usage
-----
  python scripts/pseudolabel.py \
      --hf_export   data/hf_export \
      --out         data/hf_export_v2 \
      --models_config configs/pseudolabel_models.json \
      --nnunet_results $PWD/nnunet/results \
      [--splits data/hf_export/splits_5fold.json] \
      [--device cuda] [--limit N] [--dry_run] [--skip_download]

Publish as a v2 BRANCH (main / anonymous-review URL untouched):
  HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \
      HF_EXPORT_DIR=$PWD/data/hf_export_v2 make hf-push
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.pseudolabel")

# Kept in sync with export_hf.py (re-declared, not imported, so the merge
# core stays a pure dependency-light unit that is trivially testable).
IGNORE_LABEL = 10
CANONICAL_SPINE  = frozenset({1, 2, 3, 4, 5, 6})
CANONICAL_PELVIS = frozenset({7, 8, 9})
REGION_CANONICAL = {"spine": CANONICAL_SPINE, "pelvis": CANONICAL_PELVIS}

# Which region a scoped record is MISSING (and therefore which canonical
# classes we keep from the full prediction). `config` is the exported field.
MISSING_REGION = {
    "spine_only":     "pelvis",   # manual spine, pelvis absent
    "pelvic_native":  "spine",    # manual pelvis, spine absent
}
SCOPE_CONFIGS = tuple(MISSING_REGION)


# ===========================================================================
# Model-independent core  (unit-tested in tests/test_pseudolabel.py)
# ===========================================================================

def remap_prediction(pred, label_remap: Dict[str, int],
                     supplied_canonical) -> "object":
    """Map a model's raw output classes to canonical 10-class ids.

    `label_remap` keys are model-output ints (as JSON strings). Any value
    remapping outside `supplied_canonical` — or any source class not in the
    map (incl. background) — becomes 0. This is the defensive boundary that
    keeps the model from writing into the manually-annotated region.
    """
    import numpy as np
    supplied = set(int(c) for c in supplied_canonical)
    out = np.zeros_like(pred, dtype=np.int16)
    for src_str, dst in label_remap.items():
        dst = int(dst)
        if dst not in supplied:
            continue
        out[pred == int(src_str)] = dst
    return out


def merge_pseudo_into_manual(manual, pred_canonical) -> "object":
    """Manual-preserving merge.

    Returns a copy of `manual` where voxels the annotator left as background
    (0) or IGNORE_LABEL are replaced by `pred_canonical` wherever the
    prediction is non-zero. Manual voxels with a real class (1..9) are NEVER
    touched — the hard provenance contract. Remaining IGNORE collapses to
    background (the record is no longer a partial-annotation case).
    """
    import numpy as np
    merged = np.array(manual, dtype=np.int16, copy=True)
    fillable = (merged == 0) | (merged == IGNORE_LABEL)
    take = fillable & (pred_canonical > 0)
    merged[take] = pred_canonical[take].astype(np.int16)
    merged[merged == IGNORE_LABEL] = 0
    return merged


def updated_record(record: dict, filled_region: str, merged) -> dict:
    """Return record with provenance + voxel stats updated for the v2 tree.

    Only the FILLED region's provenance flips to "pseudo"; the manually
    annotated region's prov is left exactly as-is (never downgraded).
    """
    import numpy as np
    r = dict(record)
    if filled_region == "spine":
        r["prov_spine"] = "pseudo"
    elif filled_region == "pelvis":
        r["prov_pelvis"] = "pseudo"
    r["partial_annotation"] = bool((merged == IGNORE_LABEL).any())
    r["n_voxels_ignore"] = int((merged == IGNORE_LABEL).sum())
    r["n_voxels_fg"] = int(((merged > 0) & (merged != IGNORE_LABEL)).sum())
    r["n_voxels_bg"] = int((merged == 0).sum())
    return r


def build_heldout_fold_map(splits: dict) -> Dict[str, int]:
    """token (str) -> the fold index that HELD IT OUT (token ∈ that fold's
    validation set, so that fold's model never trained on it).

    Tolerant of both splits_5fold.json schemas in play:
      * spinopelvic-seg v2:  folds:[{train_tokens, val_tokens}], test_tokens
      * CTSpinoPelvic1K v6:  folds:[{fold, train, val}]
    Tokens only in test_tokens (or absent everywhere) are intentionally
    NOT in the map → caller uses the full ensemble for those.
    """
    out: Dict[str, int] = {}
    for idx, f in enumerate(splits.get("folds", []) or []):
        fold_idx = int(f.get("fold", idx))
        val = f.get("val_tokens", f.get("val", [])) or []
        for tok in val:
            out[str(tok)] = fold_idx
    return out


# ===========================================================================
# Checkpoint download + parameterized nnU-Net inference
# ===========================================================================

def download_checkpoints(ckpt_cfg: dict, nnunet_results: Path,
                         hf_token: Optional[str]) -> Path:
    """snapshot_download the 5-fold model so it lands at the layout
    nnUNetv2_predict expects: <nnunet_results>/<results_subdir>/<trainer
    __plans__config>/{dataset.json,plans.json,fold_*/}.

    CRITICAL: the HF repo ALREADY contains `<results_subdir>/...` at its
    root, so local_dir MUST be `nnunet_results` itself — NOT
    `nnunet_results/<results_subdir>` (that double-nests it one level too
    deep and nnU-Net can't find dataset.json). Idempotent; skipped if the
    fold dirs are already correctly placed."""
    model_root = nnunet_results / ckpt_cfg["results_subdir"]
    if any(model_root.glob("*/fold_*/")):
        log.info("Checkpoints already present at %s — skip download",
                 model_root)
        return model_root
    from huggingface_hub import snapshot_download
    # An empty/blank token (HF_TOKEN="" from default.env) makes
    # huggingface_hub emit an illegal `Authorization: Bearer ` header. The
    # checkpoints repo is public, so coerce blank -> None (anonymous).
    token = (hf_token or "").strip() or None
    nnunet_results.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s  (auth=%s)", ckpt_cfg["hf_repo_id"],
             nnunet_results, "token" if token else "anonymous")
    snapshot_download(
        repo_id=ckpt_cfg["hf_repo_id"],
        repo_type=ckpt_cfg.get("hf_repo_type", "model"),
        local_dir=str(nnunet_results),     # repo already has <subdir>/ inside
        token=token,
    )
    if not any(model_root.glob("*/fold_*/")):
        log.error("Download finished but %s/<trainer>/fold_* not found — "
                  "repo layout differs from configs/pseudolabel_models.json "
                  "results_subdir=%r", model_root,
                  ckpt_cfg["results_subdir"])
    return model_root


def run_nnunet_folder(nn: dict, in_dir: Path, out_dir: Path,
                      folds: List[int], nnunet_results: str, device: str,
                      npp: int, nps: int) -> bool:
    """Run ONE `nnUNetv2_predict` over a whole input folder (model loaded
    once for the entire fold group — the dominant cost N model-loads -> 1
    per group). `--continue_prediction` skips cases already written, so a
    walltime timeout resumes instead of restarting. Returns False on a
    hard launch failure (caller then marks that group's cases failed)."""
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_in = len(list(in_dir.glob("*_0000.nii.gz")))
    if n_in == 0:
        return True  # nothing to do for this group
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir), "-o", str(out_dir),
        "-d", str(nn["dataset_id"]),
        "-tr", nn["trainer"],
        "-p", nn["plans"],
        "-c", nn["configuration"],
        "-f", *[str(f) for f in folds],
        "-chk", nn.get("checkpoint", "checkpoint_best.pth"),
        "-device", device,
        "-npp", str(npp), "-nps", str(nps),
        "--continue_prediction",
    ]
    env = dict(os.environ)
    env["nnUNet_results"] = nnunet_results
    log.info("  nnUNetv2_predict folds=%s  cases=%d  (model loaded once)",
             folds, n_in)
    try:
        # Stream nnU-Net's own progress straight through (no PIPE capture,
        # so its tqdm/ETA is visible live in the SLURM log).
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        log.error("  nnUNetv2_predict not on PATH — wrong container/env?")
        return False
    except subprocess.CalledProcessError as exc:
        # nnU-Net v2 has a known cosmetic post-completion shutdown crash;
        # validate by filesystem (caller checks per-case output presence)
        # rather than trusting the exit code.
        log.warning("  nnUNetv2_predict exit=%s — validating by filesystem",
                    exc.returncode)
    return True


def _align_to(ref_img, pred_img):
    """Nearest-neighbour resample pred onto ref's grid if they differ. The
    exported CT and label share a grid and we predict on the exported CT, so
    this is normally a no-op; the resample is a safety net."""
    import numpy as np
    if (pred_img.shape[:3] == ref_img.shape[:3]
            and np.allclose(pred_img.affine, ref_img.affine, atol=1e-4)):
        return np.asarray(pred_img.dataobj).astype(np.int16)
    from scipy.ndimage import affine_transform
    M = np.linalg.inv(pred_img.affine) @ ref_img.affine
    out = affine_transform(
        np.asarray(pred_img.dataobj).astype(np.float32),
        M[:3, :3], offset=M[:3, 3], output_shape=ref_img.shape[:3],
        order=0, mode="constant", cval=0.0)
    return np.rint(out).astype(np.int16)


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
    ap.add_argument("--hf_export", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models_config", type=Path,
                    default=Path("configs/pseudolabel_models.json"))
    ap.add_argument("--splits", type=Path, default=None,
                    help="splits_5fold.json the model was trained on "
                         "(default: <hf_export>/splits_5fold.json). MUST be "
                         "the training splits, or out-of-fold is wrong.")
    ap.add_argument("--nnunet_results", type=Path,
                    default=Path(os.environ.get("nnUNet_results",
                                                "nnunet/results")),
                    help="Where checkpoints are downloaded to / found.")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--work_dir", type=Path, default=None,
                    help="Scratch for staged inputs / raw predictions / "
                         "resume markers (default: <out>_work). Kept OUT of "
                         "the v2 tree so it is never uploaded.")
    ap.add_argument("--npp", type=int, default=4,
                    help="nnU-Net preprocessing workers (-npp).")
    ap.add_argument("--nps", type=int, default=4,
                    help="nnU-Net segmentation-export workers (-nps).")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only: copy v1→v2 verbatim, log the per-case "
                         "held-out fold, run no download/inference.")
    args = ap.parse_args()

    # Line-buffer stdout/stderr so logs stream live through Singularity /
    # SLURM redirection instead of appearing only at exit.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(line_buffering=True)
        except Exception:
            pass
    log.info("pseudolabel starting (dry_run=%s) ...", bool(args.dry_run))

    import numpy as np
    import nibabel as nib

    src, out = args.hf_export, args.out
    man_path = src / "manifest.json"
    if not man_path.exists():
        log.error("No manifest.json in %s — run `make hf-stage` first.", src)
        return 1
    cfg = json.loads(args.models_config.read_text())["checkpoints"]
    nn = cfg["nnunet"]

    splits_path = args.splits or (src / "splits_5fold.json")
    if not splits_path.exists():
        log.error("splits file not found: %s (needed for out-of-fold).",
                  splits_path)
        return 1
    heldout = build_heldout_fold_map(json.loads(splits_path.read_text()))
    all_folds = sorted(set(heldout.values())) or [0, 1, 2, 3, 4]
    log.info("Out-of-fold map: %d tokens across folds %s",
             len(heldout), all_folds)

    records = _load_manifest(man_path)
    total = len(records)
    in_scope = [r for r in records
                if r.get("ok", True) and r.get("config") in SCOPE_CONFIGS
                and r.get("ct_file") and r.get("label_file")]
    log.info("manifest: %d records, %d in scope (%s)",
             total, len(in_scope), "/".join(SCOPE_CONFIGS))

    # DRY-RUN is plan-only: classify + log the per-case held-out fold with
    # live [i/N] progress, copy NOTHING, write NO tree. (Copying the whole
    # v1 tree just to "preview" is what made this look hung.)
    if args.dry_run:
        n_oof = n_ens = 0
        for i, rec in enumerate(in_scope, 1):
            tok = str(rec.get("token", ""))
            region = MISSING_REGION[rec["config"]]
            fold = heldout.get(tok)
            if fold is None:
                n_ens += 1; mode = "ensemble(NO held-out fold!)"
            else:
                n_oof += 1; mode = f"oof fold_{fold}"
            log.info("[%d/%d] token=%s cfg=%s -> fill %s via %s",
                     i, len(in_scope), tok, rec["config"], region, mode)
        log.info("=" * 60)
        log.info("DRY-RUN plan: %d would-fill (out-of-fold=%d  "
                 "no-held-out-fold=%d), %d passthrough. Nothing written.",
                 len(in_scope), n_oof, n_ens, total - len(in_scope))
        if n_ens:
            log.warning("%d scoped tokens have NO held-out fold — they are "
                        "NOT in the training splits. If these are training "
                        "cases, --splits is wrong (out-of-fold would leak).",
                        n_ens)
        log.info("=" * 60)
        return 0

    work = args.work_dir or (out.parent / (out.name + "_work"))
    in_root, pred_root, done_dir = (work / "inputs", work / "preds",
                                    work / "done")
    for d in (in_root, pred_root, done_dir):
        d.mkdir(parents=True, exist_ok=True)
    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        if (src / extra).exists():
            shutil.copy2(str(src / extra), str(out / extra))

    if not args.skip_download:
        download_checkpoints(cfg, args.nnunet_results, args.hf_token)

    def _case_id(r) -> str:                       # unique per record
        return Path(r["ct_file"]).name[:-len(".nii.gz")]

    def _copy_into_v2(rel):
        if rel and (src / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (out / rel).exists():
                shutil.copy2(str(src / rel), str(out / rel))

    def _passthrough(r):
        _copy_into_v2(r.get("ct_file"))
        _copy_into_v2(r.get("label_file"))

    # Resume: completed cases left a marker holding their updated record.
    done: Dict[str, dict] = {}
    for m in done_dir.glob("*.json"):
        try:
            done[m.stem] = json.loads(m.read_text())
        except Exception:
            pass
    if done:
        log.info("resume: %d cases already completed (markers)", len(done))

    n_fill = n_oof = n_ens = n_pass = n_fail = n_resume = 0
    new_records: List[dict] = []
    scoped: List[tuple] = []   # (cid, rec, region, supplied, fold)

    for rec in records:
        cfg_v = rec.get("config")
        ct_rel, lbl_rel = rec.get("ct_file"), rec.get("label_file")
        if not rec.get("ok", True) or cfg_v not in SCOPE_CONFIGS \
                or not ct_rel or not lbl_rel:
            _passthrough(rec); new_records.append(rec); n_pass += 1
            continue
        cid = _case_id(rec)
        if cid in done:                       # already merged previously
            _copy_into_v2(ct_rel)             # ensure CT present (idempotent)
            new_records.append(done[cid]); n_resume += 1
            continue
        region = MISSING_REGION[cfg_v]
        fold = heldout.get(str(rec.get("token", "")))
        scoped.append((cid, rec, region, REGION_CANONICAL[region], fold))

    if args.limit and len(scoped) > args.limit:
        for (_c, r, *_x) in scoped[args.limit:]:
            _passthrough(r); new_records.append(r); n_pass += 1
        scoped = scoped[:args.limit]
        log.info("--limit %d: capping this run to %d scoped cases",
                 args.limit, len(scoped))

    # Group scoped cases by held-out fold → ONE predict per group (model
    # loaded once per group instead of once per case).
    groups: Dict[str, list] = defaultdict(list)
    for item in scoped:
        fold = item[4]
        groups["ensemble" if fold is None else f"fold_{fold}"].append(item)
    log.info("scoped=%d to fill across %d group(s): %s",
             len(scoped), len(groups),
             {k: len(v) for k, v in sorted(groups.items())})

    for key, items in sorted(groups.items()):
        folds = (all_folds if key == "ensemble"
                 else [int(key.split("_")[1])])
        in_dir, pred_dir = in_root / key, pred_root / key
        in_dir.mkdir(parents=True, exist_ok=True)
        staged = 0
        for (cid, rec, *_r) in items:
            dst = in_dir / f"{cid}_0000.nii.gz"
            if dst.exists():
                continue
            sct = src / rec["ct_file"]
            if not sct.exists():
                continue
            try:
                os.symlink(os.path.abspath(sct), dst)   # avoid GB copies
            except Exception:
                shutil.copy2(str(sct), str(dst))
            staged += 1
        log.info("group %s: %d cases  folds=%s  (staged +%d)",
                 key, len(items), folds, staged)

        launched = run_nnunet_folder(
            nn, in_dir, pred_dir, folds, str(args.nnunet_results),
            args.device, args.npp, args.nps)

        for (cid, rec, region, supplied, fold) in items:
            tok = rec.get("token", "?")
            pred_p = pred_dir / f"{cid}.nii.gz"
            if not launched or not pred_p.exists():
                _passthrough(rec); new_records.append(rec); n_fail += 1
                log.warning("token=%s cid=%s: no prediction — passthrough",
                            tok, cid)
                continue
            try:
                ref = nib.load(str(src / rec["label_file"]))
                pred_arr = _align_to(ref, nib.load(str(pred_p)))
                pred_canon = remap_prediction(
                    pred_arr, cfg["label_remap"], supplied)
                manual = np.asarray(ref.dataobj).astype(np.int16)
                merged = merge_pseudo_into_manual(manual, pred_canon)
                _copy_into_v2(rec["ct_file"])             # CT into v2
                nib.save(nib.Nifti1Image(merged, ref.affine, ref.header),
                         str(out / rec["label_file"]))    # merged label
                urec = updated_record(rec, region, merged)
                (done_dir / f"{cid}.json").write_text(json.dumps(urec))
                new_records.append(urec)
                n_fill += 1
                n_oof += int(fold is not None)
                n_ens += int(fold is None)
                log.info("token=%s cid=%s: filled %s (%d fg vox)  "
                         "[group %s done %d/%d]", tok, cid, region,
                         int(((merged > 0) & (merged != IGNORE_LABEL)).sum()),
                         key, n_fill, len(scoped))
            except Exception as exc:
                _passthrough(rec); new_records.append(rec); n_fail += 1
                log.warning("token=%s cid=%s: merge failed (%s) — "
                            "passthrough", tok, cid, exc)

    log.info("writing v2 manifest (%d records) ...", len(new_records))
    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import write_manifest
    write_manifest(new_records, out)

    log.info("=" * 60)
    log.info("pseudolabel v2 tree -> %s", out)
    log.info("  pseudo-filled (new)  : %d  (out-of-fold=%d  ensemble=%d)",
             n_fill, n_oof, n_ens)
    log.info("  resumed (prior run)  : %d", n_resume)
    log.info("  passthrough          : %d", n_pass)
    log.info("  failed (passthrough) : %d", n_fail)
    log.info("=" * 60)
    return 0


# TODO(deferred): fused-case completion. For `fused` records, locate the
# other high-fidelity diagnostic CT in the same TCIA study (filtering out
# scouts/localizers, reconciling differing convolution kernels) and fully
# pseudo-label that volume. Non-trivial series selection — tracked
# separately, intentionally NOT attempted here.

if __name__ == "__main__":
    raise SystemExit(main())
