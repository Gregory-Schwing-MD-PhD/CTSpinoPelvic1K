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
import tempfile
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
    """snapshot_download the 5-fold model into the layout nnUNetv2_predict
    expects: <nnunet_results>/<results_subdir>/. Idempotent (HF cache);
    skipped automatically if fold dirs already present."""
    dest = nnunet_results / ckpt_cfg["results_subdir"]
    if any(dest.glob("*/fold_*/")):
        log.info("Checkpoints already present at %s — skip download", dest)
        return dest
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", ckpt_cfg["hf_repo_id"], dest)
    snapshot_download(
        repo_id=ckpt_cfg["hf_repo_id"],
        repo_type=ckpt_cfg.get("hf_repo_type", "model"),
        local_dir=str(dest),
        token=hf_token,
    )
    return dest


def run_nnunet(nn: dict, ct_path: Path, work_root: Path,
               folds: List[int], nnunet_results: str,
               device: str) -> Optional[Path]:
    """Run `nnUNetv2_predict` with an explicit fold list (single held-out
    fold for training cases; all folds for never-trained cases). Returns
    the predicted seg path, or None on failure. Nothing about the label
    space / checkpoint identity is hardcoded — all from `nn`."""
    in_dir, out_dir = work_root / "in", work_root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(ct_path), str(in_dir / "case_0000.nii.gz"))

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
    ]
    env = dict(os.environ)
    env["nnUNet_results"] = nnunet_results
    log.info("    nnUNetv2_predict d=%s folds=%s", nn["dataset_id"], folds)
    try:
        subprocess.run(cmd, check=True, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        log.error("    nnUNetv2_predict not on PATH — wrong container/env?")
        return None
    except subprocess.CalledProcessError as exc:
        log.error("    nnUNetv2_predict failed (%s):\n%s", exc.returncode,
                  (exc.output or b"").decode(errors="replace")[-2000:])
        return None
    pred = out_dir / "case.nii.gz"
    return pred if pred.exists() else None


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
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only: copy v1→v2 verbatim, log the per-case "
                         "held-out fold, run no download/inference.")
    args = ap.parse_args()

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
    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        if (src / extra).exists():
            shutil.copy2(str(src / extra), str(out / extra))

    if not args.dry_run and not args.skip_download:
        download_checkpoints(cfg, args.nnunet_results, args.hf_token)

    n_fill = n_oof = n_ens = n_pass = n_fail = 0
    new_records: List[dict] = []

    def _passthrough(r):
        for rel in (r.get("ct_file"), r.get("label_file")):
            if rel and (src / rel).exists():
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                if not (out / rel).exists():
                    shutil.copy2(str(src / rel), str(out / rel))

    for rec in records:
        cfg_v = rec.get("config")
        tok = str(rec.get("token", ""))
        ct_rel, lbl_rel = rec.get("ct_file"), rec.get("label_file")

        if not rec.get("ok", True) or cfg_v not in SCOPE_CONFIGS \
                or not ct_rel or not lbl_rel:
            _passthrough(rec); new_records.append(rec); n_pass += 1
            continue

        region = MISSING_REGION[cfg_v]
        supplied = REGION_CANONICAL[region]
        fold = heldout.get(tok)
        if fold is None:
            folds = all_folds          # never trained on tok → ensemble ok
            mode = "ensemble(no-train)"
        else:
            folds = [fold]             # the ONLY fold that held tok out
            mode = f"oof fold_{fold}"

        if args.dry_run:
            _passthrough(rec); new_records.append(rec); n_pass += 1
            log.info("token=%s cfg=%s: DRY-RUN would fill %s via %s",
                     tok, cfg_v, region, mode)
            continue

        ct_src, lbl_src = src / ct_rel, src / lbl_rel
        if not ct_src.exists() or not lbl_src.exists():
            _passthrough(rec); new_records.append(rec); n_fail += 1
            log.warning("token=%s: ct/label missing on disk — passthrough", tok)
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="psl_") as td:
                pred_path = run_nnunet(
                    nn, ct_src, Path(td), folds,
                    str(args.nnunet_results), args.device)
                if pred_path is None:
                    raise RuntimeError("inference produced no output")
                ref = nib.load(str(lbl_src))
                pred_arr = _align_to(ref, nib.load(str(pred_path)))
                pred_canon = remap_prediction(
                    pred_arr, cfg["label_remap"], supplied)
                manual = np.asarray(ref.dataobj).astype(np.int16)
                merged = merge_pseudo_into_manual(manual, pred_canon)

            _passthrough(rec)  # copies the CT; label overwritten below
            nib.save(nib.Nifti1Image(merged, ref.affine, ref.header),
                     str(out / lbl_rel))
            new_records.append(updated_record(rec, region, merged))
            n_fill += 1
            n_oof += int(fold is not None)
            n_ens += int(fold is None)
            log.info("token=%s cfg=%s: filled %s via %s (%d fg vox)",
                     tok, cfg_v, region, mode,
                     int(((merged > 0) & (merged != IGNORE_LABEL)).sum()))
        except Exception as exc:
            _passthrough(rec); new_records.append(rec); n_fail += 1
            log.warning("token=%s: pseudo-fill failed (%s) — passthrough",
                        tok, exc)

        if args.limit and n_fill >= args.limit:
            done = {id(r) for r in new_records}
            for r in records:
                if id(r) not in done:
                    _passthrough(r); new_records.append(r); n_pass += 1
            log.info("--limit %d reached; remaining passed through",
                     args.limit)
            break

    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import write_manifest
    write_manifest(new_records, out)

    log.info("=" * 60)
    log.info("pseudolabel v2 tree -> %s", out)
    log.info("  pseudo-filled        : %d  (out-of-fold=%d  ensemble=%d)",
             n_fill, n_oof, n_ens)
    log.info("  passthrough          : %d", n_pass)
    log.info("  failed (passthrough) : %d", n_fail)
    log.info("=" * 60)
    if args.dry_run:
        log.info("DRY-RUN: no download/inference; v2 tree mirrors v1.")
    return 0


# TODO(deferred): fused-case completion. For `fused` records, locate the
# other high-fidelity diagnostic CT in the same TCIA study (filtering out
# scouts/localizers, reconciling differing convolution kernels) and fully
# pseudo-label that volume. Non-trivial series selection — tracked
# separately, intentionally NOT attempted here.

if __name__ == "__main__":
    raise SystemExit(main())
