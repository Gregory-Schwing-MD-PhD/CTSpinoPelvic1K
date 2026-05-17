"""
pseudolabel.py — complete the partially-annotated cases with model
predictions, producing a FULL pseudo-labelled tree for a v2 HF release.

WHAT THIS DOES (and deliberately does NOT)
==========================================
Operates ONLY on an already-staged HF export tree (`make hf-stage` →
data/hf_export/: ct/, labels/, manifest.json, splits_5fold.json, ...). It
NEVER re-runs the export and NEVER trains.

Scope (per the design decision): only `spine_only` and `pelvic_native`
records — i.e. the spine-only / pelvic-only / separate-mode scans that have
ONE region manually annotated and the other absent. For each, the MISSING
region is filled with a 5-fold nnU-Net ensemble prediction.

  spine_only     manual L1-L6      → pseudo-fill sacrum+hips (7,8,9)
  pelvic_native  manual sacrum+hip → pseudo-fill L1-L6       (1-6)

Hard contract (mirrors place_fused_masks.py / export_hf.py): a manual voxel
is NEVER overwritten. Pseudo only fills voxels the manual annotator left as
background (0) or IGNORE_LABEL (10). prov_<region> for the filled side
flips null→"pseudo"; the manual side keeps prov="manual".

`fused` records (both regions already manual) pass through UNCHANGED.
Finding a better high-fidelity CT for fused cases and fully pseudo-labelling
it is a separate, deferred follow-up (scouts / mixed kernels make CT
selection non-trivial) — see TODO at end.

MODEL IDENTITY IS EXTERNALIZED
==============================
Every model-coupled fact (nnU-Net dataset id / trainer / plans / folds /
checkpoint, AND the model-output→canonical-10-class `label_remap`) lives in
configs/pseudolabel_models.json — NOT in this file. The training scheme is
still in flux (LSTV-only, L5+L6 merged into last_lumbar, possible retrain on
normal spines); editing the JSON is sufficient to track those decisions.
A model with enabled=false (or no checkpoints) → its records are SKIPPED and
left partial, never fabricated.

Usage
-----
  python scripts/pseudolabel.py \
      --hf_export   data/hf_export \
      --out         data/hf_export_v2 \
      --models_config configs/pseudolabel_models.json \
      --nnunet_results $nnUNet_results \
      [--device cuda] [--limit N] [--dry_run]

Then publish as a v2 branch (main / anonymous-review URL untouched):
  make hf-stage   # if not already staged
  python scripts/pseudolabel.py ... --out data/hf_export_v2
  HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \
      make hf-push   # (point hf-push at data/hf_export_v2 via HF_EXPORT_DIR)
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

# Which region a scoped record is MISSING (and therefore which model fills
# it). `config` is the exported field (export_hf.py build_work).
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
    keeps a mis-specified model from writing into the wrong region.
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

    Returns a copy of `manual` where voxels that the manual annotator left
    as background (0) or IGNORE_LABEL are replaced by `pred_canonical`
    wherever the prediction is non-zero. Manual voxels with a real class
    (1..9) are NEVER touched — the hard provenance contract.
    """
    import numpy as np
    merged = np.array(manual, dtype=np.int16, copy=True)
    fillable = (merged == 0) | (merged == IGNORE_LABEL)
    take = fillable & (pred_canonical > 0)
    merged[take] = pred_canonical[take].astype(np.int16)
    # Any IGNORE the model did not fill collapses to background: the record
    # is no longer a partial-annotation case once a region is completed.
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


# ===========================================================================
# Parameterized inference  (nnU-Net native 5-fold ensemble)
# ===========================================================================

def _find_model(models_cfg: dict, region: str) -> Optional[dict]:
    for m in models_cfg.get("models", []):
        if m.get("fills_region") == region and m.get("enabled"):
            return m
    return None


def run_nnunet_ensemble(model: dict, ct_path: Path, work_root: Path,
                        nnunet_results: Optional[str],
                        device: str) -> Optional[Path]:
    """Run `nnUNetv2_predict` with the model's 5 folds (native softmax
    ensemble). Returns the predicted seg path, or None on failure.

    Parameterized entirely from `model["nnunet"]`; nothing about the label
    space or checkpoint identity is hardcoded here.
    """
    nn = model["nnunet"]
    in_dir  = work_root / "in"
    out_dir = work_root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # nnU-Net v2 expects <caseid>_0000.nii.gz for a single-channel input.
    case_in = in_dir / "case_0000.nii.gz"
    shutil.copy2(str(ct_path), str(case_in))

    folds = [str(f) for f in nn.get("folds", [0, 1, 2, 3, 4])]
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir), "-o", str(out_dir),
        "-d", str(nn["dataset_id"]),
        "-tr", nn.get("trainer", "nnUNetTrainer"),
        "-p", nn.get("plans", "nnUNetPlans"),
        "-c", nn["configuration"],
        "-f", *folds,
        "-chk", nn.get("checkpoint", "checkpoint_final.pth"),
        "-device", device,
    ]
    env = dict(os.environ)
    if nnunet_results:
        env["nnUNet_results"] = nnunet_results
    log.info("    nnUNetv2_predict d=%s folds=%s", nn["dataset_id"], folds)
    try:
        subprocess.run(cmd, check=True, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        log.error("    nnUNetv2_predict not on PATH — is the nnU-Net env active?")
        return None
    except subprocess.CalledProcessError as exc:
        log.error("    nnUNetv2_predict failed (%s):\n%s",
                  exc.returncode, (exc.output or b"").decode(errors="replace")[-2000:])
        return None
    pred = out_dir / "case.nii.gz"
    return pred if pred.exists() else None


def _align_to(ref_img, pred_img):
    """Nearest-neighbour resample pred onto ref's grid if they differ.
    The exported CT and label share a grid, and we predict on the exported
    CT, so this is normally a no-op; the resample is a safety net."""
    import numpy as np
    import nibabel as nib
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
    ap.add_argument("--hf_export", required=True, type=Path,
                    help="Staged v1 tree (from `make hf-stage`).")
    ap.add_argument("--out", required=True, type=Path,
                    help="v2 tree to create (e.g. data/hf_export_v2).")
    ap.add_argument("--models_config", type=Path,
                    default=Path("configs/pseudolabel_models.json"))
    ap.add_argument("--nnunet_results", default=os.environ.get("nnUNet_results"),
                    help="nnU-Net results root (default: $nnUNet_results).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N scoped records (debug).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only: copy v1→v2 verbatim, log what WOULD be "
                         "pseudo-filled, run no inference.")
    args = ap.parse_args()

    import numpy as np
    import nibabel as nib

    src, out = args.hf_export, args.out
    man_path = src / "manifest.json"
    if not man_path.exists():
        log.error("No manifest.json in %s — run `make hf-stage` first.", src)
        return 1
    models_cfg = json.loads(args.models_config.read_text())
    records = _load_manifest(man_path)

    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # Passthrough non-CT/label artifacts so the v2 tree is self-contained.
    for extra in ("splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        s = src / extra
        if s.exists():
            shutil.copy2(str(s), str(out / extra))

    n_fill = n_skip_nomodel = n_passthrough = n_fail = 0
    new_records: List[dict] = []

    for rec in records:
        cfg = rec.get("config")
        ct_rel, lbl_rel = rec.get("ct_file"), rec.get("label_file")

        def _passthrough(r):
            for rel in (r.get("ct_file"), r.get("label_file")):
                if rel and (src / rel).exists():
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    if not (out / rel).exists():
                        shutil.copy2(str(src / rel), str(out / rel))

        if not rec.get("ok", True) or cfg not in SCOPE_CONFIGS \
                or not ct_rel or not lbl_rel:
            _passthrough(rec)
            new_records.append(rec)
            n_passthrough += 1
            continue

        region = MISSING_REGION[cfg]
        model = _find_model(models_cfg, region)
        tok = rec.get("token", "?")

        if model is None or args.dry_run:
            _passthrough(rec)
            new_records.append(rec)
            if model is None:
                n_skip_nomodel += 1
                log.info("token=%s cfg=%s: no enabled '%s' model — left "
                         "partial (not fabricated)", tok, cfg, region)
            else:
                n_passthrough += 1
                log.info("token=%s cfg=%s: DRY-RUN would pseudo-fill %s via "
                         "model '%s'", tok, cfg, region, model["name"])
            continue

        ct_src, lbl_src = src / ct_rel, src / lbl_rel
        if not ct_src.exists() or not lbl_src.exists():
            _passthrough(rec); new_records.append(rec); n_fail += 1
            log.warning("token=%s: missing ct/label on disk — passthrough", tok)
            continue

        try:
            with tempfile.TemporaryDirectory(prefix="psl_") as td:
                pred_path = run_nnunet_ensemble(
                    model, ct_src, Path(td), args.nnunet_results, args.device)
                if pred_path is None:
                    raise RuntimeError("inference produced no output")
                ref = nib.load(str(lbl_src))
                pred_arr = _align_to(ref, nib.load(str(pred_path)))
                pred_canon = remap_prediction(
                    pred_arr, model["label_remap"],
                    model["supplies_canonical"])
                manual = np.asarray(ref.dataobj).astype(np.int16)
                merged = merge_pseudo_into_manual(manual, pred_canon)

            _passthrough(rec)  # copies the CT (label overwritten below)
            nib.save(nib.Nifti1Image(merged, ref.affine, ref.header),
                     str(out / lbl_rel))
            new_records.append(updated_record(rec, region, merged))
            n_fill += 1
            log.info("token=%s cfg=%s: filled %s (model '%s', %d vox)",
                     tok, cfg, region, model["name"],
                     int((merged > 0).sum()))
        except Exception as exc:
            _passthrough(rec); new_records.append(rec); n_fail += 1
            log.warning("token=%s: pseudo-fill failed (%s) — passthrough",
                        tok, exc)

        if args.limit and n_fill >= args.limit:
            log.info("--limit %d reached; passing remaining through",
                     args.limit)
            done = {id(r) for r in new_records}
            for r in records:
                if id(r) not in done and r not in new_records:
                    _passthrough(r); new_records.append(r); n_passthrough += 1
            break

    # Reuse export_hf's canonical-schema writer so v2 manifest is identical
    # in shape/typing to v1 (no HF-viewer CastError).
    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import write_manifest
    write_manifest(new_records, out)

    log.info("=" * 60)
    log.info("pseudolabel v2 tree -> %s", out)
    log.info("  pseudo-filled : %d", n_fill)
    log.info("  passthrough   : %d", n_passthrough)
    log.info("  skipped (no model, left partial): %d", n_skip_nomodel)
    log.info("  failed (passthrough): %d", n_fail)
    log.info("=" * 60)
    if args.dry_run:
        log.info("DRY-RUN: no inference ran; v2 tree mirrors v1.")
    return 0


# TODO(deferred): fused-case completion. For `fused` records, locate the
# other high-fidelity diagnostic CT in the same TCIA study (filtering out
# scouts/localizers and reconciling differing convolution kernels) and
# fully pseudo-label that volume. Non-trivial series selection — tracked
# separately, intentionally NOT attempted here.

if __name__ == "__main__":
    raise SystemExit(main())
