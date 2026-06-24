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

# Kept in sync with scripts/label_scheme.py (THE source of truth). Re-declared (not
# imported) so this merge core stays a pure dependency-light, trivially-testable unit.
# VerSe-native: lumbar L1-L6 = 20-25, sacrum 26, hips 30/31, ignore 255.
IGNORE_LABEL = 255
CANONICAL_SPINE  = frozenset({20, 21, 22, 23, 24, 25})       # L1-L6 (VerSe)
CANONICAL_PELVIS = frozenset({26, 30, 31})                  # sacrum, left_hip, right_hip
CLASS_NAMES_PELVIS = {26: "sacrum", 30: "left_hip", 31: "right_hip"}
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


def complete_pelvis_with_model(manual_plus, pred_canon, ct=None, *,
                               bone_hu: int = 200, complete: bool = True):
    """The GT-first union: `manual_plus` already holds the spine GT + the propagated
    REAL pelvis GT. Optionally COMPLETE the pelvis with the model wherever the GT
    left background/ignore AND (if `ct` given) the voxel is actual bone (CT>bone_hu)
    AND the model calls it a pelvis class. A manual voxel is NEVER overwritten — the
    radiologist border / junction / LSTV semantics are preserved; the model only
    fills the parts of the bone the (sometimes partial) annotation missed.

    Returns (merged, metrics). metrics[c] per pelvis class: gt_vox, pred_vox, the
    voxel Dice of the propagated GT vs the model (now that they are co-registered),
    and added_vox = the model completion = how INCOMPLETE the GT was for that bone.
    """
    import numpy as np
    merged = np.array(manual_plus, dtype=np.int16, copy=True)
    fillable = (merged == 0) | (merged == IGNORE_LABEL)
    take = fillable & (pred_canon > 0)
    if ct is not None:
        take = take & (np.asarray(ct) > bone_hu)        # complete only real bone
    if complete:
        merged[take] = pred_canon[take].astype(np.int16)
    merged[merged == IGNORE_LABEL] = 0

    metrics = {}
    for c in sorted(CANONICAL_PELVIS):
        gt_c = manual_plus == c
        pr_c = pred_canon == c
        gv, pv = int(gt_c.sum()), int(pr_c.sum())
        inter = int((gt_c & pr_c).sum())
        dice = (2.0 * inter / (gv + pv)) if (gv + pv) else float("nan")
        added = int((take & (pred_canon == c)).sum())
        completeness = gv / (gv + added) if (gv + added) else float("nan")
        metrics[c] = {"gt_vox": gv, "pred_vox": pv, "dice": dice,
                      "added_vox": added, "completeness": completeness}
    return merged, metrics


def updated_record(record: dict, filled_region: str, merged,
                   prov: str = "pseudo") -> dict:
    """Return record with provenance + voxel stats updated for the v2 tree.

    Only the FILLED region's provenance flips to `prov`; the manually annotated
    region's prov is left exactly as-is (never downgraded). `prov` is "pseudo" for
    a model fill, "manual_propagated" when the region was filled by carrying the
    patient's OWN radiologist GT across acquisitions (propagate_pelvis) — REAL GT,
    not a model guess.
    """
    import numpy as np
    r = dict(record)
    if filled_region == "spine":
        r["prov_spine"] = prov
    elif filled_region == "pelvis":
        r["prov_pelvis"] = prov
    # Review prioritisation for the human QA pass. A model-filled SPINE is the
    # riskiest pseudolabel — the ENTIRE lumbar column is inferred from a pelvic-only
    # scan (no spine ever traced for this token) — so flag it "high". A model-filled
    # pelvis (spine_only) is "normal". REAL GT carried across acquisitions
    # (manual_propagated) needs no review. Manual/null cases stay unflagged.
    if prov == "pseudo":
        r["review_priority"] = "high" if filled_region == "spine" else "normal"
    r["partial_annotation"] = bool((merged == IGNORE_LABEL).any())
    r["n_voxels_ignore"] = int((merged == IGNORE_LABEL).sum())
    r["n_voxels_fg"] = int(((merged > 0) & (merged != IGNORE_LABEL)).sum())
    r["n_voxels_bg"] = int((merged == 0).sum())
    return r


def load_propagated_map(manifest_path, propagated_dir=None) -> Dict[str, dict]:
    """token -> {path, spine_uid} for ACCEPTED propagated pelves (real GT carried
    across acquisitions by propagate_pelvis). Only accepted cases are returned, so
    a rejected/failed registration transparently falls back to the model.

    Resolves the label path from the manifest's `placed` (pelvic sub-dict), falling
    back to <propagated_dir>/<spine_uid>_pelvic_propagated.nii.gz if that path is
    not present at read time (paths in the manifest are whatever the run wrote)."""
    from pathlib import Path as _P
    out: Dict[str, dict] = {}
    if not manifest_path:
        return out
    p = _P(manifest_path)
    if not p.exists():
        log.warning("propagated manifest %s not found — all pelves via model", p)
        return out
    data = json.loads(p.read_text())
    for c in data.get("cases", []):
        if int((c.get("propagation") or {}).get("accept", 0) or 0) != 1:
            continue
        tok = str(c.get("patient_token", ""))
        pv = c.get("pelvic", {}) or {}
        suid = pv.get("series_uid", "")
        cand = pv.get("placed", "")
        path = _P(cand) if cand else None
        if (path is None or not path.exists()) and propagated_dir and suid:
            alt = _P(propagated_dir) / f"{suid}_pelvic_propagated.nii.gz"
            path = alt
        if tok and path is not None:
            out[tok] = {"path": str(path), "spine_uid": suid}
    log.info("propagated pelves (accepted, real GT): %d tokens", len(out))
    return out


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
    deep and nnU-Net can't find dataset.json).

    Idempotent, and SELF-HEALING: the skip-download guard requires the
    FULL set nnUNetv2_predict needs — dataset.json, plans.json, and at
    least one fold_*/<checkpoint> — not merely a fold dir. A stale/partial
    local tree (e.g. fold_* present but plans.json missing, the failure
    that originally cost hours) therefore re-downloads instead of being
    skipped and dying at predict."""
    nn = ckpt_cfg["nnunet"]
    model_root = nnunet_results / ckpt_cfg["results_subdir"]
    model_dir = model_root / (
        f'{nn["trainer"]}__{nn["plans"]}__{nn["configuration"]}')
    ckpt_name = nn.get("checkpoint", "checkpoint_best.pth")

    def _missing(d: Path) -> List[str]:
        miss = []
        if not (d / "dataset.json").is_file():
            miss.append("dataset.json")
        if not (d / "plans.json").is_file():
            miss.append("plans.json")
        if not any((f / ckpt_name).is_file() for f in d.glob("fold_*")):
            miss.append(f"fold_*/{ckpt_name}")
        return miss

    miss = _missing(model_dir)
    if not miss:
        log.info("Checkpoints complete at %s — skip download", model_dir)
        return model_root
    log.info("Model dir incomplete (missing: %s) — (re)downloading",
             ", ".join(miss))

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
    miss = _missing(model_dir)
    if miss:
        log.error("Download finished but %s still missing %s — the HF repo "
                  "%r is incomplete at that path (upload dataset.json/"
                  "plans.json next to fold_*), or results_subdir/trainer/"
                  "plans/configuration in configs/pseudolabel_models.json "
                  "don't match the repo layout.", model_dir,
                  ", ".join(miss), ckpt_cfg["hf_repo_id"])
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

def link_or_copy(src_path, dst_path, *, copy: bool = False) -> str:
    """Place src at dst with ONE physical copy of the data on disk: hard-link
    (same fs: 0 extra bytes, a normal file to the HF uploader), else relative
    symlink, else full copy. `copy=True` forces a real copy. Used for the big CT
    volumes so the v2 tree references the single v1 CT store, not a 280 GB dup.
    (Re-declared here, not imported, to keep this script a standalone unit.)"""
    src_path, dst_path = str(src_path), str(dst_path)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src_path, dst_path)
        return "copy"
    try:
        os.link(src_path, dst_path)
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
    ap.add_argument("--predict_fused", action="store_true",
                    help="ALSO run inference on fused cases (both regions already "
                         "manual) and cache their predictions for "
                         "eval_vs_manual --include_fused. Fused output stays "
                         "passthrough (provenance unchanged); only the cached "
                         "prediction is produced. Uses the same held-out fold, "
                         "so it is leak-safe. Reuses the scoped cache "
                         "(--continue_prediction skips already-done cases).")
    ap.add_argument("--copy_ct", action="store_true",
                    help="Copy CT volumes into the v2 tree instead of "
                         "hard-linking them to the v1 store (only for "
                         "different-filesystem trees). Default hard-links: one "
                         "physical CT copy on disk, not a 280 GB duplicate.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only: copy v1→v2 verbatim, log the per-case "
                         "held-out fold, run no download/inference.")
    ap.add_argument("--include_configs", default=None,
                    help="comma-separated configs to KEEP in the v2 tree (e.g. "
                         "'fused,spine_only', which drops pelvic_native). This is "
                         "where v2 diverges from the all-configs v1 base — we "
                         "filter HERE, not by re-exporting a filtered base. Env "
                         "INCLUDE_CONFIGS honoured if the flag is absent; empty "
                         "= keep all configs.")
    ap.add_argument("--propagated_manifest", type=Path, default=None,
                    help="placed_manifest_propagated.json from propagate_pelvis. "
                         "For its ACCEPTED tokens, the pelvis on the spine-side "
                         "(spine_only) record is filled with the propagated REAL "
                         "GT instead of the model (prov_pelvis=manual_propagated); "
                         "rejected/absent tokens fall back to the model. Env "
                         "PROPAGATED_MANIFEST honoured if the flag is absent.")
    ap.add_argument("--propagated_dir", type=Path, default=None,
                    help="dir of <spine_uid>_pelvic_propagated.nii.gz, used to "
                         "resolve a manifest path that isn't present at read time.")
    ap.add_argument("--complete_propagated", dest="complete_propagated",
                    action="store_true", default=True,
                    help="(default) for propagated cases, also let the model COMPLETE "
                         "the bone the (sometimes partial) GT missed — bone-HU, never "
                         "overwriting GT. The GT-vs-model overlap + completion is "
                         "always measured to propagated_completion_qc.csv.")
    ap.add_argument("--no_complete_propagated", dest="complete_propagated",
                    action="store_false",
                    help="ship the propagated GT pelvis as-is (measure only).")
    args = ap.parse_args()
    _inc = args.include_configs or os.environ.get("INCLUDE_CONFIGS", "")
    include_configs = {c.strip() for c in _inc.split(",") if c.strip()} or None
    _prop_man = args.propagated_manifest or (
        Path(os.environ["PROPAGATED_MANIFEST"])
        if os.environ.get("PROPAGATED_MANIFEST") else None)

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

    # Real-GT pelves carried across acquisitions (propagate_pelvis): for these
    # ACCEPTED tokens the spine-side pelvis is REAL GT, so they skip the model.
    prop_map = load_propagated_map(_prop_man, args.propagated_dir)

    records = _load_manifest(man_path)
    total = len(records)
    if include_configs:
        def _keep(r) -> bool:
            if r.get("config") in include_configs:
                return True
            # ALSO keep PURE pelvic-only tokens (match_type == pelvic_only): their
            # only acquisition is the pelvic scan, so dropping all pelvic_native
            # would erase these tokens from v2 entirely. Keep them and pseudolabel
            # their spine. (Separate-cohort pelvic SIDES are config=pelvic_native but
            # match_type=separate — those stay dropped, since the same token's spine
            # side is already kept as a spine_only record.)
            return (r.get("config") == "pelvic_native"
                    and r.get("match_type") == "pelvic_only")
        kept = [r for r in records if _keep(r)]
        n_ponly = sum(1 for r in kept if r.get("config") == "pelvic_native")
        log.info("include_configs %s (+%d pure pelvic_only, model-spine): %d -> %d "
                 "records (dropped %d redundant pelvic sides)",
                 sorted(include_configs), n_ponly, total, len(kept), total - len(kept))
        records = kept
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

    def _copy_into_v2(rel):                               # small files (labels)
        if rel and (src / rel).exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (out / rel).exists():
                shutil.copy2(str(src / rel), str(out / rel))

    def _link_ct(rel):                                    # ONE physical CT copy
        if rel and (src / rel).exists() and not (out / rel).exists():
            link_or_copy(src / rel, out / rel, copy=args.copy_ct)

    def _passthrough(r):
        _link_ct(r.get("ct_file"))
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

    n_fill = n_oof = n_ens = n_pass = n_fail = n_resume = n_prop = 0
    new_records: List[dict] = []
    # tuple: (cid, rec, region, supplied, fold, predict_only, prop_path)
    # prop_path != None -> the pelvis is laid down from the propagated REAL GT and
    # the model only COMPLETES the bone the GT missed (GT-first union).
    scoped: List[tuple] = []
    fused_predict: List[tuple] = []
    prop_metrics: List[dict] = []    # per-case GT-vs-model overlap + completion

    for rec in records:
        cfg_v = rec.get("config")
        ct_rel, lbl_rel = rec.get("ct_file"), rec.get("label_file")
        if not rec.get("ok", True) or cfg_v not in SCOPE_CONFIGS \
                or not ct_rel or not lbl_rel:
            # fused (or any non-scoped complete-GT) case: optionally PREDICT it
            # (cache only, for eval) but still pass it through unchanged.
            if (args.predict_fused and cfg_v == "fused" and ct_rel and lbl_rel
                    and rec.get("ok", True)):
                ffold = heldout.get(str(rec.get("token", "")))
                fused_predict.append((_case_id(rec), rec, "(full)", (), ffold,
                                      True, None))
            _passthrough(rec); new_records.append(rec); n_pass += 1
            continue
        cid = _case_id(rec)
        # Resume ONLY if the case was merged previously AND its output label is still
        # on disk. A `done` marker alone is not enough: ship_v2 wipes the labels dir
        # but keeps _work (for cached preds), so trusting the marker silently skips
        # re-writing a label that was just deleted -> the label goes missing. Re-check
        # the output exists; if not, fall through and re-merge/re-write it.
        done_lbl = (done.get(cid) or {}).get("label_file", lbl_rel)
        if cid in done and done_lbl and (out / done_lbl).exists():
            _link_ct(ct_rel)                  # ensure CT present (idempotent)
            new_records.append(done[cid]); n_resume += 1
            continue
        region = MISSING_REGION[cfg_v]
        fold = heldout.get(str(rec.get("token", "")))
        # Real-GT pelvis for this separate-cohort spine scan? Lay it down as GT and
        # let the model only complete the rest (still predicted, for the union+QC).
        tok = str(rec.get("token", ""))
        prop_path = (prop_map[tok]["path"]
                     if cfg_v == "spine_only" and tok in prop_map else None)
        scoped.append((cid, rec, region, REGION_CANONICAL[region], fold, False,
                       prop_path))

    if args.limit and len(scoped) > args.limit:
        for (_c, r, *_x) in scoped[args.limit:]:
            _passthrough(r); new_records.append(r); n_pass += 1
        scoped = scoped[:args.limit]
        fused_predict = fused_predict[:args.limit]
        log.info("--limit %d: capping this run to %d scoped cases",
                 args.limit, len(scoped))
    if fused_predict:
        log.info("predict_fused: %d fused case(s) will be PREDICTED (cached for "
                 "eval) but kept passthrough.", len(fused_predict))
    n_prop_cases = sum(1 for it in scoped if it[6] is not None)
    if n_prop_cases:
        log.info("propagated REAL-GT pelves: %d (spine GT + propagated pelvis GT; "
                 "model %s the bone the GT missed)", n_prop_cases,
                 "completes" if args.complete_propagated else "measured-only on")

    # Group scoped + fused-predict cases by held-out fold → ONE predict per group
    # (model loaded once per group instead of once per case).
    n_fused_pred = 0
    groups: Dict[str, list] = defaultdict(list)
    for item in scoped + fused_predict:
        fold = item[4]
        groups["ensemble" if fold is None else f"fold_{fold}"].append(item)
    log.info("to predict=%d (scoped=%d + fused=%d) across %d group(s): %s",
             len(scoped) + len(fused_predict), len(scoped), len(fused_predict),
             len(groups), {k: len(v) for k, v in sorted(groups.items())})

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

        for (cid, rec, region, supplied, fold, predict_only, prop_path) in items:
            tok = rec.get("token", "?")
            pred_p = pred_dir / f"{cid}.nii.gz"
            if predict_only:
                # fused: the cached prediction is the deliverable (for eval).
                # Output already passthrough'd; never merge into the manual GT.
                if launched and pred_p.exists():
                    n_fused_pred += 1
                else:
                    log.warning("token=%s cid=%s: fused prediction missing", tok, cid)
                continue
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

                if prop_path is not None and Path(prop_path).exists():
                    # GT-first union: lay the propagated REAL pelvis GT onto the
                    # manual, then let the model COMPLETE the bone it missed (never
                    # overwriting GT). Records the GT-vs-model overlap + completion.
                    prop_arr = _align_to(ref, nib.load(str(prop_path)))
                    prop_canon = np.where(np.isin(prop_arr, list(CANONICAL_PELVIS)),
                                          prop_arr, 0).astype(np.int16)
                    manual_plus = merge_pseudo_into_manual(manual, prop_canon)
                    ct_arr = np.asarray(nib.load(str(src / rec["ct_file"])).dataobj)
                    merged, mtr = complete_pelvis_with_model(
                        manual_plus, pred_canon, ct_arr,
                        complete=args.complete_propagated)
                    prov = ("manual_propagated_completed"
                            if args.complete_propagated else "manual_propagated")
                    row = {"token": tok, "cid": cid}
                    for c in sorted(CANONICAL_PELVIS):
                        m = mtr[c]
                        row[f"{CLASS_NAMES_PELVIS[c]}_dice"] = round(m["dice"], 4) \
                            if m["dice"] == m["dice"] else ""
                        row[f"{CLASS_NAMES_PELVIS[c]}_added_vox"] = m["added_vox"]
                        row[f"{CLASS_NAMES_PELVIS[c]}_completeness"] = \
                            round(m["completeness"], 4) \
                            if m["completeness"] == m["completeness"] else ""
                    prop_metrics.append(row)
                    n_prop += 1
                    log.info("token=%s cid=%s: pelvis <- REAL GT + model-complete "
                             "(added %d vox, sacrum dice=%.3f) [%s]", tok, cid,
                             sum(mtr[c]["added_vox"] for c in CANONICAL_PELVIS),
                             mtr[7]["dice"] if mtr[7]["dice"] == mtr[7]["dice"]
                             else float("nan"), key)
                else:
                    merged = merge_pseudo_into_manual(manual, pred_canon)
                    prov = "pseudo"

                # The model gets the hip BONE right but frequently stamps the wrong
                # laterality (dumps both hips into one class). Where the model filled
                # the pelvis, re-derive each hip voxel's side from the midline
                # (lumbar/sacrum centroid) — deterministic, no registration.
                if region == "pelvis":
                    from relabel_hips_by_midline import lateralize_hips
                    merged, n_lat = lateralize_hips(merged, ref.affine)
                    if n_lat:
                        log.info("token=%s cid=%s: lateralized %d hip voxel(s) by midline",
                                 tok, cid, n_lat)

                _link_ct(rec["ct_file"])                  # CT into v2 (one copy)
                nib.save(nib.Nifti1Image(merged, ref.affine, ref.header),
                         str(out / rec["label_file"]))    # merged label
                urec = updated_record(rec, region, merged, prov=prov)
                (done_dir / f"{cid}.json").write_text(json.dumps(urec))
                new_records.append(urec)
                n_fill += int(prop_path is None)
                n_oof += int(fold is not None and prop_path is None)
                n_ens += int(fold is None and prop_path is None)
                if prop_path is None:
                    log.info("token=%s cid=%s: filled %s (%d fg vox)  "
                             "[group %s done %d/%d]", tok, cid, region,
                             int(((merged > 0) & (merged != IGNORE_LABEL)).sum()),
                             key, n_fill, len(scoped))
            except Exception as exc:
                _passthrough(rec); new_records.append(rec); n_fail += 1
                log.warning("token=%s cid=%s: merge failed (%s) — "
                            "passthrough", tok, cid, exc)

    # propagated GT-vs-model overlap + completion (incompleteness), per case.
    if prop_metrics:
        import csv as _csv
        cols = ["token", "cid"]
        for c in sorted(CANONICAL_PELVIS):
            nm = CLASS_NAMES_PELVIS[c]
            cols += [f"{nm}_dice", f"{nm}_added_vox", f"{nm}_completeness"]
        qc_csv = out / "propagated_completion_qc.csv"
        with open(qc_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
            w.writeheader(); w.writerows(prop_metrics)
        def _avg(key):
            vals = [r[key] for r in prop_metrics
                    if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else float("nan")
        log.info("propagated completion QC -> %s", qc_csv)
        for c in sorted(CANONICAL_PELVIS):
            nm = CLASS_NAMES_PELVIS[c]
            log.info("  %-10s mean GT-vs-model Dice=%.3f  mean completeness=%.3f  "
                     "total added(completed) vox=%d", nm, _avg(f"{nm}_dice"),
                     _avg(f"{nm}_completeness"),
                     sum(r.get(f"{nm}_added_vox", 0) or 0 for r in prop_metrics))

    log.info("writing v2 manifest (%d records) ...", len(new_records))
    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import write_manifest
    # write_manifest keeps only records with a truthy `ok`, but our records
    # were read back from the v1 manifest where _coerce_manifest_record has
    # already STRIPPED `ok` — so without re-adding it every record is
    # filtered out and manifest.json lands empty. All v2 records are valid
    # (failed/rejected cases aren't produced here), so mark them ok.
    write_manifest([{**r, "ok": True} for r in new_records], out)

    log.info("=" * 60)
    log.info("pseudolabel v2 tree -> %s", out)
    log.info("  pseudo-filled (new)  : %d  (out-of-fold=%d  ensemble=%d)",
             n_fill, n_oof, n_ens)
    log.info("  REAL-GT propagated   : %d  (spine GT + propagated pelvis GT; model "
             "%s the bone the GT missed)", n_prop,
             "completed" if args.complete_propagated else "measured-only on")
    log.info("  resumed (prior run)  : %d", n_resume)
    log.info("  passthrough          : %d", n_pass)
    log.info("  failed (passthrough) : %d", n_fail)
    if args.predict_fused:
        log.info("  fused predicted (cached for eval, not merged) : %d", n_fused_pred)
    log.info("=" * 60)
    return 0


# TODO(deferred): fused-case completion. For `fused` records, locate the
# other high-fidelity diagnostic CT in the same TCIA study (filtering out
# scouts/localizers, reconciling differing convolution kernels) and fully
# pseudo-label that volume. Non-trivial series selection — tracked
# separately, intentionally NOT attempted here.

if __name__ == "__main__":
    raise SystemExit(main())
