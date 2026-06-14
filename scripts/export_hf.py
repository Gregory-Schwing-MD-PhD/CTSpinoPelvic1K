"""
export_hf.py — Export CTSpinoPelvic1K to HuggingFace-compatible flat directory,
then push directly to the Hub using upload_large_folder.

Reads placed_manifest.json (written by place_fused_masks.py) and for each case:
  1. Load CT NIfTI (from tcia_nifti/{series_uid}.nii.gz)
  2. Remap spine labels (VerSe -> 10-class) + pelvic labels (4-class -> 10-class)
  3. Merge into single 10-class label map (or partial-annotation 11-value
     map with IGNORE_LABEL=10 for separate-mode cases — see PARTIAL ANNOTATION
     CONTRACT below)
  4. Reorient CT + label to PIR canonical orientation
  5. Strip PHI from NIfTI headers
  6. Save to flat output:
       ct/{token:04d}_ct.nii.gz                       (fused)
       ct/{token:04d}_spine_ct.nii.gz                 (spine-side crop, separate mode
                                                       OR spine_only single-mask case)
       ct/{token:04d}_pelvic_ct.nii.gz                (pelvic-side crop, separate mode
                                                       OR pelvic_only single-mask case)
       labels/{token:04d}_label.nii.gz                etc.
  7. Write manifest.json, manifest.csv, data_splits.json, splits/test.json,
     splits_summary.json
  8. Push to HuggingFace

Filename schema (changed Apr 2026)
----------------------------------
The earlier convention baked `position` into every filename:
    <token:04d>_<position>_[spine|pelvic]_ct.nii.gz
That was misleading because `position` was almost always `unknown` (the
prone/supine classifier in place_fused_masks.py rarely succeeds), and
because `config` (fused / spine_only / pelvic_native) is what every
downstream consumer actually filters on.

The new convention is fully self-describing and position-free:
    fused           ->  <token:04d>_ct.nii.gz
    spine annotated ->  <token:04d>_spine_ct.nii.gz
    pelvic annotated->  <token:04d>_pelvic_ct.nii.gz

Bare `<token>_ct.nii.gz` therefore unambiguously means "fused" (both
regions present in one mask). The `_spine` / `_pelvic` suffix is now
applied uniformly to spine-side / pelvic-side files, regardless of
whether the source case is `match_type="separate"` (paired but
non-coregistered) or `match_type="spine_only" / "pelvic_only"` (only
one side has annotation upstream). Earlier the spine_only / pelvic_only
single-mask branches reused the bare `<token>` base, which collided
visually with fused — now resolved.

`position` still rides through `_export_one` and is persisted in the
manifest as a metadata column. It is no longer in the filename.

placed_manifest.json is the single source of truth. It now carries all LSTV
fields directly (lstv_pelvic, lstv_vertebral, lstv_agreement, lstv_confusion_zone)
populated by place_fused_masks.py. No secondary manifest file is needed.

PARTIAL ANNOTATION CONTRACT (May 2026)
======================================
The earlier merge_labels initialized `result` to all-zeros (background)
and then filled in non-zero classes from whichever masks were present.
For separate-mode patients (spine_only or pelvic_native exports), this
silently asserted "background everywhere the present mask doesn't speak"
— which was wrong. A spine-only export's pelvic region was getting a
hard "no sacrum, no hips" label even though the pelvic annotator hadn't
traced that scan. The model dutifully learned to suppress those classes
on those cases. Across 689 separate-mode records (out of 979 training
cases) this was systematically poisoning the loss for L6, sacrum, and
hips.

The fix: separate-mode label files now use `IGNORE_LABEL = 10` for
voxels that fall outside the present annotator's domain. nnU-Net v2's
`DC_and_CE_loss` honors `ignore_label` and produces zero gradient at
those voxels — so the network gets exactly the supervision the
annotator intended, no more.

Required downstream:
  - `dataset.json` MUST have `"ignore": 10` in its `labels` dict (already
    set in convert_hf_to_nnunet.py:LABEL_NAMES).
  - The trainer's `_IGNORE_LABEL` constant must equal 10 (already set in
    nnunet_wandb_variant.py).

Per-mode label-array contents:
  fused (both masks present):
    voxel values ∈ {0..9}.       0 = true background.
  spine-only (pelvic_path is None):
    voxel values ∈ {1..6, 7, 10}. No 0s. 7 only where the spine VerSe
    mask had id 26 (sacrum from spine annotator). 10 = ignore everywhere
    else (the spine annotator did not trace those regions).
  pelvic-only (spine_path is None):
    voxel values ∈ {7..9, 10}. No 0s. 10 everywhere outside the pelvic
    annotator's traced sacrum/hips.

CT/mask frame-mismatch fix (Apr 2026)
-------------------------------------
Earlier the CT->mask resample gate in `_export_one` was:
    if ct_img.shape[:3] != ref_shape:
        ... resample CT into mask grid ...
This missed cases where CT and placed mask shared the same shape but had
different affines (e.g., raw RPS-stored dcm2niix CT vs PIR-reoriented
placed mask). For those cases the resample was skipped and the CT data
array was wrapped with the mask's affine without being reoriented to
match — producing a saved CT whose data axes were scrambled relative to
its own affine. This affected ~9 of ~800 tokens (those where the placed
mask happened to be exactly 512^3 like the raw CT). Symptoms: TS
predictions landed nowhere near the GT mask in the audit; visualize_qc
showed CTs in non-axial slabs; HU under hip-mask voxels was air.

Fix: gate now requires BOTH shape AND affine equality before skipping
the resample. A post-write HU-at-hip-mask sanity check is also added
so any future regression is caught at export time, not 200 audit-emails
later.

Wipe-remote (Apr 2026)
----------------------
Earlier pushes left orphan files on the HF repo whenever a filename
schema changed (e.g., the position-prefix removal). `upload_large_folder`
is purely additive — it never deletes remote files that no longer exist
locally. The `--wipe_remote` flag clears all files in the HF repo (one
atomic delete commit) before pushing, giving a clean mirror of the local
export. The repo itself — its URL, git history, stars and discussions —
is PRESERVED (the repo is never deleted; the URL stays continuously live
for anonymous review). Requires `--force_wipe_remote` to skip the safety
prompt. Only valid alongside `--push_to_hub`.

Label scheme:
  0=bg  1=L1  2=L2  3=L3  4=L4  5=L5  6=L6(LSTV)  7=sacrum  8=left_hip  9=right_hip
  10=IGNORE (partial-annotation only; never present in fused exports)

Output orientation: every CT + label pair is written in PIR canonical:
  axis 0 = Posterior  (A->P as idx++)
  axis 1 = Inferior   (S->I as idx++)
  axis 2 = patient-Right (L->R as idx++)

QC figures are sliced and displayed assuming PIR, so rows correspond to:
  Row 1 "Coronal"   fix axis 0, show (I, R)  -- head at top, feet at bottom
  Row 2 "Axial"     fix axis 1, show (P, R)  -- anterior at top, spine at bottom
  Row 3 "Sagittal"  fix axis 2, show (P, I)  -- transposed for head-up

Manifest path convention: ct_file / label_file / qc_file are stored as
relative paths that INCLUDE the subdirectory prefix (e.g. 'ct/0017_ct.nii.gz'),
so that `dataset_root / ct_file` resolves to the right file under both the
local nested layout and the nested layout produced by HF upload_large_folder.
Forward slashes are used unconditionally so the manifest is portable
across OSes.

Usage (matches slurm/export_dataset.sh):
    python export_hf.py \\
        --manifest   data/placed/placed_manifest.json \\
        --nifti_dir  data/tcia_nifti \\
        --spine_dir  data/placed/spine \\
        --pelvic_dir data/placed/pelvic \\
        --out_dir    data/hf_export \\
        --workers    32 \\
        [--skip_qc] [--no_pir] [--skip_export] \\
        [--push_to_hub --hf_repo_id user/repo --hf_workers 8 [--hf_private]] \\
        [--wipe_remote --force_wipe_remote]

    # Token via env var (keeps credentials out of shell history)
    HF_TOKEN=hf_xxx python export_hf.py ... --push_to_hub
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("spinesurg.export_hf")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# -- HuggingFace config -------------------------------------------------------

# No baked-in canonical repo: the same export is pushed to several
# different HF repos (one per submission venue), so the target MUST be
# supplied explicitly at invocation via --hf_repo_id. A None default forces
# main() to fail loudly rather than silently push to a wrong/stale repo.
HF_REPO_ID   = None
HF_REPO_TYPE = "dataset"

# Any spatial axis smaller than this is treated as a scout / localizer /
# degenerate volume and rejected at ingest. Mirrors place_fused_masks.py.
MIN_VALID_SHAPE = 10

# Canonical manifest schema. This is the SINGLE source of truth for what a
# manifest record looks like in manifest.json, manifest.csv, and the
# per-split manifest_*.json files. Every emitted record carries EXACTLY
# these keys, in this order, each holding either a value of its declared
# type or JSON null (only where `nullable` is True).
#
# Inapplicable / missing values serialize as JSON null — never "" and never
# an omitted key. A clean per-column type (with null for absent) is what the
# HuggingFace dataset viewer / Parquet needs to avoid CastError.
#
# (name, py_type, nullable)
_MANIFEST_SCHEMA = [
    ("token",                  str,   False),
    ("position",               str,   True),
    # DICOM header demographics (placed_manifest schema >=2.7). Mostly
    # passed through verbatim from the internal manifest; the AGE columns
    # are the one documented exception — a public-release policy is applied
    # here via _age_public() (this is intentional, not a re-derivation of
    # other fields). All nullable: an absent header tag is an explicit
    # JSON null (never 0 / 0.0 / "").
    #   age      — patient-level integer years. Already decade-generalized
    #              and capped at 89 by TCIA de-identification (the source
    #              PatientAge tag itself is decade-coarse, e.g. "060Y").
    #              PUBLIC POLICY: age < 18 is filtered to null (implausible
    #              for this adult screening cohort). Nullable INT: null/
    #              missing/filtered -> null, never 0.
    #   age_band — decade band string DERIVED from `age` ("50-59", "60-69",
    #              ... "80-89"; 89 and any >=90 fold into "80-89"). age and
    #              age_band are kept consistent: age < 18 -> BOTH null.
    #              Honest public statement of the true decade granularity.
    #   sex      — patient-level enum: male | female | other | null.
    # patient_weight (kg) / patient_size (m) — patient-level floats.
    # convolution_kernel / manufacturer / manufacturer_model — per-
    #   acquisition strings. slice_thickness (mm) / kvp — per-acquisition
    #   floats. The float columns are the first nullable FLOATs: null ->
    #   null, never 0.0 (would corrupt the column dtype in the HF viewer).
    ("age",                    int,   True),
    ("age_band",               str,   True),
    ("sex",                    str,   True),
    ("patient_weight",         float, True),
    ("patient_size",           float, True),
    ("convolution_kernel",     str,   True),
    ("manufacturer",           str,   True),
    ("manufacturer_model",     str,   True),
    ("slice_thickness",        float, True),
    ("kvp",                    float, True),
    ("config",                 str,   False),
    ("match_type",             str,   False),
    # Per-source label provenance. prov_spine = the vertebral L1-L6 labels;
    # prov_pelvis = the sacrum + hip labels as a unit (shared source, pseudo-
    # labeled together). Enum: manual | pseudo | pseudo_corrected | null
    # (null = that structure is absent / not applicable for this record).
    # Today only manual/null are produced; pseudo / pseudo_corrected become
    # reachable when the (gated) pseudo-label + QA pipeline writes them.
    ("prov_spine",             str,   True),
    ("prov_pelvis",            str,   True),
    ("review_priority",        str,   True),   # "high" (model-filled spine on a
                                               # pelvic-only token) | "normal" (model-
                                               # filled pelvis) | null. Set by pseudolabel.
    ("lstv_label",             str,   False),
    ("lstv_class",             int,   False),
    ("lstv_pelvic",            str,   True),
    ("lstv_vertebral",         str,   True),
    ("castellvi_type",         str,   True),   # Castellvi grade, reader 1 (I..IV/IIa..)
    ("castellvi_second_read",  str,   True),   # Castellvi grade, reader 2 (null if not
                                               # double-read — only a subset were)
    ("castellvi_agreement",    bool,  True),   # read1==read2; null if no second read
    ("lstv_agreement",         bool,  True),   # true / false / null ONLY
    ("lstv_confusion_zone",    bool,  False),
    ("has_l6",                 bool,  False),
    ("n_lumbar_labels",        int,   False),
    ("has_anchor",             bool,  False),   # last rib-bearing vert (T12/T13) in FOV
    ("alignment_ok",           bool,  False),
    ("ct_resampled_to_mask",   bool,  False),
    ("postwrite_hip_bone_pct", float, True),
    ("partial_annotation",     bool,  False),
    ("n_voxels_ignore",        int,   False),
    ("n_voxels_fg",            int,   False),
    ("n_voxels_bg",            int,   False),
    ("spine_series_uid",       str,   True),
    ("pelvic_series_uid",      str,   True),
    ("spine_bone_pct",         float, True),
    ("pelvic_bone_pct",        float, True),
    ("ct_file",                str,   False),
    ("label_file",             str,   False),
    ("qc_file",                str,   False),
]
_MANIFEST_FIELDS = [name for name, _, _ in _MANIFEST_SCHEMA]


def _coerce_manifest_record(rec: dict) -> dict:
    """Project an arbitrary record onto the canonical manifest schema.

    Guarantees the output dict has EXACTLY `_MANIFEST_FIELDS` as keys, in
    schema order, each holding either a value of its declared type or JSON
    null (only where the field is declared nullable). Missing keys, None,
    and "" all collapse to null for nullable fields; for the (rare,
    defensive) case of a missing non-nullable field, a typed zero value is
    used so the column type never drifts.

    This enforces presence and type ONLY — it does not recompute any value.
    """
    out: dict = {}
    for name, py_type, nullable in _MANIFEST_SCHEMA:
        v = rec.get(name, None)
        if v == "":
            v = None
        if v is None:
            if nullable:
                out[name] = None
            elif py_type is str:
                out[name] = ""
            elif py_type is bool:
                out[name] = False
            elif py_type is int:
                out[name] = 0
            else:  # float
                out[name] = 0.0
            continue
        if py_type is bool:
            out[name] = bool(v)
        elif py_type is int:
            out[name] = int(v)
        elif py_type is float:
            out[name] = float(v)
        else:
            out[name] = str(v)
    return out

# Sanity: how much bone HU we expect under the hip-label voxels of the
# saved CT/label pair. Lower than this fires a warning at export time.
_POSTWRITE_HIP_BONE_PCT_WARN = 30.0
_POSTWRITE_MIN_HIP_VOXELS    = 1000

# -- Label maps ---------------------------------------------------------------

# Ignore label for partial-annotation cases. Voxels with this value are
# masked out of both CE and Dice loss by nnU-Net's DC_and_CE_loss when
# ignore_label is configured (see trainer's _maybe_apply_ce_reweighting
# and the dataset.json's "ignore": 10 entry).
IGNORE_LABEL = 10

VERSE_TO_10CLASS: Dict[int, int] = {
    # cervical C1-C7 (VerSe 1-7) -> 13-19. Previously DROPPED; now retained from
    # CTSpine1K GT (present on a handful of cases — see DATASET_PRINCIPLES.md).
    1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19,
    # thoracic T1-T12 (VerSe 8-19) -> 20-31. Previously DROPPED; now retained,
    # CONTIGUOUS through T12 (so the column is in order, no orphaned vertebra).
    8: 20, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25,
    14: 26, 15: 27, 16: 28, 17: 29, 18: 30,
    19: 31,                       # T12 -> per-level (NOT the anchor; see below)
    20: 1, 21: 2, 22: 3, 23: 4, 24: 5,
    25: 6,
    26: 7,
    28: 32,                       # T13 (supernumerary) -> per-level T13
}
# Append-only law (DATASET_PRINCIPLES.md): the existing 0-12 scheme is FROZEN;
# everything above is purely additive and never renumbers a published value.
#
# - Vertebrae are NATIVE and CONTIGUOUS: cervical C1-C7 -> 13-19, thoracic
#   T1-T13 -> 20-32 (T12 -> 31, T13 -> 32), lumbar L1-L6 -> 1-6, sacrum -> 7.
#   They are the spine annotator's word, written authoritatively by merge_labels.
# - The counting anchor is DERIVED, not a stored class: it is the last
#   rib-bearing vertebra = the last thoracic (T12=31 / T13=32 when present),
#   already in the native labels. The model learns rib-bearing-ness (lumbar bear
#   no ribs) from the T-vs-L distinction and from native ribs once labelled.
#   There is NO stored anchor class (the old 11/12 are retired).
# - Classes 33-62 (femur, full rib cage, sternum, costal cartilages) are
#   RESERVED names only — not in this map (no GT source yet), future annotation.
# - ignore stays 10; this change is decoupled from the pseudolabeller, which
#   only fills bg/ignore voxels and never touches these foreground labels.
PELVIC_TO_10CLASS: Dict[int, int] = {1: 7, 2: 8, 3: 9}

CLASS_NAMES = {
    # ── Lumbosacral + pelvis core (frozen; never renumber) ──────────────────
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6", 7: "sacrum", 8: "left_hip", 9: "right_hip",
    # 10 = ignore (IGNORE_LABEL), a frozen sentinel — never an anatomical class.
    # 11/12 retired: there is NO stored "last_rib_vertebra"/"rib" anchor class.
    # The counting anchor = the last rib-bearing vertebra = the last thoracic
    # (T12=31 / T13=32), which is DERIVED from the native labels; the model
    # learns rib-bearing-ness (lumbar bear no ribs) from the T-vs-L distinction
    # (and from native ribs once labelled). No special class needed.
    # ── Full native vertebral column, retained from CTSpine1K GT ────────────
    13: "C1", 14: "C2", 15: "C3", 16: "C4", 17: "C5", 18: "C6", 19: "C7",
    20: "T1", 21: "T2", 22: "T3", 23: "T4", 24: "T5", 25: "T6", 26: "T7",
    27: "T8", 28: "T9", 29: "T10", 30: "T11", 31: "T12", 32: "T13",
    # ── Reserved for future annotation (no GT source yet) ────────────────────
    33: "femur_left", 34: "femur_right",
    35: "rib_left_1", 36: "rib_left_2", 37: "rib_left_3", 38: "rib_left_4",
    39: "rib_left_5", 40: "rib_left_6", 41: "rib_left_7", 42: "rib_left_8",
    43: "rib_left_9", 44: "rib_left_10", 45: "rib_left_11", 46: "rib_left_12",
    47: "rib_left_13",
    48: "rib_right_1", 49: "rib_right_2", 50: "rib_right_3", 51: "rib_right_4",
    52: "rib_right_5", 53: "rib_right_6", 54: "rib_right_7", 55: "rib_right_8",
    56: "rib_right_9", 57: "rib_right_10", 58: "rib_right_11", 59: "rib_right_12",
    60: "rib_right_13",
    61: "sternum", 62: "costal_cartilages",
}


def _fname_base(tok) -> str:
    """Deterministic, collision-free filename stem for a patient token.

    Numeric tokens -> zero-padded NNNN (unchanged, so existing numeric
    filenames are stable). Non-numeric tokens (e.g. TCIA 'CTC-1234567890')
    -> the sanitized token verbatim, which (a) is deterministic — unlike
    the old abs(hash(tok)) % 10000, which was salted by PYTHONHASHSEED and
    changed every run — and (b) cannot collide with the numeric NNNN space
    or with another distinct token. The old scheme silently mapped
    CTC-2936751320 -> '0020', overwriting numeric token 20's files.
    """
    import re
    s = str(tok)
    if s.isdigit():
        return f"{int(s):04d}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", s).strip("_")
    return safe or "unknown"

_SEG_COLORS = {
    1: (0.15, 0.40, 0.80, 0.55), 2: (0.25, 0.55, 0.85, 0.55),
    3: (0.35, 0.65, 0.90, 0.55), 4: (0.45, 0.75, 0.92, 0.55),
    5: (0.10, 0.80, 0.85, 0.55), 6: (0.75, 0.85, 0.20, 0.65),
    7: (0.85, 0.15, 0.15, 0.55), 8: (0.95, 0.50, 0.10, 0.55),
    9: (0.95, 0.80, 0.05, 0.55),
    31: (1.00, 0.00, 1.00, 0.65),     # T12 (the last rib-bearing vertebra) — magenta
    32: (1.00, 0.45, 0.75, 0.65),     # T13 — pink
}

# -- Small helpers ------------------------------------------------------------

def _first_not_none(*vals):
    """Return the first non-None value, or None if all are None.

    Used to resolve provenance fields (series_uid, bone_pct) across the
    handful of key-name variants that have shown up in placed_manifest.json
    over the life of the pipeline.  Treats 0.0 correctly (unlike `or`).
    """
    for v in vals:
        if v is not None:
            return v
    return None


def _age_public(age):
    """Public-release age policy for the HF export (NOT applied to the
    internal placed_manifest, which keeps the precise int).

    Returns (age_int, age_band) where:
      * age below 18 -> (None, None). The TCIA screening cohort is adult;
        a sub-18 value is implausible de-identification noise, so it is
        filtered to JSON null in BOTH public columns (they must never
        disagree).
      * otherwise -> (a, band). `a` is the value verbatim — it is already
        decade-generalized + capped at 89 by TCIA de-identification /
        upstream _norm_age, so it is NOT re-floored here. `band` is the
        decade string "DD-DD" derived from it. Because the upstream cap is
        89, the maximum real input is 89 -> "80-89"; any >=90 (unreachable
        given the cap) is defensively folded into "80-89" too, since a
        capped 90+ cannot be distinguished from a real 89.

    None / unparseable -> (None, None).
    """
    if age is None:
        return (None, None)
    try:
        a = int(age)
    except (TypeError, ValueError):
        return (None, None)
    if a < 18:
        return (None, None)
    if a >= 90:
        return (a, "80-89")
    d = (a // 10) * 10
    return (a, f"{d}-{d + 9}")


def _to_optional_float(v):
    """Coerce to float, or None if value is empty/missing/unparseable."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _posix_rel(*parts: str) -> str:
    """Join path parts with '/' always, regardless of OS."""
    return "/".join(p.strip("/\\") for p in parts if p)


# -- NIfTI helpers ------------------------------------------------------------

def _load_nii(path):
    import nibabel as nib
    return nib.load(str(path))


def _validate_affine(affine, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    if affine is None or getattr(affine, "shape", None) != (4, 4):
        raise ValueError(f"{tag}affine is not 4x4: shape={getattr(affine,'shape',None)}")
    if not np.all(np.isfinite(affine)):
        bad = np.argwhere(~np.isfinite(affine)).tolist()
        raise ValueError(f"{tag}affine contains NaN/Inf at positions {bad}")
    col_norms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    if np.any(col_norms < 1e-6):
        raise ValueError(
            f"{tag}affine has near-zero column norm (degenerate series): "
            f"norms={col_norms.tolist()}"
        )


def _validate_shape(shape, label: str = "",
                    min_axis: int = MIN_VALID_SHAPE) -> None:
    tag = f"[{label}] " if label else ""
    try:
        sh = tuple(int(s) for s in shape[:3])
    except (TypeError, ValueError):
        raise ValueError(f"{tag}shape is not 3D: {shape}")
    if len(sh) != 3:
        raise ValueError(f"{tag}shape is not 3D: {shape}")
    if min(sh) < min_axis:
        raise ValueError(
            f"{tag}volume too thin for segmentation (scout / localizer): "
            f"shape={sh} min_axis_required={min_axis}"
        )


def reorient_to_pir(img):
    from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
    _validate_affine(img.affine, label="reorient_to_pir")
    target  = axcodes2ornt(('P', 'I', 'R'))
    current = io_orientation(img.affine)
    xfm     = ornt_transform(current, target)
    return img.as_reoriented(xfm)


def strip_phi(img):
    import nibabel as nib
    hdr = img.header.copy()
    for field in ('descrip', 'aux_file', 'db_name', 'intent_name'):
        try:
            hdr[field] = b''
        except (KeyError, ValueError):
            pass
    return nib.Nifti1Image(np.asarray(img.dataobj), img.affine, hdr)


def merge_labels(spine_path, pelvic_path, ref_shape):
    """Build the label volume from spine + pelvic placed masks.

    Output value range:
      Fused mode  (both masks present): voxels ∈ {0..9}
      Partial mode (only one mask):     voxels ∈ {1..9, IGNORE_LABEL=10}

    Partial-annotation contract:
      In partial mode, the result array starts as IGNORE_LABEL everywhere.
      The available mask then writes its annotated classes (1-6 for spine,
      7-9 for pelvic, plus 7 for spine where VerSe id 26 was traced as
      sacrum) into the result. Voxels that the available annotator did
      not assign to any class — including voxels where the mask file's
      value is 0 (annotated as not-of-interest) — REMAIN as IGNORE_LABEL.

      The reasoning: a "0" voxel in a spine-only mask means the spine
      annotator decided it isn't a vertebra. It does NOT mean it's
      background — it could be sacrum, hip, or true bg. The pelvic
      annotator wasn't consulted on this scan, so we don't know. nnU-Net's
      DC_and_CE_loss with ignore_label=10 will zero out the gradient at
      those voxels, giving the model "no information" rather than "this
      is bg" (which would be a lie).

      Conversely, in fused mode, both annotators traced the scan, so a
      voxel that's 0 in BOTH masks really is true background and gets
      labeled 0.

    Sacrum-from-spine fallback:
      VerSe id 26 maps to our sacrum class (7). When the spine annotator
      traced sacrum and the pelvic annotator either didn't run (partial)
      or didn't claim sacrum at this voxel (fused), we fill in sacrum
      from the spine source. The fill mask is "result is 0 (truly bg from
      pelvic) OR result is IGNORE_LABEL (not yet assigned)" — covering
      both fused and partial modes.
    """
    has_spine  = bool(spine_path  and Path(spine_path).exists())
    has_pelvic = bool(pelvic_path and Path(pelvic_path).exists())

    if has_spine and has_pelvic:
        # Fused mode: every voxel is supervised. Init to 0 = background.
        result = np.zeros(ref_shape, dtype=np.int16)
    else:
        # Partial mode: voxels not assigned by the present mask are
        # IGNORE, not background. nnU-Net will mask them out of loss.
        result = np.full(ref_shape, IGNORE_LABEL, dtype=np.int16)

    # ── Pelvic mask (sacrum + hips) ─────────────────────────────────────
    if has_pelvic:
        pelv = np.asarray(_load_nii(pelvic_path).dataobj, dtype=np.int16)
        mn   = tuple(min(a, b) for a, b in zip(ref_shape, pelv.shape))
        sl   = tuple(slice(0, m) for m in mn)
        for pid, cls in PELVIC_TO_10CLASS.items():
            result[sl][pelv[sl] == pid] = cls

    # ── Spine mask (lumbar + sacrum-from-VerSe-26) ──────────────────────
    if has_spine:
        sp = np.asarray(_load_nii(spine_path).dataobj, dtype=np.int16)
        mn = tuple(min(a, b) for a, b in zip(ref_shape, sp.shape))
        sl = tuple(slice(0, m) for m in mn)
        for vid, cls in VERSE_TO_10CLASS.items():
            if cls == 7:
                # Sacrum from VerSe id 26: only fill where pelvic didn't
                # already claim it (fused mode -> result==0) or where
                # the slot is still unassigned (partial mode -> result==IGNORE).
                # Do NOT overwrite a pelvic-claimed sacrum, hip (7-9), or
                # an already-placed lumbar vertebra (1-6).
                fill_mask = (sp[sl] == vid) & (
                    (result[sl] == 0) | (result[sl] == IGNORE_LABEL)
                )
                result[sl][fill_mask] = cls
            else:
                # Lumbar L1-L6 and the full native vertebral column (cervical,
                # thoracic): the spine annotator's word is authoritative. These
                # regions are disjoint from the pelvic classes, so a direct
                # write is safe.
                result[sl][sp[sl] == vid] = cls

    return result


# -- QC figure ----------------------------------------------------------------

def _window(arr, lo=-150, hi=700):
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """Orient a 2D slice for radiological display assuming PIR source."""
    return arr2d.T if dim == 2 else arr2d


def _overlay(bg, labels):
    rgb = np.stack([bg, bg, bg], axis=-1)
    for cls_id, (r, g, b, a) in _SEG_COLORS.items():
        mask = labels == cls_id
        if mask.any():
            for c, v in enumerate([r, g, b]):
                rgb[..., c] = np.where(mask, rgb[..., c] * (1 - a) + v * a, rgb[..., c])
    return np.clip(rgb, 0, 1)


def _center_slice(ct, lbl):
    # Use only fg classes (1-9), not IGNORE (10) or bg (0), to find the
    # spatial center for QC visualization.
    fg_mask = (lbl >= 1) & (lbl <= 9)
    nz = np.where(fg_mask)
    if not len(nz[0]):
        return ct.shape[0]//2, ct.shape[1]//2, ct.shape[2]//2
    return int(np.median(nz[0])), int(np.median(nz[1])), int(np.median(nz[2]))


def make_qc_figure(ct, lbl, out_path, token, config, lstv, position):
    """Render a 3-row QC figure assuming PIR storage order."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    # For QC display only: treat IGNORE as bg (don't render a class color).
    # The actual saved label keeps IGNORE intact for the trainer.
    lbl_disp = np.where(lbl == IGNORE_LABEL, 0, lbl).astype(np.int16)

    present    = {int(v) for v in np.unique(lbl_disp) if v > 0}
    has_lumbar = bool(present & {1,2,3,4,5,6})
    has_pelvic = bool(present & {7,8,9})

    sp_lbl = np.where(np.isin(lbl_disp, [1,2,3,4,5,6]), lbl_disp, 0).astype(np.int16)
    pv_lbl = np.where(np.isin(lbl_disp, [7,8,9]),        lbl_disp, 0).astype(np.int16)

    if has_lumbar and has_pelvic:
        cols   = ["CT (raw)", "CT + spine", "CT + pelvic", "CT + all"]
        layout = "fused"
    else:
        name   = "spine" if has_lumbar else "pelvic"
        cols   = ["CT (raw)", f"CT + {name}", f"{name.capitalize()} only"]
        layout = "single"

    i, j, k = _center_slice(ct, lbl_disp)
    planes = [(0, i, "Coronal"), (1, j, "Axial"), (2, k, "Sagittal")]

    fig, axes = plt.subplots(3, len(cols),
                              figsize=(3.5 * len(cols), 10),
                              gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    fig.patch.set_facecolor("#111111")
    for ax in axes.flat:
        ax.set_facecolor("#111111"); ax.axis("off")
    for ci, t in enumerate(cols):
        axes[0, ci].set_title(t, fontsize=8, color="#cccccc", pad=4)

    for row, (dim, idx, pname) in enumerate(planes):
        sl = [slice(None)] * 3
        sl[dim] = idx
        sl = tuple(sl)

        bg       = _display_slice(_window(ct[sl]),    dim)
        sp_slice = _display_slice(sp_lbl[sl],         dim)
        pv_slice = _display_slice(pv_lbl[sl],         dim)
        full_slice = _display_slice(lbl_disp[sl],     dim)

        axes[row, 0].imshow(np.stack([bg, bg, bg], axis=-1),
                            aspect="auto", interpolation="nearest")
        axes[row, 0].text(-0.08, 0.5, pname, transform=axes[row, 0].transAxes,
                          fontsize=7, color="#aaaaaa", rotation=90, va="center")
        if layout == "fused":
            axes[row, 1].imshow(_overlay(bg, sp_slice),  aspect="auto", interpolation="nearest")
            axes[row, 2].imshow(_overlay(bg, pv_slice),  aspect="auto", interpolation="nearest")
            axes[row, 3].imshow(_overlay(bg, full_slice), aspect="auto", interpolation="nearest")
        else:
            axes[row, 1].imshow(_overlay(bg, full_slice), aspect="auto", interpolation="nearest")
            rgb = np.zeros((*full_slice.shape, 3), dtype=np.float32)
            for cid, (r, g, b, _) in _SEG_COLORS.items():
                rgb[full_slice == cid] = [r, g, b]
            axes[row, 2].imshow(rgb, aspect="auto", interpolation="nearest")

    patches = [mpatches.Patch(facecolor=_SEG_COLORS[c][:3], label=CLASS_NAMES[c])
               for c in sorted(present) if c in _SEG_COLORS]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(9, len(patches)),
                   fontsize=7, bbox_to_anchor=(0.5, 0.0),
                   facecolor="#222222", labelcolor="#dddddd", edgecolor="#444444")

    lstv_color = "#ff9944" if lstv.upper() not in ("NORMAL","UNKNOWN","") else "#44ff88"
    fig.suptitle(f"Token {token}  [{config}]  pos={position}  LSTV={lstv}",
                 fontsize=10, y=1.002, color=lstv_color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return True


# -- Per-case export worker ---------------------------------------------------

def _export_one(args: dict) -> dict:
    import nibabel as nib

    token       = args["token"]
    position    = args["position"]
    config      = args["config"]
    ct_path     = Path(args["ct_path"])
    spine_path  = args.get("spine_path")
    pelvic_path = args.get("pelvic_path")
    out_ct      = Path(args["out_ct"])
    out_lbl     = Path(args["out_lbl"])
    out_qc      = Path(args["out_qc"])
    lstv        = args.get("lstv", "unknown")
    match_type  = args.get("match_type", "unknown")

    rel_ct_file  = _posix_rel("ct",     out_ct.name)
    rel_lbl_file = _posix_rel("labels", out_lbl.name)
    rel_qc_file  = _posix_rel("qc",     out_qc.name)

    # Label provenance, derived from which source mask is present for this
    # export. A present source mask == manual (original source-dataset)
    # annotation; an absent source == null (structure not applicable).
    # pseudo / pseudo_corrected are NOT set here — they are written by the
    # downstream pseudo-label + human-QA pipeline, which must never
    # overwrite a "manual" provenance.
    prov_spine  = "manual" if (spine_path  and Path(spine_path).exists())  else None
    prov_pelvis = "manual" if (pelvic_path and Path(pelvic_path).exists()) else None

    result = dict(
        token=token, position=position, config=config,
        match_type=match_type, lstv_label=lstv, ok=False, error=None,
        # DICOM header demographics — pass-through (resolved upstream in the
        # work-args build). age/age_band already carry the public policy
        # (<18 -> null, decade band) applied there; everything else verbatim.
        # Absent -> None -> JSON null via _coerce_manifest_record.
        age=args.get("age"),
        age_band=args.get("age_band"),
        sex=args.get("sex"),
        patient_weight=args.get("patient_weight"),
        patient_size=args.get("patient_size"),
        convolution_kernel=args.get("convolution_kernel"),
        manufacturer=args.get("manufacturer"),
        manufacturer_model=args.get("manufacturer_model"),
        slice_thickness=args.get("slice_thickness"),
        kvp=args.get("kvp"),
        prov_spine=prov_spine, prov_pelvis=prov_pelvis,
        alignment_ok=False, has_l6=False, n_lumbar_labels=0, has_anchor=False,
        ct_file=rel_ct_file, label_file=rel_lbl_file, qc_file=rel_qc_file,
        lstv_pelvic=args.get("lstv_pelvic", ""),
        lstv_vertebral=args.get("lstv_vertebral", ""),
        castellvi_type=args.get("castellvi_type") or None,
        castellvi_second_read=args.get("castellvi_second_read") or None,
        castellvi_agreement=args.get("castellvi_agreement"),
        lstv_agreement=args.get("lstv_agreement"),
        lstv_confusion_zone=args.get("lstv_confusion_zone", False),
        lstv_class=args.get("lstv_class", 0),
        spine_series_uid=args.get("spine_series_uid"),
        pelvic_series_uid=args.get("pelvic_series_uid"),
        spine_bone_pct=args.get("spine_bone_pct"),
        pelvic_bone_pct=args.get("pelvic_bone_pct"),
        ct_resampled_to_mask=False,
        postwrite_hip_bone_pct=None,
        # Partial-annotation status (NEW): True iff this case used
        # IGNORE_LABEL fill because only one of {spine, pelvic} masks
        # was present.
        partial_annotation=False,
        # Voxel-count breakdown so we can sanity-check downstream.
        n_voxels_ignore=0,
        n_voxels_fg=0,
        n_voxels_bg=0,
    )

    try:
        ref_img = None
        for p in [spine_path, pelvic_path]:
            if p and Path(p).exists():
                ref_img = _load_nii(p); break
        if ref_img is None:
            raise FileNotFoundError(f"No placed mask for token={token}")

        _validate_affine(ref_img.affine, label=f"token={token} ref_mask")
        _validate_shape(ref_img.shape,   label=f"token={token} ref_mask")

        ref_shape  = ref_img.shape[:3]
        ref_affine = ref_img.affine.copy()

        ct_img = _load_nii(ct_path)
        _validate_affine(ct_img.affine, label=f"token={token} ct")
        _validate_shape(ct_img.shape,   label=f"token={token} ct")

        ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)

        shapes_equal  = (ct_img.shape[:3] == ref_shape)
        affines_equal = np.allclose(ct_img.affine, ref_affine, atol=1e-4)
        if not (shapes_equal and affines_equal):
            from scipy.ndimage import affine_transform as _at
            M       = np.linalg.inv(ct_img.affine) @ ref_affine
            ct_data = _at(ct_data, M[:3, :3], offset=M[:3, 3],
                          output_shape=ref_shape, order=1,
                          mode="constant", cval=-1024.0)
            result["ct_resampled_to_mask"] = True

        # Partial-annotation aware label merge — see merge_labels docstring.
        lbl_data = merge_labels(spine_path, pelvic_path, ref_shape)
        # Determine partial mode by checking what merge_labels emitted.
        result["partial_annotation"] = bool((lbl_data == IGNORE_LABEL).any())

        ct_out  = nib.Nifti1Image(ct_data,  ref_affine)
        lbl_out = nib.Nifti1Image(lbl_data, ref_affine)
        lbl_out.header.set_data_dtype(np.int16)
        lbl_out.header["scl_slope"] = 1.0
        lbl_out.header["scl_inter"] = 0.0

        if not args.get("skip_pir"):
            ct_out  = reorient_to_pir(ct_out)
            lbl_out = reorient_to_pir(lbl_out)

        ct_out  = strip_phi(ct_out)
        lbl_out = strip_phi(lbl_out)

        out_ct.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        nib.save(ct_out,  str(out_ct))
        nib.save(lbl_out, str(out_lbl))

        ct_r  = nib.load(str(out_ct))
        lbl_r = nib.load(str(out_lbl))
        result["alignment_ok"] = (
            ct_r.shape[:3] == lbl_r.shape[:3]
            and np.allclose(ct_r.affine, lbl_r.affine, atol=1e-4)
        )

        # Post-write voxel count breakdown for the manifest. Helps catch
        # regressions where partial mode silently turns off.
        lbl_arr = np.asarray(lbl_r.dataobj, dtype=np.int16)
        n_ignore = int((lbl_arr == IGNORE_LABEL).sum())
        n_fg     = int(((lbl_arr >= 1) & (lbl_arr <= 9)).sum())
        n_bg     = int((lbl_arr == 0).sum())
        result["n_voxels_ignore"] = n_ignore
        result["n_voxels_fg"]     = n_fg
        result["n_voxels_bg"]     = n_bg

        # Sanity invariants:
        # - fused mode must have NO ignore voxels
        # - partial mode must have NO bg voxels (everything outside fg is ignore)
        if not result["partial_annotation"]:
            if n_ignore != 0:
                log.warning(
                    "[token=%s config=%s] FUSED mode but ignore_count=%d. "
                    "merge_labels logic regression?", token, config, n_ignore)
        else:
            if n_bg != 0:
                log.warning(
                    "[token=%s config=%s] PARTIAL mode but bg_count=%d "
                    "(should be 0; partial mode initializes to IGNORE).",
                    token, config, n_bg)

        # ── Post-write CT-vs-label HU sanity at hip-mask voxels ─────────
        try:
            ct_arr_chk = np.asarray(ct_r.dataobj, dtype=np.float32)
            hip_mask   = (lbl_arr == 8) | (lbl_arr == 9)
            n_hip      = int(hip_mask.sum())
            if n_hip >= _POSTWRITE_MIN_HIP_VOXELS:
                hu_at_hip = ct_arr_chk[hip_mask]
                bone_pct  = float((hu_at_hip > 200).sum()) / n_hip * 100.0
                result["postwrite_hip_bone_pct"] = round(bone_pct, 1)
                if bone_pct < _POSTWRITE_HIP_BONE_PCT_WARN:
                    log.warning(
                        "[token=%s config=%s] post-write HU sanity FAIL: "
                        "hip-mask voxels are %.1f%% bone (expected >%.0f). "
                        "CT data may be misaligned with label.",
                        token, config, bone_pct, _POSTWRITE_HIP_BONE_PCT_WARN,
                    )
        except Exception as _exc:
            log.debug("[token=%s] post-write sanity skipped: %s", token, _exc)

        all_vals = {int(v) for v in np.unique(lbl_arr)}
        uniq = {v for v in all_vals if 1 <= v <= 6}
        result["n_lumbar_labels"] = len(uniq)
        result["has_l6"]          = 6 in uniq
        # anchor is DERIVED: the last rib-bearing vertebra = the last thoracic,
        # T12 (31) or, when present, the supernumerary T13 (32). No stored class.
        result["has_anchor"]      = bool({31, 32} & all_vals)

        if not args.get("skip_qc"):
            ct_arr = np.asarray(ct_r.dataobj, dtype=np.float32)
            make_qc_figure(ct_arr, lbl_arr, out_qc,
                           token=str(token), config=config,
                           lstv=lstv, position=position)

        result["ok"] = True

    except Exception as exc:
        result["error"] = str(exc)
        log.error("FAIL token=%s: %s", token, exc)

    return result


# -- Build work items ---------------------------------------------------------

def _load_castellvi_map(csv_path: Optional[Path]) -> Dict[str, dict]:
    """token -> {type, second_read, agreement}, joined from _lstv_phenotypes.csv.

    Castellvi grading lives in a side CSV (radiologist phenotyping), not
    placed_manifest, so we join it in by token at export time. Most cases have a
    single read (castellvi_type); a subset were double-read (castellvi_second_read).
    We ship BOTH plus an agreement flag (True/False, or None when not double-read)
    so reader disagreement is visible rather than silently collapsed to read 1.
    Absent file / token -> no entry (the manifest fields stay JSON null).
    """
    out: Dict[str, dict] = {}
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "_lstv_phenotypes.csv"
    if not csv_path.exists():
        log.warning("castellvi: no phenotype CSV at %s — castellvi fields will be null", csv_path)
        return out
    import csv as _csv
    n_second = n_disagree = 0
    with open(csv_path, newline="") as fh:
        for r in _csv.DictReader(fh):
            tok = str(r.get("token", "")).strip()
            grade = (r.get("castellvi_type") or "").strip()
            if not (tok and grade):
                continue
            second = (r.get("castellvi_second_read") or "").strip()
            agreement = (grade == second) if second else None
            if second:
                n_second += 1
                n_disagree += int(agreement is False)
            out[tok] = {"type": grade, "second_read": second or None, "agreement": agreement}
    log.info("castellvi: loaded %d graded token(s) from %s (%d double-read, %d disagree)",
             len(out), csv_path.name, n_second, n_disagree)
    return out


def build_work(manifest_path: Path, nifti_dir: Path,
               spine_dir: Path, pelvic_dir: Path,
               include_configs: Optional[set] = None,
               castellvi_csv: Optional[Path] = None) -> List[dict]:
    """Build per-case export work from placed_manifest.json.

    include_configs: if given, keep only these configs (e.g. {"fused",
    "spine_only"} for the v2 ship, which excludes pelvic_native — a real-pelvis
    but PSEUDO-spine scan we never ship, holding it back as the pelvis-pseudo-
    label validation set). None = all configs (v1 behaviour, unchanged).

    castellvi_csv: token->Castellvi-grade source (default: repo _lstv_phenotypes.csv),
    joined into each record's castellvi_type manifest field.
    """
    data  = json.loads(manifest_path.read_text())
    cases = data.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())

    castellvi_map = _load_castellvi_map(castellvi_csv)

    work = []
    for c in cases:
        tok  = str(c.get("patient_token", "?"))
        mt   = c.get("match_type", "unknown")
        sp   = c.get("spine",  {}) or {}
        pv   = c.get("pelvic", {}) or {}

        lstv_pelvic       = c.get("lstv_pelvic",        "") or ""
        lstv_vertebral    = c.get("lstv_vertebral",      "") or ""
        lstv_agreement    = c.get("lstv_agreement")
        lstv_confusion    = c.get("lstv_confusion_zone", False)

        _cls_map = {0: "normal", 1: "LUMBARIZATION", 2: "SEMI_SACRALIZATION",
                    3: "SACRALIZATION", 4: "SACRALIZATION"}
        _cls = int(c.get("lstv_class", 0) or 0)
        if _cls > 0:
            lstv = _cls_map.get(_cls, "normal")
        else:
            _lp = lstv_pelvic    if lstv_pelvic.lower()    not in ("unknown", "", "normal") else ""
            _lv = lstv_vertebral if lstv_vertebral.lower() not in ("unknown", "", "normal") else ""
            lstv = _lp or _lv or "normal"
        if not lstv:
            mask_file = pv.get("mask_file") or pv.get("placed") or ""
            fname = Path(mask_file).name.lower()
            if   "sacrali"  in fname: lstv = "sacralization"
            elif "lumbariz" in fname: lstv = "lumbarization"
            elif "semi"     in fname: lstv = "semi"
            else:                     lstv = "normal"

        _lmap = {"lumbarization": 1, "semi": 2, "semi-sacralization": 2,
                 "sacralization": 3, "hard": 4}
        lstv_class = _lmap.get(lstv.lower(), 0)

        spine_uid  = _first_not_none(sp.get("series_uid"),
                                     c.get("spine_series_uid"))
        pelvic_uid = _first_not_none(pv.get("series_uid"),
                                     c.get("pelvic_series_uid"))

        spine_bone_pct  = _to_optional_float(_first_not_none(
            sp.get("bone_pct"),
            sp.get("spine_bone_pct"),
            c.get("spine_bone_pct"),
        ))
        pelvic_bone_pct = _to_optional_float(_first_not_none(
            pv.get("bone_pct"),
            pv.get("pelvic_bone_pct"),
            c.get("pelvic_bone_pct"),
        ))

        spine_placed  = Path(sp["placed"])  if sp.get("placed")  else None
        pelvic_placed = Path(pv["placed"])  if pv.get("placed")  else None

        if spine_placed and not spine_placed.exists() and spine_uid:
            cand = spine_dir / f"{spine_uid}_seg_placed.nii.gz"
            if cand.exists():
                spine_placed = cand
        if pelvic_placed and not pelvic_placed.exists():
            stem = Path(pv.get("placed","")).name
            if stem:
                cand = pelvic_dir / stem
                if cand.exists():
                    pelvic_placed = cand

        spine_ct  = nifti_dir / f"{spine_uid}.nii.gz"  if spine_uid  else None
        pelvic_ct = nifti_dir / f"{pelvic_uid}.nii.gz" if pelvic_uid else None

        # placed_manifest.json (schema >=2.5) emits position as an explicit
        # null when DICOM Patient Position was unavailable. Resolve in the
        # same case->spine->pelvic order as before, but a null/missing/empty
        # (or the legacy "unknown" sentinel) passes through as None — never
        # the string "unknown", which would reintroduce a sentinel string
        # into the position column. Valid labels pass through unchanged.
        pos = (c.get("position") or sp.get("position")
               or pv.get("position") or None)
        if not pos or pos == "unknown":
            pos = None

        token_base = _fname_base(tok)

        lstv_kwargs = dict(
            lstv=lstv,
            lstv_pelvic=lstv_pelvic,
            lstv_vertebral=lstv_vertebral,
            castellvi_type=(castellvi_map.get(tok) or {}).get("type"),
            castellvi_second_read=(castellvi_map.get(tok) or {}).get("second_read"),
            castellvi_agreement=(castellvi_map.get(tok) or {}).get("agreement"),
            lstv_agreement=lstv_agreement,
            lstv_confusion_zone=lstv_confusion,
            lstv_class=lstv_class,
        )
        prov_kwargs = dict(
            spine_series_uid=spine_uid,
            pelvic_series_uid=pelvic_uid,
            spine_bone_pct=spine_bone_pct,
            pelvic_bone_pct=pelvic_bone_pct,
        )
        # DICOM header demographics from placed_manifest (schema >=2.7).
        # case-level scalar first, then per-side. _first_not_none (not `or`)
        # so a real 0.0 weight / 0 age is not skipped; absent stays None ->
        # JSON null downstream. All non-age fields are pure pass-through.
        demo_kwargs = {
            f: _first_not_none(c.get(f), sp.get(f), pv.get(f))
            for f in ("sex", "patient_weight", "patient_size",
                      "convolution_kernel", "manufacturer",
                      "manufacturer_model", "slice_thickness", "kvp")
        }
        # Age: resolve the propagated integer (key is "age" — NOT
        # "patient_age"; the raw header string lives only in the sidecar's
        # age_raw and is intentionally not surfaced), then apply the
        # public-release policy: <18 -> null, decade band derived. age and
        # age_band are emitted together and stay consistent.
        _age, _age_band = _age_public(
            _first_not_none(c.get("age"), sp.get("age"), pv.get("age"))
        )
        demo_kwargs["age"] = _age
        demo_kwargs["age_band"] = _age_band

        if mt == "fused":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="fused", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed),
                    pelvic_path=str(pelvic_placed) if pelvic_placed and pelvic_placed.exists() else None,
                    fname_base=token_base, **lstv_kwargs, **prov_kwargs, **demo_kwargs,
                ))
        elif mt == "separate":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="spine_only", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed), pelvic_path=None,
                    fname_base=f"{token_base}_spine", **lstv_kwargs, **prov_kwargs, **demo_kwargs,
                ))
            if pelvic_placed and pelvic_placed.exists() and pelvic_ct and pelvic_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="pelvic_native", match_type=mt,
                    ct_path=str(pelvic_ct), spine_path=None, pelvic_path=str(pelvic_placed),
                    fname_base=f"{token_base}_pelvic", **lstv_kwargs, **prov_kwargs, **demo_kwargs,
                ))
        elif mt == "spine_only":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="spine_only", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed), pelvic_path=None,
                    fname_base=f"{token_base}_spine", **lstv_kwargs, **prov_kwargs, **demo_kwargs,
                ))
        elif mt == "pelvic_only":
            if pelvic_placed and pelvic_placed.exists() and pelvic_ct and pelvic_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="pelvic_native", match_type=mt,
                    ct_path=str(pelvic_ct), spine_path=None, pelvic_path=str(pelvic_placed),
                    fname_base=f"{token_base}_pelvic", **lstv_kwargs, **prov_kwargs, **demo_kwargs,
                ))

    if include_configs:
        before = len(work)
        work = [w for w in work if w["config"] in include_configs]
        from collections import Counter
        kept = Counter(w["config"] for w in work)
        log.info("config filter %s: %d -> %d work items  %s",
                 sorted(include_configs), before, len(work), dict(kept))
    return work


# -- Splits -------------------------------------------------------------------

def write_splits(records: List[dict], out_dir: Path, seed: int = 42) -> None:
    import random

    ok = [r for r in records if r.get("ok")]

    def _is_lstv(r, subtype):
        lbl = (r.get("lstv_label") or "NORMAL").upper()
        if subtype == "lumbarization":
            return lbl == "LUMBARIZATION"
        if subtype == "sacralization":
            return lbl in ("SACRALIZATION", "SEMI_SACRALIZATION")
        return False

    is_fused = lambda r: r.get("config") == "fused"

    strata = [
        ("lumbarization_fused",    [r for r in ok if _is_lstv(r,"lumbarization") and     is_fused(r)]),
        ("lumbarization_separate", [r for r in ok if _is_lstv(r,"lumbarization") and not is_fused(r)]),
        ("sacralization_fused",    [r for r in ok if _is_lstv(r,"sacralization") and     is_fused(r)]),
        ("sacralization_separate", [r for r in ok if _is_lstv(r,"sacralization") and not is_fused(r)]),
        ("fused_normal",           [r for r in ok if not _is_lstv(r,"lumbarization")
                                                  and not _is_lstv(r,"sacralization")
                                                  and is_fused(r)]),
        ("nonfused_normal",        [r for r in ok if not _is_lstv(r,"lumbarization")
                                                  and not _is_lstv(r,"sacralization")
                                                  and not is_fused(r)]),
    ]

    def _split_stratum(lst, name, rng):
        lst = list(lst); rng.shuffle(lst); n = len(lst)
        if n == 0: return [], [], []
        if n == 1: return lst, [], []
        if n == 2: return lst[:1], lst[1:], []
        n_test  = max(1, round(n * 0.15))
        n_val   = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = 1, max(1,(n-1)//2), n-1-max(1,(n-1)//2)
        tr = lst[n_test+n_val:]; va = lst[n_test:n_test+n_val]; te = lst[:n_test]
        log.info("  %-28s n=%3d  train=%d val=%d test=%d", name, n, len(tr), len(va), len(te))
        return tr, va, te

    train_recs: list = []
    val_recs:   list = []
    test_recs:  list = []
    strata_stats: List[dict] = []

    for name, recs in strata:
        rng = random.Random(seed + hash(name) % 1000)
        tr, va, te = _split_stratum(recs, name, rng)
        train_recs.extend(tr); val_recs.extend(va); test_recs.extend(te)
        strata_stats.append({
            "name": name, "n_total": len(recs),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        })

    log.info("Splits  train=%d  val=%d  test=%d",
             len(train_recs), len(val_recs), len(test_recs))

    def _clean(recs):
        # Project every split record onto the canonical manifest schema so
        # manifest_train/validation/test.json share the exact key set and
        # per-column types as manifest.json (no CastError in the HF viewer).
        return [_coerce_manifest_record(r) for r in recs]

    (out_dir / "manifest_train.json"     ).write_text(json.dumps(_clean(train_recs), indent=2))
    (out_dir / "manifest_validation.json").write_text(json.dumps(_clean(val_recs),   indent=2))
    (out_dir / "manifest_test.json"      ).write_text(json.dumps(_clean(test_recs),  indent=2))
    (out_dir / "data_splits.json"        ).write_text(json.dumps({
        "train": [r["ct_file"] for r in train_recs],
        "val":   [r["ct_file"] for r in val_recs],
        "test":  [r["ct_file"] for r in test_recs],
    }, indent=2))
    log.info("  manifest_train / validation / test .json written")

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    test_tokens = sorted({str(r["token"]) for r in test_recs})
    (splits_dir / "test.json").write_text(json.dumps(test_tokens, indent=2))
    log.info("  splits/test.json written (%d unique tokens)", len(test_tokens))

    summary = {
        "seed": seed,
        "n_records": {
            "train": len(train_recs), "val": len(val_recs), "test": len(test_recs),
            "total": len(ok),
        },
        "n_tokens": {
            "train": len({r["token"] for r in train_recs}),
            "val":   len({r["token"] for r in val_recs}),
            "test":  len({r["token"] for r in test_recs}),
            "total": len({r["token"] for r in ok}),
        },
        "strata": strata_stats,
    }
    (out_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("  splits_summary.json written")


# -- Manifest -----------------------------------------------------------------

def write_manifest(records: List[dict], out_dir: Path) -> None:
    ok: List[dict] = [
        _coerce_manifest_record(r) for r in records if r.get("ok")
    ]

    (out_dir / "manifest.json").write_text(json.dumps(ok, indent=2))
    if not ok:
        return

    n_confusion = sum(1 for r in ok if r.get("lstv_confusion_zone") is True)
    n_agreed    = sum(1 for r in ok
                      if r.get("lstv_agreement") is True
                      and r.get("lstv_class", 0) > 0)
    n_sp_uid    = sum(1 for r in ok if r.get("spine_series_uid"))
    n_pv_uid    = sum(1 for r in ok if r.get("pelvic_series_uid"))
    n_sp_pct    = sum(1 for r in ok if r.get("spine_bone_pct")  is not None)
    n_pv_pct    = sum(1 for r in ok if r.get("pelvic_bone_pct") is not None)
    n_resampled = sum(1 for r in ok if r.get("ct_resampled_to_mask"))
    n_partial   = sum(1 for r in ok if r.get("partial_annotation"))
    n_hu_low    = sum(1 for r in ok
                      if r.get("postwrite_hip_bone_pct") is not None
                      and r["postwrite_hip_bone_pct"] < _POSTWRITE_HIP_BONE_PCT_WARN)
    log.info("Manifest  %d cases  confusion_zone=%d  agreed_lstv=%d -> manifest.json",
             len(ok), n_confusion, n_agreed)
    log.info("  provenance:  spine_uid=%d/%d  pelvic_uid=%d/%d  "
             "spine_bone_pct=%d/%d  pelvic_bone_pct=%d/%d",
             n_sp_uid, len(ok), n_pv_uid, len(ok),
             n_sp_pct, len(ok), n_pv_pct, len(ok))
    log.info("  CT-vs-mask:  resampled=%d/%d  hu_at_hip_low=%d/%d "
             "(threshold=%.0f%% bone)",
             n_resampled, len(ok), n_hu_low, len(ok),
             _POSTWRITE_HIP_BONE_PCT_WARN)
    log.info("  partial-annotation: %d/%d cases use IGNORE_LABEL=%d "
             "(separate-mode spine-only / pelvic-only exports)",
             n_partial, len(ok), IGNORE_LABEL)
    if n_hu_low:
        bad = [r["token"] for r in ok
               if r.get("postwrite_hip_bone_pct") is not None
               and r["postwrite_hip_bone_pct"] < _POSTWRITE_HIP_BONE_PCT_WARN]
        log.warning("  TOKENS WITH LOW HU AT HIPS: %s", bad)

    n_bad_ct = sum(1 for r in ok if "/" not in str(r.get("ct_file", "")))
    n_bad_lb = sum(1 for r in ok if "/" not in str(r.get("label_file", "")))
    if n_bad_ct or n_bad_lb:
        log.error("MANIFEST PATH BUG: %d records have bare-basename ct_file, "
                  "%d have bare-basename label_file. These will not resolve "
                  "under the nested dataset layout.", n_bad_ct, n_bad_lb)

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS,
                           extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(ok)
    log.info("manifest.csv written (%d rows)", len(ok))


# -- HuggingFace push ---------------------------------------------------------

def _wipe_remote_repo(api, repo_id: str, repo_type: str, token: str,
                       force: bool = False,
                       revision: Optional[str] = None) -> None:
    """Clear ALL files from the repo WITHOUT deleting the repo.

    delete_repo() is deliberately NOT used: it would destroy git history,
    stars and discussions and briefly 404 the URL. The dataset URL is cited
    in a paper under anonymous review and must stay continuously live, so
    instead the repo is kept and its files are removed in one atomic
    delete commit. The repo, URL, history and discussions are preserved.
    `.gitattributes` is kept so the LFS tracking rules survive the re-push.
    """
    from huggingface_hub import create_repo, CommitOperationDelete

    if not force:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "wipe_remote requested without --force_wipe_remote on a "
                "non-interactive shell. Refusing to clear a HuggingFace repo "
                "without explicit confirmation. Re-submit with "
                "FORCE_WIPE_REMOTE=1 (env var) or --force_wipe_remote."
            )
        log.warning("=" * 60)
        log.warning("ABOUT TO CLEAR ALL FILES IN HF REPO: %s (type=%s)",
                    repo_id, repo_type)
        log.warning("The repo, its URL, git history, stars and discussions")
        log.warning("are PRESERVED. Only the file contents are removed.")
        log.warning("Local files are unaffected.")
        log.warning("=" * 60)
        ans = input(f"Type the repo name '{repo_id}' to confirm: ").strip()
        if ans != repo_id:
            raise RuntimeError(
                f"wipe_remote aborted: typed '{ans}', expected '{repo_id}'")

    # Ensure the repo exists and stays live (idempotent; never deletes).
    log.info("Ensuring HF repo %s exists (will be cleared, not deleted) ...",
             repo_id)
    create_repo(repo_id=repo_id, repo_type=repo_type,
                private=False, exist_ok=True, token=token)

    all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type,
                                    token=token, revision=revision)
    # Keep .gitattributes so the repo's LFS tracking rules survive the
    # re-push. Also skip anything under .cache/ — the Hub reserves that
    # prefix and rejects CommitOperationDelete on it (ValueError), and it
    # isn't dataset content anyway (it's upload state a prior push wrongly
    # uploaded; the ignore_patterns fix below stops that recurring). The
    # leftover .cache/* is inert and hidden from the dataset view.
    files = [f for f in all_files
             if f != ".gitattributes" and not f.startswith(".cache/")]
    if not files:
        log.info("  repo already empty (only .gitattributes) — nothing to do")
        return

    # Single atomic commit. Delete operations carry no payload (just the
    # path list), so even this repo's ~1600+ files fit well within the
    # Hub's per-commit limits — no batching needed. Atomicity matters: the
    # repo never observes a partially-cleared state.
    log.info("Clearing %d files from %s%s in one atomic delete commit ...",
             len(files), repo_id,
             f" (revision: {revision})" if revision else "")
    api.create_commit(
        repo_id=repo_id, repo_type=repo_type, token=token,
        revision=revision,
        commit_message="wipe: clear all files (repo/URL/history preserved)",
        operations=[CommitOperationDelete(path_in_repo=f) for f in files],
    )
    log.info("  repo contents cleared; repo / URL / git history intact")


def hf_write_preflight(repo_id, token, revision, private=False) -> None:
    """Fail fast — BEFORE the multi-hour export — if the token cannot write the
    target repo. A 403 after the export is pure waste. Also catches the silent
    trap where an EMPTY HF_TOKEN env makes HfApi fall back to a cached login
    token (often read-only or the wrong account)."""
    from huggingface_hub import HfApi, create_repo
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF preflight: no token. Set HF_TOKEN (or pass --hf_token). "
            "An EMPTY env var would silently fall back to a cached login token, "
            "so confirm your WRITE token actually reached this process.")
    api = HfApi(token=token)
    try:
        who = (api.whoami() or {}).get("name", "?")
    except Exception as e:                                # noqa: BLE001
        raise SystemExit(f"HF preflight: token invalid / not reaching HF: {e}")
    log.info("HF preflight: authenticated as '%s'.", who)
    try:
        create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE, private=private,
                    exist_ok=True, token=token)
        if revision:
            api.create_branch(repo_id=repo_id, repo_type=HF_REPO_TYPE,
                              branch=revision, token=token, exist_ok=True)
        log.info("HF preflight: WRITE access to %s@%s confirmed.",
                 repo_id, revision or "main")
    except Exception as e:                                # noqa: BLE001
        org = repo_id.split("/")[0]
        raise SystemExit(
            f"HF preflight: '{who}' authenticated but CANNOT WRITE to {repo_id}"
            f"@{revision or 'main'}. This is a permissions problem, not a script "
            f"bug:\n"
            f"  1. the token must be a WRITE token (fine-grained tokens need "
            f"'Write' access to this repo) — a read token cannot push;\n"
            f"  2. account '{who}' must have write membership in the '{org}' "
            f"org/namespace.\n"
            f"  Check: python -c \"from huggingface_hub import HfApi; "
            f"print(HfApi(token='$HF_TOKEN').whoami())\"\n"
            f"  HF said: {e}")


def push_to_hub(
    out_dir:          Path,
    repo_id:          Optional[str] = HF_REPO_ID,
    token:            Optional[str] = None,
    private:          bool          = False,
    num_workers:      int           = 8,
    interface_script: Optional[Path] = None,
    readme_path:      Optional[Path] = None,
    wipe_remote:      bool          = False,
    force_wipe_remote: bool         = False,
    revision:         Optional[str] = None,
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise RuntimeError("pip install 'huggingface_hub[hf_transfer]'")

    if not repo_id:
        raise ValueError(
            "No HuggingFace repo id. Pushing requires an explicit "
            "--hf_repo_id (or HF_REPO_ID env) — there is no default repo "
            "(the same export is pushed to multiple venue repos)."
        )

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf_token."
        )

    api = HfApi(token=token)

    log.info("Ensuring repo: %s  (private=%s) ...", repo_id, private)
    create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE,
                private=private, exist_ok=True, token=token)

    # Push to a separate revision/branch (e.g. "v2" for the pseudo-labelled
    # full release) so the reviewed main branch / URL is never disturbed.
    if revision:
        api.create_branch(repo_id=repo_id, repo_type=HF_REPO_TYPE,
                          branch=revision, token=token, exist_ok=True)
        log.info("Target revision/branch: %s", revision)

    if wipe_remote:
        _wipe_remote_repo(api, repo_id=repo_id, repo_type=HF_REPO_TYPE,
                          token=token, force=force_wipe_remote,
                          revision=revision)

    if interface_script is None:
        for cand in [Path(__file__).parent / "dataset_interface.py",
                     Path(__file__).parent.parent / "dataset_interface.py"]:
            if cand.exists():
                interface_script = cand; break

    if readme_path is None:
        for cand in [Path(__file__).parent / "README.md",
                     Path(__file__).parent.parent / "README.md"]:
            if cand.exists():
                readme_path = cand; break

    if interface_script and interface_script.exists():
        dst = out_dir / "dataset_interface.py"
        if not dst.exists():
            shutil.copy2(str(interface_script), str(dst))
        log.info("  dataset_interface.py ready")
    else:
        log.warning("  dataset_interface.py not found -- skipping")

    if readme_path and readme_path.exists():
        dst = out_dir / "README.md"
        if not dst.exists():
            shutil.copy2(str(readme_path), str(dst))
        log.info("  README.md ready")
    else:
        log.warning("  README.md not found -- skipping")

    n_ct     = sum(1 for _ in (out_dir / "ct"    ).glob("*.nii.gz")) if (out_dir / "ct"    ).exists() else 0
    n_labels = sum(1 for _ in (out_dir / "labels").glob("*.nii.gz")) if (out_dir / "labels").exists() else 0

    # upload_large_folder keeps per-file progress under out_dir/.cache and is
    # NOT repo-aware: re-using the same folder for a DIFFERENT repo/revision
    # makes it think every file is already uploaded -> the new target lands
    # EMPTY (it commits nothing). We push one export to several venue repos
    # + v2/v3 branches, so track the last target and wipe the cache when it
    # changes, forcing a real re-hash + re-upload.
    target = f"{repo_id}@{revision or 'main'}"
    marker = out_dir / ".cache" / ".hf_push_target"
    prev = marker.read_text().strip() if marker.exists() else None
    if prev and prev != target:
        shutil.rmtree(out_dir / ".cache", ignore_errors=True)
        log.info("Upload cache was for %s; cleared it for new target %s",
                 prev, target)

    log.info("=" * 60)
    log.info("Pushing to HuggingFace: %s", repo_id)
    log.info("  CTs=%d  Labels=%d  Workers=%d", n_ct, n_labels, num_workers)
    log.info("  Upload folder: %s", out_dir)
    log.info("  Excluding: qc/  (QC figures not pushed)")
    log.info("  Mode: %s", "fresh push (wipe_remote)" if wipe_remote else "additive (existing files retained)")
    log.info("  Upload is resumable -- re-run if interrupted")
    log.info("=" * 60)

    api.upload_large_folder(
        repo_id=repo_id, repo_type=HF_REPO_TYPE,
        folder_path=str(out_dir), num_workers=num_workers,
        revision=revision,
        # Never upload local-only state: QC figures, staging, and the
        # upload cache + .hf_push_target marker under .cache/. The Hub
        # rejects later deletes under .cache/, so uploading it also breaks
        # WIPE_REMOTE — exclude it here so it never lands remotely.
        ignore_patterns=["qc/*", "qc/**/*", ".hf_staging/*",
                         ".cache/*", ".cache/**/*"],
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(target)        # remember what this cache now represents
    _where = f"{repo_id}" + (f"  (revision: {revision})" if revision else "")
    log.info("Push complete -> https://huggingface.co/datasets/%s", _where)


# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",    required=True,  type=Path)
    ap.add_argument("--nifti_dir",   required=True,  type=Path)
    ap.add_argument("--spine_dir",   required=True,  type=Path)
    ap.add_argument("--pelvic_dir",  required=True,  type=Path)
    ap.add_argument("--out_dir",     required=True,  type=Path)
    ap.add_argument("--workers",     default=8,      type=int)
    ap.add_argument("--skip_qc",     action="store_true")
    ap.add_argument("--no_pir",      action="store_true")
    ap.add_argument("--debug_n",     default=0,      type=int)
    ap.add_argument("--skip_export", action="store_true")
    ap.add_argument("--castellvi_csv", default=None, type=Path,
                    help="token->Castellvi-grade CSV joined into the manifest's "
                         "castellvi_type field (default: repo _lstv_phenotypes.csv)")
    ap.add_argument("--include_configs", default=None,
                    help="comma-separated configs to ship (e.g. "
                         "'fused,spine_only' for the v2 release, which excludes "
                         "pelvic_native). Default/empty = all configs. Env "
                         "INCLUDE_CONFIGS is honoured if the flag is absent.")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id",  default=HF_REPO_ID,
                    help="Target HF repo (e.g. org/Name). REQUIRED with "
                         "--push_to_hub; there is no default repo.")
    ap.add_argument("--hf_token",    default=None)
    ap.add_argument("--hf_revision", default=None,
                    help="Push to this branch/revision instead of main "
                         "(e.g. 'v2' for the pseudo-labelled full release). "
                         "Created if absent; main/URL left untouched.")
    ap.add_argument("--hf_private",  action="store_true")
    ap.add_argument("--hf_workers",  default=8, type=int)
    ap.add_argument("--interface_script", default=None, type=Path)
    ap.add_argument("--readme_path",      default=None, type=Path)
    ap.add_argument("--wipe_remote", action="store_true",
                    help="Clear all files in the HF repo before pushing "
                         "(repo, URL, git history & discussions preserved; "
                         "the repo is NOT deleted).")
    ap.add_argument("--force_wipe_remote", action="store_true",
                    help="Skip interactive confirmation for --wipe_remote.")
    args = ap.parse_args()

    if not args.force_wipe_remote and \
       os.environ.get("FORCE_WIPE_REMOTE", "").strip().lower() in ("1", "true", "yes"):
        args.force_wipe_remote = True

    if args.push_to_hub and not args.hf_repo_id:
        log.error("--push_to_hub requires an explicit --hf_repo_id "
                  "(e.g. --hf_repo_id org/CTSpinoPelvic1K). There is no "
                  "default repo: the same export is pushed to multiple "
                  "venue repos, so a fallback would risk the wrong target.")
        sys.exit(2)

    if args.wipe_remote and not args.push_to_hub:
        log.error("--wipe_remote requires --push_to_hub. Refusing to "
                  "clear the HF repo without re-pushing.")
        sys.exit(2)

    # Verify HF write access BEFORE the (multi-hour) export, so a permission /
    # token problem aborts in seconds instead of after all the work.
    if args.push_to_hub:
        hf_write_preflight(args.hf_repo_id, args.hf_token, args.hf_revision,
                           args.hf_private)

    if not args.skip_export:
        for d in [args.out_dir/"ct", args.out_dir/"labels", args.out_dir/"qc"]:
            d.mkdir(parents=True, exist_ok=True)

        _inc = args.include_configs or os.environ.get("INCLUDE_CONFIGS", "")
        include_configs = ({c.strip() for c in _inc.split(",") if c.strip()}
                           or None)
        log.info("Building work from %s ...", args.manifest)
        work = build_work(args.manifest, args.nifti_dir, args.spine_dir,
                          args.pelvic_dir, include_configs=include_configs,
                          castellvi_csv=args.castellvi_csv)
        log.info("Work items: %d", len(work))

        if args.debug_n > 0:
            work = work[:args.debug_n]

        for item in work:
            base = item.pop("fname_base")
            item["out_ct"]   = str(args.out_dir / "ct"     / f"{base}_ct.nii.gz")
            item["out_lbl"]  = str(args.out_dir / "labels" / f"{base}_label.nii.gz")
            item["out_qc"]   = str(args.out_dir / "qc"     / f"{base}_qc.png")
            item["skip_qc"]  = args.skip_qc
            item["skip_pir"] = args.no_pir

        records, n_ok, n_fail = [], 0, 0
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_export_one, w): w["token"] for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                tok = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    rec = {"token": tok, "ok": False, "error": str(exc)}
                records.append(rec)
                if rec.get("ok"):
                    n_ok += 1
                else:
                    n_fail += 1
                    log.warning("FAIL token=%s  %s", tok, rec.get("error","?"))
                if i % 50 == 0 or i == len(work):
                    elapsed = time.time() - t0
                    eta = (len(work)-i) / max(i/elapsed, 1e-9)
                    log.info("[%d/%d]  ok=%d  fail=%d  elapsed=%.0fs  ETA=%.0fs",
                             i, len(work), n_ok, n_fail, elapsed, eta)

        write_manifest(records, args.out_dir)
        write_splits(records,   args.out_dir)

        bad = [r for r in records if r.get("ok") and not r.get("alignment_ok", True)]
        log.info("EXPORT DONE  ok=%d  fail=%d  alignment_fail=%d",
                 n_ok, n_fail, len(bad))
        if bad:
            log.error("ALIGNMENT MISMATCHES: %s", [r["token"] for r in bad])
    else:
        log.info("--skip_export: skipping export phase.")

    if args.push_to_hub:
        push_to_hub(
            out_dir=args.out_dir, repo_id=args.hf_repo_id,
            token=args.hf_token, private=args.hf_private,
            num_workers=args.hf_workers,
            interface_script=args.interface_script,
            readme_path=args.readme_path,
            wipe_remote=args.wipe_remote,
            force_wipe_remote=args.force_wipe_remote,
            revision=args.hf_revision,
        )
    else:
        log.info("HuggingFace push skipped. Add --push_to_hub to upload.")


if __name__ == "__main__":
    main()
