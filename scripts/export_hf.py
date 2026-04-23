"""
export_hf.py — Export CTSpinoPelvic1K to HuggingFace-compatible flat directory,
then push directly to the Hub using upload_large_folder.

Reads placed_manifest.json (written by place_fused_masks.py) and for each case:
  1. Load CT NIfTI (from tcia_nifti/{series_uid}.nii.gz)
  2. Remap spine labels (VerSe -> 10-class) + pelvic labels (4-class -> 10-class)
  3. Merge into single 10-class label map
  4. Reorient CT + label to PIR canonical orientation
  5. Strip PHI from NIfTI headers
  6. Save to flat output:
       ct/{token:04d}_{position}_ct.nii.gz
       labels/{token:04d}_{position}_label.nii.gz
  7. Write manifest.json, manifest.csv, data_splits.json, splits/test.json,
     splits_summary.json
  8. Push to HuggingFace

placed_manifest.json is the single source of truth. It now carries all LSTV
fields directly (lstv_pelvic, lstv_vertebral, lstv_agreement, lstv_confusion_zone)
populated by place_fused_masks.py. No secondary manifest file is needed.

Label scheme:
  0=bg  1=L1  2=L2  3=L3  4=L4  5=L5  6=L6(LSTV)  7=sacrum  8=left_hip  9=right_hip

Output orientation: every CT + label pair is written in PIR canonical:
  axis 0 = Posterior  (A->P as idx++)
  axis 1 = Inferior   (S->I as idx++)
  axis 2 = patient-Right (L->R as idx++)

QC figures are sliced and displayed assuming PIR, so rows correspond to:
  Row 1 "Coronal"   fix axis 0, show (I, R)  -- head at top, feet at bottom
  Row 2 "Axial"     fix axis 1, show (P, R)  -- anterior at top, spine at bottom
  Row 3 "Sagittal"  fix axis 2, show (P, I)  -- transposed for head-up

Usage (matches slurm/export_dataset.sh):
    python export_hf.py \\
        --manifest   data/placed/placed_manifest.json \\
        --nifti_dir  data/tcia_nifti \\
        --spine_dir  data/placed/spine \\
        --pelvic_dir data/placed/pelvic \\
        --out_dir    data/hf_export \\
        --workers    32 \\
        [--skip_qc] [--no_pir] [--skip_export] \\
        [--push_to_hub --hf_repo_id user/repo --hf_workers 8 [--hf_private]]

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

HF_REPO_ID   = "anonymous-mlhc/CTSpinoPelvic1K"
HF_REPO_TYPE = "dataset"

# Any spatial axis smaller than this is treated as a scout / localizer /
# degenerate volume and rejected at ingest. Mirrors place_fused_masks.py.
MIN_VALID_SHAPE = 10

# Optional numeric fields — written as `null` (not "") in manifest.json so
# Parquet sees a clean nullable-float column rather than mixed str/float.
_OPTIONAL_NUMERIC_FIELDS = frozenset({"spine_bone_pct", "pelvic_bone_pct"})

# -- Label maps ---------------------------------------------------------------

VERSE_TO_10CLASS: Dict[int, int] = {
    20: 1, 21: 2, 22: 3, 23: 4, 24: 5,
    25: 6,
    26: 7,
}
PELVIC_TO_10CLASS: Dict[int, int] = {1: 7, 2: 8, 3: 9}

CLASS_NAMES = {
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "L6", 7: "sacrum", 8: "left_hip", 9: "right_hip",
}

_SEG_COLORS = {
    1: (0.15, 0.40, 0.80, 0.55), 2: (0.25, 0.55, 0.85, 0.55),
    3: (0.35, 0.65, 0.90, 0.55), 4: (0.45, 0.75, 0.92, 0.55),
    5: (0.10, 0.80, 0.85, 0.55), 6: (0.75, 0.85, 0.20, 0.65),
    7: (0.85, 0.15, 0.15, 0.55), 8: (0.95, 0.50, 0.10, 0.55),
    9: (0.95, 0.80, 0.05, 0.55),
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


def _to_optional_float(v):
    """Coerce to float, or None if value is empty/missing/unparseable."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# -- NIfTI helpers ------------------------------------------------------------

def _load_nii(path):
    import nibabel as nib
    return nib.load(str(path))


def _validate_affine(affine, label: str = "") -> None:
    """
    Raise ValueError early if the affine is degenerate before nibabel's
    io_orientation crashes. Catches NaN/Inf columns and zero-norm columns
    from degenerate dcm2niix outputs (localizer/scout series with missing
    geometry tags) -- same class of inputs that broke place_fused_masks.py
    before _valid_affine() was added there.
    """
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
    """
    Raise ValueError if the spatial shape is degenerate for segmentation
    work (scout / localizer / 2-slice derivative). Mirrors the
    _valid_volume_shape filter in place_fused_masks.py so the same class
    of bad inputs gets rejected at both stages with a readable reason
    instead of an IndexError deep in the stack.
    """
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
    result = np.zeros(ref_shape, dtype=np.int16)

    if pelvic_path and Path(pelvic_path).exists():
        pelv = np.asarray(_load_nii(pelvic_path).dataobj, dtype=np.int16)
        mn   = tuple(min(a, b) for a, b in zip(ref_shape, pelv.shape))
        sl   = tuple(slice(0, m) for m in mn)
        for pid, cls in PELVIC_TO_10CLASS.items():
            result[sl][pelv[sl] == pid] = cls

    if spine_path and Path(spine_path).exists():
        sp = np.asarray(_load_nii(spine_path).dataobj, dtype=np.int16)
        mn = tuple(min(a, b) for a, b in zip(ref_shape, sp.shape))
        sl = tuple(slice(0, m) for m in mn)
        for vid, cls in VERSE_TO_10CLASS.items():
            if cls in (1, 2, 3, 4, 5, 6):
                result[sl][sp[sl] == vid] = cls
            elif cls == 7:
                # Only fill sacrum from spine seg where pelvic didn't already
                mask = (sp[sl] == vid) & (result[sl] == 0)
                result[sl][mask] = cls

    return result


# -- QC figure ----------------------------------------------------------------

def _window(arr, lo=-150, hi=700):
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """Orient a 2D slice for radiological display assuming PIR source.

    PIR: axis 0 = P (A->P as idx++), axis 1 = I (S->I as idx++),
         axis 2 = R (L->R as idx++).

    dim=0 (coronal):  slice shape (I, R). Default imshow row-0-top puts
                      superior at top. No transform needed.
    dim=1 (axial):    slice shape (P, R). Row-0-top puts anterior at top,
                      posterior (spine) at bottom. No transform needed.
    dim=2 (sagittal): slice shape (P, I). We want row=I (head up),
                      col=P. Transpose.
    """
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
    nz = np.where(lbl > 0)
    if not len(nz[0]):
        return ct.shape[0]//2, ct.shape[1]//2, ct.shape[2]//2
    return int(np.median(nz[0])), int(np.median(nz[1])), int(np.median(nz[2]))


def make_qc_figure(ct, lbl, out_path, token, config, lstv, position):
    """
    Render a 3-row QC figure assuming PIR storage order.

    Rows correspond to PIR axes:
      row 0: dim=0 (P axis fixed) -> coronal view
      row 1: dim=1 (I axis fixed) -> axial view
      row 2: dim=2 (R axis fixed) -> sagittal view (requires transpose)

    Each slice is routed through _display_slice() so that superior is
    at the top of coronal and sagittal views, and anterior is at the top
    of axial (spine at bottom, matching TotalSegmentator convention).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    present    = {int(v) for v in np.unique(lbl) if v > 0}
    has_lumbar = bool(present & {1,2,3,4,5,6})
    has_pelvic = bool(present & {7,8,9})

    sp_lbl = np.where(np.isin(lbl, [1,2,3,4,5,6]), lbl, 0).astype(np.int16)
    pv_lbl = np.where(np.isin(lbl, [7,8,9]),        lbl, 0).astype(np.int16)

    if has_lumbar and has_pelvic:
        cols   = ["CT (raw)", "CT + spine", "CT + pelvic", "CT + all"]
        layout = "fused"
    else:
        name   = "spine" if has_lumbar else "pelvic"
        cols   = ["CT (raw)", f"CT + {name}", f"{name.capitalize()} only"]
        layout = "single"

    i, j, k = _center_slice(ct, lbl)
    # PIR dims: 0 = coronal plane (fix P), 1 = axial (fix I), 2 = sagittal (fix R)
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

        # Apply PIR-aware display orientation to every slice we render.
        bg       = _display_slice(_window(ct[sl]),    dim)
        sp_slice = _display_slice(sp_lbl[sl],         dim)
        pv_slice = _display_slice(pv_lbl[sl],         dim)
        full_slice = _display_slice(lbl[sl],          dim)

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

    result = dict(
        token=token, position=position, config=config,
        match_type=match_type, lstv_label=lstv, ok=False, error=None,
        alignment_ok=False, has_l6=False, n_lumbar_labels=0,
        ct_file=out_ct.name, label_file=out_lbl.name, qc_file=out_qc.name,
        # LSTV fields -- passed through from placed_manifest.json
        lstv_pelvic=args.get("lstv_pelvic", ""),
        lstv_vertebral=args.get("lstv_vertebral", ""),
        lstv_agreement=args.get("lstv_agreement"),          # bool | None
        lstv_confusion_zone=args.get("lstv_confusion_zone", False),
        lstv_class=args.get("lstv_class", 0),
        # Provenance fields -- passed through from placed_manifest.json
        spine_series_uid=args.get("spine_series_uid"),
        pelvic_series_uid=args.get("pelvic_series_uid"),
        spine_bone_pct=args.get("spine_bone_pct"),
        pelvic_bone_pct=args.get("pelvic_bone_pct"),
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
        if ct_img.shape[:3] != ref_shape:
            from scipy.ndimage import affine_transform as _at
            M       = np.linalg.inv(ct_img.affine) @ ref_affine
            ct_data = _at(ct_data, M[:3,:3], offset=M[:3,3],
                          output_shape=ref_shape, order=1,
                          mode="constant", cval=-1024.0)

        lbl_data = merge_labels(spine_path, pelvic_path, ref_shape)

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
        result["alignment_ok"] = (ct_r.shape[:3] == lbl_r.shape[:3] and
                                   np.allclose(ct_r.affine, lbl_r.affine, atol=1e-4))

        lbl_arr = np.asarray(lbl_r.dataobj, dtype=np.int16)
        uniq    = {int(v) for v in np.unique(lbl_arr) if v > 0}
        result["n_lumbar_labels"] = len({1,2,3,4,5,6} & uniq)
        result["has_l6"]          = 6 in uniq

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

def build_work(manifest_path: Path, nifti_dir: Path,
               spine_dir: Path, pelvic_dir: Path) -> List[dict]:
    """
    Build per-case export work from placed_manifest.json.

    placed_manifest.json is the single source of truth. It carries LSTV fields
    (lstv_pelvic, lstv_vertebral, lstv_agreement, lstv_confusion_zone) and
    placement provenance (series_uid, bone_pct) written by place_fused_masks.py
    -- no secondary manifest file needed.
    """
    data  = json.loads(manifest_path.read_text())
    cases = data.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())

    work = []
    for c in cases:
        tok  = str(c.get("patient_token", "?"))
        mt   = c.get("match_type", "unknown")
        sp   = c.get("spine",  {}) or {}
        pv   = c.get("pelvic", {}) or {}

        # LSTV fields -- now native to placed_manifest.json
        lstv_pelvic       = c.get("lstv_pelvic",        "") or ""
        lstv_vertebral    = c.get("lstv_vertebral",      "") or ""
        lstv_agreement    = c.get("lstv_agreement")           # bool | None
        lstv_confusion    = c.get("lstv_confusion_zone", False)

        # Resolve a single lstv label string for filename / QC figure.
        # Prefer lstv_class (already merged/resolved by place_fused_masks.py),
        # fall back to the raw strings if class==0 in case vertebral detection
        # caught something pelvic annotation missed.
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
            # Fallback: derive from filename for legacy placed_manifest.json
            mask_file = pv.get("mask_file") or pv.get("placed") or ""
            fname = Path(mask_file).name.lower()
            if   "sacrali"  in fname: lstv = "sacralization"
            elif "lumbariz" in fname: lstv = "lumbarization"
            elif "semi"     in fname: lstv = "semi"
            else:                     lstv = "normal"

        # Integer class for the manifest (passed through separately from the string)
        _lmap = {"lumbarization": 1, "semi": 2, "semi-sacralization": 2,
                 "sacralization": 3, "hard": 4}
        lstv_class = _lmap.get(lstv.lower(), 0)

        # Provenance: series UIDs + bone_pct from placed_manifest.json.
        # Key-name variants covered for backward compat with older
        # place_fused_masks.py outputs (top-level vs nested).
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

        # Position resolution order:
        #   1. case-level c["position"]       (place_fused_masks.py >= v2.0)
        #   2. sp["position"] / pv["position"] (per-mask, also v2.0+)
        #   3. filename substring             (legacy placed_manifest.json)
        pos = (c.get("position") or sp.get("position") or pv.get("position") or "")
        if not pos or pos == "unknown":
            fname = (str(spine_placed or "") + str(pelvic_placed or "")).lower()
            pos   = "prone" if "prone" in fname else "supine" if "supine" in fname else "unknown"

        tok_int = int(tok) if tok.isdigit() else abs(hash(tok)) % 10000
        base    = f"{tok_int:04d}_{pos}"

        # Shared kwargs passed to every work item derived from this case
        lstv_kwargs = dict(
            lstv=lstv,
            lstv_pelvic=lstv_pelvic,
            lstv_vertebral=lstv_vertebral,
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

        if mt == "fused":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="fused", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed),
                    pelvic_path=str(pelvic_placed) if pelvic_placed and pelvic_placed.exists() else None,
                    fname_base=base, **lstv_kwargs, **prov_kwargs,
                ))
        elif mt == "separate":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="spine_only", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed), pelvic_path=None,
                    fname_base=f"{base}_spine", **lstv_kwargs, **prov_kwargs,
                ))
            if pelvic_placed and pelvic_placed.exists() and pelvic_ct and pelvic_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="pelvic_native", match_type=mt,
                    ct_path=str(pelvic_ct), spine_path=None, pelvic_path=str(pelvic_placed),
                    fname_base=f"{base}_pelvic", **lstv_kwargs, **prov_kwargs,
                ))
        elif mt == "spine_only":
            if spine_placed and spine_placed.exists() and spine_ct and spine_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="spine_only", match_type=mt,
                    ct_path=str(spine_ct), spine_path=str(spine_placed), pelvic_path=None,
                    fname_base=base, **lstv_kwargs, **prov_kwargs,
                ))
        elif mt == "pelvic_only":
            if pelvic_placed and pelvic_placed.exists() and pelvic_ct and pelvic_ct.exists():
                work.append(dict(
                    token=tok, position=pos, config="pelvic_native", match_type=mt,
                    ct_path=str(pelvic_ct), spine_path=None, pelvic_path=str(pelvic_placed),
                    fname_base=base, **lstv_kwargs, **prov_kwargs,
                ))

    return work


# -- Splits -------------------------------------------------------------------

def write_splits(records: List[dict], out_dir: Path, seed: int = 42) -> None:
    """
    Write stratified train/val/test splits.

    Six strata -- fused and non-fused kept separate within each LSTV subtype
    so that fused LSTV cases (full 10-class GT) are guaranteed in val and test:

      lumbarization_fused      70/15/15 -- guarantees fused lumbarization in val+test
      lumbarization_separate   70/15/15
      sacralization_fused      70/15/15 -- guarantees fused sacralization in val+test
      sacralization_separate   70/15/15
      fused_normal             70/15/15
      nonfused_normal          70/15/15

    Tiny-stratum handling:
      n >= 3  ->  at least 1 in val, 1 in test, rest in train
      n == 2  ->  1 train / 1 val / 0 test
      n == 1  ->  all train

    Outputs:
      manifest_{train,validation,test}.json   per-record splits (legacy)
      data_splits.json                         {"train":[...], "val":[...], "test":[...]}
                                               of ct_file basenames (legacy)
      splits/test.json                         flat list of unique test tokens
                                               (dataset_interface.py preferred path)
      splits_summary.json                      aggregate split stats
    """
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

    _drop = {"ok", "error"}
    def _clean(recs):
        out = []
        for r in recs:
            rec = {}
            for k, v in r.items():
                if k in _drop:
                    continue
                if v is None:
                    rec[k] = None if k in _OPTIONAL_NUMERIC_FIELDS else ""
                else:
                    rec[k] = v
            out.append(rec)
        return out

    (out_dir / "manifest_train.json"     ).write_text(json.dumps(_clean(train_recs), indent=2))
    (out_dir / "manifest_validation.json").write_text(json.dumps(_clean(val_recs),   indent=2))
    (out_dir / "manifest_test.json"      ).write_text(json.dumps(_clean(test_recs),  indent=2))
    (out_dir / "data_splits.json"        ).write_text(json.dumps({
        "train": [r["ct_file"] for r in train_recs],
        "val":   [r["ct_file"] for r in val_recs],
        "test":  [r["ct_file"] for r in test_recs],
    }, indent=2))
    log.info("  manifest_train / validation / test .json written")

    # splits/test.json -- flat unique-token list for dataset_interface.py's
    # preferred split-resolution path.
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    test_tokens = sorted({str(r["token"]) for r in test_recs})
    (splits_dir / "test.json").write_text(json.dumps(test_tokens, indent=2))
    log.info("  splits/test.json written (%d unique tokens)", len(test_tokens))

    # splits_summary.json -- at-a-glance aggregate stats
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
    """
    Write manifest.json and manifest.csv.

    HF Arrow/Parquet compatibility:
    - 'ok' and 'error' stripped (bool-always-True and null-always-None break Parquet)
    - None -> "" for string fields; None lstv_agreement -> "" (HF can't cast mixed null/bool)
    - None left as null for optional numeric fields (spine/pelvic_bone_pct) --
      Parquet handles nullable float columns natively, so stringifying would
      force a mixed str/float column.
    """
    _drop = {"ok", "error"}
    ok: List[dict] = []
    for r in records:
        if not r.get("ok"):
            continue
        rec = {}
        for k, v in r.items():
            if k in _drop:
                continue
            if v is None:
                rec[k] = None if k in _OPTIONAL_NUMERIC_FIELDS else ""
            elif isinstance(v, bool):
                rec[k] = v
            else:
                rec[k] = v
        ok.append(rec)

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
    log.info("Manifest  %d cases  confusion_zone=%d  agreed_lstv=%d -> manifest.json",
             len(ok), n_confusion, n_agreed)
    log.info("  provenance:  spine_uid=%d/%d  pelvic_uid=%d/%d  "
             "spine_bone_pct=%d/%d  pelvic_bone_pct=%d/%d",
             n_sp_uid, len(ok), n_pv_uid, len(ok),
             n_sp_pct, len(ok), n_pv_pct, len(ok))

    keys = [
        "token", "position", "config", "match_type",
        "lstv_label", "lstv_class",
        "lstv_pelvic", "lstv_vertebral", "lstv_agreement", "lstv_confusion_zone",
        "has_l6", "n_lumbar_labels", "alignment_ok",
        "spine_series_uid", "pelvic_series_uid",
        "spine_bone_pct", "pelvic_bone_pct",
        "ct_file", "label_file", "qc_file",
    ]
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)
    log.info("manifest.csv written (%d rows)", len(ok))


# -- HuggingFace push ---------------------------------------------------------

def push_to_hub(
    out_dir:          Path,
    repo_id:          str           = HF_REPO_ID,
    token:            Optional[str] = None,
    private:          bool          = False,
    num_workers:      int           = 8,
    interface_script: Optional[Path] = None,
    readme_path:      Optional[Path] = None,
) -> None:
    """
    Push ct/, labels/, manifest.json, manifest.csv, data_splits.json,
    splits/, splits_summary.json, dataset_interface.py, and README.md to
    HuggingFace using upload_large_folder.

    qc/ PNGs are excluded (large, derivative, regenerable).
    Authenticates via HF_TOKEN bearer auth -- no git credentials embedded.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise RuntimeError("pip install 'huggingface_hub[hf_transfer]'")

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf_token."
        )

    api = HfApi(token=token)
    log.info("Ensuring repo: %s  (private=%s) ...", repo_id, private)
    create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE,
                private=private, exist_ok=True, token=token)

    # Auto-detect interface script and README if staged next to or above this file
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

    log.info("=" * 60)
    log.info("Pushing to HuggingFace: %s", repo_id)
    log.info("  CTs=%d  Labels=%d  Workers=%d", n_ct, n_labels, num_workers)
    log.info("  Upload folder: %s", out_dir)
    log.info("  Excluding: qc/  (QC figures not pushed)")
    log.info("  Upload is resumable -- re-run if interrupted")
    log.info("=" * 60)

    api.upload_large_folder(
        repo_id=repo_id, repo_type=HF_REPO_TYPE,
        folder_path=str(out_dir), num_workers=num_workers,
        ignore_patterns=["qc/*", "qc/**/*", ".hf_staging/*"],
    )
    log.info("Push complete -> https://huggingface.co/datasets/%s", repo_id)


# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required I/O
    ap.add_argument("--manifest",    required=True,  type=Path)
    ap.add_argument("--nifti_dir",   required=True,  type=Path)
    ap.add_argument("--spine_dir",   required=True,  type=Path)
    ap.add_argument("--pelvic_dir",  required=True,  type=Path)
    ap.add_argument("--out_dir",     required=True,  type=Path)
    # Export behaviour
    ap.add_argument("--workers",     default=8,      type=int)
    ap.add_argument("--skip_qc",     action="store_true")
    ap.add_argument("--no_pir",      action="store_true")
    ap.add_argument("--debug_n",     default=0,      type=int)
    ap.add_argument("--skip_export", action="store_true")
    # HuggingFace push
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id",  default=HF_REPO_ID)
    ap.add_argument("--hf_token",    default=None)
    ap.add_argument("--hf_private",  action="store_true")
    ap.add_argument("--hf_workers",  default=8, type=int)
    ap.add_argument("--interface_script", default=None, type=Path)
    ap.add_argument("--readme_path",      default=None, type=Path)
    args = ap.parse_args()

    # -- Export ---------------------------------------------------------------
    if not args.skip_export:
        for d in [args.out_dir/"ct", args.out_dir/"labels", args.out_dir/"qc"]:
            d.mkdir(parents=True, exist_ok=True)

        log.info("Building work from %s ...", args.manifest)
        work = build_work(args.manifest, args.nifti_dir, args.spine_dir, args.pelvic_dir)
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

    # -- Push -----------------------------------------------------------------
    if args.push_to_hub:
        push_to_hub(
            out_dir=args.out_dir, repo_id=args.hf_repo_id,
            token=args.hf_token, private=args.hf_private,
            num_workers=args.hf_workers,
            interface_script=args.interface_script,
            readme_path=args.readme_path,
        )
    else:
        log.info("HuggingFace push skipped. Add --push_to_hub to upload.")


if __name__ == "__main__":
    main()
