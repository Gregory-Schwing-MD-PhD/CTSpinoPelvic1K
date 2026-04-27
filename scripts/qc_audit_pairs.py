"""
qc_audit_pairs.py — Render CT + GT + TS-prediction triplets for L/R-audit
verification, using the SAME orientation conventions as visualize_qc.py.

This addresses the situation where the L/R audit (TS-vs-GT hip Dice) flags
tokens, and you need to visually verify whether the GT mask is L/R-correct
or actually inverted.

Display convention (matches visualize_qc.py):
  Volumes are reoriented to canonical PIR (P=axis 0, I=axis 1, R=axis 2).
  Axial slice (dim=1) has shape (P, R):
    rows: anterior (top) -> posterior (bottom)
    cols: patient-LEFT (left) -> patient-RIGHT (right)   <- NEUROLOGICAL convention
  This is the SAME convention as visualize_qc.py, so direct comparison works.

INPUTS
======
  --manifest    : path to a manifest JSON. Auto-detects format:
                  - placed_manifest.json / placed_manifest_orientation_fixed.json
                    (uses cases[].pelvic.placed for GT masks; CT from nifti_dir)
                  - hf_export manifest list (uses ct_file / label_file)
  --tokens      : comma-separated tokens to render. Required.
  --ts_glob     : glob pattern for TS predictions. Default uses results/.
  --nifti_dir   : directory of CT NIfTIs (placed-manifest mode only).
  --hf_export   : root of hf_export dir (hf-manifest mode only).
  --out_dir     : output directory for PNGs.

OUTPUT
======
  3-panel PNG per token: [CT only] [CT + GT overlay] [CT + TS overlay]
  Both GT and TS use the same color scheme (yellow=left_hip, green=right_hip).
  R/L annotations match visualize_qc.py: L on viewer-left, R on viewer-right.

USAGE
=====
  # placed-manifest mode
  python scripts/qc_audit_pairs.py \\
      --manifest data/placed/placed_manifest_orientation_fixed.json \\
      --nifti_dir data/tcia_nifti \\
      --tokens 3,460,480,616 \\
      --out_dir data/qc_audit_pairs

  # hf-export-manifest mode
  python scripts/qc_audit_pairs.py \\
      --manifest data/hf_export/manifest.json \\
      --hf_export data/hf_export \\
      --tokens 642,210,302 \\
      --out_dir data/qc_audit_pairs
"""

from __future__ import annotations

import argparse
import json
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qc_audit_pairs")

_HU_MIN, _HU_MAX = -200, 800

# Match visualize_qc.py colors for hips
# In placed-manifest mode the pelvic mask uses: 1=sacrum, 2=left_hip, 3=right_hip
# In hf-export mode the unified label uses:     7=sacrum, 8=left_hip, 9=right_hip
# TS predictions use TS class IDs:              25=sacrum, 77=hip_left, 78=hip_right
_LEFT_HIP_COLOR  = (0.95, 0.85, 0.10, 0.55)   # yellow
_RIGHT_HIP_COLOR = (0.20, 0.85, 0.30, 0.55)   # green


# ── Reorientation (matches visualize_qc.py) ──────────────────────────────────

def _load_pir(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI and reorient to canonical PIR. Returns (data, new_affine)."""
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff,
    )
    img      = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)
    data     = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff  = img.affine @ inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _load_pir_label(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Same as _load_pir but preserves integer label values."""
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff,
    )
    img      = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)
    arr_int  = np.asarray(img.dataobj).astype(np.int32)
    data     = apply_orientation(arr_int, xfm).squeeze().astype(np.int32)
    new_aff  = img.affine @ inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _resample_to_pir_grid(
    src_path: Path, ref_pir_data: np.ndarray, ref_pir_affine: np.ndarray,
) -> np.ndarray:
    """
    Resample a NIfTI onto a reference PIR grid using SimpleITK, returning
    a numpy array in PIR axis order matching ref_pir_data.shape.

    SimpleITK uses LPS internally. We construct a reference image from
    ref_pir_affine + ref_pir_data.shape, resample, then return the array.

    SimpleITK GetArrayFromImage returns (z, y, x) (reverse of physical
    storage), so we transpose back to PIR.
    """
    import SimpleITK as sitk
    import nibabel as nib

    # Build a reference SimpleITK image with the PIR grid metadata.
    # SimpleITK's storage order is reverse of nibabel-style PIR, so we
    # construct the reference from the underlying NIfTI on disk that
    # corresponds to ref data via a temp file? No -- simpler: read the
    # ORIGINAL src and ref via nibabel, resample with nibabel-friendly
    # logic instead of SimpleITK.
    # Cleaner approach: use scipy to map_coordinates from the source's
    # voxel grid into ref PIR voxel coordinates.

    from scipy.ndimage import map_coordinates

    src_img = nib.load(str(src_path))
    src_arr = np.asarray(src_img.dataobj).astype(np.int32)
    src_aff = src_img.affine.astype(np.float64)
    src_inv = np.linalg.inv(src_aff)

    # For each voxel in ref PIR grid, compute its world position via
    # ref_pir_affine, then map to src voxel index via src_inv.
    s = ref_pir_data.shape
    ii, jj, kk = np.meshgrid(
        np.arange(s[0], dtype=np.float64),
        np.arange(s[1], dtype=np.float64),
        np.arange(s[2], dtype=np.float64),
        indexing="ij",
    )
    # World coordinates per ref voxel
    aff = ref_pir_affine.astype(np.float64)
    wx = aff[0, 0]*ii + aff[0, 1]*jj + aff[0, 2]*kk + aff[0, 3]
    wy = aff[1, 0]*ii + aff[1, 1]*jj + aff[1, 2]*kk + aff[1, 3]
    wz = aff[2, 0]*ii + aff[2, 1]*jj + aff[2, 2]*kk + aff[2, 3]

    # Map back to source voxel index
    si = src_inv[0, 0]*wx + src_inv[0, 1]*wy + src_inv[0, 2]*wz + src_inv[0, 3]
    sj = src_inv[1, 0]*wx + src_inv[1, 1]*wy + src_inv[1, 2]*wz + src_inv[1, 3]
    sk = src_inv[2, 0]*wx + src_inv[2, 1]*wy + src_inv[2, 2]*wz + src_inv[2, 3]

    coords = np.stack([si, sj, sk], axis=0)
    out = map_coordinates(src_arr, coords, order=0, mode="constant", cval=0)
    return out.astype(np.int32)


# ── Display helpers (matches visualize_qc.py) ────────────────────────────────

def _window(ct: np.ndarray) -> np.ndarray:
    return np.clip((ct - _HU_MIN) / (_HU_MAX - _HU_MIN), 0.0, 1.0)


def _overlay_color(base_rgb: np.ndarray, mask_2d: np.ndarray,
                   value: int, color_rgba: Tuple[float, float, float, float]
                   ) -> np.ndarray:
    out = base_rgb.copy()
    m = (mask_2d == value)
    if not m.any():
        return out
    r, g, b, a = color_rgba
    out[m, 0] = out[m, 0] * (1 - a) + r * a
    out[m, 1] = out[m, 1] * (1 - a) + g * a
    out[m, 2] = out[m, 2] * (1 - a) + b * a
    return np.clip(out, 0.0, 1.0)


def _safe_slice(arr: np.ndarray, dim: int, idx: int) -> np.ndarray:
    clamped = int(np.clip(idx, 0, arr.shape[dim] - 1))
    s = [slice(None)] * arr.ndim
    s[dim] = clamped
    return arr[tuple(s)]


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    """Match visualize_qc.py's display orientation exactly.

    PIR: axis 0 = P, axis 1 = I, axis 2 = R
    dim=0 (coronal):  slice (I, R). Row 0 = I=0 = superior. OK as-is.
    dim=1 (axial):    slice (P, R). Row 0 = P=0 = anterior. OK as-is.
    dim=2 (sagittal): slice (P, I). Transpose so row=I, col=P.
    """
    return arr2d.T if dim == 2 else arr2d


# ── Manifest loading ─────────────────────────────────────────────────────────

def _load_pairs_from_manifest(
    manifest_path: Path, tokens: List[str],
    nifti_dir: Optional[Path], hf_export: Optional[Path],
) -> List[Dict]:
    """
    Returns a list of dicts: {token, ct_path, gt_path, label_scheme}
    where label_scheme is "placed" (1=sac, 2=Lh, 3=Rh) or "exported"
    (7=sac, 8=Lh, 9=Rh).
    """
    doc = json.loads(manifest_path.read_text())

    # Auto-detect format
    if isinstance(doc, list):
        # hf_export manifest: list of records
        records = doc
        is_hf = True
    elif isinstance(doc, dict) and "cases" in doc:
        # placed_manifest.json (with or without orientation fix)
        records = doc["cases"]
        is_hf = False
    elif isinstance(doc, dict) and "records" in doc:
        records = doc["records"]
        is_hf = True
    else:
        raise ValueError(f"unrecognized manifest format: {manifest_path}")

    pairs: List[Dict] = []
    token_set = set(tokens)

    for rec in records:
        if is_hf:
            tok = str(rec.get("token", ""))
        else:
            tok = str(rec.get("patient_token", ""))
        if tok not in token_set:
            continue

        if is_hf:
            if hf_export is None:
                raise ValueError("--hf_export required for hf manifest mode")
            ct_path  = hf_export / rec["ct_file"]
            gt_path  = hf_export / rec["label_file"]
            label_scheme = "exported"
            cfg = rec.get("config", "fused")
        else:
            if nifti_dir is None:
                raise ValueError("--nifti_dir required for placed-manifest mode")
            pv = rec.get("pelvic") or {}
            sp = rec.get("spine")  or {}
            # Prefer the orientation-fixed CT/mask paths if present
            if "ct_nifti" in pv and pv["ct_nifti"]:
                ct_path = Path(pv["ct_nifti"])
            elif pv.get("series_uid"):
                ct_path = nifti_dir / f"{pv['series_uid']}.nii.gz"
            elif "ct_nifti" in sp and sp["ct_nifti"]:
                ct_path = Path(sp["ct_nifti"])
            elif sp.get("series_uid"):
                ct_path = nifti_dir / f"{sp['series_uid']}.nii.gz"
            else:
                log.warning("token=%s: no CT path resolvable", tok); continue

            gt = pv.get("placed")
            if not gt:
                log.warning("token=%s: no pelvic.placed mask", tok); continue
            gt_path = Path(gt)
            label_scheme = "placed"
            cfg = rec.get("match_type", "fused")

        if not ct_path.exists():
            log.warning("token=%s: CT not found: %s", tok, ct_path); continue
        if not gt_path.exists():
            log.warning("token=%s: GT not found: %s", tok, gt_path); continue

        pairs.append({
            "token":        tok,
            "config":       cfg,
            "ct_path":      ct_path,
            "gt_path":      gt_path,
            "label_scheme": label_scheme,
        })

    found = {p["token"] for p in pairs}
    missing = sorted(token_set - found)
    if missing:
        log.warning("Tokens not found in manifest: %s", missing)
    return pairs


def _find_ts_pred(token: str, config: str, ts_glob: str) -> Optional[Path]:
    """Find latest TS prediction for token/config."""
    pattern = ts_glob.format(token=token, config=config)
    candidates = sorted(glob(pattern))
    return Path(candidates[-1]) if candidates else None


# ── Per-pair figure ──────────────────────────────────────────────────────────

def _render_pair(pair: Dict, ts_glob: str, out_dir: Path,
                 audit_csv: Optional[Path] = None) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tok    = pair["token"]
    cfg    = pair["config"]
    scheme = pair["label_scheme"]

    # Label values to show
    if scheme == "placed":
        gt_left, gt_right = 2, 3
        scheme_desc = "placed (2=L, 3=R)"
    else:  # exported
        gt_left, gt_right = 8, 9
        scheme_desc = "exported (8=L, 9=R)"

    # Load CT + GT in PIR
    try:
        ct_pir, ct_aff_pir = _load_pir(pair["ct_path"])
    except Exception as e:
        log.error("token=%s: CT load failed: %s", tok, e); return False

    try:
        gt_pir_native, gt_aff_pir = _load_pir_label(pair["gt_path"])
    except Exception as e:
        log.error("token=%s: GT load failed: %s", tok, e); return False

    # Resample GT onto CT's PIR grid if shapes differ. Even if shapes match,
    # they share the same affine in PIR after reorientation, so we only need
    # to handle the shape mismatch case explicitly.
    if gt_pir_native.shape != ct_pir.shape:
        gt_pir = _resample_to_pir_grid(pair["gt_path"], ct_pir, ct_aff_pir)
    else:
        gt_pir = gt_pir_native

    # TS prediction
    ts_path = _find_ts_pred(tok, cfg, ts_glob)
    ts_pir: Optional[np.ndarray] = None
    if ts_path is not None:
        try:
            ts_pir = _resample_to_pir_grid(ts_path, ct_pir, ct_aff_pir)
        except Exception as e:
            log.warning("token=%s: TS resample failed: %s", tok, e)

    # Pick axial slice (dim=1) with most hip activity in GT or TS
    hip_activity = (gt_pir == gt_left) | (gt_pir == gt_right)
    if ts_pir is not None:
        hip_activity = hip_activity | (ts_pir == 77) | (ts_pir == 78)
    if not hip_activity.any():
        log.warning("token=%s: no hip voxels in GT or TS", tok); return False
    j_proj = hip_activity.sum(axis=(0, 2))  # along axis 1 (I)
    j_idx = int(np.argmax(j_proj))

    ct_sl = _display_slice(_safe_slice(ct_pir, 1, j_idx), 1)
    gt_sl = _display_slice(_safe_slice(gt_pir, 1, j_idx), 1)
    if ts_pir is not None:
        ts_sl = _display_slice(_safe_slice(ts_pir, 1, j_idx), 1)
    else:
        ts_sl = None

    # Audit values for title
    audit_str = ""
    if audit_csv is not None and audit_csv.exists():
        import csv
        with open(audit_csv) as f:
            for row in csv.DictReader(f):
                if str(row.get("token")) == tok and row.get("config") == cfg:
                    audit_str = (
                        f"  audit: {row.get('verdict','?'):>7}  "
                        f"no-swap={float(row.get('noswap_L', 0) or 0):.2f}/"
                        f"{float(row.get('noswap_R', 0) or 0):.2f}  "
                        f"swap={float(row.get('swap_L', 0) or 0):.2f}/"
                        f"{float(row.get('swap_R', 0) or 0):.2f}"
                    )
                    break

    # Build figure
    ct_win = _window(ct_sl)
    base = np.stack([ct_win, ct_win, ct_win], axis=-1).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#111111")

    for a in ax:
        a.set_facecolor("#111111")
        a.imshow(base, aspect="equal", interpolation="nearest")
        a.axis("off")
        # PIR axial (P, R): col 0 = patient-LEFT, col N = patient-RIGHT
        # (NEUROLOGICAL convention -- matches visualize_qc.py)
        a.text(0.02, 0.5, "L", transform=a.transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")
        a.text(0.96, 0.5, "R", transform=a.transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")
        a.text(0.5, 0.02, "(neurologic conv: viewer-LEFT = patient-LEFT)",
               transform=a.transAxes, fontsize=8, color="#888888",
               ha="center", va="bottom")

    ax[0].set_title(f"Token {tok} / {cfg}  CT (axial j={j_idx})",
                    color="#dddddd", fontsize=11)

    # GT panel
    gt_rgb = base.copy()
    gt_rgb = _overlay_color(gt_rgb, gt_sl, gt_left,  _LEFT_HIP_COLOR)
    gt_rgb = _overlay_color(gt_rgb, gt_sl, gt_right, _RIGHT_HIP_COLOR)
    ax[1].clear()
    ax[1].set_facecolor("#111111")
    ax[1].imshow(gt_rgb, aspect="equal", interpolation="nearest")
    ax[1].axis("off")
    ax[1].text(0.02, 0.5, "L", transform=ax[1].transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")
    ax[1].text(0.96, 0.5, "R", transform=ax[1].transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")
    ax[1].set_title(f"GT  yellow=label_{gt_left}(left_hip)  green=label_{gt_right}(right_hip)\n"
                    f"scheme: {scheme_desc}", color="#dddddd", fontsize=10)

    # TS panel
    ax[2].clear()
    ax[2].set_facecolor("#111111")
    if ts_sl is not None:
        ts_rgb = base.copy()
        ts_rgb = _overlay_color(ts_rgb, ts_sl, 77, _LEFT_HIP_COLOR)
        ts_rgb = _overlay_color(ts_rgb, ts_sl, 78, _RIGHT_HIP_COLOR)
        ax[2].imshow(ts_rgb, aspect="equal", interpolation="nearest")
        ax[2].set_title(f"TS  yellow=77(hip_left)  green=78(hip_right){audit_str}",
                        color="#dddddd", fontsize=10)
    else:
        ax[2].imshow(base, aspect="equal", interpolation="nearest")
        ax[2].text(0.5, 0.5, "no TS prediction", transform=ax[2].transAxes,
                   color="#666666", fontsize=12, ha="center", va="center")
        ax[2].set_title(f"TS (missing){audit_str}", color="#dddddd", fontsize=10)
    ax[2].axis("off")
    ax[2].text(0.02, 0.5, "L", transform=ax[2].transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")
    ax[2].text(0.96, 0.5, "R", transform=ax[2].transAxes, fontsize=28,
               fontweight="bold", color="yellow", va="center")

    out_path = out_dir / f"qc_audit_{tok}_{cfg}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("token=%s/%s -> %s", tok, cfg, out_path)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--tokens",   required=True, type=str,
                    help="Comma-separated tokens to render (e.g. 3,460,480,616)")
    ap.add_argument("--nifti_dir", type=Path, default=None,
                    help="CT directory (for placed-manifest mode)")
    ap.add_argument("--hf_export", type=Path, default=None,
                    help="HF export root (for hf-manifest mode)")
    ap.add_argument("--ts_glob",  type=str,
                    default="results/totalseg_bench_*/ts_preds/{token}_{config}/segmentation.nii.gz",
                    help="Glob pattern for TS predictions; {token} and {config} substituted")
    ap.add_argument("--audit_csv", type=Path, default=None,
                    help="Optional audit CSV to annotate Dice values in titles")
    ap.add_argument("--out_dir",  type=Path, required=True)
    args = ap.parse_args()

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    if not tokens:
        log.error("No tokens specified"); return

    pairs = _load_pairs_from_manifest(
        args.manifest, tokens, args.nifti_dir, args.hf_export,
    )
    if not pairs:
        log.error("No matching pairs found"); return

    log.info("Rendering %d pairs", len(pairs))
    n_ok = 0
    for pair in pairs:
        if _render_pair(pair, args.ts_glob, args.out_dir, args.audit_csv):
            n_ok += 1
    log.info("Done: %d/%d rendered to %s", n_ok, len(pairs), args.out_dir)


if __name__ == "__main__":
    main()
