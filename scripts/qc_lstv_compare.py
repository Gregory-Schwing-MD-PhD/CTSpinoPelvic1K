"""
qc_lstv_compare.py — 5-panel LSTV-case QC: CT | GT spine | GT pelvic | Fused | TS

For each LSTV-flagged token, render one PNG with five panels.

Layout
------
  Panel 1: CT only (windowed, no overlay) — uses the spine-side CT for
           'separate' tokens (since panels 2 and 5 are also spine-side).
           Uses the single CT for 'fused' tokens.
  Panel 2: GT spine labels (lumbar 1..6 + sacrum from spine annotation),
           overlaid on whichever CT contains the spine annotations.
  Panel 3: GT pelvic labels (sacrum + hips), overlaid on whichever CT
           contains the pelvic annotations. For 'separate' tokens this
           is the PELVIC-side CT, not the spine CT — pelvic_native and
           spine_only acquisitions in the source datasets are
           independent scans whose world coordinates do not overlap.
           Trying to resample the pelvic mask onto the spine CT grid
           produces an empty panel because mask voxels fall outside
           the spine CT's bounding box.
  Panel 4: GT fused (BOTH spine + pelvic on a single CT) — only meaningful
           for match_type='fused'. For 'separate' tokens the panel stays
           blank with a note explaining why fusion is impossible.
  Panel 5: TS prediction, on the spine-side CT.

Each panel picks its own coronal slice through its own data, so panel 3
slicing through the pelvic acquisition is independent of panel 2 slicing
through the spine acquisition. The figure stays a single file but
shows two physically distinct slabs side by side when match_type='separate'.

Per-class Dice
--------------
For 'fused' tokens, computed in the shared frame on the full volume.
For 'separate' tokens, computed in EACH side's own frame:
  - L1..L5 vs TS using the spine CT + spine label + spine TS pred
  - sacrum/hip_L/hip_R vs TS using the pelvic CT + pelvic label + pelvic TS pred
The dice line at the bottom shows both, separated by " | ".

Selection
---------
Default renders ALL non-normal LSTV tokens. Override with --tokens or --classes.

Usage
-----
  python scripts/qc_lstv_compare.py \\
      --manifest /data/hf_export/manifest.json \\
      --hf_export /data/hf_export \\
      --ts_glob 'results/totalseg_bench_*/ts_preds/{token}_{config}/segmentation.nii.gz' \\
      --out_dir /data/qc_lstv
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qc_lstv_compare")

_HU_MIN, _HU_MAX = -150, 700

GT_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
            7: "sacrum", 8: "hip_L", 9: "hip_R"}

SPINE_GT_LABELS  = {1, 2, 3, 4, 5, 6}
PELVIC_GT_LABELS = {7, 8, 9}
SACRUM_LABEL     = 7

TS_TO_GT_VERT = {31: 1, 30: 2, 29: 3, 28: 4, 27: 5}
TS_NAMES_VERT = {31: "L1", 30: "L2", 29: "L3", 28: "L4", 27: "L5"}
TS_SAC = 25
TS_HIP_L, TS_HIP_R = 77, 78

_GT_COLORS = {
    1: (0.20, 0.40, 0.95, 0.55),
    2: (0.10, 0.65, 0.95, 0.55),
    3: (0.10, 0.85, 0.55, 0.55),
    4: (0.85, 0.85, 0.10, 0.55),
    5: (0.95, 0.55, 0.10, 0.55),
    6: (0.95, 0.20, 0.20, 0.55),
    7: (0.85, 0.15, 0.85, 0.55),
    8: (0.95, 0.50, 0.10, 0.50),
    9: (0.95, 0.80, 0.05, 0.50),
}
_TS_VERT_COLORS = {ts_id: _GT_COLORS[gt_id] for ts_id, gt_id in TS_TO_GT_VERT.items()}
_TS_HIP_COLORS  = {TS_HIP_L: _GT_COLORS[8], TS_HIP_R: _GT_COLORS[9],
                   TS_SAC:   _GT_COLORS[7]}


# ── Reorientation ────────────────────────────────────────────────────────────

def _load_pir(path: Path, *, as_int: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff,
    )
    img = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm = ornt_transform(src_ornt, dst_ornt)
    if as_int:
        arr = np.asarray(img.dataobj).astype(np.int32)
        data = apply_orientation(arr, xfm).squeeze().astype(np.int32)
    else:
        data = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff = img.affine @ inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _resample_to_pir_grid(src_path: Path, ref_pir_data: np.ndarray,
                           ref_pir_affine: np.ndarray) -> np.ndarray:
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    src_img = nib.load(str(src_path))
    src_arr = np.asarray(src_img.dataobj).astype(np.int32)
    src_aff = src_img.affine.astype(np.float64)
    src_inv = np.linalg.inv(src_aff)

    s = ref_pir_data.shape
    ii, jj, kk = np.meshgrid(
        np.arange(s[0], dtype=np.float64),
        np.arange(s[1], dtype=np.float64),
        np.arange(s[2], dtype=np.float64),
        indexing="ij",
    )
    aff = ref_pir_affine.astype(np.float64)
    wx = aff[0, 0]*ii + aff[0, 1]*jj + aff[0, 2]*kk + aff[0, 3]
    wy = aff[1, 0]*ii + aff[1, 1]*jj + aff[1, 2]*kk + aff[1, 3]
    wz = aff[2, 0]*ii + aff[2, 1]*jj + aff[2, 2]*kk + aff[2, 3]

    si = src_inv[0, 0]*wx + src_inv[0, 1]*wy + src_inv[0, 2]*wz + src_inv[0, 3]
    sj = src_inv[1, 0]*wx + src_inv[1, 1]*wy + src_inv[1, 2]*wz + src_inv[1, 3]
    sk = src_inv[2, 0]*wx + src_inv[2, 1]*wy + src_inv[2, 2]*wz + src_inv[2, 3]

    coords = np.stack([si, sj, sk], axis=0)
    return map_coordinates(src_arr, coords, order=0,
                           mode="constant", cval=0).astype(np.int32)


# ── Display helpers ──────────────────────────────────────────────────────────

def _window(arr):
    return np.clip((arr - _HU_MIN) / (_HU_MAX - _HU_MIN), 0, 1)


def _overlay(bg_rgb, label_2d, color_map):
    out = bg_rgb.copy()
    for lid, (r, g, b, a) in color_map.items():
        m = (label_2d == lid)
        if not m.any():
            continue
        out[m, 0] = out[m, 0] * (1 - a) + r * a
        out[m, 1] = out[m, 1] * (1 - a) + g * a
        out[m, 2] = out[m, 2] * (1 - a) + b * a
    return np.clip(out, 0, 1)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    sz    = int(a.sum()) + int(b.sum())
    if sz == 0:
        return float("nan")
    return 2 * inter / sz


def _decompose_fused(lbl_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    spine  = np.where(np.isin(lbl_arr, list(SPINE_GT_LABELS)),  lbl_arr, 0)
    pelvic = np.where(np.isin(lbl_arr, list(PELVIC_GT_LABELS)), lbl_arr, 0)
    return spine.astype(np.int32), pelvic.astype(np.int32)


def _coronal_slice_idx(activity: np.ndarray) -> Optional[int]:
    """Pick PIR axis 0 (P) index with most activity. Returns None if none."""
    if not activity.any():
        return None
    p_proj = activity.sum(axis=(1, 2))
    return int(np.argmax(p_proj))


def _pad_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    """Center-pad a 2D RGB slice vertically (in axis-0) with black.
    Used so panels with different (I, R) shapes line up cleanly in a row."""
    h, w, _ = rgb.shape
    if h >= target_h:
        return rgb
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    out = np.zeros((target_h, w, 3), dtype=rgb.dtype)
    out[pad_top:pad_top + h] = rgb
    return out


# ── Side-loader: load CT + label + TS for one side (spine or pelvic) ────────

def _load_side(ct_path: Path, lbl_path: Path,
                ts_path: Optional[Path],
                keep_labels: set) -> Optional[Dict]:
    """
    Load CT + label + (optional) TS prediction for ONE side of a separate case
    or for the unified frame of a fused case. All arrays come back in PIR
    order on the CT's grid.

    keep_labels filters the label volume to just the labels of interest
    (spine vertebrae or pelvic sacrum+hips). Other labels (e.g., ignore=10)
    are zeroed.
    """
    if not ct_path.exists() or not lbl_path.exists():
        return None
    try:
        ct_pir, ct_aff = _load_pir(ct_path, as_int=False)
        lbl_native, _  = _load_pir(lbl_path, as_int=True)
        if lbl_native.shape == ct_pir.shape:
            lbl_pir = lbl_native
        else:
            lbl_pir = _resample_to_pir_grid(lbl_path, ct_pir, ct_aff)
        lbl_filtered = np.where(np.isin(lbl_pir, list(keep_labels)),
                                 lbl_pir, 0).astype(np.int32)
    except Exception as e:
        log.warning("load_side failed for %s + %s: %s", ct_path.name, lbl_path.name, e)
        return None

    ts_pir: Optional[np.ndarray] = None
    if ts_path is not None and ts_path.exists():
        try:
            ts_pir = _resample_to_pir_grid(ts_path, ct_pir, ct_aff)
        except Exception as e:
            log.warning("ts resample failed for %s: %s", ts_path.name, e)

    return dict(
        ct=ct_pir, ct_aff=ct_aff,
        label=lbl_filtered, label_full=lbl_pir,
        ts=ts_pir,
    )


def _find_ts(ts_glob: str, token: str, config: str) -> Optional[Path]:
    pattern = ts_glob.format(token=token, config=config)
    paths = sorted(glob(pattern))
    return Path(paths[-1]) if paths else None


# ── Per-token render ─────────────────────────────────────────────────────────

def _render_token(token: str,
                   cfg_recs: Dict[str, dict],
                   hf_root: Path,
                   ts_glob: str,
                   out_dir: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not cfg_recs:
        log.warning("token=%s: no records", token); return False

    # Determine match_type
    if "fused" in cfg_recs:
        match_type = "fused"
    elif "spine_only" in cfg_recs and "pelvic_native" in cfg_recs:
        match_type = "separate"
    elif "spine_only" in cfg_recs:
        match_type = "spine_only"
    elif "pelvic_native" in cfg_recs:
        match_type = "pelvic_only"
    else:
        log.warning("token=%s: unknown config set %s", token, list(cfg_recs))
        return False

    rep = (cfg_recs.get("fused") or cfg_recs.get("spine_only")
           or cfg_recs.get("pelvic_native"))
    lstv   = rep.get("lstv_label", "")
    has_l6 = bool(rep.get("has_l6", False))

    # Per-side data ----------------------------------------------------------
    spine_side  : Optional[Dict] = None
    pelvic_side : Optional[Dict] = None
    fused_side  : Optional[Dict] = None

    if match_type == "fused":
        rec = cfg_recs["fused"]
        ct_path  = hf_root / rec["ct_file"]
        lbl_path = hf_root / rec["label_file"]
        ts_path  = _find_ts(ts_glob, token, "fused")
        try:
            ct_pir, ct_aff = _load_pir(ct_path, as_int=False)
            lbl_native, _  = _load_pir(lbl_path, as_int=True)
            if lbl_native.shape == ct_pir.shape:
                lbl_pir = lbl_native
            else:
                lbl_pir = _resample_to_pir_grid(lbl_path, ct_pir, ct_aff)
            sp_arr, pv_arr = _decompose_fused(lbl_pir)
            ts_pir = None
            if ts_path is not None and ts_path.exists():
                ts_pir = _resample_to_pir_grid(ts_path, ct_pir, ct_aff)
            shared = dict(ct=ct_pir, ct_aff=ct_aff,
                          label_full=lbl_pir, ts=ts_pir)
            spine_side  = dict(shared, label=sp_arr)
            pelvic_side = dict(shared, label=pv_arr)
            fused_side  = dict(shared, label=lbl_pir.astype(np.int32))
        except Exception as e:
            log.warning("token=%s fused load failed: %s", token, e)
            return False
    else:
        # Spine side
        if "spine_only" in cfg_recs:
            rec = cfg_recs["spine_only"]
            spine_side = _load_side(
                ct_path=hf_root / rec["ct_file"],
                lbl_path=hf_root / rec["label_file"],
                ts_path=_find_ts(ts_glob, token, "spine_only"),
                keep_labels=SPINE_GT_LABELS | {SACRUM_LABEL},
            )
        # Pelvic side
        if "pelvic_native" in cfg_recs:
            rec = cfg_recs["pelvic_native"]
            pelvic_side = _load_side(
                ct_path=hf_root / rec["ct_file"],
                lbl_path=hf_root / rec["label_file"],
                ts_path=_find_ts(ts_glob, token, "pelvic_native"),
                keep_labels=PELVIC_GT_LABELS,
            )

    if spine_side is None and pelvic_side is None:
        log.warning("token=%s: neither side loadable", token); return False

    # Per-class Dice (3D, in each side's own frame) --------------------------
    dice: Dict[str, float] = {}
    if spine_side is not None and spine_side.get("ts") is not None:
        sp_lbl = spine_side["label"]; ts_arr = spine_side["ts"]
        for ts_id, gt_id in TS_TO_GT_VERT.items():
            d = _dice(ts_arr == ts_id, sp_lbl == gt_id)
            if not np.isnan(d):
                dice[GT_NAMES[gt_id]] = d
        if (sp_lbl == 6).any():
            dice["L6"] = float("nan")
    if pelvic_side is not None and pelvic_side.get("ts") is not None:
        pv_lbl = pelvic_side["label"]; ts_arr = pelvic_side["ts"]
        d_sac = _dice(ts_arr == TS_SAC, pv_lbl == SACRUM_LABEL)
        if not np.isnan(d_sac):
            dice["sacrum"] = d_sac
        for ts_id, gt_id in [(TS_HIP_L, 8), (TS_HIP_R, 9)]:
            d = _dice(ts_arr == ts_id, pv_lbl == gt_id)
            if not np.isnan(d):
                dice[GT_NAMES[gt_id]] = d

    # Per-panel slicing ------------------------------------------------------
    # Each panel slices its own data through the coronal axis (PIR axis 0).
    # For 'fused' all panels share the CT; for 'separate' each side slices
    # its own CT.

    def _slice_2d(side: Optional[Dict], extra_activity: Optional[np.ndarray] = None
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Returns (ct_2d, lbl_2d, ts_2d_or_None) or None."""
        if side is None:
            return None
        ct, lbl, ts = side["ct"], side["label"], side.get("ts")
        activity = (lbl > 0)
        if extra_activity is not None:
            activity |= extra_activity
        if ts is not None:
            activity |= (((ts >= 27) & (ts <= 31))
                         | (ts == TS_SAC) | (ts == TS_HIP_L) | (ts == TS_HIP_R))
        idx = _coronal_slice_idx(activity)
        if idx is None:
            idx = ct.shape[0] // 2
        return (ct[idx], lbl[idx], ts[idx] if ts is not None else None, idx)

    sp_sl  = _slice_2d(spine_side)
    pv_sl  = _slice_2d(pelvic_side)
    fu_sl  = _slice_2d(fused_side) if fused_side is not None else None

    # Build figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(28, 9))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d"); ax.axis("off")

    # Choose which side anchors panel 1 (the "CT only" panel).
    # For fused: shared CT. For separate: spine if it exists, else pelvic.
    if match_type == "fused" and fu_sl is not None:
        anchor = fu_sl; anchor_name = "fused"
    elif sp_sl is not None:
        anchor = sp_sl; anchor_name = "spine-side"
    else:
        anchor = pv_sl; anchor_name = "pelvic-side"

    bg_anchor = _window(anchor[0])
    bg_anchor_rgb = np.stack([bg_anchor]*3, axis=-1).astype(np.float32)

    axes[0].imshow(bg_anchor_rgb, aspect="equal", interpolation="nearest")
    axes[0].set_title(f"CT  ({anchor_name}, coronal P={anchor[3]})",
                      color="#dddddd", fontsize=11)

    # Panel 2: GT spine
    if sp_sl is not None:
        bg = _window(sp_sl[0])
        bg_rgb = np.stack([bg]*3, axis=-1).astype(np.float32)
        axes[1].imshow(_overlay(bg_rgb, sp_sl[1], _GT_COLORS),
                       aspect="equal", interpolation="nearest")
        axes[1].set_title(f"GT spine (vertebrae)  P={sp_sl[3]}",
                          color="#dddddd", fontsize=11)
    else:
        axes[1].imshow(bg_anchor_rgb, aspect="equal", interpolation="nearest")
        axes[1].set_title("GT spine — N/A (no spine annotation)",
                          color="#888888", fontsize=11)

    # Panel 3: GT pelvic — uses PELVIC-side CT for separate cases
    if pv_sl is not None:
        bg = _window(pv_sl[0])
        bg_rgb = np.stack([bg]*3, axis=-1).astype(np.float32)
        axes[2].imshow(_overlay(bg_rgb, pv_sl[1], _GT_COLORS),
                       aspect="equal", interpolation="nearest")
        ct_label = "fused" if match_type == "fused" else "pelvic-side CT"
        axes[2].set_title(f"GT pelvic  ({ct_label}, P={pv_sl[3]})",
                          color="#dddddd", fontsize=11)
    else:
        axes[2].imshow(bg_anchor_rgb, aspect="equal", interpolation="nearest")
        axes[2].set_title("GT pelvic — N/A (no pelvic annotation)",
                          color="#888888", fontsize=11)

    # Panel 4: Fused (only meaningful when match_type=='fused')
    if match_type == "fused" and fu_sl is not None:
        bg = _window(fu_sl[0])
        bg_rgb = np.stack([bg]*3, axis=-1).astype(np.float32)
        axes[3].imshow(_overlay(bg_rgb, fu_sl[1], _GT_COLORS),
                       aspect="equal", interpolation="nearest")
        axes[3].set_title("GT fused (spine + pelvic)",
                          color="#dddddd", fontsize=11)
    else:
        axes[3].imshow(bg_anchor_rgb, aspect="equal", interpolation="nearest")
        msg = {
            "separate":     "(separate — different CTs, no shared frame)",
            "spine_only":   "(spine_only — no pelvic)",
            "pelvic_only":  "(pelvic_only — no spine)",
        }.get(match_type, f"({match_type})")
        axes[3].set_title(f"GT fused {msg}",
                          color="#888888", fontsize=11)

    # Panel 5: TS prediction
    # Prefer spine-side TS for the panel since it shows vertebrae;
    # if only pelvic-side exists, show that instead.
    ts_panel_side = None
    ts_panel_label = None
    if sp_sl is not None and sp_sl[2] is not None:
        ts_panel_side = sp_sl
        ts_panel_label = "spine_only" if match_type != "fused" else "fused"
    elif pv_sl is not None and pv_sl[2] is not None:
        ts_panel_side = pv_sl
        ts_panel_label = "pelvic_native"
    if ts_panel_side is not None:
        bg = _window(ts_panel_side[0])
        bg_rgb = np.stack([bg]*3, axis=-1).astype(np.float32)
        ts_2d = ts_panel_side[2]
        axes[4].imshow(_overlay(bg_rgb, ts_2d,
                                 {**_TS_VERT_COLORS, **_TS_HIP_COLORS}),
                       aspect="equal", interpolation="nearest")
        axes[4].set_title(f"TS prediction ({ts_panel_label})",
                          color="#dddddd", fontsize=11)
    else:
        axes[4].imshow(bg_anchor_rgb, aspect="equal", interpolation="nearest")
        axes[4].set_title("TS prediction — N/A",
                          color="#888888", fontsize=11)

    title = (f"Token {token}  match_type={match_type}  "
             f"LSTV={lstv}  has_L6={has_l6}")
    fig.suptitle(title, color="#ffffff", fontsize=12, y=0.995)

    # Per-class Dice annotation
    dice_lines = []
    for k in ["L1", "L2", "L3", "L4", "L5", "L6", "sacrum", "hip_L", "hip_R"]:
        if k not in dice:
            continue
        v = dice[k]
        if np.isnan(v):
            dice_lines.append(f"{k}: GT-only (no TS class)")
        else:
            arrow = "" if v >= 0.5 else "  ⬇"
            dice_lines.append(f"{k}: {v:.3f}{arrow}")
    if dice_lines:
        fig.text(0.5, 0.04, "  |  ".join(dice_lines),
                 ha="center", color="#cccccc", fontsize=10, family="monospace")

    # Legend (only labels actually present in any panel)
    present_ids = set()
    for arr_pair in (sp_sl, pv_sl, fu_sl):
        if arr_pair is None: continue
        present_ids |= {int(v) for v in np.unique(arr_pair[1]) if v > 0}
    for side in (spine_side, pelvic_side):
        if side is None: continue
        ts = side.get("ts")
        if ts is None: continue
        for ts_id, gt_id in TS_TO_GT_VERT.items():
            if (ts == ts_id).any():
                present_ids.add(gt_id)
        if (ts == TS_SAC  ).any(): present_ids.add(7)
        if (ts == TS_HIP_L).any(): present_ids.add(8)
        if (ts == TS_HIP_R).any(): present_ids.add(9)
    legend_handles = []
    for gt_id in sorted(present_ids):
        if gt_id not in _GT_COLORS: continue
        r, g, b, _ = _GT_COLORS[gt_id]
        legend_handles.append(mpatches.Patch(facecolor=(r, g, b),
                                              label=GT_NAMES[gt_id]))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 9), fontsize=9,
                   bbox_to_anchor=(0.5, 0.0),
                   facecolor="#222222", labelcolor="#dddddd",
                   edgecolor="#444444")

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_lstv = (lstv or "unknown").lower().replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"lstv_{safe_lstv}_{token}_{match_type}.png"
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    dice_summary = "/".join(
        f"{dice.get(k, float('nan')):.2f}" for k in ["L1", "L2", "L3", "L4", "L5"]
    )
    log.info("rendered %s  match=%s  L1-L5=%s",
             out_path.name, match_type, dice_summary)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--hf_export", required=True, type=Path)
    ap.add_argument("--ts_glob", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--classes", default="all",
                    help="Comma-separated lstv_label classes ('all' = every "
                         "non-normal LSTV case). Examples: 'sacralization', "
                         "'lumbarization', 'sacralization,lumbarization'.")
    ap.add_argument("--n_per_class", type=int, default=0,
                    help="Limit per class (0 = unlimited)")
    ap.add_argument("--tokens", default="",
                    help="Override class selection: comma-separated token list")
    args = ap.parse_args()

    doc = json.loads(args.manifest.read_text())
    records = doc if isinstance(doc, list) else doc.get("records", [])
    log.info("Manifest has %d records", len(records))

    token_to_recs: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in records:
        token_to_recs[str(r.get("token"))][r.get("config", "?")] = r

    if args.tokens.strip():
        wanted = {t.strip() for t in args.tokens.split(",") if t.strip()}
        selected = [t for t in token_to_recs if t in wanted]
        log.info("Token filter: %d tokens selected", len(selected))
    else:
        if args.classes.strip().lower() == "all":
            wanted_classes = None
        else:
            wanted_classes = {c.strip().lower()
                              for c in args.classes.split(",") if c.strip()}
        buckets: Dict[str, List[str]] = defaultdict(list)
        for tok, cfg_recs in token_to_recs.items():
            for r in cfg_recs.values():
                lbl = str(r.get("lstv_label", "")).strip().lower()
                if not lbl or lbl == "normal": continue
                if lbl in ("sacralization", "semi_sacralization", "semi-sacralization"):
                    bucket = "sacralization"
                elif lbl == "lumbarization":
                    bucket = "lumbarization"
                else:
                    bucket = lbl
                if wanted_classes is None or bucket in wanted_classes:
                    buckets[bucket].append(tok)
                    break
        for k in list(buckets):
            buckets[k] = sorted(set(buckets[k]))
        selected = []
        for k, lst in sorted(buckets.items()):
            log.info("  class=%-15s : %d tokens", k, len(lst))
            if args.n_per_class > 0:
                selected.extend(lst[:args.n_per_class])
            else:
                selected.extend(lst)
        seen = set(); uniq = []
        for t in selected:
            if t not in seen: seen.add(t); uniq.append(t)
        selected = uniq

    log.info("Rendering %d tokens", len(selected))
    n_ok = 0
    for tok in selected:
        if _render_token(tok, token_to_recs[tok], args.hf_export,
                          args.ts_glob, args.out_dir):
            n_ok += 1
    log.info("Done: %d/%d rendered to %s", n_ok, len(selected), args.out_dir)


if __name__ == "__main__":
    main()
