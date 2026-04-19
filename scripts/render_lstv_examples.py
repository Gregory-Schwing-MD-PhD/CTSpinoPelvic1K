#!/usr/bin/env python3
"""
render_lstv_examples.py — Publication-quality 3D LSTV panel renders.

Produces a 4-panel composite (Normal / Sacralization-morph / Sacralization-count
/ Lumbarization) showing the same anatomical selection under:

  --source gt    raw placed label volumes (placed/spine + placed/{fused,pelvic_native})
  --source ts    TotalSegmentator predictions, remapped to the unified label scheme

Use both to produce GT vs TS side-by-side figures for the paper (Figure: TS limitations
on LSTV morphology).

Inputs (canonical)
------------------
  --manifest   data/placed/placed_manifest.json     (source of patient_token → LSTV mapping)
  --spine_dir  data/placed/spine                    (GT mode only)
  --fused_dir  data/placed/fused                    (GT mode only)
  --pelv_dir   data/placed/pelvic_native            (GT mode only)
  --ts_pred_dir  data/results/totalseg_bench_*/ts_preds
                                                    (TS mode only)
  --hf_export_dir  data/hf_export                   (for case lookup in TS mode)

Output
------
  <out_dir>/A_normal.jpg
  <out_dir>/B_sacralization_morph.jpg
  <out_dir>/C_sacralization_count.jpg
  <out_dir>/D_lumbarization.jpg
  <out_dir>/LSTV_panel_4x1.jpg
"""

import argparse
import json
import warnings
import tempfile
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import marching_cubes

warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────────

SPINE_COLORS = {
    20: np.array([0.08, 0.38, 1.00]),   # L1
    21: np.array([0.00, 0.68, 1.00]),   # L2
    22: np.array([0.00, 0.88, 0.80]),   # L3
    23: np.array([0.05, 0.85, 0.30]),   # L4
    24: np.array([1.00, 0.82, 0.00]),   # L5
    25: np.array([1.00, 0.38, 0.00]),   # L6 / LSTV
    26: np.array([0.95, 0.08, 0.08]),   # spine sacrum
}
PELV_COLORS = {
    1:  np.array([0.82, 0.00, 0.95]),   # pelvic sacrum
    2:  np.array([0.68, 0.68, 0.82]),   # L hip
    3:  np.array([0.65, 0.68, 0.80]),   # R hip
}
BG_COLOR = np.array([0.97, 0.97, 0.97])

# TS ML output → same (CTSpine1K-style) label codes we render
# so the TS mode can share the GT rendering pipeline unchanged.
TS_TO_SPINE = {31: 20, 30: 21, 29: 22, 28: 23, 27: 24}    # L1..L5
TS_TO_PELV  = {25: 1,  77: 2,  78: 3}                       # sacrum + hips
# (TS has no L6 / L6 == 25 slot; that's the whole point of showing TS figure.)

_LSTV_FIELDS = ("lstv_pelvic", "lstv_label", "lstv_qualifier",
                "lstv_type", "lstv_morph", "lstv")


def _token_matches(token, name):
    if not token:
        return False
    if token in name:
        return True
    try:
        t = int(token)
        for fmt in (f".{t:04d}.", f"_{t:04d}_", f".{t}.", f"_{t}_"):
            if fmt in name:
                return True
    except ValueError:
        pass
    return False


def _extract_lstv_label(case_dict, pelvic_file):
    for field in _LSTV_FIELDS:
        val = (case_dict.get(field, "") or "").strip().lower()
        if val:
            return val
    if pelvic_file:
        fname = Path(pelvic_file).name.lower()
        for canonical, tok in [
            ("sacralization",       "sacralization"),
            ("semi_sacralization",  "semi"),
            ("lumbarization",       "lumbarization"),
            ("normal",              "normal"),
        ]:
            if tok in fname:
                return canonical
    return ""


def load_nifti(path):
    img = nib.load(str(path))
    return (np.asarray(img.dataobj, dtype=np.int16),
            np.abs(img.header.get_zooms()[:3]),
            img.affine)


def crop_to_lumbosacral(vol, labels_present, margin_vox=8):
    mask = np.zeros_like(vol, dtype=bool)
    for lbl in labels_present:
        mask |= (vol == lbl)
    if not mask.any():
        return vol, (slice(None), slice(None), slice(None))
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin_vox, 0)
    hi = np.minimum(coords.max(0) + margin_vox + 1, np.array(vol.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return vol[sl], sl


def downsample(vol, factor=2):
    if factor <= 1:
        return vol
    from skimage.measure import block_reduce
    return block_reduce(vol, (factor, factor, factor), np.max).astype(vol.dtype)


def extract_surface(vol, label, step=1, smooth_iter=0):
    binary = (vol == label).astype(np.uint8)
    if binary.sum() < 8:
        return None, None
    if smooth_iter > 0:
        binary = (ndimage.gaussian_filter(binary.astype(float),
                                           sigma=smooth_iter) > 0.4).astype(np.uint8)
    try:
        verts, faces, _, _ = marching_cubes(binary, level=0.5, step_size=step,
                                             allow_degenerate=False)
        return verts.astype(np.float32), faces.astype(np.int32)
    except Exception:
        return None, None


def compute_face_shading(verts, faces, light_dir=None, ambient=0.28):
    if light_dir is None:
        light_dir = np.array([0.35, 0.50, 0.80], dtype=float)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-12)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return (ambient + (1.0 - ambient) * np.clip(normals @ light_dir, 0, 1)).astype(np.float32)


def laplacian_smooth(verts, faces, n_iter=4, relax=0.5):
    verts = verts.copy().astype(np.float64)
    for _ in range(n_iter):
        accum = np.zeros_like(verts)
        counts = np.zeros(len(verts), dtype=np.float64)
        for ci in range(3):
            a = faces[:, ci]; b = faces[:, (ci + 1) % 3]
            np.add.at(accum, a, verts[b]); np.add.at(accum, b, verts[a])
            np.add.at(counts, a, 1.);      np.add.at(counts, b, 1.)
        mask = counts > 0
        verts[mask] = (1 - relax) * verts[mask] + relax * accum[mask] / counts[mask, np.newaxis]
    return verts.astype(np.float32)


def faces_to_poly(verts, faces, shading, base_color, alpha=1.0):
    fc = (base_color[np.newaxis, :] * shading[:, np.newaxis]).clip(0, 1)
    return verts[faces], np.column_stack([fc, np.full(len(fc), alpha)])


# ── TS prediction loader (resamples + remaps to spine/pelv label codes) ─────

def _load_ts_as_rendering_volume(ts_path: Path, ref_label_path: Path):
    """Resample TS ML output to GT grid, split into spine & pelvic volumes."""
    import SimpleITK as sitk
    moving = sitk.ReadImage(str(ts_path), sitk.sitkInt32)
    fixed  = sitk.ReadImage(str(ref_label_path), sitk.sitkInt32)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(fixed); rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0); rs.SetTransform(sitk.Transform())
    resampled = rs.Execute(moving)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
        tmp = tf.name
    try:
        sitk.WriteImage(resampled, tmp)
        arr = np.asarray(nib.load(tmp).dataobj, dtype=np.int32)
        aff = nib.load(str(ref_label_path)).affine
    finally:
        try: os.unlink(tmp)
        except OSError: pass
    spine_vol = np.zeros_like(arr, dtype=np.int16)
    pelv_vol  = np.zeros_like(arr, dtype=np.int16)
    for ts, ct in TS_TO_SPINE.items():
        spine_vol[arr == ts] = ct
    for ts, ct in TS_TO_PELV.items():
        pelv_vol[arr == ts] = ct
    zooms = np.abs(np.diag(aff)[:3])
    return spine_vol, pelv_vol, zooms, aff


def render_case(spine_vol, pelv_vol, affine,
                out_path, elev=20, azim=-90, downsample_factor=2,
                figsize=(4.5, 4.5)):
    """Render one 3D case and save as JPG."""
    spine_labels_render = [20, 21, 22, 23, 24, 25, 26]
    pelv_labels_render  = [1] if pelv_vol is not None else []

    if pelv_vol is not None:
        if spine_vol.shape != pelv_vol.shape:
            mn = tuple(min(a, b) for a, b in zip(spine_vol.shape, pelv_vol.shape))
            spine_vol = spine_vol[:mn[0], :mn[1], :mn[2]]
            pelv_vol  = pelv_vol[:mn[0], :mn[1], :mn[2]]
        combined = spine_vol.copy()
        for lbl in pelv_labels_render:
            m = pelv_vol == lbl
            combined[m] = lbl + 100
        _, sl = crop_to_lumbosacral(combined,
                                     spine_labels_render + [l + 100 for l in pelv_labels_render],
                                     margin_vox=20)
    else:
        _, sl = crop_to_lumbosacral(spine_vol, spine_labels_render, margin_vox=15)

    crop_offset = np.array([sl[0].start or 0, sl[1].start or 0, sl[2].start or 0],
                            dtype=float)
    scale_affine = affine.copy()
    scale_affine[:3, 3] += affine[:3, :3] @ (crop_offset * downsample_factor)
    scale_affine[:3, :3] *= downsample_factor

    spine_crop = downsample(spine_vol[sl], downsample_factor)
    pelv_crop  = downsample(pelv_vol[sl], downsample_factor) if pelv_vol is not None else None

    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(BG_COLOR)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    all_bounds = []

    def _add_mesh(vol, label, color, alpha, smooth=2, lap=6):
        verts, faces = extract_surface(vol, label, step=1, smooth_iter=smooth)
        if verts is None:
            return
        verts = laplacian_smooth(verts, faces, n_iter=lap)
        ones  = np.ones((len(verts), 1), dtype=np.float32)
        world = (scale_affine @ np.hstack([verts, ones]).T).T[:, :3]
        shading = compute_face_shading(world, faces,
                                        light_dir=np.array([1.0, -2.0, 3.0]))
        tris, fc = faces_to_poly(world, faces, shading, color, alpha)
        ax.add_collection3d(Poly3DCollection(tris, zsort="average",
                                               facecolor=fc[:, :3],
                                               edgecolor="none", alpha=alpha))
        all_bounds.append(world)

    for lbl in spine_labels_render:
        col   = SPINE_COLORS.get(lbl, np.array([0.7, 0.7, 0.7]))
        alpha = 0.88 if (lbl == 26 and pelv_crop is not None) else 1.0
        if lbl == 26:
            _add_mesh(spine_crop, lbl, col, alpha, smooth=0, lap=1)
        else:
            _add_mesh(spine_crop, lbl, col, alpha, smooth=2, lap=5)
    if pelv_crop is not None:
        for lbl in pelv_labels_render:
            col   = PELV_COLORS.get(lbl, np.array([0.7, 0.7, 0.7]))
            alpha = 0.92 if lbl == 1 else 0.72
            _add_mesh(pelv_crop, lbl, col, alpha, smooth=0, lap=1)

    if all_bounds:
        all_v = np.vstack(all_bounds)
        ctr   = (all_v.max(0) + all_v.min(0)) / 2
        r     = (all_v.max(0) - all_v.min(0)).max() / 2 * 1.05
        ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
        ax.set_zlim(ctr[2]-r, ctr[2]+r)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.dist = 7.0
    plt.subplots_adjust(left=0, right=1, top=1.0, bottom=0.0)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor=BG_COLOR, format="jpg",
                pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    print(f"    Saved: {out_path}")
    return True


# ── Case discovery + loaders for GT vs TS ────────────────────────────────────

def _find_spine_path(spine_dir, token, spine_uid):
    sd = Path(spine_dir)
    if not sd.exists():
        return None
    if spine_uid:
        p = sd / f"{spine_uid}_seg_placed.nii.gz"
        if p.exists():
            return p
    for p in sd.glob("*.nii.gz"):
        if _token_matches(token, p.name):
            return p
    return None


def _find_pelv_path(token, pelvic_placed, fused_dir, pelv_dir):
    if pelvic_placed and Path(pelvic_placed).exists():
        return Path(pelvic_placed)
    for d in [Path(fused_dir), Path(pelv_dir)]:
        if not d.exists():
            continue
        for p in d.glob("*.nii.gz"):
            if token and _token_matches(token, p.name):
                return p
    return None


def discover_cases(manifest_path, spine_dir, fused_dir, pelv_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    cases = manifest.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())

    results = {"sacralization_morph": [], "sacralization_count": [],
               "lumbarization": [], "normal": []}

    for c in cases:
        token = str(c.get("patient_token", ""))
        mt    = c.get("match_type", "")
        sp    = c.get("spine",  {}) or {}
        pv    = c.get("pelvic", {}) or {}

        spine_uid     = sp.get("series_uid", "") or ""
        pelvic_placed = pv.get("placed", "")
        pelvic_file   = pv.get("mask_file", "") or pelvic_placed or ""
        bone_ok       = ((sp.get("bone_pct") or 0) > 40) or sp.get("bone_pct") is None
        is_fused      = (mt == "fused")

        lstv_class = c.get("lstv_class", None)
        if lstv_class is not None:
            lstv_label = {0:"normal", 1:"lumbarization", 2:"semi",
                          3:"sacralization", 4:"sacralization"}.get(
                              int(lstv_class), "normal")
        else:
            lstv_label = _extract_lstv_label(c, pelvic_file) or "normal"

        spine_path = _find_spine_path(spine_dir, token, spine_uid)
        pelv_path  = (_find_pelv_path(token, pelvic_placed, fused_dir, pelv_dir)
                      if (is_fused or pelvic_placed) else None)
        if spine_path is None or not bone_ok:
            continue

        entry = (token, spine_path, pelv_path, is_fused, bone_ok, spine_uid)
        lstv_vert = (c.get("lstv_vertebral") or "").lower()

        if   "sacrali" in lstv_label and "sacrali" in lstv_vert:
            results["sacralization_count"].append(entry)
        elif "sacrali" in lstv_label:
            results["sacralization_morph"].append(entry)
        elif "sacrali" in lstv_vert:
            results["sacralization_count"].append(entry)
        elif "lumbariz" in lstv_label:
            results["lumbarization"].append(entry)
        elif "semi" in lstv_label:
            results["sacralization_morph"].append(entry)
        elif "normal" in lstv_label or lstv_label == "":
            results["normal"].append(entry)

    for k in results:
        results[k].sort(key=lambda e: (not e[3], -e[4]))

    print("\n  Case discovery summary:")
    for k, lst in results.items():
        print(f"    {k:28s}: {len(lst):4d} cases  "
              f"({sum(1 for e in lst if e[3])} fused)")
    return results


def check_l5_absent(spine_path):
    try:
        return 24 not in np.asarray(nib.load(str(spine_path)).dataobj,
                                      dtype=np.int16)
    except Exception:
        return False


# ── Composite assembly (autocrop + 4-panel strip + legend) ──────────────────

def _autocrop(img, bg=(247, 247, 247), tol=12):
    arr  = np.array(img)
    mask = np.any(np.abs(arr.astype(int) - np.array(bg)) > tol, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    pad = 6
    r0, r1 = max(0, rows[0]-pad), min(arr.shape[0], rows[-1]+pad+1)
    c0, c1 = max(0, cols[0]-pad), min(arr.shape[1], cols[-1]+pad+1)
    from PIL import Image as _PI
    return _PI.fromarray(arr[r0:r1, c0:c1])


def _make_composite(paths, out_path):
    try:
        from PIL import Image
    except ImportError:
        print("    WARNING: Pillow not installed — composite skipped.")
        return
    imgs = [_autocrop(Image.open(p).convert("RGB")) if p and Path(p).exists() else None
            for p in paths]
    real = [i for i in imgs if i is not None]
    if not real:
        return
    target_h = max(i.size[1] for i in real)
    max_w    = max(i.size[0] for i in real)
    resized  = [img.resize((max_w, target_h), Image.LANCZOS)
                 if img is not None else None for img in imgs]
    bg_rgb = (247, 247, 247)

    label_col = [("L1", SPINE_COLORS[20]), ("L2", SPINE_COLORS[21]),
                 ("L3", SPINE_COLORS[22]), ("L4", SPINE_COLORS[23]),
                 ("L5", SPINE_COLORS[24]), ("L6", SPINE_COLORS[25]),
                 ("Sacrum", PELV_COLORS[1])]
    n_lbls = len(label_col)
    row_in = (target_h / 100.0) / n_lbls
    fs_pt  = max(12, int(row_in * 100 * 0.42))

    import io as _io
    fig_leg = plt.figure(figsize=(2.6, target_h / 100.0),
                          facecolor=tuple(c/255 for c in bg_rgb))
    ltr_fs = int(fs_pt * 1.4)
    for i, (nm, col) in enumerate(label_col):
        y_c = 1.0 - (i + 0.5) / n_lbls
        h   = 0.55 / n_lbls
        ax_s = fig_leg.add_axes([0.04, y_c - h/2, 0.20, h])
        ax_s.set_facecolor(tuple(col)); ax_s.set_xticks([]); ax_s.set_yticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        fig_leg.text(0.30, y_c, nm,
                      fontsize=fs_pt, fontweight="bold", color="black",
                      va="center", ha="left")
    buf = _io.BytesIO()
    fig_leg.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                     pad_inches=0.06, facecolor=tuple(c/255 for c in bg_rgb))
    plt.close(fig_leg); buf.seek(0)
    lbl_pil = Image.open(buf).convert("RGB")
    lbl_w   = int(lbl_pil.width * target_h / lbl_pil.height)
    lbl_img = lbl_pil.resize((lbl_w, target_h), Image.LANCZOS)
    col_w   = lbl_w

    letters = ["A", "B", "C", "D"]
    lettered = []
    for i, img in enumerate(resized):
        if img is None:
            lettered.append(None); continue
        img = img.copy()
        ltr = letters[i] if i < len(letters) else ""
        if ltr:
            fig_l = plt.figure(figsize=(img.width/100, img.height/100),
                                facecolor="none")
            fig_l.text(0.0, 0.95, ltr,
                        fontsize=ltr_fs, fontweight="bold", color="black",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.85))
            buf_l = _io.BytesIO()
            fig_l.savefig(buf_l, format="png", dpi=100,
                           bbox_inches=None, pad_inches=0, transparent=True)
            plt.close(fig_l); buf_l.seek(0)
            overlay = Image.open(buf_l).convert("RGBA")
            overlay = overlay.resize((img.width, img.height), Image.LANCZOS)
            base    = img.convert("RGBA")
            base.alpha_composite(overlay)
            img = base.convert("RGB")
        lettered.append(img)

    strip = Image.new("RGB", (max_w * len(lettered), target_h), bg_rgb)
    for i, img in enumerate(lettered):
        if img is not None:
            strip.paste(img, (i * max_w, 0))
    final = Image.new("RGB", (col_w + strip.width, target_h), bg_rgb)
    final.paste(lbl_img, (0, 0))
    final.paste(strip, (col_w, 0))
    final.save(str(out_path), "JPEG", quality=96, subsampling=0)
    print(f"    Composite saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="data/placed/placed_manifest.json")
    ap.add_argument("--source",   default="gt", choices=["gt", "ts"],
                    help="Render ground-truth labels or TS predictions")
    ap.add_argument("--spine_dir", default="data/placed/spine")
    ap.add_argument("--fused_dir", default="data/placed/fused")
    ap.add_argument("--pelv_dir",  default="data/placed/pelvic_native")
    ap.add_argument("--ts_pred_dir", default="",
                    help="For --source ts: dir containing {token}_{config}/segmentation.nii.gz")
    ap.add_argument("--hf_export_dir", default="data/hf_export",
                    help="For --source ts: used to locate GT label NIfTIs as resample ref")
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--token_sacral_morph", default="")
    ap.add_argument("--token_sacral_count", default="")
    ap.add_argument("--token_lumbar",       default="")
    ap.add_argument("--token_normal",       default="")
    ap.add_argument("--downsample", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                             "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
    })

    discovered = discover_cases(args.manifest, args.spine_dir,
                                 args.fused_dir, args.pelv_dir)

    def _pick(case_type, override_token):
        if override_token:
            tok = override_token.strip()
            sd  = Path(args.spine_dir)
            sp  = next((p for p in sd.glob("*.nii.gz")
                         if _token_matches(tok, p.name)), None)
            pv = None
            for d in [Path(args.fused_dir), Path(args.pelv_dir)]:
                if not d.exists():
                    continue
                pv = next((p for p in d.glob("*.nii.gz")
                            if _token_matches(tok, p.name)), None)
                if pv:
                    break
            if sp:
                return (tok, sp, pv, pv is not None, 99, "")
            print(f"  WARNING: token {tok} not found")
            return None
        lst = discovered.get(case_type, [])
        if not lst:
            print(f"  WARNING: no {case_type} cases found")
            return None
        fused = [e for e in lst if e[3]]
        return fused[0] if fused else lst[0]

    render_specs = [
        ("normal",              args.token_normal,
         "A_normal.jpg"),
        ("sacralization_morph", args.token_sacral_morph,
         "B_sacralization_morph.jpg"),
        ("sacralization_count", args.token_sacral_count,
         "C_sacralization_count.jpg"),
        ("lumbarization",       args.token_lumbar,
         "D_lumbarization.jpg"),
    ]

    rendered = []
    for case_type, override, fname in render_specs:
        print(f"\n  Rendering ({args.source}): {case_type}")
        entry = _pick(case_type, override)
        if entry is None:
            rendered.append(None); continue
        token, spine_path, pelv_path, is_fused, _, spine_uid = entry
        print(f"    token={token}{' [fused]' if is_fused else ''}")

        # For count-type sacralization, prefer a case where L5 is absent
        if case_type == "sacralization_count" and not override:
            if not check_l5_absent(spine_path):
                for e in (discovered.get("normal", []) +
                           discovered.get("sacralization_morph", [])):
                    if check_l5_absent(e[1]):
                        token, spine_path, pelv_path, is_fused, _, spine_uid = e
                        print(f"    (switched to token={token} with absent L5)")
                        break

        # ── Load volumes (GT mode vs TS mode) ────────────────────────────
        if args.source == "gt":
            spine_vol, _, affine = load_nifti(spine_path)
            if pelv_path and Path(pelv_path).exists():
                pelv_vol, _, _ = load_nifti(pelv_path)
            else:
                pelv_vol = None
        else:
            if not args.ts_pred_dir:
                print("    ERROR: --source ts requires --ts_pred_dir")
                rendered.append(None); continue
            # Find a TS prediction for this token (prefer the fused config)
            ts_roots = [Path(args.ts_pred_dir) / f"{token}_fused",
                        Path(args.ts_pred_dir) / f"{token}_spine_only",
                        Path(args.ts_pred_dir) / f"{token}_pelvic_native",
                        Path(args.ts_pred_dir) / token]
            ts_seg = None
            for r in ts_roots:
                p = r / "segmentation.nii.gz"
                if p.exists():
                    ts_seg = p; break
            if ts_seg is None:
                print(f"    ERROR: no TS prediction found under "
                      f"{args.ts_pred_dir} for token={token}")
                rendered.append(None); continue
            # Use the HF label NIfTI as resample reference
            hf_export = Path(args.hf_export_dir)
            candidates = []
            try:
                t04 = f"{int(token):04d}"
            except ValueError:
                t04 = token
            for cfg in ("fused", "spine_only", "pelvic_native"):
                candidates.append(hf_export / "labels" / f"{t04}_{cfg}_label.nii.gz")
            ref = next((p for p in candidates if p.exists()), None)
            if ref is None:
                print(f"    ERROR: no reference label under {hf_export}/labels/ "
                      f"for token={token}")
                rendered.append(None); continue
            spine_vol, pelv_vol, _, affine = _load_ts_as_rendering_volume(ts_seg, ref)

        ok = render_case(spine_vol=spine_vol, pelv_vol=pelv_vol, affine=affine,
                          out_path=out_dir / fname,
                          elev=25, azim=-90,
                          downsample_factor=args.downsample)
        rendered.append((out_dir / fname) if ok else None)

    print("\n  Building composite panel...")
    composite_name = f"LSTV_panel_4x1_{args.source}.jpg"
    _make_composite(rendered, out_dir / composite_name)

    print(f"\n  Done.  Output: {out_dir}\n")


if __name__ == "__main__":
    main()
