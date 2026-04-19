"""
place_fused_masks.py — Patient-centric mask placement.

REFACTOR NOTE (vs. the previous SpineSurg-CT version)
-----------------------------------------------------
This version consumes **patient_db.json** produced by build_db.py.  The old
version read `data/matched/colonog_matched_pairs.json` and had to augment it
at runtime by rescanning TCIA directories to recover the full per-patient
series list (the `_augment_patient_uids_from_tcia` helper).  That entire
step is now unnecessary:

  • build_db.py already indexes EVERY TCIA series grouped by canonical
    PatientID (via `tcia_index.build_tcia_patient_index`), so the list of
    candidate UIDs for each patient is already complete and sitting inside
    patient_db.json at `patients[uid].tcia_series[*].series_uid`.

  • build_db.py already attaches spine and pelvic masks to patients via
    DICOM PatientID equality, so there is no need to re-derive the
    spine-path / pelvic-path / sp_nii-path mapping from a secondary manifest.

  • LSTV fields (lstv_pelvic, lstv_vertebral, lstv_agreement,
    lstv_confusion_zone, lstv_class) are derived directly from the per-mask
    `lstv_label` fields in patient_db.json.  No cross-check JSON is needed.

Everything below the input-parsing layer (dcm2niix, spine placement,
pelvic placement, manifest writing) is functionally unchanged from the
original.

ARCHITECTURE
============
For each patient, independently finds the TCIA CT series that maximises bone
coverage (HU > 200) under each placed label volume:

  Spine seg   (CTSpine1K)  → world-space affine NN resample, best bone_pct
  Pelvic mask (CTPelvic1K) → bone z-profile cross-correlation, best bone_pct

Every TCIA series for the patient is tried for each mask.  match_type is
determined post-hoc from results:
  same winning series for both → fused
  different winning series     → separate
  one mask only                → spine_only / pelvic_only

FALLBACK CHAIN (spine only)
============================
  1. World-space affine NN resample (all series, best bone_pct)
  2. Phase cross-correlation + 8 axis-flips
  3. CTSpine1K NIfTI anchor (last resort, bone gate bypassed)

OUTPUTS
=======
  tcia_nifti/{uid}.nii.gz                   dcm2niix reference NIfTIs (PIR)
  placed/spine/{uid}_seg_placed.nii.gz      best spine seg per patient
  placed/pelvic/{stem}_pelvic_placed.nii.gz best pelvic mask per patient
  placed/placed_manifest.json               winning series + bone_pct per case
"""
from __future__ import annotations

import argparse
import os
import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spinesurg.place_masks")

BONE_HU             = 200.0
MIN_VOXELS          = 50
MIN_SPINE_VOXELS    = 10_000
SPINE_BONE_WARN     = 40.0
BONE_ACCEPT_THRESH  = 20.0


# ===========================================================================
# Orientation helper (PIR)
# ===========================================================================

def _compute_bone_pct_from_files(placed_path, ct_path, bone_hu=200):
    """Recompute bone fraction from a placed mask + CT NIfTI (legacy cache)."""
    try:
        import nibabel as _nib_bp
        import numpy  as _np_bp
        from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation

        placed_img = _nib_bp.load(str(placed_path))
        ct_img     = _nib_bp.load(str(ct_path))

        _target  = axcodes2ornt(("P", "I", "R"))
        _current = io_orientation(ct_img.affine)
        _xfm     = ornt_transform(_current, _target)
        ct_pir   = ct_img.as_reoriented(_xfm)

        placed_arr = _np_bp.asarray(placed_img.dataobj) > 0

        if ct_pir.shape[:3] == placed_arr.shape[:3]:
            ct_data  = _np_bp.asarray(ct_pir.dataobj, dtype=_np_bp.float32)
            n_vox    = int(placed_arr.sum())
            if n_vox == 0:
                return None
            return round(float((ct_data[placed_arr] > bone_hu).sum() / n_vox * 100), 1)

        if (ct_pir.shape[0] == placed_arr.shape[0]
                and ct_pir.shape[1] == placed_arr.shape[1]):
            placed_aff = placed_img.affine
            ct_aff     = ct_pir.affine
            dz_world = placed_aff[2, 3] - ct_aff[2, 3]
            vox_sz_z = abs(ct_aff[2, 2]) or 1.0
            z0 = int(round(dz_world / vox_sz_z))
            z1 = z0 + placed_arr.shape[2]
            z0c = max(0, z0); z1c = min(ct_pir.shape[2], z1)
            if z1c <= z0c:
                return None
            ct_slice = _np_bp.asarray(ct_pir.dataobj[:, :, z0c:z1c], dtype=_np_bp.float32)
            mask_slice = placed_arr[:, :, :ct_slice.shape[2]]
            n_vox = int(mask_slice.sum())
            if n_vox == 0:
                return None
            return round(float((ct_slice[mask_slice] > bone_hu).sum() / n_vox * 100), 1)

        return None
    except Exception:
        return None


def _to_pir(data, affine):
    import nibabel as nib
    import numpy as np
    from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation

    img_tmp  = nib.Nifti1Image(data, affine)
    src_ornt = nib.io_orientation(affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)

    if np.issubdtype(data.dtype, np.integer):
        pir_data = apply_orientation(data.astype(np.int32), xfm).astype(data.dtype)
    else:
        pir_data = apply_orientation(data.astype(np.float32), xfm)

    pir_aff = affine @ nib.orientations.inv_ornt_aff(xfm, img_tmp.shape[:3])
    return pir_data, pir_aff


def _world_bbox(affine, shape):
    import numpy as np
    si, sj, sk = int(shape[0]) - 1, int(shape[1]) - 1, int(shape[2]) - 1
    corners = np.array([
        [0,  0,  0,  1], [si, 0,  0,  1], [0,  sj, 0,  1], [0,  0,  sk, 1],
        [si, sj, 0,  1], [si, 0,  sk, 1], [0,  sj, sk, 1], [si, sj, sk, 1],
    ], dtype=np.float64).T
    world = (affine @ corners)[:3, :]
    return world.min(axis=1), world.max(axis=1)


def _bbox_overlap_frac(ref_affine, ref_shape, seg_affine, seg_shape):
    import numpy as np
    ref_min, ref_max = _world_bbox(ref_affine, ref_shape)
    seg_min, seg_max = _world_bbox(seg_affine, seg_shape)

    overlap_min  = np.maximum(ref_min, seg_min)
    overlap_max  = np.minimum(ref_max, seg_max)
    overlap_size = np.maximum(0.0, overlap_max - overlap_min)

    seg_size = np.abs(seg_max - seg_min)
    seg_vol  = float(np.prod(seg_size))
    frac     = float(np.prod(overlap_size)) / max(1.0, seg_vol)

    return (
        frac,
        (float(ref_min[2]), float(ref_max[2])),
        (float(seg_min[2]), float(seg_max[2])),
    )


def _log_header_diff(token, ref_img, sp_img, overlap):
    import numpy as np
    def _sp(a): return np.sqrt((a[:3,:3]**2).sum(axis=0))
    ra, sa = ref_img.affine, sp_img.affine
    rsh = tuple(int(s) for s in ref_img.shape[:3])
    ssh = tuple(int(s) for s in sp_img.shape[:3])
    od  = float(np.linalg.norm(ra[:3,3] - sa[:3,3]))
    sr  = np.round(_sp(ra), 3)
    ss  = np.round(_sp(sa), 3)
    dcr = ra[:3,:3] / np.maximum(1e-9, _sp(ra))
    dcs = sa[:3,:3] / np.maximum(1e-9, _sp(sa))
    om  = bool(np.allclose(np.abs(dcr), np.abs(dcs), atol=0.01))
    flp = [bool(np.dot(dcr[:,i], dcs[:,i]) < -0.9) for i in range(3)]
    if any(flp):   diag = "AXIS_FLIP"
    elif od > 10:  diag = f"ORIGIN_SHIFT {od:.0f}mm"
    elif od > 1:   diag = f"SMALL_DIFF {od:.1f}mm"
    else:          diag = "NEAR_IDENTICAL"
    log.warning(
        "[nifti_anchor] token=%s overlap=%.1f%% ref=%s sp=%s "
        "spacing ref=%s sp=%s origin ref=[%.0f %.0f %.0f] sp=[%.0f %.0f %.0f] "
        "diff=%.0fmm orient=%s flips=%s  %s",
        token, overlap*100, rsh, ssh, sr, ss,
        ra[0,3], ra[1,3], ra[2,3], sa[0,3], sa[1,3], sa[2,3],
        od, om, flp, diag,
    )


def _spine_placement_checks(spine_arr, ref_data, placed_labels, ref_aff, token):
    """PCA-based vertebral ordering check — handles lordosis robustly."""
    import numpy as np

    IS_ORDER_TOL_MM = 8.0

    spine_bool = (spine_arr > 0)
    if not spine_bool.any():
        return dict(bone_pct=0., mean_hu=0., is_ordered=True, label_z_centroids={})

    mn  = tuple(min(a, b) for a, b in zip(ref_data.shape, spine_arr.shape))
    rd  = ref_data  [:mn[0], :mn[1], :mn[2]]
    sa  = spine_arr [:mn[0], :mn[1], :mn[2]]
    sb  = (sa > 0)

    hu  = rd[sb].astype("float32") if sb.any() else np.array([], dtype="float32")
    bp  = float((hu > 200).sum()) / max(1, len(hu)) * 100
    mhu = float(hu.mean()) if len(hu) else 0.

    if bp < SPINE_BONE_WARN:
        log.warning(
            "  [spine_check] token=%-8s bone=%.0f%% hu=%.0f"
            " < SPINE_BONE_WARN=%.0f%% -- likely misaligned (soft tissue)",
            token, bp, mhu, SPINE_BONE_WARN,
        )
    else:
        log.info("  [spine_check] token=%-8s bone=%.0f%% hu=%.0f  OK",
                 token, bp, mhu)

    label_world: dict = {}
    label_z:     dict = {}
    for lbl in placed_labels:
        vox_idx = np.where(spine_arr == lbl)
        if not len(vox_idx[0]):
            continue
        hv = np.vstack([
            np.array(vox_idx[0], dtype="float32"),
            np.array(vox_idx[1], dtype="float32"),
            np.array(vox_idx[2], dtype="float32"),
            np.ones(len(vox_idx[0]), dtype="float32"),
        ])
        world = (ref_aff @ hv)[:3]
        cx, cy, cz = float(world[0].mean()), float(world[1].mean()), float(world[2].mean())
        label_world[lbl] = np.array([cx, cy, cz], dtype=np.float64)
        label_z[lbl]     = cz

    is_ordered = True
    if len(label_world) >= 2:
        sl     = sorted(label_world.items(), key=lambda x: x[0])
        labels = [lbl for lbl, _ in sl]
        pts    = np.stack([c for _, c in sl], axis=0)

        if len(pts) >= 3:
            centered = pts - pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            spine_axis = Vt[0]
        else:
            spine_axis = pts[1] - pts[0]
            spine_axis = spine_axis / (np.linalg.norm(spine_axis) + 1e-9)

        if spine_axis[2] < 0:
            spine_axis = -spine_axis

        projections = {lbl: float(np.dot(c, spine_axis)) for lbl, c in label_world.items()}
        proj_sorted = [projections[lbl] for lbl in labels]

        violations = []
        for i in range(len(proj_sorted) - 1):
            delta = proj_sorted[i] - proj_sorted[i + 1]
            if delta < -IS_ORDER_TOL_MM:
                violations.append((labels[i], labels[i + 1], round(-delta, 1)))

        is_ordered = len(violations) == 0
        if not is_ordered:
            log.warning(
                "  [spine_check] token=%-8s IS_ORDER_FAIL (PCA axis) "
                "violations (lbl_a, lbl_b, reversal_mm): %s",
                token, violations,
            )

    return dict(bone_pct=bp, mean_hu=mhu, is_ordered=is_ordered,
                label_z_centroids=label_z)


def _bone_pct_of_placed(arr, ref_data):
    import numpy as np
    mask = arr > 0
    if not mask.any():
        return 0.0
    mn = tuple(min(a, b) for a, b in zip(ref_data.shape, arr.shape))
    rd = ref_data[:mn[0], :mn[1], :mn[2]]
    mk = mask[:mn[0], :mn[1], :mn[2]]
    hu = rd[mk].astype(np.float32)
    return float((hu > BONE_HU).sum()) / max(1, len(hu)) * 100


def _sorted_candidates(seg_affine, seg_shape, candidate_uids, primary_uid, nifti_dir):
    import nibabel as nib
    scored = []
    for uid in candidate_uids:
        if uid == primary_uid:
            continue
        nii_path = Path(nifti_dir) / f"{uid}.nii.gz"
        if not nii_path.exists():
            continue
        try:
            img = nib.load(str(nii_path))
            ov, _, _ = _bbox_overlap_frac(
                seg_affine, seg_shape, img.affine, img.shape[:3])
            scored.append((ov, uid, img))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[0])
    return [(uid, img) for _, uid, img in scored]


def _phase_xcorr_with_flips(seg_data, seg_affine, refs_to_try, token):
    import numpy as np
    from scipy.ndimage import affine_transform, shift as nd_shift

    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        log.warning("  [xcorr] token=%-8s skimage not available; phase_xcorr skipped", token)
        return None, None, 0.0, "skimage_unavailable"

    best_bone_pct  = 0.0
    best_arr       = None
    best_ref_img   = None
    best_desc      = "all_flips_refs_failed"

    for uid_label, ref_img in refs_to_try:
        try:
            ref_data    = ref_img.get_fdata(dtype=np.float32)
            ref_bone    = (ref_data > BONE_HU).astype(np.float32)
            ref_shape   = tuple(ref_img.shape[:3])
            ref_bone_ds = ref_bone[::4, ::4, ::4]

            for f0 in (False, True):
                for f1 in (False, True):
                    for f2 in (False, True):
                        flips    = [f0, f1, f2]
                        flip_aff = seg_affine.copy()
                        for i, f in enumerate(flips):
                            if f:
                                flip_aff[:3, 3] += seg_affine[:3, i] * (seg_data.shape[i] - 1)
                                flip_aff[:3, i] *= -1

                        M    = np.linalg.inv(flip_aff) @ ref_img.affine
                        cand = affine_transform(
                            seg_data, matrix=M[:3, :3], offset=M[:3, 3],
                            output_shape=ref_shape,
                            order=0, mode="constant", cval=0, prefilter=False,
                        ).astype(np.int32)

                        cand_bone_ds = (cand > 0).astype(np.float32)[::4, ::4, ::4]
                        if cand_bone_ds.sum() < (MIN_SPINE_VOXELS // 64):
                            continue

                        try:
                            raw_shift, _, _ = phase_cross_correlation(
                                ref_bone_ds, cand_bone_ds,
                                upsample_factor=1, normalization=None,
                            )
                            shift_full = raw_shift * 4.0
                        except Exception:
                            shift_full = np.zeros(3)

                        cand_shifted = nd_shift(
                            cand.astype(np.float32), shift_full,
                            order=0, mode="constant", cval=0,
                        ).astype(np.int32)

                        n_vox = int((cand_shifted > 0).sum())
                        if n_vox < MIN_SPINE_VOXELS:
                            continue

                        bone_pct = _bone_pct_of_placed(cand_shifted, ref_data)

                        if bone_pct > best_bone_pct:
                            best_bone_pct = bone_pct
                            best_arr      = cand_shifted
                            best_ref_img  = ref_img
                            best_desc     = (
                                f"uid={uid_label}  flips={flips}  "
                                f"shift={[round(float(s), 1) for s in shift_full]}  "
                                f"bone={bone_pct:.0f}%  vox={n_vox}"
                            )
        except Exception:
            continue

    return best_arr, best_ref_img, best_bone_pct, best_desc


# ===========================================================================
# dcm2niix
# ===========================================================================

def _convert_one_series(args):
    series_uid, series_dir, nifti_dir = args
    out_path   = Path(nifti_dir) / f"{series_uid}.nii.gz"
    if out_path.exists():
        return series_uid, True, "skip"
    series_dir = Path(series_dir)
    if not series_dir.exists() or not list(series_dir.glob("*.dcm")):
        return series_uid, False, f"no_dcm:{series_dir}"

    def _attempt(extra_flags, label):
        cmd = ["dcm2niix", "-z", "y", "-f", series_uid,
               "-o", str(nifti_dir)] + extra_flags + [str(series_dir)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return False, "timeout", ""
        except Exception as e:
            return False, str(e), ""

        if out_path.exists():
            return True, f"ok_{label}", res.stderr
        cands = sorted(Path(nifti_dir).glob(f"{series_uid}*.nii.gz"))
        if cands:
            cands[0].rename(out_path)
            for extra in Path(nifti_dir).glob(f"{series_uid}*"):
                if extra != out_path:
                    extra.unlink(missing_ok=True)
            return True, f"ok_renamed_{label}", res.stderr
        return False, f"no_output_{label}", res.stderr

    ok, msg, stderr = _attempt([], "std")
    if ok:
        return series_uid, True, msg

    if "myInstanceNumberOrderIsNotSpatial" in stderr or "InstanceNumber" in stderr:
        ok, msg, _ = _attempt(["-n", "y"], "filename_sort")
        if ok:
            return series_uid, True, msg

    if not ok:
        ok, msg, _ = _attempt(["-i", "y"], "ignore_derived")
        if ok:
            return series_uid, True, msg

    return series_uid, False, f"no_output:{stderr[-300:]}"


def _try_tqdm(iterable=None, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        class _DummyBar:
            def update(self, n=1): pass
            def close(self): pass
            def set_postfix(self, **kw): pass
            def __iter__(self): return iter([])
        return _DummyBar()


def convert_series(uid_dir_pairs, nifti_dir, workers):
    nifti_dir.mkdir(parents=True, exist_ok=True)
    work   = [(uid, d, str(nifti_dir)) for uid, d in uid_dir_pairs]
    n_pre  = sum(1 for w in work if (nifti_dir / f"{w[0]}.nii.gz").exists())
    n_todo = len(work) - n_pre
    log.info("dcm2niix: %d to convert  %d already done  workers=%d",
             n_todo, n_pre, workers)
    if n_todo == 0:
        return n_pre, 0

    failures = []
    n_ok = n_fail = 0
    bar = _try_tqdm(total=len(work), desc="dcm2niix",
                    unit="series", dynamic_ncols=True)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert_one_series, w): w[0] for w in work}
        for fut in as_completed(futs):
            uid, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append({"uid": uid, "reason": msg})
                log.warning("  FAIL dcm2niix %s: %s", uid, msg[:200])
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
                bar.update(1)

    if hasattr(bar, "close"):
        bar.close()

    elapsed = time.time() - t0
    log.info(
        "dcm2niix complete: ok=%d  fail=%d  skip=%d  total=%.0fs  %.1f s/series",
        n_ok, n_fail, n_pre, elapsed,
        elapsed / max(1, n_ok + n_fail),
    )

    if failures:
        log.warning("dcm2niix failures (%d):", len(failures))
        for f in failures[:20]:
            log.warning("  UID=%s  reason=%s", f["uid"], f["reason"])
        if len(failures) > 20:
            log.warning("  ... and %d more", len(failures) - 20)

    return n_ok, n_fail


# ===========================================================================
# Spine placement worker
# ===========================================================================

def _place_spine_best_series(args):
    (token, seg_path, sp_nii_path,
     candidate_uids, nifti_dir, out_dir) = args
    out_dir = Path(out_dir)

    import logging as _log
    import json as _json_c
    _spine_log = _log.getLogger("spinesurg.place_masks")
    for _cuid in candidate_uids:
        _cached_file = out_dir / f"{_cuid}_seg_placed.nii.gz"
        if _cached_file.exists():
            _sidecar = out_dir / f"{_cuid}_seg_placed.json"
            if _sidecar.exists():
                try:
                    _r = _json_c.loads(_sidecar.read_text())
                    _r["method"] = "cached"
                    _spine_log.info(
                        "  [spine] token=%-8s  CACHED (sidecar) %s  bone=%.1f%%",
                        token, _cached_file.name, _r.get("bone_pct") or 0.0,
                    )
                    return token, True, _r
                except Exception:
                    pass

            _bone_pct = _vox = _labels = _is_ok = None
            try:
                import nibabel as _nib2
                import numpy as _np2
                _arr    = _np2.asarray(_nib2.load(str(_cached_file)).dataobj,
                                       dtype=_np2.int32)
                _vox    = int((_arr > 0).sum())
                _labels = sorted(set(_arr.ravel().tolist()) - {0})
                _is_ok  = True
            except Exception:
                pass

            if _bone_pct is None:
                _ct_p = Path(nifti_dir) / f"{_cuid}.nii.gz"
                _bone_pct = _compute_bone_pct_from_files(_cached_file, _ct_p)

            _result = {
                "token":      token,
                "series_uid": _cuid,
                "placed":     str(_cached_file),
                "bone_pct":   _bone_pct,
                "vox":        _vox,
                "labels":     _labels,
                "method":     "cached",
                "IS_ok":      _is_ok,
            }
            try:
                _sidecar.write_text(_json_c.dumps(_result, indent=2, default=str))
            except Exception:
                pass
            return token, True, _result

    try:
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import affine_transform

        seg_img  = nib.load(str(seg_path))
        seg_data = np.array(seg_img.dataobj, dtype=np.int32).squeeze()
        if seg_data.ndim != 3:
            return token, False, {"error": f"seg_ndim={seg_data.ndim}"}

        sorted_cands = _sorted_candidates(
            seg_img.affine, seg_img.shape[:3], candidate_uids, "", nifti_dir,
        )
        log.info("  [spine] token=%-8s  %d candidates", token, len(sorted_cands))

        best_uid = best_arr = best_ref = None
        best_bp  = 0.0
        for uid, ref_img in sorted_cands:
            _ov, _, _ = _bbox_overlap_frac(
                ref_img.affine, ref_img.shape[:3], seg_img.affine, seg_img.shape[:3])
            M    = np.linalg.inv(seg_img.affine) @ ref_img.affine
            cand = affine_transform(
                seg_data, matrix=M[:3,:3], offset=M[:3,3],
                output_shape=tuple(ref_img.shape[:3]),
                order=0, mode="constant", cval=0, prefilter=False,
            ).astype(np.int32)
            rd  = ref_img.get_fdata(dtype=np.float32)
            bp  = _bone_pct_of_placed(cand, rd)
            vox = int((cand > 0).sum())
            log.info(
                "  [spine] token=%-8s  uid=...%s  overlap=%.0f%%  vox=%d  bone=%.0f%%",
                token, uid[-10:], _ov * 100, vox, bp,
            )
            if vox >= MIN_SPINE_VOXELS and bp > best_bp:
                best_uid, best_arr, best_ref, best_bp = uid, cand, ref_img, bp

        method = f"world_space:{best_uid}" if best_uid else None

        if best_bp < BONE_ACCEPT_THRESH:
            log.warning(
                "  [spine] token=%-8s  world_space best=%.0f%% -> phase_xcorr_flips",
                token, best_bp,
            )
            _arr, _xref, _xbp, _xdesc = _phase_xcorr_with_flips(
                seg_data, seg_img.affine, sorted_cands, token,
            )
            if _arr is not None and _xbp > best_bp:
                _xuid = _xdesc.split("uid=")[-1].split()[0] if "uid=" in _xdesc else "xcorr"
                best_uid, best_arr, best_ref, best_bp = _xuid, _arr, _xref, _xbp
                method = f"phase_xcorr:{_xdesc}"
                log.warning("  [spine] token=%-8s  XCORR_HIT  bone=%.0f%%", token, _xbp)

        if best_bp < BONE_ACCEPT_THRESH and sp_nii_path and Path(sp_nii_path).exists():
            if sorted_cands:
                _fuid, _fref = sorted_cands[0]
                _ov2, _, _ = _bbox_overlap_frac(
                    _fref.affine, _fref.shape[:3], seg_img.affine, seg_img.shape[:3])
                _spn = nib.load(str(sp_nii_path))
                _log_header_diff(token, _fref, _spn, _ov2)
                Ma   = np.linalg.inv(seg_img.affine) @ _spn.affine
                cand = affine_transform(
                    seg_data, matrix=Ma[:3,:3], offset=Ma[:3,3],
                    output_shape=tuple(_fref.shape[:3]),
                    order=0, mode="constant", cval=0, prefilter=False,
                ).astype(np.int32)
                if int((cand > 0).sum()) >= MIN_SPINE_VOXELS:
                    rd  = _fref.get_fdata(dtype=np.float32)
                    bp  = _bone_pct_of_placed(cand, rd)
                    log.warning(
                        "  [spine] token=%-8s  NIFTI_ANCHOR  uid=%s  bone=%.0f%%  (last resort)",
                        token, _fuid, bp,
                    )
                    best_uid, best_arr, best_ref, best_bp = _fuid, cand, _fref, bp
                    method = f"nifti_anchor:{_fuid}"

        if best_arr is None or best_ref is None:
            return token, False, {"error": "all_methods_failed", "token": token}

        n_vox  = int((best_arr > 0).sum())
        labels = sorted(set(best_arr.ravel().tolist()) - {0})
        checks = _spine_placement_checks(
            best_arr, best_ref.get_fdata(dtype=np.float32),
            labels, best_ref.affine, token,
        )
        out_path = out_dir / f"{best_uid}_seg_placed.nii.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pir_data, pir_aff = _to_pir(best_arr, best_ref.affine)
        nib.save(nib.Nifti1Image(pir_data, pir_aff), str(out_path))

        result = {
            "token":      token,
            "series_uid": best_uid,
            "placed":     str(out_path),
            "bone_pct":   round(best_bp, 1),
            "vox":        n_vox,
            "labels":     labels,
            "method":     method,
            "IS_ok":      checks["is_ordered"],
        }

        sidecar_path = out_dir / f"{best_uid}_seg_placed.json"
        import json as _json_s
        sidecar_path.write_text(_json_s.dumps(result, indent=2, default=str))

        log.info(
            "  [spine] token=%-8s  BEST uid=...%s  bone=%.0f%%  vox=%d  IS_ok=%s",
            token, best_uid[-10:], best_bp, n_vox, checks["is_ordered"],
        )
        return token, True, result

    except Exception:
        import traceback
        return token, False, {"error": traceback.format_exc(), "token": token}


# ===========================================================================
# Pelvic placement worker
# ===========================================================================

def _place_pelvic_best_series(args):
    (token, mask_path, candidate_uids, nifti_dir, out_dir) = args
    out_dir = Path(out_dir)

    mask_stem = Path(mask_path).name.replace(".nii.gz", "").replace(".nii", "")
    existing_p = out_dir / f"{mask_stem}_pelvic_placed.nii.gz"
    if existing_p.exists():
        import json as _json_pc, logging as _log_pc
        _pelv_log = _log_pc.getLogger("spinesurg.place_masks")

        _sidecar_p = Path(str(existing_p).replace(".nii.gz", ".json"))
        if _sidecar_p.exists():
            try:
                _r = _json_pc.loads(_sidecar_p.read_text())
                _r["method"] = "cached"
                return token, True, _r
            except Exception:
                pass

        result = {"token": token, "placed": str(existing_p),
                  "series_uid": None, "bone_pct": None,
                  "vox": None, "z_off": None, "method": "cached"}

        if result["series_uid"] is None:
            _best_uid_r = None
            _best_bp_r  = -1.0
            for _uid_r in candidate_uids:
                _ct_r = Path(nifti_dir) / f"{_uid_r}.nii.gz"
                _bp_r = _compute_bone_pct_from_files(existing_p, _ct_r)
                if _bp_r is not None and _bp_r > _best_bp_r:
                    _best_bp_r  = _bp_r
                    _best_uid_r = _uid_r
            if _best_uid_r:
                result["series_uid"] = _best_uid_r
                result["bone_pct"]   = _best_bp_r
                result["method"]     = "cached_bone_rematch"

        try:
            _sidecar_p.write_text(_json_pc.dumps(result, indent=2, default=str))
        except Exception:
            pass
        return token, True, result

    try:
        import nibabel as nib
        import numpy as np
        from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

        mask_img      = nib.load(str(mask_path))
        orig_mask_aff = mask_img.affine

        sorted_cands = _sorted_candidates(
            orig_mask_aff, mask_img.shape[:3], candidate_uids, "", nifti_dir,
        )
        log.info("  [pelv]  token=%-8s  %d candidates", token, len(sorted_cands))
        if not sorted_cands:
            return token, False, {"error": "no_candidate_niftis", "token": token}

        best_uid = best_pir = best_pir_aff = None
        best_bp  = 0.0
        best_z   = best_score_val = 0

        for uid, ref_img in sorted_cands:
            ref_aff  = ref_img.affine
            ref_data = ref_img.get_fdata(dtype=np.float32)
            ref_ornt = io_orientation(ref_aff)
            ref_nz   = ref_data.shape[2]

            try:
                xfm       = ornt_transform(io_orientation(orig_mask_aff), ref_ornt)
                mask_data = apply_orientation(
                    np.array(mask_img.dataobj, dtype=np.float32).astype(np.int32), xfm,
                ).astype(np.float32)
            except ValueError:
                log.warning(
                    "  [pelv]  token=%-8s  uid=...%s  ornt_transform NaN",
                    token, uid[-10:] if sorted_cands else "?",
                )
                mask_data = np.array(mask_img.dataobj, dtype=np.float32).astype(np.float32)
            mask_nz = mask_data.shape[2]

            ref_bone = np.array(
                [(ref_data[:, :, z] > BONE_HU).sum() for z in range(ref_nz)],
                dtype=np.float32,
            )
            mask_lbl = np.array(
                [(mask_data[:, :, z] > 0).sum() for z in range(mask_nz)],
                dtype=np.float32,
            )

            z_off = 0; score = -1.0
            for zo in range(max(1, ref_nz - mask_nz + 1)):
                win = ref_bone[zo: zo + mask_nz]
                s   = float(np.dot(win[:len(win)], mask_lbl[:len(win)]))
                if s > score:
                    score, z_off = s, zo
            if score <= 0:
                z_off = max(0, ref_nz - mask_nz)

            md = mask_data.copy()
            if orig_mask_aff[0, 0] < 0:
                tmp       = (md == 2).copy()
                md[md == 3] = 2
                md[tmp]    = 3

            z_end = min(ref_nz, z_off + mask_nz)
            if z_end > z_off:
                rw  = ref_data[:, :, z_off:z_end]
                mw  = md[:, :, :z_end - z_off]
                mn  = tuple(min(a, b) for a, b in zip(rw.shape, mw.shape))
                msk = mw[:mn[0], :mn[1], :mn[2]] > 0
                bp  = (float((rw[:mn[0],:mn[1],:mn[2]][msk].astype(np.float32) > BONE_HU).sum())
                       / max(1, int(msk.sum())) * 100) if msk.any() else 0.0
            else:
                bp = 0.0

            log.info(
                "  [pelv]  token=%-8s  uid=...%s  z_off=%d  score=%.0f  bone=%.0f%%",
                token, uid[-10:], z_off, score, bp,
            )

            if bp > best_bp:
                placed_aff = ref_aff.copy()
                placed_aff[:3, 3] = ref_aff[:3, 3] + z_off * ref_aff[:3, 2]
                best_pir, best_pir_aff = _to_pir(md.astype(np.int32), placed_aff)
                best_uid, best_bp, best_z, best_score_val = uid, bp, z_off, score

        if best_pir is None:
            return token, False, {"error": "all_candidates_zero_bone", "token": token}

        out_path = out_dir / f"{mask_stem}_pelvic_placed.nii.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(best_pir, best_pir_aff), str(out_path))

        n_nz = int((np.asarray(nib.load(str(out_path)).dataobj) > 0).sum())
        if n_nz < MIN_VOXELS:
            out_path.unlink(missing_ok=True)
            return token, False, {"error": "placed_mask_empty", "token": token}

        result = {
            "token":      token,
            "series_uid": best_uid,
            "placed":     str(out_path),
            "bone_pct":   round(best_bp, 1),
            "vox":        n_nz,
            "z_off":      best_z,
        }

        sidecar_path = Path(str(out_path).replace(".nii.gz", ".json"))
        import json as _json_p
        sidecar_path.write_text(_json_p.dumps(result, indent=2, default=str))

        log.info(
            "  [pelv]  token=%-8s  BEST uid=...%s  z_off=%d  bone=%.0f%%  vox=%d",
            token, best_uid[-10:], best_z, best_bp, n_nz,
        )
        return token, True, result

    except Exception:
        import traceback
        return token, False, {"error": traceback.format_exc(), "token": token}


# ===========================================================================
# Parallel runner
# ===========================================================================

def _run_parallel(work, worker_fn, workers, label):
    n_ok = n_fail = n_skip = 0
    t0        = time.time()
    effective = 1 if len(work) <= 5 else workers
    failures  = []
    results   = []

    bar = _try_tqdm(total=len(work), desc=f"[{label}]",
                    unit="case", dynamic_ncols=True)

    def _handle(token, ok, msg):
        nonlocal n_ok, n_fail, n_skip
        if isinstance(msg, str) and msg == "skip":
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_fail += 1
            err = msg.get("error","") if isinstance(msg,dict) else str(msg)
            failures.append((token, err))
            log.warning("  FAIL [%s] token=%s: %s", label, token,
                        err.strip().splitlines()[-1] if err else "unknown")
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
            bar.update(1)

    if effective == 1:
        for w in work:
            token, ok, msg = worker_fn(w)
            _handle(token, ok, msg)
            results.append((token, ok, msg))
    else:
        with ProcessPoolExecutor(max_workers=effective) as ex:
            futs = {ex.submit(worker_fn, w): w[0] for w in work}
            for fut in as_completed(futs):
                token, ok, msg = fut.result()
                _handle(token, ok, msg)
                results.append((token, ok, msg))

    if hasattr(bar, "close"):
        bar.close()

    elapsed = time.time() - t0
    log.info("[%s] done: ok=%d  skip=%d  fail=%d  total=%.0fs",
             label, n_ok, n_skip, n_fail, elapsed)

    if failures:
        log.warning("[%s] FAILURES (%d):", label, len(failures))
        for tok, err in failures:
            log.warning("  token=%-8s  %s", tok, err)

    return n_ok, n_fail, results


def run_spine_placement(work, workers):
    return _run_parallel(work, _place_spine_best_series, workers, "spine")


def run_pelvic_placement(work, workers):
    return _run_parallel(work, _place_pelvic_best_series, workers, "pelvic")


# ===========================================================================
# PatientDB → registry (the refactored input layer)
# ===========================================================================

def _lstv_derived_fields(spine_label: str, pelvic_label: str) -> Dict:
    """
    Cross-check spine (vertebral-counting) vs pelvic (filename) LSTV labels,
    derive lstv_agreement and lstv_class.  Mirrors the old resolve_dataset.py
    agreement logic so downstream consumers (export_hf.py) see the same fields.
    """
    LSTV_CLS = {
        "LUMBARIZATION":      1,
        "SEMI_SACRAL":        2,
        "SEMI_SACRALIZATION": 2,
        "SACRALIZATION":      3,
    }
    UNINFORMATIVE = {"UNKNOWN", "AMBIGUOUS", "INCOMPLETE_SCAN", "", None}

    sp = (spine_label  or "").strip()
    pv = (pelvic_label or "").strip()

    is_lstv = lambda s: s and s.upper() not in UNINFORMATIVE and s.upper() != "NORMAL"

    # Agreement is None when either source is uninformative
    if (not sp or sp.upper() in UNINFORMATIVE) or (not pv or pv.upper() in UNINFORMATIVE):
        agreement = None
    else:
        # Agreement is True when both say "normal" or both say some flavour of LSTV
        agreement = (sp.upper() == pv.upper()) or (is_lstv(sp) and is_lstv(pv))

    # Confusion zone: sources disagree AND at least one calls it LSTV
    confusion = (agreement is False) and (is_lstv(sp) or is_lstv(pv))

    # Pelvic label takes priority for integer class (matches place_fused_masks v1)
    lstv_class = LSTV_CLS.get(pv.upper(), 0) or LSTV_CLS.get(sp.upper(), 0)

    return {
        "lstv_pelvic":         pv,
        "lstv_vertebral":      sp,
        "lstv_agreement":      agreement,
        "lstv_confusion_zone": confusion,
        "lstv_class":          lstv_class,
    }


def _build_registry_from_patient_db(
    db_path:       Path,
    token_filter:  Optional[Set[str]],
    debug_n:       int,
) -> Dict[str, dict]:
    """
    Build the placement registry from patient_db.json.

    Registry schema (per-patient):
      {
        "segs":    [Path, ...]       spine seg files for this patient
        "masks":   [Path, ...]       pelvic mask files for this patient
        "sp_niis": [str, ...]        full-volume spine image NIfTIs (for nifti_anchor)
        "uids":    set[str]          ALL TCIA series UIDs for this patient
        "lstv_pelvic":    str
        "lstv_vertebral": str
        "lstv_agreement": bool | None
        "lstv_confusion_zone": bool
        "lstv_class": int
      }
    """
    # Import locally so the container need not have patient_db.py in its default path.
    from patient_db import PatientDB

    db = PatientDB.from_json(db_path)
    log.info("Loaded patient_db.json: %d patients  (%d complete)",
             len(db.patients), len(db.complete_patients()))

    patients: Dict[str, dict] = {}

    for uid, rec in db.patients.items():
        tok = rec.patient_token
        if token_filter and tok not in token_filter:
            continue
        if not rec.spine_masks and not rec.pelvic_masks:
            continue

        segs    = [Path(m.mask_file) for m in rec.spine_masks
                   if m.mask_file and Path(m.mask_file).exists()]
        masks   = [Path(m.mask_file) for m in rec.pelvic_masks
                   if m.mask_file and Path(m.mask_file).exists()]
        sp_niis = [m.image_file for m in rec.spine_masks
                   if m.image_file and Path(m.image_file).exists()]

        series_uids: Set[str] = set(s.series_uid for s in rec.tcia_series if s.series_uid)

        # LSTV: use first mask of each type (patients typically have one of each)
        spine_lstv  = rec.spine_masks[0].lstv_label  if rec.spine_masks  else ""
        pelvic_lstv = rec.pelvic_masks[0].lstv_label if rec.pelvic_masks else ""
        lstv_fields = _lstv_derived_fields(spine_lstv, pelvic_lstv)

        patients[tok] = {
            "segs":    segs,
            "masks":   masks,
            "sp_niis": sp_niis,
            "uids":    series_uids,
            **lstv_fields,
        }

    # Debug limit
    if debug_n > 0 and not token_filter:
        keep = sorted(
            patients.keys(),
            key=lambda t: (0, int(t)) if t.isdigit() else (1, t),
        )[:debug_n]
        patients = {k: patients[k] for k in keep}

    return patients


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patient_db",       required=True, type=Path,
                   help="patient_db.json from build_db.py (REPLACES --pairs)")
    p.add_argument("--spine_root",       required=True, type=Path,
                   help="CTSpine1K root (unused for data; kept for path interpolation)")
    p.add_argument("--pelvis_root",      required=True, type=Path,
                   help="CTPelvic1K root (unused for data; kept for path interpolation)")
    p.add_argument("--tcia_dir",         required=True, type=Path)
    p.add_argument("--nifti_dir",        required=True, type=Path)
    p.add_argument("--out_dir",          required=True, type=Path)
    p.add_argument("--workers",          default=32, type=int)
    p.add_argument("--dcm2niix_workers", default=16, type=int)
    p.add_argument("--debug_n",          default=0,  type=int)
    p.add_argument("--tokens",           default="", type=str,
                   help="Comma-separated patient tokens (debug mode)")
    p.add_argument("--convert_only",     action="store_true")
    p.add_argument("--skip_convert",     action="store_true")
    args = p.parse_args()

    token_filter: Optional[Set[str]] = None
    if args.tokens.strip():
        token_filter = {t.strip() for t in args.tokens.split(",") if t.strip()}
        log.info("Token filter: %d → %s", len(token_filter), sorted(token_filter))

    # ── Step 0: Build patient registry from patient_db.json ──────────────
    if not args.patient_db.exists():
        log.error("patient_db.json not found: %s", args.patient_db)
        log.error("       Run build_db.py first (Stage 2 Step A).")
        raise SystemExit(1)

    patients = _build_registry_from_patient_db(
        args.patient_db, token_filter, args.debug_n,
    )

    log.info("Patient registry: %d patients", len(patients))
    log.info("  with spine segs  : %d", sum(1 for r in patients.values() if r["segs"]))
    log.info("  with pelvic masks: %d", sum(1 for r in patients.values() if r["masks"]))
    log.info("  with both        : %d", sum(1 for r in patients.values()
                                            if r["segs"] and r["masks"]))
    log.info("  total TCIA UIDs  : %d", sum(len(r["uids"]) for r in patients.values()))

    # ── Build work items ──────────────────────────────────────────────────
    spine_work:  List[Tuple] = []
    pelvic_work: List[Tuple] = []

    spine_out  = args.out_dir / "spine"
    pelvic_out = args.out_dir / "pelvic"
    for d in (spine_out, pelvic_out, args.nifti_dir):
        d.mkdir(parents=True, exist_ok=True)

    uid_dirs: Dict[str, str] = {}

    for token, data in patients.items():
        cand_uids = sorted(
            uid for uid in data["uids"]
            if (args.tcia_dir / uid).is_dir()
        )
        for uid in cand_uids:
            if uid not in uid_dirs:
                series_dir = args.tcia_dir / uid
                if series_dir.is_dir() and any(series_dir.glob("*.dcm")):
                    uid_dirs[uid] = str(series_dir)

        sp_nii_path = data["sp_niis"][0] if data["sp_niis"] else ""

        for seg_path in data["segs"]:
            spine_work.append((
                token, str(seg_path), sp_nii_path,
                cand_uids, str(args.nifti_dir), str(spine_out),
            ))

        for mask_path in data["masks"]:
            pelvic_work.append((
                token, str(mask_path),
                cand_uids, str(args.nifti_dir), str(pelvic_out),
            ))

    log.info("======================================================================")
    log.info("  Patients total          : %d", len(patients))
    log.info("  Spine placements to run : %d", len(spine_work))
    log.info("  Pelvic placements to run: %d", len(pelvic_work))
    log.info("  DICOM UIDs for dcm2niix : %d", len(uid_dirs))
    if token_filter:
        log.info("  Token filter            : %s", sorted(token_filter))
    log.info("======================================================================")

    # ── Step 1: dcm2niix ──────────────────────────────────────────────────
    if not args.skip_convert:
        if uid_dirs:
            convert_series(list(uid_dirs.items()), args.nifti_dir, args.dcm2niix_workers)
    else:
        log.info("Step 1: skipped (--skip_convert)")

    if args.convert_only:
        log.info("--convert_only: done.")
        return

    # ── Step 2: Spine placement ───────────────────────────────────────────
    force = str(os.environ.get("FORCE_PLACEMENT", "0")).strip() == "1"
    if force:
        log.warning("FORCE_PLACEMENT=1: deleting cached spine + pelvic placements")
        for p in spine_out.glob("*_seg_placed.nii.gz"):
            p.unlink()
        for p in pelvic_out.glob("*_pelvic_placed.nii.gz"):
            p.unlink()
        if (args.out_dir / "placed_manifest.json").exists():
            (args.out_dir / "placed_manifest.json").unlink()

    spine_results:  Dict[str, dict] = {}
    if spine_work:
        log.info("Step 2: Spine placement  %d cases  workers=%d",
                 len(spine_work), args.workers)
        _, _, raw_results = run_spine_placement(spine_work, args.workers)
        for tok, ok, msg in raw_results:
            if ok and isinstance(msg, dict):
                spine_results[tok] = msg

    # ── Step 3: Pelvic placement ──────────────────────────────────────────
    pelvic_results: Dict[str, dict] = {}
    if pelvic_work:
        log.info("Step 3: Pelvic placement  %d cases  workers=%d",
                 len(pelvic_work), args.workers)
        _, _, raw_results = run_pelvic_placement(pelvic_work, args.workers)
        for tok, ok, msg in raw_results:
            if ok and isinstance(msg, dict):
                pelvic_results[tok] = msg

    # ── Step 4: Merge results → determine match type ─────────────────────
    all_tokens = sorted(set(spine_results) | set(pelvic_results),
                        key=lambda t: (0, int(t)) if t.isdigit() else (1, t))
    manifest_cases = []
    for tok in all_tokens:
        sp = spine_results.get(tok)
        pv = pelvic_results.get(tok)

        if sp and pv:
            match_type = "fused" if sp["series_uid"] == pv["series_uid"] else "separate"
        elif sp:
            match_type = "spine_only"
        else:
            match_type = "pelvic_only"

        pdata = patients.get(tok, {})
        case = {
            "patient_token":       tok,
            "match_type":          match_type,
            "lstv_pelvic":         pdata.get("lstv_pelvic",    ""),
            "lstv_vertebral":      pdata.get("lstv_vertebral", ""),
            "lstv_agreement":      pdata.get("lstv_agreement"),
            "lstv_confusion_zone": pdata.get("lstv_confusion_zone", False),
            "lstv_class":          pdata.get("lstv_class", 0),
        }
        if sp:
            case["spine"] = sp
        if pv:
            case["pelvic"] = pv
        manifest_cases.append(case)

    # ── Step 5: Write placed_manifest.json ────────────────────────────────
    manifest = {
        "n_cases":       len(manifest_cases),
        "n_fused":       sum(1 for c in manifest_cases if c["match_type"] == "fused"),
        "n_separate":    sum(1 for c in manifest_cases if c["match_type"] == "separate"),
        "n_spine_only":  sum(1 for c in manifest_cases if c["match_type"] == "spine_only"),
        "n_pelvic_only": sum(1 for c in manifest_cases if c["match_type"] == "pelvic_only"),
        "cases":         manifest_cases,
    }
    manifest_path = args.out_dir / "placed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # ── Summary ───────────────────────────────────────────────────────────
    import numpy as np
    log.info("======================================================================")
    log.info("  dcm2niix NIfTIs:    %d", len(list(args.nifti_dir.glob("*.nii.gz"))))
    log.info("  Spine placed:       %d", len(list(spine_out.glob("*_seg_placed.nii.gz"))))
    log.info("  Pelvic placed:      %d", len(list(pelvic_out.glob("*_pelvic_placed.nii.gz"))))
    log.info("  Cases in manifest:  %d  (fused=%d  separate=%d  spine_only=%d  pelvic_only=%d)",
             manifest["n_cases"], manifest["n_fused"], manifest["n_separate"],
             manifest["n_spine_only"], manifest["n_pelvic_only"])
    log.info("======================================================================")

    spine_bps = [
        c["spine"]["bone_pct"] for c in manifest_cases
        if "spine" in c and c["spine"].get("bone_pct") is not None
    ]
    if spine_bps:
        log.info("  Spine bone_pct:  mean=%.1f%%  p10=%.1f%%  p50=%.1f%%  min=%.1f%%",
                 float(np.mean(spine_bps)), float(np.percentile(spine_bps, 10)),
                 float(np.percentile(spine_bps, 50)), min(spine_bps))
        low = sum(1 for v in spine_bps if v < 20)
        if low:
            log.warning("  %d spine cases below BONE_ACCEPT_THRESH (20%%) — check QC", low)

    pelvic_bps = [
        c["pelvic"]["bone_pct"] for c in manifest_cases
        if "pelvic" in c and c["pelvic"].get("bone_pct") is not None
    ]
    if pelvic_bps:
        log.info("  Pelvic bone_pct: mean=%.1f%%  p10=%.1f%%  p50=%.1f%%  min=%.1f%%",
                 float(np.mean(pelvic_bps)), float(np.percentile(pelvic_bps, 10)),
                 float(np.percentile(pelvic_bps, 50)), min(pelvic_bps))

    log.info("  Manifest →  %s", manifest_path)
    log.info("======================================================================")


if __name__ == "__main__":
    main()
