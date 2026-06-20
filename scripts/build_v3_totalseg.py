"""
build_v3_totalseg.py — derive the v3 tree from v2 with a TotalSegmentator pass:
reordered spinopelvic core + the GT thoracic column + femurs (+ an optional S1 carve).

v2 ships radiologist spine GT + model-pseudolabelled pelves (lumbar 1..6, sacrum 7,
hips 8/9, ignore 10). v3, per case:
  1. REMAPS the v2 core into the reordered scheme (lumbar 1..6 unchanged; S1 is
     inserted at 7, so sacrum 7->8, hips 8/9->9/10, ignore 10->50).
  2. Adds the GT THORACIC COLUMN (T1..T13 -> 13..25) from the placed VerSe spine
     mask — these were always in the GT but dropped from v2; v3 ships them.
  3. Runs ONE TotalSegmentator inference and adds the femurs (femur_left/right ->
     11/12) on background.
  4. carves S1 (id 7) out of the GT sacrum (default on; --no_carve_s1 to disable):
     only sacrum voxels that TS calls vertebrae_S1 become S1, so the sacrum's outer
     boundary stays GT.
GT voxels are never overwritten: additions land on background, and the S1 carve
only subdivides the existing sacrum.

Why no ribs?
------------
These are FOV-limited spinopelvic scans — usually only the lower thoracic (~T8 down)
is in view, so there is no full rib cage to count from and neither TS nor a
point-cloud labeler (RibSeg) can NUMBER ribs reliably. v3 therefore does not emit
ribs; ids 26-49 are reserved-but-empty for future manual / AI-assisted annotation.

Output label scheme (v3) — contiguous, ignore highest
-----------------------------------------------------
0 bg | 1-6 L1-L6 | 7 S1 | 8 sacrum | 9 left_hip | 10 right_hip | 11 femur_left |
12 femur_right | 13-25 T1-T13 | (26-49 rib_left/right_1..12 RESERVED, not populated) |
50 ignore.  v3_label_dict() is the exact map.

This is the v3 build stage invoked by slurm/ship_v3.sh inside the TS container.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple

import nibabel as nib
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import relabel_ribs as RR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_v3_totalseg")

# TotalSegmentator "total"-task CT ROI names requested in one pass.
RIB_NAMES: List[str] = (
    [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)]
)
FEMUR_NAMES: List[str] = ["femur_left", "femur_right"]
S1_TS_NAME = "vertebrae_S1"
# Ribs are NOT segmented in v3 (TS can't number them on FOV-limited spinopelvic
# scans). RIB_NAMES is kept only to reserve the rib ids 26-49 in the label scheme.
TS_ROI_NAMES: List[str] = FEMUR_NAMES + [S1_TS_NAME]

# ---------------------------------------------------------------------------
# v3 label scheme (reordered core + GT thoracic column + bone). Contiguous, ignore
# HIGHEST:
#   0 bg | 1-6 L1-L6 | 7 S1 | 8 sacrum | 9 left_hip | 10 right_hip
#   11 femur_left | 12 femur_right | 13-25 T1-T13 (GT thoracic)
#   26-37 rib_left_1..12 | 38-49 rib_right_1..12 | 50 ignore
S1_ID, SACRUM_ID = 7, 8
LEFT_HIP_ID, RIGHT_HIP_ID = 9, 10
FEMUR_LEFT, FEMUR_RIGHT = 11, 12
FEMUR_ID = {"femur_left": FEMUR_LEFT, "femur_right": FEMUR_RIGHT}
THORACIC_BASE = 12                                # T_N -> 12 + N  (T1=13 … T13=25)
V2_IGNORE, V3_IGNORE = 10, 50

# v2 (1-9, ignore 10) -> v3 ids: lumbar 1-6 unchanged; sacrum/hips shift down by the
# S1 insertion; v2 ignore -> 50. (S1 itself is carved from the sacrum, below.)
V2_TO_V3 = {7: SACRUM_ID, 8: LEFT_HIP_ID, 9: RIGHT_HIP_ID, V2_IGNORE: V3_IGNORE}

# GT thoracic column from the placed VerSe spine masks: VerSe 8..19 = T1..T12,
# VerSe 28 = T13. Emitted as output classes 13..25 AND used to number the ribs.
VERSE_THORACIC = {v: THORACIC_BASE + (v - 7) for v in range(8, 20)}
VERSE_THORACIC[28] = THORACIC_BASE + 13

# rib output ids via relabel_ribs offsets: rib_left_N -> 25+N (26..37),
# rib_right_N -> 37+N (38..49).
RR.LEFT_OFFSET = 25
RR.RIGHT_OFFSET = 37


def v3_label_dict() -> Dict[str, int]:
    """The full v3 {name: id} label map (background..ignore), for dataset.json."""
    d = {"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
         "S1": S1_ID, "sacrum": SACRUM_ID, "left_hip": LEFT_HIP_ID,
         "right_hip": RIGHT_HIP_ID, "femur_left": FEMUR_LEFT, "femur_right": FEMUR_RIGHT}
    for n in range(1, 14):                         # T1..T13 -> 13..25
        d[f"T{n}"] = THORACIC_BASE + n
    for n in range(1, 13):
        d[f"rib_left_{n}"] = RR.LEFT_OFFSET + n
    for n in range(1, 13):
        d[f"rib_right_{n}"] = RR.RIGHT_OFFSET + n
    d["ignore"] = V3_IGNORE
    return d


def _nib_to_sitk_ref(ref_img: "nib.Nifti1Image") -> "object":
    """A SimpleITK image with `ref_img`'s geometry (empty pixels) to resample onto."""
    import SimpleITK as sitk
    tmp = nib.Nifti1Image(np.zeros(ref_img.shape[:3], np.uint8), ref_img.affine,
                          ref_img.header)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as fh:
        ref_path = fh.name
    nib.save(tmp, ref_path)
    img = sitk.ReadImage(ref_path, sitk.sitkUInt8)
    Path(ref_path).unlink(missing_ok=True)
    return img


def _sitk_to_nib_array(sitk_img, target_shape: Tuple[int, ...]) -> np.ndarray:
    """SimpleITK image -> numpy in nibabel (i,j,k) data order, shape-checked."""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk_img)            # (z,y,x)
    arr = np.transpose(arr, (2, 1, 0))                # -> (x,y,z) == (i,j,k)
    if arr.shape != tuple(target_shape):
        raise ValueError(f"resampled anchor shape {arr.shape} != ref {tuple(target_shape)}")
    return arr.astype(np.int32)


# ===========================================================================
# GT thoracic column (output classes + rib anchors) + the TotalSegmentator pass
# ===========================================================================
def gt_thoracic_labels(
    spine_mask_path: Optional[Path], ref_img: "nib.Nifti1Image",
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Thoracic vertebrae from the placed VerSe spine mask, remapped to v3 ids
    (T1..T13 -> 13..25) and resampled onto the ref grid. Returns (label array,
    {thoracic number N: world-Z centroid mm}). Empty if no mask / no thoracic GT.
    """
    out = np.zeros(ref_img.shape[:3], dtype=np.int32)
    zmap: Dict[int, float] = {}
    if not spine_mask_path or not Path(spine_mask_path).exists():
        return out, zmap
    import SimpleITK as sitk
    img = sitk.ReadImage(str(spine_mask_path), sitk.sitkInt32)
    arr = sitk.GetArrayFromImage(img)
    remap = np.zeros_like(arr)
    for verse_id, v3id in VERSE_THORACIC.items():
        remap[arr == verse_id] = v3id
    if not remap.any():
        return out, zmap
    rimg = sitk.GetImageFromArray(remap); rimg.CopyInformation(img)
    rs = sitk.ResampleImageFilter(); rs.SetReferenceImage(_nib_to_sitk_ref(ref_img))
    rs.SetInterpolator(sitk.sitkNearestNeighbor); rs.SetTransform(sitk.Transform())
    out = _sitk_to_nib_array(rs.Execute(rimg), ref_img.shape[:3])
    aff = ref_img.affine
    for v3id in (int(v) for v in np.unique(out)):
        if v3id == 0:
            continue
        ijk = np.array(np.nonzero(out == v3id)).mean(axis=1)
        zmap[v3id - THORACIC_BASE] = float(nib.affines.apply_affine(aff, ijk)[2])
    return out, zmap


def _run_ts_ml(ct_path: Path, ref_img: "nib.Nifti1Image", device: str, roi_names):
    """Run TS (valid roi_names, ml) -> (label array on ref grid, {roi_name: value}).

    roi_names not in the CT 'total' task are dropped (with a warning), so MR-only
    names like intervertebral_discs are safe to request.
    """
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    name_to_ts = {name: idx for idx, name in class_map["total"].items()}
    valid = [n for n in roi_names if n in name_to_ts]
    missing = [n for n in roi_names if n not in name_to_ts]
    if missing:
        log.warning("TS 'total' (CT) has no class %s -- skipping it", missing)
    if not valid:
        return np.zeros(ref_img.shape[:3], dtype=np.int32), {}

    pred = totalsegmentator(input=nib.load(str(ct_path)), output=None, task="total",
                            ml=True, device=device, roi_subset=valid, verbose=False)
    arr = np.asarray(pred.dataobj).astype(np.int32)
    if arr.shape[:3] != ref_img.shape[:3]:
        import SimpleITK as sitk                                # rare grid drift -> resample
        m = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)).astype(np.int32))
        m.CopyInformation(_nib_to_sitk_ref(pred))
        rs = sitk.ResampleImageFilter(); rs.SetReferenceImage(_nib_to_sitk_ref(ref_img))
        rs.SetInterpolator(sitk.sitkNearestNeighbor); rs.SetTransform(sitk.Transform())
        arr = _sitk_to_nib_array(rs.Execute(m), ref_img.shape[:3]).astype(np.int32)
    present = set(int(v) for v in np.unique(arr)) - {0}
    name_val = {name: name_to_ts[name] for name in valid}
    if present and not (present & set(name_val.values())):     # compacted roi_subset fallback
        name_val = {name: k for k, name in enumerate(valid, start=1)}
    return arr, name_val


def ts_femurs_and_s1(
    ct_path: Path, ref_img: "nib.Nifti1Image",
    device: str = "gpu", min_voxels: int = 150,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    """Femurs + the TS S1 mask from ONE TS run.

    Femurs are written to their fixed v3 ids on background. The TS vertebrae_S1
    binary is returned separately (the caller carves it into the GT sacrum by
    default; --no_carve_s1 disables it).

    Ribs are deliberately NOT emitted: on these FOV-limited spinopelvic scans there
    is no full rib cage to count from, so neither TS nor a point-cloud labeler
    (RibSeg) can number ribs reliably. Rib ids 26-49 are left reserved-but-empty for
    future manual / AI-assisted annotation. Returns (additions-on-bg, s1_mask, meta).
    """
    arr, name_val = _run_ts_ml(ct_path, ref_img, device, TS_ROI_NAMES)
    present = set(int(v) for v in np.unique(arr)) - {0}
    out = np.zeros(ref_img.shape[:3], dtype=np.int32)
    meta: Dict[str, object] = {"femurs": []}

    # ---- femurs: direct, on background ----
    for name in FEMUR_NAMES:
        v = name_val.get(name)
        if v is not None and v in present:
            mask = arr == v
            if int(mask.sum()) >= min_voxels:
                out[mask] = FEMUR_ID[name]
                meta["femurs"].append(name)

    # ---- TS S1 mask (optionally carved into the GT sacrum by the caller) ----
    s1_mask = None
    sv = name_val.get(S1_TS_NAME)
    if sv is not None and sv in present:
        m = arr == sv
        if int(m.sum()) >= min_voxels:
            s1_mask = m
    return out, s1_mask, meta


# ===========================================================================
# Per-case + driver
# ===========================================================================
@contextlib.contextmanager
def case_tmpdir(base: Path, cid: str) -> Iterator[Path]:
    """Pin TS/nnUNet temp I/O to a fresh per-case dir and delete it afterwards.

    The TS python API (output=None) and nnUNet write hundreds of MB of temp
    NIfTIs per case under tempfile.gettempdir() (the NFS-bound container /tmp) and
    do NOT clean them up. Left to accumulate they fill the scratch after ~100
    cases, and every TS call past that point dies with
    `FileNotFoundError: /tmp/...` — silently shipping bone-less labels. Pinning
    TMPDIR per case and rmtree-ing it keeps /tmp bounded to a single case's
    footprint, so the run can't fill. Restores the env on exit even on error."""
    d = base / f"case_{cid}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in ("TMPDIR", "TMP", "TEMP")}
    prev_tempdir = tempfile.tempdir
    for k in ("TMPDIR", "TMP", "TEMP"):
        os.environ[k] = str(d)
    tempfile.tempdir = str(d)
    try:
        yield d
    finally:
        tempfile.tempdir = prev_tempdir
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)


def _save_label(arr, affine, header, out_path: Path) -> None:
    """Write a uint16 label, FIRST breaking any pre-existing hardlink at the target.

    The v2->v3 mirror may have hardlinked this path to the v2 label (same inode);
    `nib.save` truncates in place, so writing without unlinking first would corrupt
    the v2 file. Unlinking guarantees a fresh inode — v3 writes never touch v2."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    nib.save(nib.Nifti1Image(np.asarray(arr).astype(np.uint16), affine, header),
             str(out_path))


def _carve_s1_slab(merged: np.ndarray, s1_mask: np.ndarray, affine) -> int:
    """Carve S1 (id 7) as a clean slab of the GT sacrum (id 8), split along the
    sacrum's PRINCIPAL AXIS so the S1/S2 plane follows pelvic tilt, not world-Z.

    TS-S1 only locates the cut; the whole cranial slab of the sacrum becomes S1, so
    no sacrum speckles remain inside the S1 body. Returns S1 voxel count."""
    sac = merged == SACRUM_ID
    seed = sac & s1_mask
    if not seed.any() or not sac.any():
        return 0
    sac_ijk = np.array(np.nonzero(sac)).T
    sac_w = nib.affines.apply_affine(affine, sac_ijk)
    center = sac_w.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((sac_w - center).T))
    # principal (long) axis of the sacrum — highest-variance axis that still has a
    # real cranio-caudal component (so the ala width can't pick a left-right axis).
    axis = None
    for k in np.argsort(evals)[::-1]:
        if abs(evecs[2, k]) > 0.3:
            axis = evecs[:, k].copy(); break
    if axis is None:
        axis = evecs[:, int(np.argmax(evals))].copy()
    if axis[2] < 0:                                  # orient cranially (superior = +)
        axis = -axis
    sac_proj = (sac_w - center) @ axis
    seed_proj = (nib.affines.apply_affine(affine, np.array(np.nonzero(seed)).T) - center) @ axis
    cut = float(np.percentile(seed_proj, 10))        # S1/S2 boundary along the axis
    promote = sac_ijk[sac_proj >= cut]
    merged[promote[:, 0], promote[:, 1], promote[:, 2]] = S1_ID
    return int(promote.shape[0])


def process_case(
    ct_path: Path, v2_label_path: Path, spine_mask_path: Optional[Path],
    out_label_path: Path, *, device: str = "gpu", min_voxels: int = 150,
    carve_s1: bool = True,
) -> Dict[str, object]:
    """Build the reordered v3 label: remap v2 core -> add GT thoracic -> add TS
    ribs/femurs (-> optionally carve S1 out of the sacrum)."""
    lbl_img = nib.load(str(v2_label_path))
    v2 = np.asarray(lbl_img.dataobj).astype(np.int32)
    qc: Dict[str, object] = {"ct": ct_path.name, "femur_vox": 0,
                             "status": "ok", "note": ""}

    # 1) remap the v2 core labels into the reordered v3 ids (lumbar 1-6 unchanged;
    #    sacrum 7->8, hips 8/9->9/10, ignore 10->50). Index by the original v2 so the
    #    shifts can't collide.
    merged = v2.copy()
    for old, new in V2_TO_V3.items():
        merged[v2 == old] = new

    # 2) GT thoracic column (output classes 13-25), on background
    thor_vol, vert_z = gt_thoracic_labels(spine_mask_path, lbl_img)
    pl = (merged == 0) & (thor_vol > 0)
    merged[pl] = thor_vol[pl]

    # 3) TS femurs (+ the S1 mask) on background. Ribs are NOT emitted in v3 — TS
    #    can't number them on the FOV-limited spinopelvic scans; ids 26-49 stay
    #    reserved for future manual / AI-assisted annotation.
    add_vol, s1_mask, meta = ts_femurs_and_s1(ct_path, lbl_img,
                                              device=device, min_voxels=min_voxels)
    pl = (merged == 0) & (add_vol > 0)
    n_bone = int(pl.sum())
    merged[pl] = add_vol[pl]

    # 4) carve S1 (id 7) as a slab of the GT sacrum, split along the sacrum's
    #    PRINCIPAL AXIS so the S1/S2 plane follows pelvic tilt (not world-Z). On by
    #    default (--no_carve_s1 to disable). Only subdivides the existing sacrum in
    #    place — the sacrum's outer boundary stays radiologist GT.
    n_s1 = (_carve_s1_slab(merged, s1_mask, lbl_img.affine)
            if (carve_s1 and s1_mask is not None) else 0)

    _save_label(merged, lbl_img.affine, lbl_img.header, out_label_path)
    n_thor = len(vert_z)
    qc.update(femur_vox=n_bone, status="ok",
              note=f"thoracic={n_thor} femurs={meta['femurs']} s1_vox={n_s1}")
    log.info("  %s: %d thoracic vertebra(e) + %d femur(s) + S1(%d vox)",
             ct_path.name, n_thor, len(meta["femurs"]), n_s1)
    return qc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2_dir", required=True, type=Path, help="v2 tree (ct/, labels/, manifest.json)")
    ap.add_argument("--v3_dir", required=True, type=Path, help="v3 output tree")
    ap.add_argument("--spine_dir", required=False, type=Path, default=None,
                    help="placed VerSe spine masks ({uid}_seg_placed.nii.gz); used "
                         "for the T12 numbering anchor. Missing -> TS-native numbers.")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--min_voxels", type=int, default=150,
                    help="drop a TS rib whose voxel count is below this (spurious blob)")
    ap.add_argument("--carve_s1", action="store_true", default=True,
                    help="carve an S1 (id 7) slab out of the GT sacrum (default ON)")
    ap.add_argument("--no_carve_s1", dest="carve_s1", action="store_false",
                    help="disable the S1 carve (id 7 left empty)")
    ap.add_argument("--dilation_radius", type=int, default=4)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap cases (debug)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip cases already rib-processed (default on) — a timed-out "
                         "or preempted job continues instead of restarting")
    ap.add_argument("--no_resume", dest="resume", action="store_false",
                    help="force a full rebuild (ignore .totalseg_done markers)")
    args = ap.parse_args()

    manifest = json.loads((args.v2_dir / "manifest.json").read_text())
    records = manifest["records"] if isinstance(manifest, dict) and "records" in manifest else manifest

    # Mirror the v2 tree IDEMPOTENTLY: hardlink (fallback copy) each CT/label only if
    # it is absent in v3, so a RESUME does NOT re-copy ~188 GB of CTs every run.
    # process_case overwrites the labels it ribs; everything else is left in place.
    args.v3_dir.mkdir(parents=True, exist_ok=True)
    def _mirror(src: Path, dst: Path, *, hardlink: bool) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            try:
                os.link(src, dst)              # CTs ONLY: never modified, safe to share inode
                return
            except OSError:
                pass
        shutil.copy2(src, dst)                 # labels: independent copy (v3 OVERWRITES them;
                                               # a hardlink here would corrupt the v2 label)
    for sub in ("ct", "labels"):
        sd = args.v2_dir / sub
        if sd.exists():
            for f in sd.glob("*.nii.gz"):
                _mirror(f, args.v3_dir / sub / f.name, hardlink=(sub == "ct"))
    for f in args.v2_dir.glob("*.json"):
        shutil.copy2(f, args.v3_dir / f.name)

    # Resume: a per-case marker (holding that case's QC row) is written once a case
    # is fully rib-processed. On restart, completed cases are skipped — so a job that
    # times out / is preempted continues instead of re-running TotalSegmentator on
    # the cases it already finished. Clear .totalseg_done to force a full rebuild.
    # Markers live in a _work sibling, NOT inside the v3 tree, so they never ship to HF.
    done_dir = args.v3_dir.parent / (args.v3_dir.name + "_work") / "totalseg_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    done: Dict[str, dict] = {}
    if args.resume:
        for m in done_dir.glob("*.json"):
            try:
                done[m.stem] = json.loads(m.read_text())
            except Exception:
                pass
        if done:
            log.info("resume: %d case(s) already rib-processed — skipping", len(done))

    qc_rows: List[Dict[str, object]] = []
    # Rib the RELEASED set = 802: 342 fused + 440 spine_only + the 20 PURE
    # pelvic-only orphans (config=pelvic_native AND match_type=pelvic_only) whose
    # ONLY acquisition is the pelvic scan, so they were pseudo-spined and shipped.
    # The ~351 separate-mode pelvic sides (match_type=separate) are NOT ribbed:
    # that patient's spine acquisition is the released spine_only volume instead.
    # (Mirrors the scoping in pseudolabel.py.)
    from collections import Counter
    def _released(r) -> bool:
        if r.get("config") in ("fused", "spine_only"):
            return True
        return (r.get("config") == "pelvic_native"
                and r.get("match_type") == "pelvic_only")
    todo = [r for r in records if _released(r)]
    if args.limit:
        todo = todo[: args.limit]
    log.info("v3 TotalSegmentator: %d case(s) to process  breakdown=%s",
             len(todo), dict(Counter(r.get("config") for r in todo)))

    # Root for per-case temp dirs (see case_tmpdir): a sibling of the original
    # tempdir so TS's temp NIfTIs stay on the same fs but get purged each case.
    tmp_root = Path(tempfile.gettempdir()) / f"v3_work_{os.getpid()}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(todo, 1):
        label_rel = r.get("label_file") or ""
        ct_rel = r.get("ct_file") or ""
        if not label_rel or not ct_rel:
            continue
        cid = Path(label_rel).name[: -len(".nii.gz")]
        out_label_path = args.v3_dir / label_rel
        # Skip ONLY if marked done AND the output label is actually present.
        if args.resume and cid in done and out_label_path.exists():
            qc_rows.append(done[cid])
            continue
        ct_path = args.v2_dir / ct_rel
        v2_label_path = args.v2_dir / label_rel
        spine_uid = r.get("spine_series_uid")
        spine_mask = (args.spine_dir / f"{spine_uid}_seg_placed.nii.gz") \
            if (args.spine_dir and spine_uid) else None
        log.info("[%d/%d] token=%s config=%s", i, len(todo), r.get("token"), r.get("config"))
        try:
            with case_tmpdir(tmp_root, cid):
                qc = process_case(ct_path, v2_label_path, spine_mask, out_label_path,
                                  device=args.device, min_voxels=args.min_voxels,
                                  carve_s1=args.carve_s1)
        except Exception as exc:                                       # noqa: BLE001
            # log.exception so the REAL TS/nnUNet traceback lands in the .err log —
            # a bare str(exc) ("FileNotFoundError: /tmp/...") hides the root cause.
            log.exception("  token=%s FAILED: %s — shipping v2 label (core remapped, no bone)",
                          r.get("token"), exc)
            # Still apply the v2->v3 core remap so the scheme stays uniform.
            li = nib.load(str(v2_label_path))
            la = np.asarray(li.dataobj).astype(np.int32)
            base = la.copy()
            for old, new in V2_TO_V3.items():
                la[base == old] = new
            _save_label(la, li.affine, li.header, out_label_path)
            qc = {"ct": ct_path.name, "status": "error", "note": str(exc)[:200],
                  "femur_vox": 0}
        qc["token"] = r.get("token")
        qc_rows.append(qc)
        # Mark done only after the output label is on disk (a timeout mid-case leaves
        # no marker -> that case re-runs next time; finished cases never re-run).
        if out_label_path.exists():
            (done_dir / f"{cid}.json").write_text(json.dumps(qc))

    import csv
    qc_path = args.v3_dir / "totalseg_qc.csv"
    with open(qc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "ct", "status", "femur_vox", "note"])
        w.writeheader()
        for row in qc_rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    # Emit the v3 label scheme (training-contiguous, ignore=34) for dataset.json.
    (args.v3_dir / "dataset_labels.json").write_text(json.dumps(v3_label_dict(), indent=2))

    shutil.rmtree(tmp_root, ignore_errors=True)

    n_ok = sum(1 for r in qc_rows if r.get("status") == "ok")
    n_err = sum(1 for r in qc_rows if r.get("status") == "error")
    log.info("v3 TotalSegmentator done: %d ok / %d error of %d -> %s  (labels: dataset_labels.json)",
             n_ok, n_err, len(qc_rows), qc_path)

    # Fail-fast: a TS crash (filled /tmp, bad container, …) makes every case ship a
    # bone-less label while main() still returns 0 — which lets ship_v3's afterok
    # push upload a half-bone tree. Refuse to signal success when TS errored on more
    # than a tolerated handful, so the push does NOT run on a broken build. Resume
    # markers keep the cases that DID succeed, so a fixed resubmit only redoes the rest.
    tolerance = max(5, len(qc_rows) // 100)
    if n_err > tolerance:
        log.error("ABORT: %d/%d cases errored (> tolerance %d) — likely a TS temp/IO "
                  "failure. NOT signalling success so the push is blocked; check the "
                  "traceback above, fix, and resubmit (the %d good cases are kept).",
                  n_err, len(qc_rows), tolerance, n_ok)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
