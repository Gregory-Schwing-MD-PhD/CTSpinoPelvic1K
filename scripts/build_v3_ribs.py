"""
build_v3_ribs.py — derive the v3 tree from v2 with a TotalSegmentator pass:
GT-matched ribs + femurs + spinal_cord.

v2 ships radiologist spine GT + model-pseudolabelled pelves (classes 1..9, ignore
10). v3 = v2 + a TotalSegmentator pass (one inference/case) adding GT-matched ribs
plus extra structures (femurs, spinal_cord). Ribs are emitted
ONLY where a GT thoracic vertebra backs them: for each GT vertebra T_N present in
the spine mask, the TS rib whose head sits at that level (nearest in Z, within ~1
vertebra) is labelled rib N; ribs with no GT vertebra at their level are dropped,
and a case with no thoracic GT gets no ribs. The extra structures are unambiguous
and added directly (any name TS's CT task lacks is skipped). Everything is merged
into the v2 label volume WITHOUT ever overwriting an existing v2 voxel (additions
land only on background).

Why GT-vertebra-matched ribs?
-----------------------------
Rib numbering comes entirely from the radiologist GT vertebrae, so nothing depends
on TotalSegmentator's (un-reviewed) vertebra numbering, and every rib is grounded in
a real GT vertebra. A mislabelled GT vertebra would simply have no rib near its
level and produce nothing. (The earlier overlap-with-dilated-vertebra method
missed ribs across the costovertebral joint gap; this Z-level match does not.)

Output label scheme (v3)
------------------------
Spine/pelvis GT 1..9 untouched | rib_left N -> 9+N (10..21) | rib_right N -> 21+N
(22..33) | then EXTRA_ROIS at 34.. (femur_left, femur_right, spinal_cord) |
ignore highest. Additions land on background only; GT voxels are never
overwritten. v3_label_dict() is the exact map.

This is the v3 build stage invoked by slurm/ship_v3.sh inside the TS container.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import relabel_ribs as RR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_v3_ribs")

# TotalSegmentator "total"-task ROI names: 12 ribs per side + the two femurs.
RIB_NAMES: List[str] = (
    [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)]
)
# Extra single-label ROIs added directly (no GT anchoring), in label-id order.
# Any name not in TS's CT "total" task is skipped at runtime with a warning.
# (intervertebral_discs is MR-only on TS, so per-level discs are derived
# geometrically from the GT vertebrae instead -- not requested here.)
EXTRA_ROIS: List[str] = ["femur_left", "femur_right", "spinal_cord"]
TS_ROI_NAMES: List[str] = RIB_NAMES + EXTRA_ROIS

# VerSe ids 8..19 == thoracic T1..T12 (rib anchors).
VERSE_THORACIC_LO, VERSE_THORACIC_HI = 8, 19

# ---------------------------------------------------------------------------
# v3 TRAINING-CONTIGUOUS label scheme. nnU-Net requires consecutive label ids
# with the ignore label HIGHEST:
#   0 bg | 1..6 L1..L6 | 7 sacrum | 8 left_hip | 9 right_hip
#   10..21 rib_left_1..12 | 22..33 rib_right_1..12
#   34.. EXTRA_ROIS (femur_left, femur_right, spinal_cord, intervertebral_discs)
#   then ignore (highest). v2 ignore (10) is remapped to V3_IGNORE so it never
#   collides with rib id 10.
RR.LEFT_OFFSET = 9
RR.RIGHT_OFFSET = 21
EXTRA_BASE = 34                                   # first id after rib_right_12 (33)
EXTRA_IDS = {name: EXTRA_BASE + i for i, name in enumerate(EXTRA_ROIS)}
V2_IGNORE = 10
V3_IGNORE = EXTRA_BASE + len(EXTRA_ROIS)           # after the extras


def v3_label_dict() -> Dict[str, int]:
    """The full v3 {name: id} label map (background..ignore), for dataset.json."""
    d = {"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
         "sacrum": 7, "left_hip": 8, "right_hip": 9}
    for n in range(1, 13):
        d[f"rib_left_{n}"] = 9 + n
    for n in range(1, 13):
        d[f"rib_right_{n}"] = 21 + n
    for name, fid in EXTRA_IDS.items():
        d[name] = fid
    d["ignore"] = V3_IGNORE
    return d


# ===========================================================================
# Thoracic anchors from the placed VerSe spine mask
# ===========================================================================
def thoracic_anchor_on_grid(
    spine_mask_path: Path, ref_img: "nib.Nifti1Image",
) -> Optional[np.ndarray]:
    """Extract T1..T12 from a VerSe spine mask and resample onto `ref_img`'s grid.

    Returns an int array (ref grid) with voxel value N = thoracic T-N (1..12), or
    None if no thoracic vertebra is present in the mask. Resampling is physical-
    space nearest-neighbour so the anchors line up with the TS ribs regardless of
    the two files' stored orientations.
    """
    import SimpleITK as sitk
    verse = sitk.ReadImage(str(spine_mask_path), sitk.sitkInt32)
    arr = sitk.GetArrayFromImage(verse)
    thoracic = (arr >= VERSE_THORACIC_LO) & (arr <= VERSE_THORACIC_HI)
    if not thoracic.any():
        return None
    # Remap 8..19 -> 1..12 in place (0 elsewhere), keep the VerSe geometry.
    remap = np.where(thoracic, arr - (VERSE_THORACIC_LO - 1), 0).astype(np.int32)
    remap_img = sitk.GetImageFromArray(remap)
    remap_img.CopyInformation(verse)

    # Build a SimpleITK reference from the nibabel ref so we resample to ITS grid.
    ref_sitk = _nib_to_sitk_ref(ref_img)
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref_sitk)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    out = rs.Execute(remap_img)
    # SimpleITK array is (z,y,x); nibabel/relabel_ribs work in (i,j,k) = data order.
    return _sitk_to_nib_array(out, ref_img.shape[:3])


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
# TotalSegmentator ribs, numbered from the GT thoracic vertebrae
# ===========================================================================
def _gt_thoracic_z(spine_mask_path: Optional[Path]) -> Dict[int, float]:
    """{thoracic number N: world-Z centroid (mm)} for VerSe T1..T12 (ids 8..19)
    actually present in the GT spine mask. Empty if none / no mask."""
    vert: Dict[int, float] = {}
    if spine_mask_path and Path(spine_mask_path).exists():
        img = nib.load(str(spine_mask_path))
        arr, aff = np.asarray(img.dataobj), img.affine
        for vid in range(8, 20):                   # VerSe 8..19 == T1..T12
            m = arr == vid
            if m.any():
                ijk = np.array(np.nonzero(m)).mean(axis=1)
                vert[vid - 7] = float(nib.affines.apply_affine(aff, ijk)[2])
    return vert


def _vertebra_spacing(spine_mask_path: Path, default: float = 25.0) -> float:
    """Median consecutive vertebra-centroid Z gap (mm) from the spine mask."""
    img = nib.load(str(spine_mask_path))
    arr, aff = np.asarray(img.dataobj), img.affine
    zs = []
    for vid in range(8, 26):                        # thoracic + lumbar
        m = arr == vid
        if m.any():
            ijk = np.array(np.nonzero(m)).mean(axis=1)
            zs.append(float(nib.affines.apply_affine(aff, ijk)[2]))
    zs.sort()
    gaps = np.diff(zs) if len(zs) > 1 else np.array([])
    gaps = gaps[(gaps > 10) & (gaps < 50)]          # plausible vertebra heights
    return float(np.median(gaps)) if gaps.size else default


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


def ts_ribs_and_extras(
    ct_path: Path, ref_img: "nib.Nifti1Image", spine_mask_path: Optional[Path],
    device: str = "gpu", min_voxels: int = 150,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """GT-vertebra-matched ribs + the EXTRA_ROIS (femurs, cord, discs) from ONE TS run.

    Ribs: emit rib N only where GT thoracic vertebra T_N exists, matched to the TS
    rib at that Z-level (numbering entirely from the GT vertebra). Extras: each
    EXTRA_ROIS class that TS actually produced is written to its fixed v3 id
    directly (unambiguous, no anchoring). Returns (v3-id volume, meta).
    """
    arr, name_val = _run_ts_ml(ct_path, ref_img, device, TS_ROI_NAMES)
    present = set(int(v) for v in np.unique(arr)) - {0}
    affine = ref_img.affine
    out = np.zeros(ref_img.shape[:3], dtype=np.int32)
    meta: Dict[str, object] = {"gt_thoracic": [], "n_matched": 0, "extras": []}

    # ---- ribs: GT-vertebra-matched ----
    vert = _gt_thoracic_z(spine_mask_path)
    meta["gt_thoracic"] = sorted(vert)
    rib_val = {name_val[f"rib_{s}_{n}"]: (s, n)
               for s in ("left", "right") for n in range(1, 13)
               if name_val.get(f"rib_{s}_{n}") is not None}
    rib_vals = [v for v in rib_val if v in present]
    if vert and rib_vals:
        x_mid = float(np.median(nib.affines.apply_affine(
            affine, np.array(np.nonzero(np.isin(arr, rib_vals))).T)[:, 0]))
        byside: Dict[str, list] = {"left": [], "right": []}    # [head_z, mask]
        for v in rib_vals:
            s, _n = rib_val[v]
            mask = arr == v
            if int(mask.sum()) < min_voxels:
                continue
            world = nib.affines.apply_affine(affine, np.array(np.nonzero(mask)).T)
            dx = np.abs(world[:, 0] - x_mid)
            head = dx <= np.quantile(dx, 0.30)
            byside[s].append([float(world[head, 2].mean()), mask])
        tol = _vertebra_spacing(spine_mask_path) * 0.9         # head within ~1 vertebra
        for s in ("left", "right"):
            ribs = byside[s]
            used = [False] * len(ribs)
            for N, zN in sorted(vert.items()):                 # one rib per GT vertebra/side
                best, bestd = -1, tol
                for j, (hz, _m) in enumerate(ribs):
                    if used[j]:
                        continue
                    if abs(hz - zN) < bestd:
                        best, bestd = j, abs(hz - zN)
                if best >= 0:
                    used[best] = True
                    out[ribs[best][1]] = (RR.LEFT_OFFSET + N) if s == "left" else (RR.RIGHT_OFFSET + N)
                    meta["n_matched"] = int(meta["n_matched"]) + 1

    # ---- extras: direct single-label ROIs (femurs, spinal_cord, discs) ----
    for name in EXTRA_ROIS:
        v = name_val.get(name)
        if v is not None and v in present:
            mask = arr == v
            if int(mask.sum()) >= min_voxels:
                out[mask] = EXTRA_IDS[name]
                meta["extras"].append(name)
    return out, meta


# ===========================================================================
# GT-safe merge
# ===========================================================================
def merge_ribs_into_label(v2_label: np.ndarray, rib_vol: np.ndarray) -> Tuple[np.ndarray, int]:
    """Lay ribs onto v2 labels ONLY where v2 is background (0). GT is never touched.

    Returns (merged, n_written). Ribs that would land on an existing v2 voxel
    (spine/pelvis GT or ignore=10) are dropped, so the merge can never corrupt the
    shipped ground truth.
    """
    merged = v2_label.copy()
    place = (v2_label == 0) & (rib_vol > 0)
    merged[place] = rib_vol[place].astype(merged.dtype)
    return merged, int(place.sum())


# ===========================================================================
# Per-case + driver
# ===========================================================================
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


def process_case(
    ct_path: Path, v2_label_path: Path, spine_mask_path: Optional[Path],
    out_label_path: Path, *, device: str = "gpu", min_voxels: int = 150,
) -> Dict[str, object]:
    """Add GT-matched ribs + TS extras (femurs/cord/discs) to one case."""
    lbl_img = nib.load(str(v2_label_path))
    v2_label = np.asarray(lbl_img.dataobj).astype(np.int32)
    # Move the v2 ignore (10) -> V3_IGNORE so it never collides with rib_left_1 (10).
    v2_label[v2_label == V2_IGNORE] = V3_IGNORE

    qc: Dict[str, object] = {"ct": ct_path.name, "ribs_written": 0, "n_ribs": 0,
                             "status": "ok", "note": ""}

    add_vol, meta = ts_ribs_and_extras(ct_path, lbl_img, spine_mask_path,
                                       device=device, min_voxels=min_voxels)
    merged, n_written = merge_ribs_into_label(v2_label, add_vol)
    _save_label(merged, lbl_img.affine, lbl_img.header, out_label_path)

    has = bool(np.any(add_vol))
    status = "ok" if has else ("no_thoracic_gt" if not meta["gt_thoracic"] else "no_labels")
    qc.update(ribs_written=n_written, n_ribs=meta["n_matched"], status=status,
              note=f"gt_thoracic={meta['gt_thoracic']} ribs={meta['n_matched']} "
                   f"extras={meta['extras']}")
    log.info("  %s: %d rib(s) + extras %s (GT thoracic %s), %d vox merged",
             ct_path.name, meta["n_matched"], meta["extras"], meta["gt_thoracic"],
             n_written)
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
    ap.add_argument("--dilation_radius", type=int, default=4)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap cases (debug)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip cases already rib-processed (default on) — a timed-out "
                         "or preempted job continues instead of restarting")
    ap.add_argument("--no_resume", dest="resume", action="store_false",
                    help="force a full rebuild (ignore .rib_done markers)")
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
    # the cases it already finished. Clear .rib_done to force a full rebuild.
    # Markers live in a _work sibling, NOT inside the v3 tree, so they never ship to HF.
    done_dir = args.v3_dir.parent / (args.v3_dir.name + "_work") / "rib_done"
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
    log.info("v3 ribs: %d case(s) to process  breakdown=%s",
             len(todo), dict(Counter(r.get("config") for r in todo)))

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
            qc = process_case(ct_path, v2_label_path, spine_mask, out_label_path,
                              device=args.device, min_voxels=args.min_voxels)
        except Exception as exc:                                       # noqa: BLE001
            log.error("  token=%s FAILED: %s — shipping v2 label (ignore remapped, no ribs)",
                      r.get("token"), exc)
            # Still apply the ignore 10->34 remap so v3's ignore id stays uniform.
            li = nib.load(str(v2_label_path))
            la = np.asarray(li.dataobj).astype(np.int32)
            la[la == V2_IGNORE] = V3_IGNORE
            _save_label(la, li.affine, li.header, out_label_path)
            qc = {"ct": ct_path.name, "status": "error", "note": str(exc)[:200],
                  "ribs_written": 0, "n_ribs": 0}
        qc["token"] = r.get("token")
        qc_rows.append(qc)
        # Mark done only after the output label is on disk (a timeout mid-case leaves
        # no marker -> that case re-runs next time; finished cases never re-run).
        if out_label_path.exists():
            (done_dir / f"{cid}.json").write_text(json.dumps(qc))

    import csv
    qc_path = args.v3_dir / "rib_qc.csv"
    with open(qc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "ct", "status", "n_ribs",
                                           "ribs_written", "note"])
        w.writeheader()
        for row in qc_rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    # Emit the v3 label scheme (training-contiguous, ignore=34) for dataset.json.
    (args.v3_dir / "dataset_labels.json").write_text(json.dumps(v3_label_dict(), indent=2))

    n_ok = sum(1 for r in qc_rows if r["status"] == "ok")
    log.info("v3 ribs done: %d/%d cases got ribs -> %s  (labels: dataset_labels.json)",
             n_ok, len(qc_rows), qc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
