"""
build_v3_ribs.py — derive the v3 tree from v2 by adding instance-labelled ribs.

v2 ships radiologist spine GT + model-pseudolabelled pelves (classes 1..9, ignore
10). v3 = v2 + ribs: for each case we run TotalSegmentator restricted to the 24 rib
ROIs, then re-number those ribs anatomically from the GROUND-TRUTH thoracic
vertebrae (TotalSegmentator's own rib numbers are wrong on a partial FOV — see
relabel_ribs.py), and merge the result into the v2 label volume WITHOUT ever
overwriting an existing v2 voxel (ribs land only on background).

Thoracic anchors
----------------
The shipped v2 labels are the canonical 1..9 set and contain NO thoracic
vertebrae, so the anchors come from the placed VerSe spine masks (--spine_dir),
where thoracic T1..T12 are VerSe ids 8..19. We remap 8..19 -> 1..12 and resample
into the v2 CT grid (SimpleITK, physical space, nearest-neighbour) so the anchors
and the TS ribs share one voxel grid.

Output label scheme (v3 ribs)
-----------------------------
Left rib N  -> relabel_ribs.LEFT_OFFSET  + N  (default 100+N)
Right rib N -> relabel_ribs.RIGHT_OFFSET + N  (default 200+N)
Spine/pelvis GT (1..9) and ignore (10) are untouched.

This is the v3 build stage invoked by slurm/ship_v3.sh inside the TS container.
"""
from __future__ import annotations

import argparse
import json
import logging
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

# TotalSegmentator "total"-task rib ROI names (12 per side).
RIB_NAMES: List[str] = (
    [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)]
)

# VerSe ids 8..19 == thoracic T1..T12. Remap to the 1..12 that relabel_ribs expects.
VERSE_THORACIC_LO, VERSE_THORACIC_HI = 8, 19


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
# TotalSegmentator ribs -> binary
# ===========================================================================
def ts_rib_binary(ct_path: Path, ref_img: "nib.Nifti1Image", device: str = "gpu") -> np.ndarray:
    """Run TS restricted to ribs and return a binary rib mask on the CT grid.

    With roi_subset = the 24 ribs and ml=True, EVERY non-zero voxel in the output
    is a rib, so binarising is just `> 0`. TS already runs on `ct_path` (the v2 CT)
    so the result is on that grid; we only sanity-check the shape.
    """
    from totalsegmentator.python_api import totalsegmentator
    pred = totalsegmentator(input=nib.load(str(ct_path)), output=None, task="total",
                            ml=True, device=device, roi_subset=RIB_NAMES, verbose=False)
    arr = np.asarray(pred.dataobj)
    if arr.shape[:3] != ref_img.shape[:3]:
        import SimpleITK as sitk                                # rare grid drift -> resample
        m = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)).astype(np.int32))
        m.CopyInformation(_nib_to_sitk_ref(pred))
        rs = sitk.ResampleImageFilter(); rs.SetReferenceImage(_nib_to_sitk_ref(ref_img))
        rs.SetInterpolator(sitk.sitkNearestNeighbor); rs.SetTransform(sitk.Transform())
        arr = _sitk_to_nib_array(rs.Execute(m), ref_img.shape[:3])
    return (arr > 0).astype(np.uint8)


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
def process_case(
    ct_path: Path, v2_label_path: Path, spine_mask_path: Optional[Path],
    out_label_path: Path, *, device: str = "gpu", min_voxels: int = 500,
    dilation_radius: int = 4, pad: int = 10,
) -> Dict[str, object]:
    """Add ribs to one case; write the merged v3 label. Returns a QC dict."""
    lbl_img = nib.load(str(v2_label_path))
    v2_label = np.asarray(lbl_img.dataobj).astype(np.int32)

    qc: Dict[str, object] = {"ct": ct_path.name, "ribs_written": 0, "n_ribs": 0,
                             "status": "ok", "note": ""}

    anchor = (thoracic_anchor_on_grid(spine_mask_path, lbl_img)
              if spine_mask_path and spine_mask_path.exists() else None)
    if anchor is None:
        # No thoracic ruler -> we cannot number ribs; ship v2 label unchanged.
        nib.save(nib.Nifti1Image(v2_label.astype(np.uint16), lbl_img.affine, lbl_img.header),
                 str(out_label_path))
        qc.update(status="no_thoracic_anchor",
                  note="no T1..T12 in placed spine mask / not in FOV")
        log.info("  %s: no thoracic anchor — v2 label copied unchanged", ct_path.name)
        return qc

    binary = ts_rib_binary(ct_path, lbl_img, device=device)
    labeled, kept = RR.label_and_filter_components(binary, min_voxels=min_voxels)
    dil = RR.dilate_vertebrae_local(anchor, dilation_radius=dilation_radius, pad=pad)
    assignments = RR.assign_ribs(labeled, kept, anchor, dil, lbl_img.affine)
    rib_vol = RR.build_output_volume(labeled, assignments)

    merged, n_written = merge_ribs_into_label(v2_label, rib_vol)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(merged.astype(np.uint16), lbl_img.affine, lbl_img.header),
             str(out_label_path))
    qc.update(ribs_written=n_written, n_ribs=len(assignments))
    log.info("  %s: %d rib(s) assigned, %d voxel(s) merged onto background",
             ct_path.name, len(assignments), n_written)
    return qc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2_dir", required=True, type=Path, help="v2 tree (ct/, labels/, manifest.json)")
    ap.add_argument("--v3_dir", required=True, type=Path, help="v3 output tree")
    ap.add_argument("--spine_dir", required=True, type=Path,
                    help="placed VerSe spine masks ({uid}_seg_placed.nii.gz) for thoracic anchors")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--min_voxels", type=int, default=500)
    ap.add_argument("--dilation_radius", type=int, default=4)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap cases (debug)")
    args = ap.parse_args()

    manifest = json.loads((args.v2_dir / "manifest.json").read_text())
    records = manifest["records"] if isinstance(manifest, dict) and "records" in manifest else manifest

    # Mirror the v2 tree, then overwrite the labels we touch (CT + manifest unchanged).
    args.v3_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("ct", "labels"):
        if (args.v2_dir / sub).exists():
            shutil.copytree(args.v2_dir / sub, args.v3_dir / sub, dirs_exist_ok=True)
    for f in args.v2_dir.glob("*.json"):
        shutil.copy2(f, args.v3_dir / f.name)

    qc_rows: List[Dict[str, object]] = []
    todo = [r for r in records if r.get("config") in ("fused", "spine_only")]
    if args.limit:
        todo = todo[: args.limit]
    log.info("v3 ribs: %d case(s) to process", len(todo))

    for i, r in enumerate(todo, 1):
        label_rel = r.get("label_file") or ""
        ct_rel = r.get("ct_file") or ""
        if not label_rel or not ct_rel:
            continue
        ct_path = args.v2_dir / ct_rel
        v2_label_path = args.v2_dir / label_rel
        out_label_path = args.v3_dir / label_rel
        spine_uid = r.get("spine_series_uid")
        spine_mask = (args.spine_dir / f"{spine_uid}_seg_placed.nii.gz") if spine_uid else None
        log.info("[%d/%d] token=%s config=%s", i, len(todo), r.get("token"), r.get("config"))
        try:
            qc = process_case(ct_path, v2_label_path, spine_mask, out_label_path,
                              device=args.device, min_voxels=args.min_voxels,
                              dilation_radius=args.dilation_radius, pad=args.pad)
        except Exception as exc:                                       # noqa: BLE001
            log.error("  token=%s FAILED: %s — shipping v2 label unchanged",
                      r.get("token"), exc)
            shutil.copy2(v2_label_path, out_label_path)
            qc = {"ct": ct_path.name, "status": "error", "note": str(exc)[:200],
                  "ribs_written": 0, "n_ribs": 0}
        qc["token"] = r.get("token")
        qc_rows.append(qc)

    import csv
    qc_path = args.v3_dir / "rib_qc.csv"
    with open(qc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "ct", "status", "n_ribs",
                                           "ribs_written", "note"])
        w.writeheader()
        for row in qc_rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    n_ok = sum(1 for r in qc_rows if r["status"] == "ok")
    log.info("v3 ribs done: %d/%d cases got ribs -> %s", n_ok, len(qc_rows), qc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
