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

# TotalSegmentator "total"-task rib ROI names (12 per side).
RIB_NAMES: List[str] = (
    [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)]
)

# VerSe ids 8..19 == thoracic T1..T12. Remap to the 1..12 that relabel_ribs expects.
VERSE_THORACIC_LO, VERSE_THORACIC_HI = 8, 19

# ---------------------------------------------------------------------------
# v3 TRAINING-CONTIGUOUS label scheme. nnU-Net requires consecutive label ids
# with the ignore label HIGHEST, so adding 24 ribs pushes ignore off 10:
#   0 bg | 1..6 L1..L6 | 7 sacrum | 8 left_hip | 9 right_hip
#   10..21 rib_left_1..12 | 22..33 rib_right_1..12 | 34 ignore
# We set relabel_ribs' offsets so left rib n -> 9+n (10..21), right n -> 21+n
# (22..33), and remap the v2 ignore (10) -> 34 so it never collides with rib id 10.
RR.LEFT_OFFSET = 9
RR.RIGHT_OFFSET = 21
V2_IGNORE = 10
V3_IGNORE = 34


def v3_label_dict() -> Dict[str, int]:
    """The full v3 {name: id} label map (background..ignore), for dataset.json."""
    d = {"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
         "sacrum": 7, "left_hip": 8, "right_hip": 9}
    for n in range(1, 13):
        d[f"rib_left_{n}"] = 9 + n
    for n in range(1, 13):
        d[f"rib_right_{n}"] = 21 + n
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
    # Move the ignore label off 10 (which is now rib_left_1) -> 34, for every case
    # so v3's ignore id is uniform whether or not the case gets ribs.
    v2_label[v2_label == V2_IGNORE] = V3_IGNORE

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
    def _mirror(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)                  # cheap: no data copied (same filesystem)
        except OSError:
            shutil.copy2(src, dst)             # cross-device fallback
    for sub in ("ct", "labels"):
        sd = args.v2_dir / sub
        if sd.exists():
            for f in sd.glob("*.nii.gz"):
                _mirror(f, args.v3_dir / sub / f.name)
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
    todo = [r for r in records if r.get("config") in ("fused", "spine_only")]
    if args.limit:
        todo = todo[: args.limit]
    log.info("v3 ribs: %d case(s) to process", len(todo))

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
        spine_mask = (args.spine_dir / f"{spine_uid}_seg_placed.nii.gz") if spine_uid else None
        log.info("[%d/%d] token=%s config=%s", i, len(todo), r.get("token"), r.get("config"))
        try:
            qc = process_case(ct_path, v2_label_path, spine_mask, out_label_path,
                              device=args.device, min_voxels=args.min_voxels,
                              dilation_radius=args.dilation_radius, pad=args.pad)
        except Exception as exc:                                       # noqa: BLE001
            log.error("  token=%s FAILED: %s — shipping v2 label (ignore remapped, no ribs)",
                      r.get("token"), exc)
            # Still apply the ignore 10->34 remap so v3's ignore id stays uniform.
            li = nib.load(str(v2_label_path))
            la = np.asarray(li.dataobj).astype(np.int32)
            la[la == V2_IGNORE] = V3_IGNORE
            nib.save(nib.Nifti1Image(la.astype(np.uint16), li.affine, li.header),
                     str(out_label_path))
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
