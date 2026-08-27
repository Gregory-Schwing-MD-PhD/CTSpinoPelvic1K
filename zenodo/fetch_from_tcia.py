"""fetch_from_tcia.py — rebuild the CT half of CTSpinoPelvic1K, end to end.

This deposit ships labels, not images: the CTs are 193 GB against 1.8 GB of labels and are
already public in The Cancer Imaging Archive. What the source collections never published,
and what `manifest.json` here does, is which CT series each annotation belongs on.

This reproduces the three steps the release itself used, in the same order and with the same
settings, so a rebuilt volume matches the released one:

  1. DOWNLOAD the series named in the manifest, by SeriesInstanceUID, with `tcia_utils`.
  2. CONVERT the DICOM series to NIfTI with `dcm2niix`.
  3. RESAMPLE the result onto the label's grid -- trilinear, -1024 HU outside the original
     extent -- so image and label share an affine exactly.

STEP 3 IS NOT OPTIONAL AND IS THE ONE PEOPLE SKIP. Each label was drawn on one grid, and at
export the CT was resampled onto THAT grid. A fresh conversion of the same DICOM series is
the right anatomy on the wrong grid, and the mask will not sit on the bone. Everything
needed is already in the label file, because the label's own affine is the target -- which
also means the released PIR orientation comes along for free and there is nothing separate
to rotate.

WHY IT DOWNLOADS A NAMED SERIES AND NOT A PATIENT. Every patient in this cohort was scanned
twice, prone and supine, often reconstructed with more than one kernel. A patient identifier
resolves to several volumes; only one of them is the one the annotation was drawn on. That
is the entire point of the crosswalk, so the UID is used and never the patient.

    pip install tcia_utils nibabel scipy numpy      # and dcm2niix on PATH
    python fetch_from_tcia.py --manifest manifest.json --labels labels --out ct
    python fetch_from_tcia.py --manifest manifest.json --labels labels --out ct \\
        --cases 0007 0033 --keep-dicom
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform

BONE_HU = 200.0


def rec_id(r):
    return Path(str(r.get("label_file", ""))).name.split("_")[0]


def series_for(rec):
    """The UID this record's labels were drawn on.

    spine_series_uid where a spine annotation exists, otherwise the pelvic one: the 20
    pelvic-only records have no spine UID because they have no spine annotation.
    """
    return (str(rec.get("spine_series_uid") or "").strip()
            or str(rec.get("pelvic_series_uid") or "").strip())


def download_series(uid, dest):
    """DICOMs for one SeriesInstanceUID into dest/.

    tcia_utils.downloadSeries SILENTLY DOES NOTHING when its target directory already
    exists, even when empty, so the directory must not be pre-created and a partial one
    from an interrupted run has to be removed first.
    """
    from tcia_utils import nbia
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # input_type="list" means series_data is a list of UID STRINGS. Passing the
    # list-of-dicts form here -- which is what the DEFAULT input type wants, being the shape
    # getSeries returns -- raises "unhashable type: dict" and downloads nothing. The two
    # arguments have to agree, and this combination was shipped without ever being run.
    nbia.downloadSeries([uid], input_type="list",
                        path=str(dest.parent), as_zip=False)
    got = dest.parent / uid
    if got.exists() and got != dest:
        got.rename(dest)
    n = len(list(dest.rglob("*.dcm"))) if dest.exists() else 0
    if not n:
        raise RuntimeError(f"no DICOM files downloaded for {uid}")
    return n


def dicom_to_nifti(dcm_dir, work):
    """dcm2niix, the same converter the release used."""
    work.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["dcm2niix", "-z", "y", "-f", "vol", "-o", str(work), str(dcm_dir)],
                       capture_output=True, text=True)
    out = sorted(work.glob("*.nii.gz"))
    if not out:
        raise RuntimeError(f"dcm2niix produced nothing: {r.stderr[-300:] or r.stdout[-300:]}")
    # a series occasionally splits; the largest volume is the reconstruction, the rest are
    # localisers and derived series
    return max(out, key=lambda p: p.stat().st_size)


def resample_to_label(ct_path, label_path):
    """CT on the label's grid. The same operation, order and fill value the export used."""
    ref = nib.load(str(label_path))
    ct = nib.load(str(ct_path))
    data = np.asarray(ct.dataobj, dtype=np.float32)
    if ct.shape[:3] == ref.shape[:3] and np.allclose(ct.affine, ref.affine, atol=1e-4):
        return nib.Nifti1Image(data, ref.affine)
    M = np.linalg.inv(ct.affine) @ ref.affine
    out = affine_transform(data, M[:3, :3], offset=M[:3, 3],
                           output_shape=ref.shape[:3], order=1,
                           mode="constant", cval=-1024.0)
    return nib.Nifti1Image(out, ref.affine)


def bone_fraction(ct_img, label_path):
    """How much of the label sits on bone. The check that catches the wrong series."""
    lab = np.asanyarray(nib.load(str(label_path)).dataobj)
    img = np.asarray(ct_img.dataobj)
    m = (lab > 0) & (lab != 255)
    return float((img[m] >= BONE_HU).mean()) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="ct")
    ap.add_argument("--cases", nargs="*", default=None, help="default: every record")
    ap.add_argument("--keep-dicom", action="store_true",
                    help="keep the downloaded DICOMs instead of discarding them")
    ap.add_argument("--dicom-dir", default=None,
                    help="where to keep DICOMs (implies --keep-dicom)")
    ap.add_argument("--min-bone", type=float, default=0.5,
                    help="reject a rebuild whose label sits on less bone than this")
    a = ap.parse_args()

    if shutil.which("dcm2niix") is None:
        print("  ! dcm2niix is not on PATH. It is what the release used to convert DICOM;")
        print("  ! install it (https://github.com/rordenlab/dcm2niix) and re-run.")
        return 2

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    if a.cases:
        want = set(a.cases)
        recs = [r for r in recs if rec_id(r) in want]
    print(f"  {len(recs)} record(s) to rebuild")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = Path(a.dicom_dir) if a.dicom_dir else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)

    ok = warn = fail = 0
    for i, r in enumerate(recs, 1):
        cid = rec_id(r)
        dst = out / f"{cid}_ct.nii.gz"
        if dst.exists():
            print(f"  [{i}/{len(recs)}] {cid}: already built")
            ok += 1
            continue
        lab = Path(a.labels) / f"{cid}_label.nii.gz"
        uid = series_for(r)
        if not lab.exists():
            print(f"  [{i}/{len(recs)}] {cid}: no label file, skipped")
            fail += 1
            continue
        if not uid:
            print(f"  [{i}/{len(recs)}] {cid}: no series identifier, skipped")
            fail += 1
            continue

        tmp = Path(tempfile.mkdtemp(prefix=f"ctsp_{cid}_"))
        try:
            dcm = (keep / uid) if keep else (tmp / uid)
            if not (dcm.exists() and any(dcm.rglob("*.dcm"))):
                download_series(uid, dcm)
            nii = dicom_to_nifti(dcm, tmp / "nii")
            img = resample_to_label(nii, lab)
            frac = bone_fraction(img, lab)
            if frac == frac and frac < a.min_bone:
                # not a resampling failure: almost always the other acquisition of the
                # same patient, which is the mistake the crosswalk exists to prevent
                print(f"  [{i}/{len(recs)}] {cid}: ONLY {frac*100:.0f}% of the label is on "
                      f"bone -- check the UID, not the resampling")
                warn += 1
            nib.save(img, str(dst))
            print(f"  [{i}/{len(recs)}] {cid}: built, {frac*100:.0f}% of label on bone",
                  flush=True)
            ok += 1
        except Exception as e:                                        # noqa: BLE001
            print(f"  [{i}/{len(recs)}] {cid}: FAILED {type(e).__name__}: {e}")
            fail += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n  built {ok}, {warn} with a low bone fraction, {fail} failed")
    print(f"  CTs in {out}, each sharing its label's affine exactly")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
