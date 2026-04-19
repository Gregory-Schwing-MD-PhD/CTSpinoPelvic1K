"""
diagnose_series.py -- Placement diagnostic: winning series vs CTSpine1K annotation source.

For each token reports:
  1. placed_series_uid   (winning series from placed_manifest.json — bone_pct-maximised)
  2. ctspine1k_src_uid   (series CTSpine1K originally annotated, from Path.csv / filename)
  3. PatientPosition DICOM tag
  4. Affine comparison: CTSpine1K NIfTI vs dcm2niix of winning series
  5. VERDICT

VERDICT LOGIC
  SAME_SERIES
      placed_series_uid matches the CTSpine1K annotation source.
      CTSpine1K and placement agree on which CT to use.
  DIFFERENT_SERIES
      placed_series_uid differs from the CTSpine1K annotation source.
      This is expected and OK when bone_pct is good — the exhaustive search
      found a better-aligned series than what CTSpine1K originally annotated.
      Check placed_manifest.json: spine.bone_pct > 40% and IS_ok=True.
  PRONE_SUPINE_MISMATCH
      UIDs differ with an axis flip and >50% Z-overlap.  Placement found the
      supine/prone counterpart of the CTSpine1K annotation.

Usage
  python scripts/diagnose_series.py \\
      --tokens   149,153,604 \\
      --manifest data/placed/placed_manifest.json \\
      --nifti_dir data/tcia_nifti \\
      --tcia_dir  data/tcia \\
      --ctspine1k data/ctspine1k
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.diagnose_series")


def _read_dicom_tags(series_dir: Path) -> dict:
    try:
        import pydicom
    except ImportError:
        return {"error": "pydicom not installed"}
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return {"error": f"no .dcm in {series_dir}"}
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    except Exception as e:
        return {"error": str(e)}
    def _g(tag, default="UNKNOWN"):
        try: return str(getattr(ds, tag, default) or default)
        except: return default
    iop = []
    try: iop = [float(v) for v in ds.ImageOrientationPatient]
    except: pass
    return {
        "PatientPosition":    _g("PatientPosition"),
        "SeriesDescription":  _g("SeriesDescription"),
        "AcquisitionDate":    _g("AcquisitionDate"),
        "ImageOrientationPatient": iop,
        "n_dcm_files":        len(dcm_files),
    }


def _nifti_summary(path: Path) -> dict:
    if not path.exists():
        return {"error": f"not found: {path}"}
    try:
        import nibabel as nib
        img = nib.load(str(path))
        aff = img.affine
        sp  = np.sqrt((aff[:3,:3]**2).sum(axis=0))
        return {
            "shape":   tuple(int(s) for s in img.shape[:3]),
            "origin":  [round(float(aff[i,3]),1) for i in range(3)],
            "spacing": [round(float(s),3) for s in sp],
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_axis_flips(aff_a, aff_b):
    def _dc(a):
        sp = np.sqrt((a[:3,:3]**2).sum(axis=0))
        return a[:3,:3] / np.maximum(1e-9, sp)
    dca, dcb = _dc(aff_a), _dc(aff_b)
    return [bool(np.dot(dca[:,i], dcb[:,i]) < -0.9) for i in range(3)]


def _find_path_csv(root: Path) -> Optional[Path]:
    for c in [root/"Path.csv", root/"all_labels.csv",
              root/"metadata"/"Path.csv", root/"rawdata"/"Path.csv"]:
        if c.exists(): return c
    hits = list(root.rglob("Path.csv"))
    return hits[0] if hits else None


def _ctspine1k_src_uid(token: str, root: Path) -> Optional[str]:
    csv = _find_path_csv(root)
    if csv:
        try:
            import csv as _csv
            with open(csv, newline="") as f:
                for row in _csv.DictReader(f):
                    if token.lower() in json.dumps(row).lower():
                        for col in ("SeriesUID","SeriesInstanceUID","UID","Path","FileName"):
                            val = row.get(col,"").strip()
                            if val and "." in val and len(val) > 10:
                                return val
        except Exception as e:
            log.debug("Path.csv error: %s", e)

    for d_rel in ["rawdata/volumes/COLONOG","volumes/COLONOG","raw_data/volumes/COLONOG"]:
        d = root / d_rel
        if not d.exists(): continue
        for hit in sorted(d.glob(f"*{token}*.nii.gz")):
            stem = hit.stem.replace(".nii","")
            if "." in stem and all(c in "0123456789." for c in stem):
                return stem
    return None


def _find_ctspine1k_nii(token: str, spine: dict, ctspine1k_root: Path) -> Optional[Path]:
    raw_path = spine.get("nifti_image","")
    if raw_path:
        from pathlib import PurePosixPath
        parts = PurePosixPath(raw_path).parts
        for i, part in enumerate(parts):
            if part == "ctspine1k":
                p = ctspine1k_root / Path(*parts[i+1:]) if len(parts) > i+1 else None
                if p and p.exists(): return p
    for d_rel in ["rawdata/volumes/COLONOG","volumes/COLONOG","raw_data/volumes/COLONOG"]:
        hits = sorted((ctspine1k_root / d_rel).glob(f"*{token}*.nii.gz"))
        if hits: return hits[0]
    return None


def diagnose_token(token: str, manifest_cases: dict,
                   nifti_dir: Path, tcia_dir: Path,
                   ctspine1k_root: Path) -> dict:
    r: dict = {"token": token}

    case = manifest_cases.get(token)
    if not case:
        r["error"] = "token not in placed_manifest.json"; return r

    r["match_type"] = case.get("match_type","?")
    sp_manifest = case.get("spine") or {}
    pv_manifest = case.get("pelvic") or {}

    placed_uid = sp_manifest.get("series_uid")
    r["placed_series_uid"]  = placed_uid
    r["placed_bone_pct"]    = sp_manifest.get("bone_pct","?")
    r["placed_IS_ok"]       = sp_manifest.get("IS_ok","?")
    r["placed_method"]      = sp_manifest.get("method","?")
    if pv_manifest:
        r["pelvic_series_uid"] = pv_manifest.get("series_uid")
        r["pelvic_bone_pct"]   = pv_manifest.get("bone_pct","?")

    if placed_uid:
        sd = tcia_dir / placed_uid
        r["tcia_series_dir_exists"] = sd.exists()
        r["dicom_tags"] = _read_dicom_tags(sd) if sd.exists() else {"error": f"dir missing: {sd}"}
    else:
        r["dicom_tags"] = {"error": "no placed_series_uid in manifest"}

    dcm2niix_nii = nifti_dir / f"{placed_uid}.nii.gz" if placed_uid else None
    r["dcm2niix_nii_exists"] = bool(dcm2niix_nii and dcm2niix_nii.exists())
    if r["dcm2niix_nii_exists"]:
        r["dcm2niix_header"] = _nifti_summary(dcm2niix_nii)

    ctspine1k_src = _ctspine1k_src_uid(token, ctspine1k_root)
    r["ctspine1k_src_uid"] = ctspine1k_src
    if ctspine1k_src and placed_uid:
        r["uid_match"] = (ctspine1k_src.strip() == placed_uid.strip())
    else:
        r["uid_match"] = None

    sp_nii = _find_ctspine1k_nii(token, sp_manifest, ctspine1k_root)
    r["ctspine1k_nii_exists"] = bool(sp_nii and sp_nii.exists())
    if r["ctspine1k_nii_exists"]:
        r["ctspine1k_header"] = _nifti_summary(sp_nii)

    if r.get("dcm2niix_nii_exists") and r.get("ctspine1k_nii_exists"):
        try:
            import nibabel as nib
            aff_d = nib.load(str(dcm2niix_nii)).affine
            aff_s = nib.load(str(sp_nii)).affine
            r["axis_flips"]     = _compute_axis_flips(aff_s, aff_d)
            r["any_flip"]       = any(r["axis_flips"])
            r["origin_diff_mm"] = round(float(np.linalg.norm(aff_d[:3,3]-aff_s[:3,3])),1)

            def _zr(aff, shape):
                zs = [(aff @ [0,0,k,1])[2] for k in [0, shape[2]-1]]
                return min(zs), max(zs)
            sd_z = _zr(aff_s, nib.load(str(sp_nii)).shape)
            dd_z = _zr(aff_d, nib.load(str(dcm2niix_nii)).shape)
            z_ov = max(0.0, min(sd_z[1],dd_z[1]) - max(sd_z[0],dd_z[0]))
            r["z_range_ctspine1k"] = [round(sd_z[0],0), round(sd_z[1],0)]
            r["z_range_dcm2niix"]  = [round(dd_z[0],0), round(dd_z[1],0)]
            r["z_overlap_pct"]     = round(z_ov / max(1., sd_z[1]-sd_z[0]) * 100, 1)
        except Exception as e:
            r["geom_error"] = str(e)

    r["verdict"] = _verdict(r)
    return r


def _verdict(r: dict) -> str:
    uid_match = r.get("uid_match")
    flip      = r.get("any_flip", False)
    zov       = r.get("z_overlap_pct", 0)
    patpos    = (r.get("dicom_tags") or {}).get("PatientPosition","?")
    bone      = r.get("placed_bone_pct","?")
    is_ok     = r.get("placed_IS_ok","?")

    if uid_match is True:
        return f"SAME_SERIES  bone={bone}%  IS_ok={is_ok}"

    if uid_match is False:
        if flip and zov > 50:
            return (f"DIFFERENT_SERIES / PRONE_SUPINE_MISMATCH  "
                    f"patpos={patpos}  z_overlap={zov:.0f}%  "
                    f"flip={r.get('axis_flips')}  bone={bone}%  IS_ok={is_ok}")
        return (f"DIFFERENT_SERIES  z_overlap={zov:.0f}%  "
                f"bone={bone}%  IS_ok={is_ok}  "
                "(OK if bone_pct good — exhaustive search found better series)")

    odiff = r.get("origin_diff_mm", 0)
    if odiff < 5:
        return f"LIKELY_SAME_SERIES  origin_diff={odiff:.0f}mm  bone={bone}%"
    return f"UNCERTAIN  origin_diff={odiff:.0f}mm  flip={flip}  z_overlap={zov:.0f}%  bone={bone}%"


def print_report(results: List[dict]) -> None:
    print("\n" + "="*80)
    print("  CTSpinoPelvic1K  Placement Diagnostic (winning series vs CTSpine1K source)")
    print("="*80)
    for r in results:
        print(f"\n── token {r['token']:>6}  →  {r.get('verdict','?')}")
        print(f"   match_type         : {r.get('match_type','?')}")
        print(f"   placed_series_uid  : {r.get('placed_series_uid','N/A')}")
        print(f"   ctspine1k_src_uid  : {r.get('ctspine1k_src_uid','N/A')}")
        print(f"   uid_match          : {r.get('uid_match','N/A')}")
        print(f"   placed  bone_pct   : {r.get('placed_bone_pct','?')}%  "
              f"IS_ok={r.get('placed_IS_ok','?')}  method={r.get('placed_method','?')}")
        if r.get("pelvic_series_uid"):
            same = r.get("placed_series_uid") == r.get("pelvic_series_uid")
            print(f"   pelvic  bone_pct   : {r.get('pelvic_bone_pct','?')}%  "
                  f"series={'same' if same else 'different'}")
        dtags = r.get("dicom_tags") or {}
        if "error" not in dtags:
            print(f"   PatientPosition    : {dtags.get('PatientPosition','?')}")
            print(f"   SeriesDescription  : {dtags.get('SeriesDescription','?')}")
            iop = dtags.get("ImageOrientationPatient",[])
            if iop: print(f"   ImageOrientationPt : {[round(v,2) for v in iop]}")
            print(f"   n_dcm_files        : {dtags.get('n_dcm_files','?')}")
        else:
            print(f"   DICOM error        : {dtags['error']}")
        dh = r.get("dcm2niix_header") or {}
        sh = r.get("ctspine1k_header") or {}
        if dh and "error" not in dh:
            print(f"   dcm2niix  shape={dh.get('shape')}  "
                  f"origin={dh.get('origin')}  spacing={dh.get('spacing')}")
        if sh and "error" not in sh:
            print(f"   CTSpine1K shape={sh.get('shape')}  "
                  f"origin={sh.get('origin')}  spacing={sh.get('spacing')}")
        if "origin_diff_mm" in r:
            print(f"   origin_diff_mm     : {r['origin_diff_mm']}")
            print(f"   axis_flips         : {r.get('axis_flips')}")
            print(f"   z_range CTSpine1K  : {r.get('z_range_ctspine1k')}  "
                  f"dcm2niix={r.get('z_range_dcm2niix')}  "
                  f"overlap={r.get('z_overlap_pct')}%")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        patpos = (r.get("dicom_tags") or {}).get("PatientPosition","?")
        print(f"  token={r['token']:>6}  patpos={patpos:>4}  "
              f"bone={r.get('placed_bone_pct','?'):>5}%  "
              f"IS_ok={r.get('placed_IS_ok','?')}  "
              f"{r.get('verdict','?')[:60]}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens",     default="",   type=str)
    p.add_argument("--token_file", default=None, type=Path)
    p.add_argument("--manifest",   required=True, type=Path,
                   help="placed_manifest.json from place_fused_masks.py")
    p.add_argument("--nifti_dir",  required=True, type=Path)
    p.add_argument("--tcia_dir",   required=True, type=Path)
    p.add_argument("--ctspine1k",  required=True, type=Path)
    p.add_argument("--out_json",   default=None, type=Path)
    args = p.parse_args()

    tokens: Set[str] = set()
    if args.tokens.strip():
        tokens.update(t.strip() for t in args.tokens.split(",") if t.strip())
    if args.token_file and args.token_file.exists():
        tokens.update(l.strip() for l in args.token_file.read_text().splitlines() if l.strip())
    if not tokens:
        p.error("Provide --tokens or --token_file")

    csv = _find_path_csv(args.ctspine1k)
    log.info("CTSpine1K Path.csv: %s", csv or "NOT FOUND")

    manifest = json.loads(args.manifest.read_text())
    manifest_cases: dict = {
        str(c.get("patient_token","?")): c
        for c in manifest.get("cases", [])
    }
    log.info("Manifest: %d cases loaded", len(manifest_cases))

    results = []
    for tok in sorted(tokens):
        log.info("token=%s ...", tok)
        results.append(
            diagnose_token(tok, manifest_cases, args.nifti_dir,
                           args.tcia_dir, args.ctspine1k)
        )

    print_report(results)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=str))
        log.info("JSON -> %s", args.out_json)


if __name__ == "__main__":
    main()
