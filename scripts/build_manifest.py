"""
build_manifest.py — External training manifest for CTSpinoPelvic1K.

For pipelines that prefer NIfTI file paths over the HF NIfTI-pair export.
Reads canonical sources (patient_db.json + placed_manifest.json) and emits
colonog_training_manifest.json with four entry types:

  fused         (match_type=fused)       dcm2niix CT + placed spine seg + placed pelvic mask
  spine_only    (match_type=spine_only)  dcm2niix CT + placed spine seg
  pelvic_only   (match_type=pelvic_only) dcm2niix CT + placed pelvic mask (own acquisition)
  pelvic_native (match_type=separate)    dcm2niix CT + placed pelvic mask from the pelvic half

Paths (CTSpinoPelvic1K convention)
----------------------------------
  image         = nifti_dir/{series_uid}.nii.gz
  spine_label   = placed/spine/{spine_uid}_seg_placed.nii.gz
  pelvic_mask   = placed/fused/{stem}_pelvic_placed.nii.gz     (fused)
                = placed/pelvic_native/{stem}_pelvic_placed.nii.gz (pelvic_only / pelvic_native)

LSTV
----
  lstv_class int via pelvic-priority resolution:
    0 Normal  1 Lumbarization  2 Semi-sacralization  3 Sacralization
    Pelvic filename label takes priority over vertebral count.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ctspinopelvic1k.build_manifest")


_LSTV_STR_TO_INT: Dict[str, int] = {
    "NORMAL": 0, "LUMBARIZATION": 1, "SEMI_SACRAL": 2,
    "SEMI_SACRALIZATION": 2, "SACRALIZATION": 3, "HARD_SACRAL": 4,
    "UNKNOWN": 0, "AMBIGUOUS": 0, "INCOMPLETE_SCAN": 0, "EXCLUDED": 0,
}
_LSTV_INT_TO_NAME = {
    0: "Normal", 1: "Lumbarization", 2: "Semi-sacralization",
    3: "Sacralization", 4: "Hard/Complex sacralization",
}


def _resolve_lstv_class(lstv_pelvic: str, lstv_vertebral: str) -> Tuple[int, str]:
    pv = (lstv_pelvic or "").upper()
    vt = (lstv_vertebral or "").upper()
    pv_int = _LSTV_STR_TO_INT.get(pv, 0)
    if pv_int > 0:
        return pv_int, f"pelvic_filename:{lstv_pelvic}"
    if pv in ("NORMAL", "UNKNOWN", "AMBIGUOUS", "INCOMPLETE_SCAN", "EXCLUDED", ""):
        vt_int = _LSTV_STR_TO_INT.get(vt, 0)
        if vt_int > 0:
            return vt_int, f"vertebral_scan:{lstv_vertebral}"
    return 0, "default_normal"


def _mask_stem(mask_raw_path: str) -> str:
    return Path(mask_raw_path).name.replace(".nii.gz", "").replace(".nii", "")


def _load_patient_db(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"patient_db not found: {path}")
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return json.loads(path.read_text())


def _spine_image_path(
    spine_info: Dict, nifti_dir: Optional[Path]
) -> Optional[str]:
    """Resolve the dcm2niix-produced CT for a spine series."""
    if not nifti_dir or not spine_info:
        return None
    sid = spine_info.get("series_uid")
    if not sid:
        return None
    p = nifti_dir / f"{sid}.nii.gz"
    return str(p) if p.exists() else None


def _pelvic_image_path(
    pelvic_info: Dict, nifti_dir: Optional[Path]
) -> Optional[str]:
    if not nifti_dir or not pelvic_info:
        return None
    sid = pelvic_info.get("series_uid")
    if not sid:
        return None
    p = nifti_dir / f"{sid}.nii.gz"
    return str(p) if p.exists() else None


def _placed_spine_path(
    spine_info: Dict, placed_spine_dir: Optional[Path]
) -> Optional[str]:
    if not placed_spine_dir:
        return spine_info.get("placed") if spine_info else None
    sid = (spine_info or {}).get("series_uid")
    if sid:
        p = placed_spine_dir / f"{sid}_seg_placed.nii.gz"
        if p.exists():
            return str(p)
    # Fallback to whatever is in the manifest
    return (spine_info or {}).get("placed")


def _placed_pelvic_path(
    pelvic_info: Dict,
    placed_mask_dir: Optional[Path],           # fused case pelvic masks
    placed_pelvic_native_dir: Optional[Path],  # pelvic-native case masks
    is_fused: bool,
) -> Optional[str]:
    if not pelvic_info:
        return None
    # Canonical source
    placed = pelvic_info.get("placed")
    if placed and Path(placed).exists():
        return placed
    # Derive stem from mask_file if we have a search dir
    mask_file = pelvic_info.get("mask_file", "")
    if mask_file:
        stem = _mask_stem(mask_file)
        for d in ([placed_mask_dir] if is_fused else [placed_pelvic_native_dir, placed_mask_dir]):
            if d:
                p = d / f"{stem}_pelvic_placed.nii.gz"
                if p.exists():
                    return str(p)
    return placed


def _build_fused_entry(case: Dict, nifti_dir, placed_spine_dir,
                       placed_mask_dir) -> Optional[Dict]:
    token = str(case.get("patient_token", "?"))
    sp, pv = case.get("spine") or {}, case.get("pelvic") or {}

    image = _spine_image_path(sp, nifti_dir)
    seg   = _placed_spine_path(sp, placed_spine_dir)
    mask  = _placed_pelvic_path(pv, placed_mask_dir, None, is_fused=True)

    if not image or not seg or not mask:
        log.warning("fused token=%s missing  img=%s seg=%s mask=%s",
                    token,
                    "OK" if image else "MISS",
                    "OK" if seg   else "MISS",
                    "OK" if mask  else "MISS")
        return None

    lstv_int, lstv_src = _resolve_lstv_class(
        case.get("lstv_pelvic", ""), case.get("lstv_vertebral", ""),
    )
    return {
        "patient_token":  token,
        "match_type":     "fused",
        "dataset":        "spine",
        "label_role":     "spineimg",
        "image":          image,
        "spine_label":    seg,
        "pelvic_mask":    mask,
        "lstv_class":     lstv_int,
        "lstv_source":    lstv_src,
        "series_uid":     sp.get("series_uid"),
        "series_match":   case.get("series_agreement"),
        "image_source":   "dcm2niix",
        "seg_source":     "placed_dcm2niix",
        "lstv_pelvic":    case.get("lstv_pelvic"),
        "lstv_vertebral": case.get("lstv_vertebral"),
        "lstv_agreement": case.get("lstv_agreement"),
        "lstv_confusion_zone": case.get("lstv_confusion_zone", False),
        "bone_pct_spine":  sp.get("bone_pct"),
        "bone_pct_pelvic": pv.get("bone_pct"),
        "warnings":       case.get("warnings", []),
    }


def _build_spine_only_entry(case: Dict, nifti_dir, placed_spine_dir) -> Optional[Dict]:
    token = str(case.get("patient_token", "?"))
    sp = case.get("spine") or {}
    image = _spine_image_path(sp, nifti_dir)
    seg   = _placed_spine_path(sp, placed_spine_dir)
    if not image or not seg:
        log.warning("spine_only token=%s missing  img=%s seg=%s",
                    token,
                    "OK" if image else "MISS",
                    "OK" if seg   else "MISS")
        return None
    lstv_int, lstv_src = _resolve_lstv_class(
        case.get("lstv_pelvic", ""), case.get("lstv_vertebral", ""),
    )
    return {
        "patient_token":  token,
        "match_type":     "spine_only",
        "dataset":        "spine",
        "label_role":     "spineimg",
        "image":          image,
        "spine_label":    seg,
        "pelvic_mask":    None,
        "lstv_class":     lstv_int,
        "lstv_source":    lstv_src,
        "series_uid":     sp.get("series_uid"),
        "series_match":   None,
        "image_source":   "dcm2niix",
        "seg_source":     "placed_dcm2niix",
        "lstv_pelvic":    case.get("lstv_pelvic"),
        "lstv_vertebral": case.get("lstv_vertebral"),
        "lstv_agreement": case.get("lstv_agreement"),
        "lstv_confusion_zone": case.get("lstv_confusion_zone", False),
        "bone_pct_spine": sp.get("bone_pct"),
        "warnings":       case.get("warnings", []),
    }


def _build_pelvic_only_entry(case: Dict, nifti_dir,
                              placed_mask_dir, placed_pelvic_native_dir,
                              source_tag: str = "pelvic_only") -> Optional[Dict]:
    token = str(case.get("patient_token", "?"))
    pv = case.get("pelvic") or {}
    image = _pelvic_image_path(pv, nifti_dir)
    mask  = _placed_pelvic_path(pv, placed_mask_dir,
                                 placed_pelvic_native_dir, is_fused=False)
    if not image or not mask:
        log.warning("%s token=%s missing  img=%s mask=%s",
                    source_tag, token,
                    "OK" if image else "MISS",
                    "OK" if mask  else "MISS")
        return None
    lstv_int, lstv_src = _resolve_lstv_class(
        case.get("lstv_pelvic", ""), case.get("lstv_vertebral", ""),
    )
    return {
        "patient_token":  token,
        "match_type":     "pelvic_only",
        "dataset":        "pelvic",
        "label_role":     "pelvimg",
        "image":          image,
        "pelvic_image":   image,
        "spine_label":    None,
        "pelvic_mask":    mask,
        "lstv_class":     lstv_int,
        "lstv_source":    lstv_src,
        "series_uid":     pv.get("series_uid"),
        "series_dir":     pv.get("series_dir"),
        "series_match":   None,
        "source":         source_tag,
        "image_source":   "dcm2niix",
        "lstv_pelvic":    case.get("lstv_pelvic"),
        "lstv_vertebral": case.get("lstv_vertebral"),
        "lstv_agreement": case.get("lstv_agreement"),
        "lstv_confusion_zone": case.get("lstv_confusion_zone", False),
        "bone_pct_pelvic": pv.get("bone_pct"),
        "warnings":       case.get("warnings", []),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def build_manifest(
    placed_manifest_path:     Path,
    patient_db_path:          Optional[Path],
    out_path:                 Path,
    nifti_dir:                Optional[Path],
    placed_spine_dir:         Optional[Path],
    placed_mask_dir:          Optional[Path],
    placed_pelvic_native_dir: Optional[Path],
) -> Dict:
    if not placed_manifest_path.exists():
        raise FileNotFoundError(f"placed_manifest.json not found: {placed_manifest_path}")

    placed = json.loads(placed_manifest_path.read_text())
    cases  = placed.get("cases", [])
    log.info("placed_manifest: %d cases", len(cases))

    # patient_db is optional — we don't actually need extra fields for the
    # training manifest, but augment warnings/lstv if present and missing.
    patient_db = {}
    if patient_db_path and patient_db_path.exists():
        try:
            pdb = _load_patient_db(patient_db_path)
            patient_db = pdb.get("patients", {})
            log.info("patient_db: %d patients", len(patient_db))
        except Exception as e:
            log.warning("Could not load patient_db %s: %s", patient_db_path, e)

    def _augment(case: Dict) -> Dict:
        tok = str(case.get("patient_token", ""))
        pdata = patient_db.get(tok) if patient_db else None
        if pdata:
            for k in ("lstv_pelvic", "lstv_vertebral", "lstv_agreement"):
                if not case.get(k) and pdata.get(k):
                    case[k] = pdata[k]
        return case

    fused: List[Dict]     = []
    spine_only: List[Dict] = []
    pelvic_only: List[Dict] = []
    n_separate_spine = n_separate_pelvic = 0

    for c in cases:
        c = _augment(c)
        mt = c.get("match_type", "")
        if mt == "fused":
            e = _build_fused_entry(c, nifti_dir, placed_spine_dir, placed_mask_dir)
            if e: fused.append(e)
        elif mt == "spine_only":
            e = _build_spine_only_entry(c, nifti_dir, placed_spine_dir)
            if e: spine_only.append(e)
        elif mt == "pelvic_only":
            e = _build_pelvic_only_entry(c, nifti_dir, placed_mask_dir,
                                          placed_pelvic_native_dir,
                                          source_tag="pelvic_only")
            if e: pelvic_only.append(e)
        elif mt == "separate":
            e_s = _build_spine_only_entry(c, nifti_dir, placed_spine_dir)
            if e_s:
                spine_only.append(e_s)
                n_separate_spine += 1
            e_p = _build_pelvic_only_entry(c, nifti_dir, placed_mask_dir,
                                            placed_pelvic_native_dir,
                                            source_tag="pelvic_native")
            if e_p:
                pelvic_only.append(e_p)
                n_separate_pelvic += 1

    all_entries = fused + spine_only + pelvic_only
    lstv_counts = Counter(e["lstv_class"] for e in all_entries)

    log.info("Manifest: fused=%d  spine_only=%d (separate=%d)  "
             "pelvic_only=%d (separate=%d)",
             len(fused), len(spine_only), n_separate_spine,
             len(pelvic_only), n_separate_pelvic)
    log.info("LSTV: %s",
             "  ".join(f"{_LSTV_INT_TO_NAME[k]}={v}"
                        for k, v in sorted(lstv_counts.items()) if v))

    n_missing = 0
    for e in all_entries:
        for field in ("image", "spine_label", "pelvic_mask"):
            v = e.get(field)
            if v and not Path(v).exists():
                log.warning("MISSING  token=%-6s  %s=%s",
                            e["patient_token"], field, v)
                n_missing += 1
    if n_missing:
        log.warning("%d file references missing "
                    "(run place_fused_masks.py first, check --placed_spine_dir)",
                    n_missing)
    else:
        log.info("All file references OK.")

    manifest = {
        "metadata": {
            "generated_by":         "build_manifest.py",
            "source_placed":        str(placed_manifest_path),
            "source_patient_db":    str(patient_db_path) if patient_db_path else None,
            "n_fused":              len(fused),
            "n_spine_only":         len(spine_only),
            "n_spine_from_separate": n_separate_spine,
            "n_pelvic_only":        len(pelvic_only),
            "n_pelvic_from_separate": n_separate_pelvic,
            "n_missing_files":      n_missing,
            "lstv_counts":          dict(lstv_counts),
        },
        "fused":       fused,
        "spine_only":  spine_only,
        "pelvic_only": pelvic_only,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Manifest -> %s  (%d entries)", out_path, len(all_entries))
    return manifest


def verify_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)
    m    = json.loads(manifest_path.read_text())
    meta = m.get("metadata", {})
    log.info("Manifest: fused=%d  spine_only=%d  pelvic_only=%d",
             meta.get("n_fused", 0),
             meta.get("n_spine_only", 0),
             meta.get("n_pelvic_only", 0))
    all_entries = (m.get("fused", []) + m.get("spine_only", [])
                   + m.get("pelvic_only", []))
    n_ok = n_missing = n_none = 0
    for e in all_entries:
        for field in ("image", "spine_label", "pelvic_mask"):
            v = e.get(field)
            if v is None:
                n_none += 1
            elif not Path(v).exists():
                n_missing += 1
                log.warning("MISSING  token=%-6s  %s=%s",
                            e["patient_token"], field, v)
            else:
                n_ok += 1
    log.info("Files: %d OK  %d missing  %d null", n_ok, n_missing, n_none)
    if n_missing:
        log.error("%d missing files", n_missing)
        sys.exit(1)
    log.info("Manifest OK.")


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--placed_manifest",
                   default="data/placed/placed_manifest.json",
                   help="Canonical output of place_fused_masks.py")
    p.add_argument("--patient_db",
                   default="data/patient_db/patient_db.json",
                   help="Optional: augments LSTV / warnings fields")
    p.add_argument("--out",
                   default="data/matched/colonog_training_manifest.json")
    p.add_argument("--nifti_dir",          default="data/tcia_nifti")
    p.add_argument("--placed_spine_dir",   default="data/placed/spine")
    p.add_argument("--placed_mask_dir",    default="data/placed/fused",
                   help="Pelvic masks on fused spine CTs")
    p.add_argument("--placed_pelvic_native_dir",
                   default="data/placed/pelvic_native",
                   help="Pelvic masks on their own pelvic CTs")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    if a.verify:
        verify_manifest(Path(a.out))
    else:
        build_manifest(
            placed_manifest_path      = Path(a.placed_manifest),
            patient_db_path           = Path(a.patient_db) if a.patient_db else None,
            out_path                  = Path(a.out),
            nifti_dir                 = Path(a.nifti_dir) if a.nifti_dir else None,
            placed_spine_dir          = Path(a.placed_spine_dir) if a.placed_spine_dir else None,
            placed_mask_dir           = Path(a.placed_mask_dir)  if a.placed_mask_dir  else None,
            placed_pelvic_native_dir  = Path(a.placed_pelvic_native_dir)
                                        if a.placed_pelvic_native_dir else None,
        )
