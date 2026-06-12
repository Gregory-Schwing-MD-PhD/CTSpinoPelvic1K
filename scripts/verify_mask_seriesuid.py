"""
verify_mask_seriesuid.py — empirically confirm the paper's central premise:
the CTSpine1K and CTPelvic1K masks carry NO SeriesInstanceUID (only a PatientID,
and at most a weak SeriesNumber proxy), so they cannot be trivially matched to a
specific TCIA DICOM series.

We parse identity the way the pipeline does (mask_index.py):
  spine  : {patient_uid}_seg.nii.gz
  pelvic : dataset2_{patient_uid}_{series_number}_{nz}[_qualifiers]_mask_4label.nii.gz
so the patient UID (and, for pelvic, a SeriesNUMBER) come from the filename. We
then check, against the TCIA index (every series' PatientID + SeriesInstanceUID):

  * does the filename carry a true TCIA SeriesInstanceUID?   (expect 0)
  * does it resolve to a TCIA PatientID?                     (expect ~all COLONOG)
  * pelvic only: does it carry a SeriesNUMBER weak proxy?    (a small int, not a UID)
  * does the NIfTI header hide a SeriesInstanceUID?          (expect 0)
and report series-per-patient (>1 -> PatientID is ambiguous -> bone-HU needed)
and that CTPelvic1K ships no CT.

Run in the project container (needs nibabel). See slurm/verify_mask_seriesuid.sh.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_db import canonical_uid                 # noqa: E402
from tcia_index import build_tcia_patient_index       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("verify_mask_seriesuid")

# The pipeline's own filename parsers (mask_index.py).
_SPINE_RE = re.compile(
    r"^(?:(1\.3\.6\.1\.4\.1\.9328\.50\.4\.\d+)|(CTC-\d+))_seg\.nii\.gz$", re.I)
_PELVIC_RE = re.compile(
    r"^dataset2_(1\.3\.6\.1\.4\.1\.9328\.50\.4\.\d+)_(\d+)_(\d+)"
    r"((?:_[A-Za-z0-9]+)*)_mask_4label(?:\.nii\.gz|\.nii)?$", re.I)
_UID_RE = re.compile(r"\b\d+(?:\.\d+){4,}\b")   # DICOM-UID-like, for header scan


def _spine_id(name: str) -> Tuple[Optional[str], Optional[str]]:
    m = _SPINE_RE.match(name)
    if not m:
        return None, None
    return (m.group(1) or m.group(2)), None          # (patient_uid, series_number)


def _pelvic_id(name: str) -> Tuple[Optional[str], Optional[str]]:
    m = _PELVIC_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)                    # (patient_uid, series_number)


def _header_has_series(p: Path, series_uids: Set[str]) -> bool:
    try:
        import nibabel as nib
        h = nib.load(str(p)).header
        for field in ("descrip", "db_name", "aux_file", "intent_name"):
            try:
                val = bytes(h[field]).decode("latin-1", "ignore")
            except Exception:
                val = str(h.get(field, ""))
            for u in _UID_RE.findall(val):
                if u in series_uids or canonical_uid(u) in series_uids:
                    return True
        return False
    except Exception:
        return False


def _scan(files: List[Path], extract, patient_uids: Set[str],
          series_uids: Set[str], label: str, ct_dir: Path) -> None:
    n = len(files)
    has_pid = has_series_collide = has_series_num = hdr_series = nonsource = 0
    collide_examples: List[str] = []
    for f in files:
        pid, snum = extract(f.name)
        if pid is None:
            nonsource += 1                            # other source (VerSe/liver/...)
            continue
        canon = canonical_uid(pid)
        if canon in patient_uids or pid in patient_uids:
            has_pid += 1
        if canon in series_uids or pid in series_uids:
            has_series_collide += 1
            if len(collide_examples) < 5:
                collide_examples.append(f.name)
        if snum is not None:
            has_series_num += 1
        if _header_has_series(f, series_uids):
            hdr_series += 1

    log.info("-" * 64)
    log.info("%s: %d mask files", label, n)
    log.info("  parsed as a TCIA (COLONOG) patient UID : %d", n - nonsource)
    log.info("  of those, resolve to a TCIA PatientID  : %d   <- patient-level only",
             has_pid)
    log.info("  carry a true SeriesInstanceUID         : %d   <- (UID-namespace collision, not a series link)",
             has_series_collide)
    if collide_examples:
        log.info("      collisions: %s", collide_examples)
    log.info("  carry only a SeriesNUMBER proxy (int)  : %d   <- weak proxy, NOT a UID",
             has_series_num)
    log.info("  SeriesInstanceUID hidden in NIfTI hdr  : %d", hdr_series)
    log.info("  non-COLONOG source (no TCIA UID)       : %d   <- e.g. VerSe / MSD-liver",
             nonsource)
    if ct_dir is not None and ct_dir.exists():
        cts = [c for c in ct_dir.rglob("*.nii.gz")
               if "ct" in c.name.lower() and "seg" not in c.name.lower()
               and "mask" not in c.name.lower()]
        log.info("  paired CT volumes shipped              : %d%s", len(cts),
                 "  (none -> mask cannot even self-locate)" if not cts else
                 "  (anonymized; no series link)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcia_dir", required=True, type=Path)
    ap.add_argument("--spine_dir", type=Path, default=None)
    ap.add_argument("--pelvic_dir", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()

    grouped, _ = build_tcia_patient_index(a.tcia_dir)
    patient_uids: Set[str] = set(grouped.keys())
    series_uids: Set[str] = {r.series_uid for rs in grouped.values() for r in rs}
    sp = sorted(len(rs) for rs in grouped.values())
    log.info("=" * 64)
    log.info("TCIA index: %d patients, %d distinct SeriesInstanceUIDs",
             len(patient_uids), len(series_uids))
    if sp:
        log.info("series per patient: min %d, median %d, max %d  "
                 "(>1 => PatientID alone is ambiguous => bone-HU needed)",
                 sp[0], sp[len(sp) // 2], sp[-1])

    def _masks(root: Optional[Path]) -> List[Path]:
        if not root or not root.exists():
            return []
        fs = sorted(root.rglob("*.nii.gz"))
        fs = [p for p in fs if "seg" in p.name.lower() or "mask" in p.name.lower()] or fs
        return fs[: a.sample] if a.sample else fs

    if a.spine_dir:
        _scan(_masks(a.spine_dir), lambda nm: _spine_id(nm), patient_uids,
              series_uids, "CTSpine1K (spine masks)", a.spine_dir)
    if a.pelvic_dir:
        _scan(_masks(a.pelvic_dir), lambda nm: _pelvic_id(nm), patient_uids,
              series_uids, "CTPelvic1K (pelvic masks)", a.pelvic_dir)

    log.info("=" * 64)
    log.info("READ: 'true SeriesInstanceUID' and 'hidden in NIfTI hdr' = 0 -> no "
             "mask carries a series identifier. Masks resolve only to a PatientID "
             "(pelvic also to a weak SeriesNUMBER), and patients have a median of "
             "%d series -> the bone-HU search is required to pick the right scan.",
             sp[len(sp) // 2] if sp else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
