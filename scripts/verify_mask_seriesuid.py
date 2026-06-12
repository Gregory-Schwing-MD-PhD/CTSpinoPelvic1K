"""
verify_mask_seriesuid.py — empirically confirm the paper's central premise:
the CTSpine1K and CTPelvic1K masks carry NO SeriesInstanceUID, only a
PatientID, so they cannot be trivially matched to a specific TCIA DICOM series.

For each mask file we check three things against the TCIA index (which holds,
per patient, every series' PatientID and SeriesInstanceUID):

  1. Does the mask's filename identifier match a TCIA *SeriesInstanceUID*?
     (If yes, the mask WOULD be trivially mappable. Expect: 0.)
  2. Does it match a TCIA *PatientID* (after canonicalization)?
     (Expect: ~all — the mask resolves only to a patient.)
  3. Does the NIfTI header (descrip/db_name/aux_file/intent_name) hide a
     SeriesInstanceUID? (Expect: none.)

Then it reports, for the matched patients, how many TCIA series each has —
because >1 series per patient is exactly why a PatientID is not enough and the
bone-HU search is needed. It also confirms CTPelvic1K ships no paired CT.

Run inside the project container (needs nibabel):

  singularity exec --bind "$(pwd):/workspace,$DATA_DIR:/data" "$SIF_PATH" \
    python3 /workspace/scripts/verify_mask_seriesuid.py \
      --tcia_dir /data/tcia --spine_dir /data/ctspine1k --pelvic_dir /data/ctpelvic1k
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_db import canonical_uid                 # noqa: E402
from tcia_index import build_tcia_patient_index       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("verify_mask_seriesuid")

# A DICOM UID: digits and dots, reasonably long. Used to scan filenames + headers.
_UID_RE = re.compile(r"\b\d+(?:\.\d+){4,}\b")
# strip the mask-file decorations to recover the bare identifier in the name
_STRIP = ("_seg.nii.gz", "_mask.nii.gz", "_CT-iso", ".nii.gz", ".nii")


def _name_uid(p: Path) -> str:
    s = p.name
    for suf in _STRIP:
        s = s.replace(suf, "")
    return s


def _header_uids(p: Path) -> List[str]:
    """Any DICOM-UID-like string hiding in the NIfTI text header fields."""
    try:
        import nibabel as nib
        h = nib.load(str(p)).header
        found = []
        for field in ("descrip", "db_name", "aux_file", "intent_name"):
            try:
                val = bytes(h[field]).decode("latin-1", "ignore")
            except Exception:
                val = str(h.get(field, ""))
            found += _UID_RE.findall(val)
        return found
    except Exception:
        return []


def _scan(mask_files: List[Path], patient_uids: Set[str], series_uids: Set[str],
          label: str, ct_check_dir: Optional[Path]) -> None:
    n = len(mask_files)
    hits_series = hits_patient = hits_header_series = no_match = 0
    examples_series, examples_patient_only = [], []
    for f in mask_files:
        raw = _name_uid(f)
        canon = canonical_uid(raw)
        in_series = (raw in series_uids) or (canon in series_uids)
        in_patient = (canon in patient_uids) or (raw in patient_uids)
        hdr = _header_uids(f)
        hdr_series = any((u in series_uids) or (canonical_uid(u) in series_uids)
                         for u in hdr)
        hits_series += int(in_series)
        hits_patient += int(in_patient)
        hits_header_series += int(hdr_series)
        if in_series and len(examples_series) < 3:
            examples_series.append(f.name)
        if in_patient and not in_series and len(examples_patient_only) < 3:
            examples_patient_only.append(f.name)
        if not in_series and not in_patient and not hdr_series:
            no_match += 1

    log.info("-" * 64)
    log.info("%s: %d mask files", label, n)
    log.info("  filename == a TCIA SeriesInstanceUID : %d   (trivially mappable)",
             hits_series)
    log.info("  filename -> a TCIA PatientID         : %d   (patient-level only)",
             hits_patient)
    log.info("  SeriesInstanceUID hidden in NIfTI hdr: %d", hits_header_series)
    log.info("  matched neither (other source/format): %d", no_match)
    if examples_patient_only:
        log.info("  e.g. patient-only: %s", examples_patient_only)
    if ct_check_dir is not None:
        cts = list(ct_check_dir.rglob("*.nii.gz"))
        n_ct = sum(1 for c in cts if "ct" in c.name.lower()
                   and "seg" not in c.name.lower() and "mask" not in c.name.lower())
        log.info("  paired CT volumes shipped alongside  : %d  %s",
                 n_ct, "(none -> mask cannot even be self-located)" if n_ct == 0 else "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcia_dir", required=True, type=Path)
    ap.add_argument("--spine_dir", type=Path, default=None,
                    help="CTSpine1K mask root (e.g. data/ctspine1k).")
    ap.add_argument("--pelvic_dir", type=Path, default=None,
                    help="CTPelvic1K mask root (e.g. data/ctpelvic1k).")
    ap.add_argument("--sample", type=int, default=0,
                    help="cap masks scanned per source (0 = all).")
    a = ap.parse_args()

    grouped, _ = build_tcia_patient_index(a.tcia_dir)
    patient_uids: Set[str] = set(grouped.keys())
    series_uids: Set[str] = {r.series_uid for rs in grouped.values() for r in rs}
    series_per_pat = [len(rs) for rs in grouped.values()]
    log.info("=" * 64)
    log.info("TCIA index: %d patients, %d distinct SeriesInstanceUIDs",
             len(patient_uids), len(series_uids))
    if series_per_pat:
        sp = sorted(series_per_pat)
        med = sp[len(sp) // 2]
        log.info("series per patient: min %d, median %d, max %d  "
                 "(>1 => PatientID alone is ambiguous)",
                 sp[0], med, sp[-1])

    def _masks(root: Optional[Path]) -> List[Path]:
        if not root or not root.exists():
            return []
        fs = sorted(p for p in root.rglob("*.nii.gz")
                    if "seg" in p.name.lower() or "mask" in p.name.lower())
        fs = fs or sorted(root.rglob("*.nii.gz"))
        return fs[: a.sample] if a.sample else fs

    if a.spine_dir:
        _scan(_masks(a.spine_dir), patient_uids, series_uids,
              "CTSpine1K (spine masks)", ct_check_dir=a.spine_dir)
    if a.pelvic_dir:
        _scan(_masks(a.pelvic_dir), patient_uids, series_uids,
              "CTPelvic1K (pelvic masks)", ct_check_dir=a.pelvic_dir)

    log.info("=" * 64)
    log.info("READ: 'filename == SeriesInstanceUID' and 'hidden in NIfTI hdr' "
             "should be 0 -> no mask carries a series identifier. "
             "'-> a PatientID' should be ~all -> the mask resolves only to a "
             "patient, who has >1 series -> the bone-HU search is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
