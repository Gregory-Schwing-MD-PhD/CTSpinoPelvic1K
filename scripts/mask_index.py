"""
mask_index.py — Parse all spine and pelvic mask files.

Extracts patient identity and metadata from filenames.  NIfTI headers are read
for geometry (affine + shape) to support downstream series matching, but patient
identity is NEVER derived from NIfTI content — only from the filename-encoded UID.

SPINE MASK FILENAMES
--------------------
Format: {patient_uid}_seg.nii.gz

PELVIC MASK FILENAMES
---------------------
Format: dataset2_{patient_uid}_{series_number}_{nz}[_qualifiers]_mask_4label.nii.gz

QUALIFIER PARSING (Apr 2026 fix)
================================
Pre-fix bug
-----------
_parse_pelvic_flags() did substring matching on the lowercased qualifier
block:

    if "sacralization" in rem:
        lstv = LSTV.SACRALIZATION
    else:
        lstv = LSTV.NORMAL

Because "sacralization" is a substring of "semisacralization", the 2 patients
with `_semisacralization_` qualifier (tokens 22 and 120 in COLONOG) were
silently labeled as SACRALIZATION instead of SEMI_SACRAL. The
LSTV.SEMI_SACRAL enum value existed in patient_db.py but was never assigned.

Fix
---
Tokenize the qualifier block by underscore, then do exact-token matching.
Order of LSTV resolution:

    1. "semisacralization" token  → LSTV.SEMI_SACRAL
    2. "sacralization" token      → LSTV.SACRALIZATION
    3. otherwise                  → LSTV.NORMAL

The `hard_sacralization` qualifier (1 patient, token 123) decomposes into
two tokens: ['hard', 'sacralization']. The parser treats it as SACRALIZATION
(LSTV) AND sets flag_hard (image quality), not as a separate LSTV grade.
"hard" / "veryhard" are CTPelvic1K's annotation-difficulty flags, orthogonal
to anatomical LSTV classification.

Quality / annotation-difficulty flags exposed via PelvicMaskRecord.lstv_flags:
  - flag_hard, flag_veryhard       — annotator difficulty markers
  - flag_lowdose, flag_metal       — scan quality
  - flag_crop                      — cropped scan
  - flag_dqjoint, flag_ydjoint     — annotator-specific markers
  - flag_intestinal_calculus       — pathology
  - flag_rl_flip                   — orientation marker
  - flag_supine, flag_prone        — patient position
  - flag_sacralization, flag_semisacralization  — also drive lstv_label
"""

from __future__ import annotations

import concurrent.futures
import logging
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from patient_db import (
    SpineMaskRecord,
    PelvicMaskRecord,
    LSTV,
    LUMBAR_LABEL_IDS,
    canonical_uid,
    patient_token,
)

log = logging.getLogger(__name__)

try:
    import numpy as np
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False
    warnings.warn("nibabel/numpy not installed — NIfTI geometry unavailable.", stacklevel=1)

_SPINE_CACHE_NAME  = ".spine_mask_cache.pkl"
_PELVIC_CACHE_NAME = ".pelvic_mask_cache.pkl"


def _cache_is_valid(cache_path: Path, source_dir: Path, glob: str) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for f in source_dir.rglob(glob):
        if f.stat().st_mtime > cache_mtime:
            log.info("Cache stale: %s is newer than cache", f.name)
            return False
    return True


def _load_cache(cache_path: Path, label: str) -> Optional[List]:
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        log.info("%s cache loaded: %d records from %s", label, len(data), cache_path)
        return data
    except Exception as exc:
        log.warning("%s cache load failed (%s) — will rescan.", label, exc)
        return None


def _save_cache(records: List, cache_path: Path, label: str) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz = cache_path.stat().st_size / 1e3
        log.info("%s cache saved: %d records → %s  (%.0f KB)", label, len(records), cache_path, sz)
    except Exception as exc:
        log.warning("Failed to save %s cache: %s", label, exc)


# ── Filename patterns ─────────────────────────────────────────────────────────

_SPINE_SEG_RE = re.compile(
    r"^(?:(1\.3\.6\.1\.4\.1\.9328\.50\.4\.(\d+))|(CTC-\d+))_seg\.nii\.gz$",
    re.IGNORECASE,
)

_PELVIC_MASK_RE = re.compile(
    r"^dataset2_"
    r"(1\.3\.6\.1\.4\.1\.9328\.50\.4\.\d+)"
    r"_(\d+)"
    r"_(\d+)"
    r"((?:_[A-Za-z0-9]+)*)"
    r"_mask_4label"
    r"(?:\.nii\.gz|\.nii)?$",
    re.IGNORECASE,
)

# Per-token qualifier flags. Each entry maps a single underscore-delimited
# token in the filename's qualifier block to a boolean flag column on
# PelvicMaskRecord.lstv_flags. With token-based matching there is no
# substring collision between 'sacralization' and 'semisacralization'.
_PELVIC_QUALIFIER_FLAGS = [
    # LSTV qualifiers (also drive lstv_label resolution)
    ("semisacralization", "flag_semisacralization"),
    ("sacralization",     "flag_sacralization"),
    # Annotation difficulty / scan quality (orthogonal to LSTV)
    ("hard",              "flag_hard"),
    ("veryhard",          "flag_veryhard"),
    ("lowdose",           "flag_lowdose"),
    ("metal",             "flag_metal"),
    ("crop",              "flag_crop"),
    # Annotator markers
    ("dqjoint",           "flag_dqjoint"),
    ("ydjoint",           "flag_ydjoint"),
    ("intestinalcalculus","flag_intestinal_calculus"),
    # Position / orientation
    ("rl",                "flag_rl_flip"),
    ("supine",            "flag_supine"),
    ("prone",             "flag_prone"),
]


def _read_nifti_header(path: Path) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    if not HAS_NIB:
        return None, None
    try:
        img    = nib.load(str(path))
        shape  = list(img.shape)
        affine = img.affine.flatten().tolist()
        return shape, affine
    except Exception as exc:
        log.debug("_read_nifti_header failed for %s: %s", path, exc)
        return None, None


def _count_lumbar_labels(seg_path: Path) -> Tuple[str, int, List[str]]:
    """Count lumbar labels to infer LSTV status."""
    if not HAS_NIB:
        return LSTV.UNKNOWN, 0, []
    try:
        data    = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.int16)
        present = {int(v) for v in np.unique(data) if v != 0}
        labels  = [LUMBAR_LABEL_IDS[lid]
                   for lid in sorted(LUMBAR_LABEL_IDS)
                   if lid in present]
        n = len(labels)

        if n == 0:   return LSTV.AMBIGUOUS,       0, labels
        if n <= 3:   return LSTV.INCOMPLETE_SCAN, n, labels
        if n == 4:   return LSTV.SACRALIZATION,   n, labels
        if n == 5:   return LSTV.NORMAL,          n, labels
        if n == 6:   return LSTV.LUMBARIZATION,   n, labels
        return LSTV.AMBIGUOUS, n, labels

    except Exception as exc:
        log.debug("_count_lumbar_labels failed for %s: %s", seg_path, exc)
        return LSTV.UNKNOWN, 0, []


# ── Spine ────────────────────────────────────────────────────────────────────

def _find_spine_image(seg_path: Path, spine_root: Path) -> Optional[Path]:
    base      = seg_path.name.replace("_seg.nii.gz", "")
    raw_dir   = seg_path.parent.parent.parent
    for d in (raw_dir, spine_root / "raw_data", spine_root / "rawdata", spine_root):
        for sub in ("volumes/COLONOG", "volumes", ""):
            for suffix in ("", "_ct"):
                p = (d / sub / f"{base}{suffix}.nii.gz") if sub else (d / f"{base}{suffix}.nii.gz")
                if p.exists():
                    return p
    return None


def _parse_one_spine_mask(seg_path: Path, spine_root: Path) -> Optional[SpineMaskRecord]:
    fname = seg_path.name
    m     = _SPINE_SEG_RE.match(fname)
    if not m:
        log.warning("Spine seg filename did not match expected pattern: %s", fname)
        return None

    if m.group(3):
        uid_raw = m.group(3)
        uid     = uid_raw.upper()
    else:
        uid_raw = m.group(1)
        uid     = canonical_uid(uid_raw)

    tok = patient_token(uid)

    img_path         = _find_spine_image(seg_path, spine_root)
    geom_path        = img_path if img_path else seg_path
    shape, affine    = _read_nifti_header(geom_path)

    lstv_label, n_lumbar, lumbar_labels = _count_lumbar_labels(seg_path)

    return SpineMaskRecord(
        mask_file       = str(seg_path),
        image_file      = str(img_path) if img_path else None,
        patient_uid     = uid,
        patient_token   = tok,
        nifti_shape     = shape,
        nifti_affine    = affine,
        lstv_label      = lstv_label,
        n_lumbar_found  = n_lumbar,
        lumbar_labels   = lumbar_labels,
        candidates      = [],
    )


def _parse_spine_mask_worker(args: Tuple) -> Optional["SpineMaskRecord"]:
    seg_path_str, spine_root_str = args
    return _parse_one_spine_mask(Path(seg_path_str), Path(spine_root_str))


def parse_spine_masks(
    spine_root:      Path,
    workers:         int  = 16,
    debug_n:         Optional[int] = None,
    rebuild_cache:   bool = False,
) -> List[SpineMaskRecord]:
    seg_dirs = [
        spine_root / "rawdata"  / "labels" / "COLONOG",
        spine_root / "raw_data" / "labels" / "COLONOG",
        spine_root / "labels"   / "COLONOG",
        spine_root / "COLONOG",
        spine_root,
    ]
    seg_dir: Optional[Path] = None
    seg_files: List[Path] = []
    for d in seg_dirs:
        if d.exists():
            seg_files = sorted(d.rglob("*_seg.nii.gz"))
            if seg_files:
                seg_dir = d
                log.info("Spine segs: %d files found under %s", len(seg_files), d)
                break
    if not seg_files:
        seg_files = sorted(spine_root.rglob("*_seg.nii.gz"))
        seg_dir   = spine_root
        log.info("Spine segs (broad search): %d files", len(seg_files))

    cache_path = spine_root / _SPINE_CACHE_NAME
    if not debug_n and not rebuild_cache and seg_dir:
        if _cache_is_valid(cache_path, seg_dir, "*_seg.nii.gz"):
            cached = _load_cache(cache_path, "spine mask")
            if cached is not None:
                from collections import Counter
                log.info("  LSTV labels (cached): %s",
                         dict(Counter(r.lstv_label for r in cached)))
                return cached
        else:
            log.info("Spine mask cache missing or stale — rescanning.")

    if debug_n:
        seg_files = seg_files[:debug_n]
        log.info("Spine segs: limited to first %d for debug (cache skipped).", debug_n)

    total  = len(seg_files)
    work   = [(str(f), str(spine_root)) for f in seg_files]
    records: List[SpineMaskRecord] = []
    n_fail = 0
    t0     = time.time()

    log.info("Parsing %d spine seg files (workers=%d) ...", total, workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parse_spine_mask_worker, w): w[0] for w in work}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                r = fut.result()
                if r:
                    records.append(r)
                else:
                    n_fail += 1
            except Exception as exc:
                n_fail += 1
                log.error("Spine parse failed %s: %s",
                          Path(futures[fut]).name, exc)

            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate    = i / max(elapsed, 0.001)
                eta     = (total - i) / max(rate, 0.001)
                log.info(
                    "  spine [%d/%d]  ok=%d  fail=%d  "
                    "elapsed=%.0fs  ETA=%.0fs  (%.1f files/s)",
                    i, total, len(records), n_fail, elapsed, eta, rate,
                )

    records.sort(key=lambda r: (0, int(r.patient_token)) if r.patient_token.isdigit()
                                else (1, r.patient_token))
    log.info("Parsed %d spine masks (from %d files, %d failed).",
             len(records), total, n_fail)

    from collections import Counter
    lstv_counts = Counter(r.lstv_label for r in records)
    log.info("  LSTV labels: %s", dict(lstv_counts))

    if not debug_n:
        _save_cache(records, cache_path, "spine mask")
    return records


# ── Pelvic ───────────────────────────────────────────────────────────────────

def _parse_pelvic_flags(qualifier_block: str) -> Tuple[Dict[str, bool], str, str]:
    """
    Parse the pelvic mask qualifier block into:
      - flags dict: every entry in _PELVIC_QUALIFIER_FLAGS gets a bool
      - lstv label: NORMAL / SEMI_SACRAL / SACRALIZATION
      - position:   PRONE / SUPINE / UNKNOWN

    Apr 2026 fix
    ------------
    Pre-fix did substring matching:

        if "sacralization" in rem:
            lstv = LSTV.SACRALIZATION

    which fired on both 'sacralization' AND 'semisacralization' tokens
    because the latter contains the former as a substring. Result: the
    2 patients with `_semisacralization_` qualifier (tokens 22 and 120)
    were misclassified as full sacralization instead of semi-sacralization.

    Post-fix: split the qualifier block by underscore into individual
    tokens, then check membership exactly. 'semisacralization' is checked
    BEFORE 'sacralization' so that tokens like `hard_sacralization`
    (which decomposes into ['hard', 'sacralization']) correctly resolve
    to SACRALIZATION without spurious semi-matching.
    """
    # Initialize all flags to False
    flags: Dict[str, bool] = {col: False for _, col in _PELVIC_QUALIFIER_FLAGS}

    # Tokenize qualifier block. Empty block → no tokens, all flags False.
    tokens = [t for t in qualifier_block.lower().strip("_").split("_") if t]
    token_set = set(tokens)

    # Set per-token boolean flags with EXACT match (no substring confusion).
    for key, col in _PELVIC_QUALIFIER_FLAGS:
        if key in token_set:
            flags[col] = True

    # Resolve LSTV label. Order matters: semi BEFORE full.
    if flags["flag_semisacralization"]:
        lstv = LSTV.SEMI_SACRAL
    elif flags["flag_sacralization"]:
        lstv = LSTV.SACRALIZATION
    else:
        lstv = LSTV.NORMAL

    # Resolve position from per-token flags first; fall back to substring
    # match on the original qualifier block for legacy uppercase tokens.
    if flags["flag_prone"]:
        position = "PRONE"
    elif flags["flag_supine"]:
        position = "SUPINE"
    else:
        upper = qualifier_block.upper()
        if "PRONE" in upper:
            position = "PRONE"
        elif "SUPINE" in upper:
            position = "SUPINE"
        else:
            position = "UNKNOWN"

    return flags, lstv, position


def _parse_one_pelvic_mask(mask_path: Path) -> Optional[PelvicMaskRecord]:
    name = mask_path.name
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break

    m = _PELVIC_MASK_RE.match(mask_path.name)
    if not m:
        log.warning("Pelvic mask filename did not match expected pattern: %s",
                    mask_path.name)
        return None

    uid_raw         = m.group(1)
    series_num      = int(m.group(2))
    nz              = int(m.group(3))
    qualifier_block = (m.group(4) or "").strip("_")

    uid = canonical_uid(uid_raw)
    tok = patient_token(uid)

    flags, lstv_label, position = _parse_pelvic_flags(qualifier_block)

    shape, affine = _read_nifti_header(mask_path)

    return PelvicMaskRecord(
        mask_file                = str(mask_path),
        patient_uid              = uid,
        patient_token            = tok,
        series_number_from_fname = series_num,
        nz_from_fname            = nz,
        position_from_fname      = position,
        nifti_shape              = shape,
        nifti_affine             = affine,
        lstv_label               = lstv_label,
        lstv_qualifier_block     = qualifier_block,
        lstv_flags               = flags,
        candidates               = [],
    )


def _parse_pelvic_mask_worker(path_str: str) -> Optional["PelvicMaskRecord"]:
    return _parse_one_pelvic_mask(Path(path_str))


def parse_pelvic_masks(
    pelvis_root:   Path,
    workers:       int  = 16,
    debug_n:       Optional[int] = None,
    rebuild_cache: bool = False,
) -> List[PelvicMaskRecord]:
    mask_dir: Optional[Path] = None
    for candidate in [
        pelvis_root / "masks" / "CTPelvic1K_dataset2_mask_mappingback",
        pelvis_root / "masks",
    ]:
        if candidate.exists():
            mask_dir = candidate
            break

    if mask_dir is None:
        log.error("Pelvic mask directory not found under %s", pelvis_root)
        return []

    mask_files = sorted(mask_dir.glob("dataset2_*.nii.gz"))
    log.info("Pelvic masks: %d files found under %s", len(mask_files), mask_dir)

    cache_path = pelvis_root / _PELVIC_CACHE_NAME
    if not debug_n and not rebuild_cache:
        if _cache_is_valid(cache_path, mask_dir, "dataset2_*.nii.gz"):
            cached = _load_cache(cache_path, "pelvic mask")
            if cached is not None:
                from collections import Counter
                log.info("  LSTV labels (cached): %s",
                         dict(Counter(r.lstv_label for r in cached)))
                log.info("  Positions   (cached): %s",
                         dict(Counter(r.position_from_fname for r in cached)))
                return cached
        else:
            log.info("Pelvic mask cache missing or stale — rescanning.")

    if debug_n:
        mask_files = mask_files[:debug_n]
        log.info("Pelvic masks: limited to first %d for debug (cache skipped).", debug_n)

    total  = len(mask_files)
    work   = [str(f) for f in mask_files]
    records: List[PelvicMaskRecord] = []
    n_fail = 0
    t0     = time.time()

    log.info("Parsing %d pelvic mask files (workers=%d) ...", total, workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parse_pelvic_mask_worker, w): w for w in work}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                r = fut.result()
                if r:
                    records.append(r)
                else:
                    n_fail += 1
            except Exception as exc:
                n_fail += 1
                log.error("Pelvic parse failed %s: %s",
                          Path(futures[fut]).name, exc)

            if i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate    = i / max(elapsed, 0.001)
                eta     = (total - i) / max(rate, 0.001)
                log.info(
                    "  pelvic [%d/%d]  ok=%d  fail=%d  "
                    "elapsed=%.0fs  ETA=%.0fs  (%.1f files/s)",
                    i, total, len(records), n_fail, elapsed, eta, rate,
                )

    records.sort(key=lambda r: (0, int(r.patient_token)) if r.patient_token.isdigit()
                                else (1, r.patient_token))
    log.info("Parsed %d pelvic masks (from %d files, %d failed).",
             len(records), total, n_fail)

    from collections import Counter
    lstv_counts = Counter(r.lstv_label for r in records)
    pos_counts  = Counter(r.position_from_fname for r in records)
    flag_counts = Counter()
    for r in records:
        for col, val in r.lstv_flags.items():
            if val:
                flag_counts[col] += 1
    log.info("  LSTV labels: %s", dict(lstv_counts))
    log.info("  Positions:   %s", dict(pos_counts))
    if flag_counts:
        log.info("  Active flags: %s", dict(sorted(flag_counts.items(),
                                                    key=lambda x: -x[1])))

    if not debug_n:
        _save_cache(records, cache_path, "pelvic mask")

    return records
