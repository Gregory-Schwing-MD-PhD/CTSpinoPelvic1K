"""
dataset_interface.py — Runtime interface for the CTSpinoPelvic1K HF dataset.

Two classes:
  CTSpinoPelvic1K        dict-style dataset wrapping an HF export directory.
                         Used by benchmark_totalseg.py / viz_ts_case.py /
                         render_lstv_examples.py.  No torch dependency.
  CTSpinoPelvicDataset   PyTorch Dataset adapter on top of CTSpinoPelvic1K.

Expected layout (produced by scripts/export_hf.py):
  <root>/
    ct/<token:04d>_ct.nii.gz                  # fused
    ct/<token:04d>_spine_ct.nii.gz            # spine-side (separate or
                                                spine_only single-mask)
    ct/<token:04d>_pelvic_ct.nii.gz           # pelvic-side (separate or
                                                pelvic_only single-mask)
    labels/<token:04d>_label.nii.gz           # parallel naming
    labels/<token:04d>_spine_label.nii.gz
    labels/<token:04d>_pelvic_label.nii.gz
    manifest.json                 per-record metadata (flat list OR
                                  {"records": [...]} wrapper — both accepted)
    splits_5fold.json             PREFERRED: unified splits file (schema v3+,
                                  current generator writes v4 with LSTV-first
                                  stratification).  Carries test_tokens +
                                  folds in one file.
    splits/                       legacy layout (still read as fallback):
      test.json                   flat list of unique test patient tokens
      cv_5fold.json               5-fold CV on trainval pool
    data_splits.json              earliest format: {"train": [...], "val":
                                  [...], "test": [...]} of ct_file entries
                                  (last-resort fallback)
    splits_summary.json           aggregate split stats (optional)

Filename schema note (changed Apr 2026)
---------------------------------------
The earlier schema baked `position` (supine / prone) into every filename.
That was misleading because the prone/supine classifier rarely succeeded,
and `config` (fused / spine_only / pelvic_native) is what every downstream
consumer actually filters on. The current schema uses suffixes alone:

    fused                      ->  <token:04d>_ct.nii.gz
    spine annotated            ->  <token:04d>_spine_ct.nii.gz
    pelvic annotated           ->  <token:04d>_pelvic_ct.nii.gz

Bare `<token>_ct.nii.gz` therefore unambiguously means a `fused` case
(both regions present in one mask). `position` still rides through to
the manifest as a metadata column — it is no longer in the filename.

Manifest ct_file / label_file paths can be either a relative path
("ct/0017_ct.nii.gz") or a bare basename ("0017_ct.nii.gz"). The path
resolver tries the value verbatim first and falls back to
`root/ct/{basename}` (or `root/labels/{basename}`) if the primary misses.
This keeps the class tolerant of manifests that predate the path-prefix
fix in export_hf.py.

Splits resolution order (first hit wins):
  1. splits_5fold.json                 (unified, schema v3+ from
                                        generate_5fold_splits.py;
                                        v4 is the current schema)
  2. splits/test.json + splits/cv_5fold.json (legacy pair)
  3. data_splits.json                  (earliest export_hf.py format)

Quickstart (benchmarking / viz — no splits):
  >>> from dataset_interface import CTSpinoPelvic1K
  >>> ds = CTSpinoPelvic1K("data/hf_export")
  >>> print(ds.stats())
  >>> fused = ds.filter(config="fused", present_only=True)

Quickstart (HF Hub — lazy NIfTI fetch on first access):
  >>> ds = CTSpinoPelvic1K.from_hub(repo_id="anonymous-neurips-ED/CTSpinoPelvic1K")
  >>> ct_arr, affine = ds.cases[0].load_ct()   # downloads on first call
  >>> # subsequent loads of the same case hit the local HF cache

Quickstart (training):
  >>> from dataset_interface import CTSpinoPelvicDataset
  >>> ds_tr = CTSpinoPelvicDataset("data/hf_export", split=("fold", 0, "train"))
  >>> ds_va = CTSpinoPelvicDataset("data/hf_export", split=("fold", 0, "val"))
  >>> ds_te = CTSpinoPelvicDataset("data/hf_export", split="test")

Quickstart (annotation-aware filtering, placed_manifest schema v2.1+):
  >>> sp = ds.filter(has_annotation="spinous", present_only=True)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore


LABEL_NAMES = [
    "background", "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "left_hip", "right_hip",
]
NUM_CLASSES = len(LABEL_NAMES)


# ── Case record ──────────────────────────────────────────────────────────────

@dataclass
class Case:
    """One NIfTI pair (CT + label) with metadata.

    For HF-backed datasets, `ct_path` and `label_path` may not exist on
    disk yet — the file is fetched lazily on first call to
    `load_ct()` / `load_label()` via the back-reference to the parent
    dataset.  For local roots the back-reference is None and load_*
    just opens the file directly.
    """
    token:               str
    config:              str         # "fused" | "spine_only" | "pelvic_native"
    match_type:          str
    ct_path:             Path
    label_path:          Path
    split:               str = "unknown"
    lstv_label:          str = ""
    lstv_pelvic:         str = ""
    lstv_vertebral:      str = ""
    lstv_agreement:      Optional[bool] = None
    lstv_confusion_zone: bool = False
    lstv_class:          int = 0
    has_l6:              bool = False
    n_lumbar_labels:     int = 0
    position:            str = "unknown"
    spine_series_uid:    Optional[str] = None
    pelvic_series_uid:   Optional[str] = None
    spine_bone_pct:      Optional[float] = None
    pelvic_bone_pct:     Optional[float] = None
    # Per-export diagnostics (added v4 of export_hf.py).
    alignment_ok:        bool = True
    ct_resampled_to_mask: bool = False
    postwrite_hip_bone_pct: Optional[float] = None
    # Mask-type flags (placed_manifest schema v2.1+, forwarded through
    # export_hf.py).  Default to just `core` so records from older
    # manifests look like they always did.
    annotations:         Dict[str, bool] = field(default_factory=lambda: {"core": True})

    # Manifest-relative paths (e.g. "ct/0017_ct.nii.gz"). Populated at
    # construction time. Used by the lazy-fetch path to ask the parent
    # dataset for an HF-backed download — even when ct_path/label_path
    # have been resolved against a local root that hasn't actually
    # received the bytes yet.
    ct_file_rel:         str = ""
    label_file_rel:      str = ""
    # Back-reference to the parent dataset for HF-lazy-fetch. None for
    # purely local datasets. Set as a weakref-style attribute (not a
    # @dataclass field) so equality / repr / serialization don't try to
    # walk into the dataset.
    _parent: object = field(default=None, repr=False, compare=False)

    def exists(self) -> bool:
        """True iff both files are present on disk RIGHT NOW.

        For HF-backed datasets this returns False until the case has
        been fetched. Use `load_ct()` / `load_label()` to trigger a
        fetch before calling exists() if you want present-on-disk
        semantics.
        """
        return self.ct_path.exists() and self.label_path.exists()

    def aligned(self) -> bool:
        """Whether the saved CT/label pair passed the post-write affine
        check at export time. False for tokens where alignment_ok was
        False in the manifest."""
        return bool(self.alignment_ok)

    def has_annotation(self, kind: str) -> bool:
        """True iff this case has the given annotation kind available
        (e.g. 'spinous', 'tp', 'discs', 'facets').  'core' (the 10-class
        spine + pelvis fusion) is True for every case."""
        return bool(self.annotations.get(kind, False))

    def _ensure_local(self) -> None:
        """Make sure ct_path and label_path exist on disk, downloading
        from HF if this is an HF-backed dataset and the files are
        missing.

        No-op for fully-local datasets (no parent or parent isn't HF).
        """
        if self._parent is None:
            return
        fetcher = getattr(self._parent, "_hf_fetch", None)
        if fetcher is None:
            return
        if not self.ct_path.exists():
            new_ct = fetcher(self.ct_file_rel)
            if new_ct is not None:
                self.ct_path = Path(new_ct)
        if not self.label_path.exists():
            new_lbl = fetcher(self.label_file_rel)
            if new_lbl is not None:
                self.label_path = Path(new_lbl)

    def load_ct(self):
        import nibabel as nib
        import numpy as np
        self._ensure_local()
        img = nib.load(str(self.ct_path))
        return np.asarray(img.dataobj, dtype=np.float32), img.affine

    def load_label(self):
        import nibabel as nib
        import numpy as np
        self._ensure_local()
        img = nib.load(str(self.label_path))
        return np.asarray(img.dataobj, dtype=np.int16), img.affine


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_optional_bool(v):
    """HF Parquet doesn't tolerate mixed null/bool columns, so export_hf.py
    may write None as "".  Reverse that here so lstv_agreement is back to
    Optional[bool]."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):  return True
        if s in ("false", "0", "no"):  return False
    return None


def _coerce_optional_float(v):
    """Normalize numeric fields to Optional[float].

    Current export_hf.py writes None for missing bone_pct (Parquet-native),
    but older exports converted None to "" for all columns.  Handle both,
    plus stringified floats from hand-edited manifests and NaN.
    """
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # reject NaN


def _coerce_optional_str(v):
    """Normalize strings, treating None/"" as missing."""
    if v is None:
        return None
    s = str(v)
    return s if s else None


def _coerce_annotations(v):
    """Normalize the annotations field from a manifest record.

    Accepts:
      None / ""              -> {"core": True}
      {"core": True, ...}    -> as-is with bools coerced
      '{"core": true, ...}'  -> parsed JSON (some Parquet round-trips stringify)
    Any other shape falls back to the default {"core": True}.
    """
    if v is None or v == "":
        return {"core": True}
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (TypeError, ValueError):
            return {"core": True}
    if isinstance(v, dict):
        return {str(k): bool(val) for k, val in v.items()}
    return {"core": True}


def _resolve_file(root: Path, rel: str, canonical_subdir: str) -> Path:
    """Resolve a manifest-declared file path against the dataset root.

    Handles two manifest generations uniformly:

      * New exports (export_hf.py with the subdir-prefix fix) store
        ct_file='ct/XXXX_ct.nii.gz'.  `root / rel` already points to the
        right file; the fallback branch is a no-op.

      * Old exports stored just the basename ct_file='XXXX_ct.nii.gz',
        but the files actually live at root/ct/XXXX_ct.nii.gz.  The
        fallback catches that and returns the canonical location.

    Guardrails:
      * Empty `rel` returns `root` unchanged (caller will see missing file).
      * If `rel` contains any path separator, assume the manifest author
        meant it — don't second-guess with the fallback.
      * If the fallback location doesn't exist either, return the primary
        path so `exists()` downstream produces a meaningful 'missing file'
        error pointing at the manifest's declared location.
    """
    if not rel:
        return root
    primary = root / rel
    if primary.exists():
        return primary
    if "/" in rel or "\\" in rel:
        return primary
    fallback = root / canonical_subdir / rel
    return fallback if fallback.exists() else primary


# ── Main dataset class ───────────────────────────────────────────────────────

class CTSpinoPelvic1K:
    """Directory-backed dataset with rich per-case metadata.

    For HF-backed instances (constructed via `from_hub`), only the
    metadata files are downloaded eagerly.  CT and label NIfTIs are
    fetched lazily on first call to `Case.load_ct()` / `load_label()`
    via `_hf_fetch()`, and cached for future calls under the
    `huggingface_hub` cache.  This keeps from_hub fast (~MB of metadata)
    while still letting any code that loads bytes work transparently.
    """

    # Splits schema version read from splits_5fold.json. Recorded on the
    # instance so callers can introspect which schema actually fed the
    # in-memory splits without re-reading the file.
    splits_schema_version: Optional[int] = None
    splits_scheme:         Optional[str] = None

    # HF lazy-fetch state. None for local-only datasets.
    _hf_repo_id:    Optional[str] = None
    _hf_token:      Optional[str] = None
    _hf_cache_dir:  Optional[str] = None

    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._load()

    # ── HF lazy-fetch ─────────────────────────────────────────────────────
    def _hf_fetch(self, rel_path: str) -> Optional[str]:
        """Ensure the file at `rel_path` (relative to the HF repo root)
        is on disk locally. Returns the local path as a string, or None
        if this dataset isn't HF-backed.

        Race-safe across processes (huggingface_hub uses file locks).
        Network / API errors propagate up to the caller — the assumption
        is that anyone reaching this point WANTS the bytes and would
        rather see the error than silently skip.
        """
        if not self._hf_repo_id or not rel_path:
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed. pip install huggingface_hub"
            ) from e
        return hf_hub_download(
            repo_id   = self._hf_repo_id,
            repo_type = "dataset",
            filename  = rel_path,
            token     = self._hf_token,
            cache_dir = self._hf_cache_dir,
        )

    # ── Internal: splits resolution ─────────────────────────────────────
    def _resolve_splits(self) -> Tuple[Dict[str, str], Dict[str, str], Optional[Dict]]:
        """
        Return (token_to_split, ctfile_to_split, cv_doc).

        `token_to_split` maps patient tokens to "test" / "trainval" (we do
        not fill val-specific labels here — fold membership is looked up
        separately via self.cv and self.fold()).

        `ctfile_to_split` is the fallback for legacy data_splits.json,
        mapping ct_file entries to "train"/"val"/"test".  Both the full
        relative path and the basename are registered so lookup works
        regardless of which form the manifest uses for ct_file.

        `cv_doc` is the unified splits document (schema v3+) when read
        from splits_5fold.json, OR the legacy splits/cv_5fold.json
        content. `None` if no CV folds are available.
        """
        token_to_split:  Dict[str, str] = {}
        ctfile_to_split: Dict[str, str] = {}
        cv_doc:          Optional[Dict] = None

        # ── Source 1 (preferred): splits_5fold.json ────────────────────
        unified_path = self.root / "splits_5fold.json"
        if unified_path.exists():
            try:
                doc = json.loads(unified_path.read_text())
                schema = int(doc.get("schema_version", 0) or 0)
                if schema < 3:
                    import warnings as _w
                    _w.warn(
                        f"{unified_path} has schema_version={schema}; "
                        "expected >=3 (v4 is the current generator). "
                        "Ignoring and falling back to legacy splits files.",
                        stacklevel=3,
                    )
                else:
                    self.splits_schema_version = schema
                    self.splits_scheme = str(
                        doc.get("strata_scheme") or doc.get("strata_scheme_intended") or ""
                    )
                    for tok in doc.get("test_tokens", []) or []:
                        token_to_split[str(tok)] = "test"
                    if "folds" in doc:
                        cv_doc = {"folds": doc["folds"]}
                    return token_to_split, ctfile_to_split, cv_doc
            except (OSError, ValueError, TypeError) as e:
                import warnings as _w
                _w.warn(
                    f"Could not read unified splits at {unified_path}: {e}. "
                    "Falling back to legacy splits files.",
                    stacklevel=3,
                )

        # ── Source 2 (legacy): splits/test.json + splits/cv_5fold.json ──
        test_path = self.root / "splits" / "test.json"
        if test_path.exists():
            try:
                for tok in json.loads(test_path.read_text()):
                    token_to_split[str(tok)] = "test"
            except (OSError, ValueError):
                pass

        cv_path = self.root / "splits" / "cv_5fold.json"
        if cv_path.exists():
            try:
                cv_doc = json.loads(cv_path.read_text())
            except (OSError, ValueError):
                cv_doc = None

        # ── Source 3 (earliest): data_splits.json ──────────────────────
        data_splits_path = self.root / "data_splits.json"
        if data_splits_path.exists() and not token_to_split:
            try:
                ds_splits = json.loads(data_splits_path.read_text())
                for side in ("test", "val", "train"):
                    for ctfile in ds_splits.get(side, []) or []:
                        # Register both the full entry and its basename so
                        # lookup works whether the manifest records
                        # ct_file as 'ct/X.nii.gz' or just 'X.nii.gz'.
                        s = str(ctfile)
                        ctfile_to_split[s] = side
                        bn = Path(s).name
                        if bn and bn != s:
                            ctfile_to_split[bn] = side
            except (OSError, ValueError):
                pass

        return token_to_split, ctfile_to_split, cv_doc

    def _load(self) -> None:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json missing under {self.root}. "
                "Re-run scripts/export_hf.py.")
        manifest = json.loads(manifest_path.read_text())

        if isinstance(manifest, list):
            raw_records = manifest
        elif isinstance(manifest, dict):
            raw_records = manifest.get("records", [])
        else:
            raise ValueError(
                f"manifest.json has unexpected type {type(manifest).__name__}; "
                "expected list or dict.")

        token_to_split, ctfile_to_split, self.cv = self._resolve_splits()

        self.cases: List[Case] = []
        for r in raw_records:
            token    = str(r.get("token", ""))
            cfg      = str(r.get("config", ""))
            ct_file  = r.get("ct_file", "") or ""
            lbl_file = r.get("label_file", "") or ""

            split = r.get("split")
            if not split:
                split = token_to_split.get(token)
            if not split:
                mapped = ctfile_to_split.get(ct_file) or \
                         ctfile_to_split.get(Path(ct_file).name)
                if mapped == "test":
                    split = "test"
                elif mapped in ("train", "val"):
                    split = "trainval"
            if not split:
                split = "trainval"

            self.cases.append(Case(
                token               = token,
                config              = cfg,
                match_type          = r.get("match_type", "") or "",
                ct_path             = _resolve_file(self.root, ct_file,  "ct"),
                label_path          = _resolve_file(self.root, lbl_file, "labels"),
                split               = split,
                lstv_label          = r.get("lstv_label", "") or "",
                lstv_pelvic         = r.get("lstv_pelvic", "") or "",
                lstv_vertebral      = r.get("lstv_vertebral", "") or "",
                lstv_agreement      = _coerce_optional_bool(r.get("lstv_agreement")),
                lstv_confusion_zone = bool(r.get("lstv_confusion_zone", False)),
                lstv_class          = int(r.get("lstv_class", 0) or 0),
                has_l6              = bool(r.get("has_l6", False)),
                n_lumbar_labels     = int(r.get("n_lumbar_labels", 0) or 0),
                position            = r.get("position", "unknown") or "unknown",
                spine_series_uid    = _coerce_optional_str(r.get("spine_series_uid")),
                pelvic_series_uid   = _coerce_optional_str(r.get("pelvic_series_uid")),
                spine_bone_pct      = _coerce_optional_float(r.get("spine_bone_pct")),
                pelvic_bone_pct     = _coerce_optional_float(r.get("pelvic_bone_pct")),
                # Per-export diagnostics from export_hf.py (Apr 2026+).
                # Default alignment_ok=True for legacy manifests that
                # didn't carry the field — matches the pre-diagnostic
                # behavior of always-True in earlier dataset_interface
                # versions.
                alignment_ok        = bool(r.get("alignment_ok", True)),
                ct_resampled_to_mask = bool(r.get("ct_resampled_to_mask", False)),
                postwrite_hip_bone_pct = _coerce_optional_float(r.get("postwrite_hip_bone_pct")),
                annotations         = _coerce_annotations(r.get("annotations")),
                ct_file_rel         = ct_file,
                label_file_rel      = lbl_file,
                _parent             = self,
            ))

        self._by_token_config: Dict[Tuple[str, str], Case] = {
            (c.token, c.config): c for c in self.cases
        }

    # ── Construction from the Hub ────────────────────────────────────────
    @classmethod
    def from_hub(cls, repo_id: str, token: Optional[str] = None,
                 cache_dir: Optional[str] = None) -> "CTSpinoPelvic1K":
        """Construct a dataset backed by a HuggingFace dataset repo.

        Eagerly downloads only the metadata files (manifest, splits,
        README). NIfTI volumes are fetched lazily on first
        `Case.load_ct()` / `load_label()` call. Subsequent loads of the
        same case hit the local huggingface_hub cache.

        Args:
          repo_id:   "user/repo" on huggingface.co
          token:     auth token if the repo is private; reads HF_TOKEN
                     env var if None
          cache_dir: where huggingface_hub puts downloaded files. None
                     uses the default (~/.cache/huggingface).
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed. pip install huggingface_hub"
            ) from e
        local_dir = snapshot_download(
            repo_id   = repo_id,
            repo_type = "dataset",
            token     = token,
            cache_dir = str(Path(os.path.expanduser(cache_dir))) if cache_dir else None,
            allow_patterns = [
                "manifest.json",
                "manifest.csv",
                "splits_5fold.json",
                "splits/**",
                "data_splits.json",
                "splits_summary.json",
                "README.md",
                "dataset_interface.py",
            ],
        )
        inst = cls(local_dir)
        # Stash the HF config so each Case can lazily fetch its own
        # NIfTIs via _hf_fetch when load_ct / load_label is called.
        inst._hf_repo_id   = repo_id
        inst._hf_token     = token
        inst._hf_cache_dir = (
            str(Path(os.path.expanduser(cache_dir))) if cache_dir else None
        )
        return inst

    # ── Filtering ─────────────────────────────────────────────────────────
    def filter(self,
               config:         Optional[str]  = None,
               match_type:     Optional[str]  = None,
               lstv_label:     Optional[str]  = None,
               split:          Optional[str]  = None,
               has_annotation: Optional[str]  = None,
               aligned_only:   bool = False,
               present_only:   bool = False) -> List[Case]:
        """Filter cases by metadata attributes.

        Note: `present_only=True` means "present on disk RIGHT NOW".
        For HF-backed datasets where files haven't been fetched yet,
        this will return an empty list. Use `aligned_only` instead if
        you want quality-filtered cases regardless of fetch state.
        """
        out = list(self.cases)
        if config:
            out = [c for c in out if c.config == config]
        if match_type:
            out = [c for c in out if c.match_type == match_type]
        if lstv_label is not None:
            lc = lstv_label.lower()
            out = [c for c in out if c.lstv_label.lower() == lc]
        if split:
            out = [c for c in out if c.split == split]
        if has_annotation:
            out = [c for c in out if c.has_annotation(has_annotation)]
        if aligned_only:
            out = [c for c in out if c.aligned()]
        if present_only:
            out = [c for c in out if c.exists()]
        return out

    # ── Split accessors ───────────────────────────────────────────────────
    def test_set(self) -> List[Case]:
        """Fixed test holdout (patient-level)."""
        return [c for c in self.cases if c.split == "test"]

    def trainval(self) -> List[Case]:
        """Trainval pool (everything not in the test holdout)."""
        return [c for c in self.cases if c.split == "trainval"]

    def fold(self, i: int) -> Tuple[List[Case], List[Case]]:
        """Return (train_cases, val_cases) for fold i ∈ [0, n_folds).

        Raises RuntimeError if no CV splits are available in the dataset.
        """
        if self.cv is None:
            raise RuntimeError(
                "No 5-fold CV found. Looked for splits_5fold.json (schema "
                "v3+ from generate_5fold_splits.py; v4 is current) and "
                "splits/cv_5fold.json (legacy). Either re-export with "
                "scripts/export_hf.py or run "
                "scripts/generate_5fold_splits.py to produce one."
            )
        folds = self.cv.get("folds", [])
        if not 0 <= i < len(folds):
            raise IndexError(f"fold {i} out of range [0, {len(folds)})")
        train_toks = set(folds[i]["train_tokens"])
        val_toks   = set(folds[i]["val_tokens"])
        train = [c for c in self.cases if c.token in train_toks]
        val   = [c for c in self.cases if c.token in val_toks]
        return train, val

    @property
    def n_folds(self) -> int:
        return len(self.cv["folds"]) if self.cv else 0

    def splits(self) -> Tuple[List[Case], List[Case], List[Case]]:
        """
        Backward-compatible 3-tuple (train, val, test).

        Returns (trainval_pool, [], test_set) — `train` is the full trainval
        pool and `val` is empty.  Callers that need real train/val should use
        `fold(i)` instead.
        """
        return self.trainval(), [], self.test_set()

    # ── Lookup ────────────────────────────────────────────────────────────
    def get(self, token: str, config: str) -> Optional[Case]:
        return self._by_token_config.get((str(token), str(config)))

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    # ── Stats ─────────────────────────────────────────────────────────────
    def stats(self) -> str:
        from collections import Counter
        cfg  = Counter(c.config for c in self.cases)
        sp   = Counter(c.split for c in self.cases)
        mt   = Counter(c.match_type for c in self.cases)
        lstv = Counter(c.lstv_class for c in self.cases)
        n_present = sum(1 for c in self.cases if c.exists())
        n_sp_uid  = sum(1 for c in self.cases if c.spine_series_uid)
        n_pv_uid  = sum(1 for c in self.cases if c.pelvic_series_uid)
        n_sp_pct  = sum(1 for c in self.cases if c.spine_bone_pct  is not None)
        n_pv_pct  = sum(1 for c in self.cases if c.pelvic_bone_pct is not None)
        n_aligned = sum(1 for c in self.cases if c.aligned())
        n_resampled = sum(1 for c in self.cases if c.ct_resampled_to_mask)
        n_hu_low  = sum(1 for c in self.cases
                        if c.postwrite_hip_bone_pct is not None
                        and c.postwrite_hip_bone_pct < 30.0)

        ann_keys = set()
        for c in self.cases:
            ann_keys.update(c.annotations.keys())
        ann_counts = {
            k: sum(1 for c in self.cases if c.annotations.get(k))
            for k in sorted(ann_keys)
        }

        lines = [
            "CTSpinoPelvic1K",
            f"  root:            {self.root}",
            f"  records:         {len(self.cases)}  (present on disk: {n_present})",
            f"  configs:         {dict(cfg)}",
            f"  splits:          {dict(sp)}  (patient-level)",
            f"  match_types:     {dict(mt)}",
            f"  lstv_class dist: {dict(lstv)}",
            f"  cv folds:        {self.n_folds}",
            f"  splits source:   schema_v{self.splits_schema_version}  scheme={self.splits_scheme or '-'}"
            if self.splits_schema_version
            else f"  splits source:   (legacy splits/ or data_splits.json)",
            f"  annotations:     {ann_counts}",
            f"  provenance:      spine_uid={n_sp_uid}  pelvic_uid={n_pv_uid}  "
            f"spine_bone_pct={n_sp_pct}  pelvic_bone_pct={n_pv_pct}",
            f"  diagnostics:     aligned={n_aligned}/{len(self.cases)}  "
            f"resampled={n_resampled}  hu_at_hip<30%={n_hu_low}",
        ]
        if self._hf_repo_id:
            lines.append(
                f"  hf-backed:       {self._hf_repo_id}  "
                f"(NIfTIs fetched lazily; cache_dir={self._hf_cache_dir or 'default'})"
            )
        return "\n".join(lines)


# ── PyTorch Dataset adapter ──────────────────────────────────────────────────

class CTSpinoPelvicDataset(Dataset):
    """
    PyTorch Dataset yielding per-case tensors from NIfTI files.

    Split selection:
        split="trainval"           — whole trainval pool (use for a single run)
        split="test"               — fixed test holdout (final reporting only)
        split=("fold", 0, "train") — fold 0 train side of 5-fold CV
        split=("fold", 0, "val")   — fold 0 val side of 5-fold CV

    HF-backed roots: NIfTIs are fetched lazily on first __getitem__ for
    each case. With num_workers>0 in the DataLoader, multiple workers
    may race to fetch the same case — huggingface_hub uses file locks
    internally to make this safe (the second worker waits and reads the
    cached result).
    """

    def __init__(self, root: str,
                 split="trainval",
                 config: Optional[str] = None,
                 transform=None,
                 cache_dir: Optional[str] = None):
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for CTSpinoPelvicDataset")
        if not Path(os.path.expanduser(str(root))).exists():
            self._ds = CTSpinoPelvic1K.from_hub(repo_id=root, cache_dir=cache_dir)
        else:
            self._ds = CTSpinoPelvic1K(root)
        self.split  = split
        self.config = config
        self.transform = transform

        if isinstance(split, tuple) and len(split) == 3 and split[0] == "fold":
            _, fold_i, side = split
            tr, va = self._ds.fold(int(fold_i))
            cases = tr if side == "train" else va
        elif split == "test":
            cases = self._ds.test_set()
        elif split == "trainval":
            cases = self._ds.trainval()
        elif split == "all":
            cases = list(self._ds.cases)
        else:
            raise ValueError(f"Unknown split spec: {split!r}")

        if config:
            cases = [c for c in cases if c.config == config]

        # For HF-backed datasets, files don't exist yet — Case.load_*
        # will fetch them on demand. Skip the present_only filter so we
        # don't return an empty dataset before the first fetch.
        if self._ds._hf_repo_id:
            self.cases: List[Case] = list(cases)
        else:
            self.cases = [c for c in cases if c.exists()]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        c = self.cases[idx]
        ct_np,  affine = c.load_ct()
        lbl_np, _      = c.load_label()
        ct    = torch.from_numpy(ct_np.astype("float32")).unsqueeze(0)   # (1,Z,Y,X)
        label = torch.from_numpy(lbl_np.astype("int64"))                   # (Z,Y,X)
        item = {
            "ct":     ct,
            "label":  label,
            "affine": torch.from_numpy(affine.astype("float32")),
            "token":  c.token,
            "config": c.config,
            "meta":   {
                "match_type":          c.match_type,
                "lstv_label":          c.lstv_label,
                "lstv_class":          c.lstv_class,
                "lstv_confusion_zone": c.lstv_confusion_zone,
                "has_l6":              c.has_l6,
                "n_lumbar_labels":     c.n_lumbar_labels,
                "position":            c.position,
                "spine_series_uid":    c.spine_series_uid,
                "pelvic_series_uid":   c.pelvic_series_uid,
                "spine_bone_pct":      c.spine_bone_pct,
                "pelvic_bone_pct":     c.pelvic_bone_pct,
                "alignment_ok":        c.alignment_ok,
                "ct_resampled_to_mask": c.ct_resampled_to_mask,
                "postwrite_hip_bone_pct": c.postwrite_hip_bone_pct,
                "annotations":         dict(c.annotations),
            },
        }
        if self.transform is not None:
            item = self.transform(item)
        return item


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke test: load + print.")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    ds = CTSpinoPelvic1K(args.root)
    print(ds.stats())
    print(f"\ntest / trainval: {len(ds.test_set())} / {len(ds.trainval())}")
    if ds.n_folds > 0:
        tr, va = ds.fold(0)
        print(f"fold 0 train/val: {len(tr)} / {len(va)}")
    sample = (ds.trainval() or ds.cases)
    if sample:
        c = sample[0]
        print(f"\nfirst case:")
        print(f"  token={c.token}  config={c.config}  split={c.split}  lstv={c.lstv_label}")
        print(f"  ct:    {c.ct_path}  (exists={c.ct_path.exists()})")
        print(f"  label: {c.label_path}  (exists={c.label_path.exists()})")
        print(f"  provenance: spine_uid={c.spine_series_uid}  "
              f"pelvic_uid={c.pelvic_series_uid}  "
              f"spine_bone_pct={c.spine_bone_pct}  "
              f"pelvic_bone_pct={c.pelvic_bone_pct}")
        print(f"  diagnostics: aligned={c.aligned()}  "
              f"resampled={c.ct_resampled_to_mask}  "
              f"hu_at_hip%={c.postwrite_hip_bone_pct}")
        print(f"  annotations: {c.annotations}")
