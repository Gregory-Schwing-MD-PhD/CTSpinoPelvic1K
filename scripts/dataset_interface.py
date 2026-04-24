"""
dataset_interface.py — Runtime interface for the CTSpinoPelvic1K HF dataset.

Two classes:
  CTSpinoPelvic1K        dict-style dataset wrapping an HF export directory.
                         Used by benchmark_totalseg.py / viz_ts_case.py /
                         render_lstv_examples.py.  No torch dependency.
  CTSpinoPelvicDataset   PyTorch Dataset adapter on top of CTSpinoPelvic1K.

Expected layout (produced by scripts/export_hf.py):
  <root>/
    ct/<token:04d>_<config>_ct.nii.gz
    labels/<token:04d>_<config>_label.nii.gz
    manifest.json                 per-record metadata (flat list OR
                                  {"records": [...]} wrapper — both accepted)
    splits_5fold.json             PREFERRED: unified splits file (schema v3+)
                                  from scripts/generate_5fold_splits.py.
                                  Carries test_tokens + folds in one file.
    splits/                       legacy layout (still read as fallback):
      test.json                   flat list of unique test patient tokens
      cv_5fold.json               5-fold CV on trainval pool
    data_splits.json              earliest format: {"train": [...], "val":
                                  [...], "test": [...]} of ct_file entries
                                  (last-resort fallback)
    splits_summary.json           aggregate split stats (optional)

Manifest ct_file / label_file paths can be either a relative path
("ct/0017_supine_ct.nii.gz") or a bare basename ("0017_supine_ct.nii.gz").
The path resolver tries the value verbatim first and falls back to
`root/ct/{basename}` (or `root/labels/{basename}`) if the primary misses.
This keeps the class tolerant of manifests that predate the path-prefix
fix in export_hf.py.

Splits resolution order (first hit wins):
  1. splits_5fold.json                 (unified, schema v3 from
                                        generate_5fold_splits.py)
  2. splits/test.json + splits/cv_5fold.json (legacy pair)
  3. data_splits.json                  (earliest export_hf.py format)

Quickstart (benchmarking / viz — no splits):
  >>> from dataset_interface import CTSpinoPelvic1K
  >>> ds = CTSpinoPelvic1K("data/hf_export")
  >>> print(ds.stats())
  >>> fused = ds.filter(config="fused", present_only=True)

Quickstart (HF Hub):
  >>> ds = CTSpinoPelvic1K.from_hub(repo_id="anonymous-mlhc/CTSpinoPelvic1K")

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
    """One NIfTI pair (CT + label) with metadata."""
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
    position:            str = "unknown"
    spine_series_uid:    Optional[str] = None
    pelvic_series_uid:   Optional[str] = None
    spine_bone_pct:      Optional[float] = None
    pelvic_bone_pct:     Optional[float] = None
    # Mask-type flags (placed_manifest schema v2.1+, forwarded through
    # export_hf.py).  Default to just `core` so records from older
    # manifests look like they always did.
    annotations:         Dict[str, bool] = field(default_factory=lambda: {"core": True})

    def exists(self) -> bool:
        return self.ct_path.exists() and self.label_path.exists()

    def aligned(self) -> bool:
        """CT and label share the same affine by construction in export_hf.py."""
        return True

    def has_annotation(self, kind: str) -> bool:
        """True iff this case has the given annotation kind available
        (e.g. 'spinous', 'tp', 'discs', 'facets').  'core' (the 10-class
        spine + pelvis fusion) is True for every case."""
        return bool(self.annotations.get(kind, False))

    def load_ct(self):
        import nibabel as nib
        import numpy as np
        img = nib.load(str(self.ct_path))
        return np.asarray(img.dataobj, dtype=np.float32), img.affine

    def load_label(self):
        import nibabel as nib
        import numpy as np
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
    """Directory-backed dataset with rich per-case metadata."""

    # Splits schema version read from splits_5fold.json. Recorded on the
    # instance so callers can introspect which schema actually fed the
    # in-memory splits without re-reading the file.
    splits_schema_version: Optional[int] = None
    splits_scheme:         Optional[str] = None

    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._load()

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

        `cv_doc` is the unified splits document (schema v3) when read from
        splits_5fold.json, OR the legacy splits/cv_5fold.json content.
        `None` if no CV folds are available.
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
                        "expected >=3. Ignoring and falling back to legacy "
                        "splits files.",
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
                position            = r.get("position", "unknown") or "unknown",
                spine_series_uid    = _coerce_optional_str(r.get("spine_series_uid")),
                pelvic_series_uid   = _coerce_optional_str(r.get("pelvic_series_uid")),
                spine_bone_pct      = _coerce_optional_float(r.get("spine_bone_pct")),
                pelvic_bone_pct     = _coerce_optional_float(r.get("pelvic_bone_pct")),
                annotations         = _coerce_annotations(r.get("annotations")),
            ))

        self._by_token_config: Dict[Tuple[str, str], Case] = {
            (c.token, c.config): c for c in self.cases
        }

    # ── Construction from the Hub ────────────────────────────────────────
    @classmethod
    def from_hub(cls, repo_id: str, token: Optional[str] = None,
                 cache_dir: Optional[str] = None) -> "CTSpinoPelvic1K":
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
                "splits_5fold.json",
                "splits/**",
                "data_splits.json",
                "splits_summary.json",
                "README.md",
            ],
        )
        return cls(local_dir)

    # ── Filtering ─────────────────────────────────────────────────────────
    def filter(self,
               config:         Optional[str]  = None,
               match_type:     Optional[str]  = None,
               lstv_label:     Optional[str]  = None,
               split:          Optional[str]  = None,
               has_annotation: Optional[str]  = None,
               aligned_only:   bool = False,
               present_only:   bool = False) -> List[Case]:
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
                "No 5-fold CV found. Looked for splits_5fold.json (schema v3 "
                "from generate_5fold_splits.py) and splits/cv_5fold.json "
                "(legacy). Either re-export with scripts/export_hf.py or run "
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
        ]
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
        self.cases: List[Case] = [c for c in cases if c.exists()]

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
                "position":            c.position,
                "spine_series_uid":    c.spine_series_uid,
                "pelvic_series_uid":   c.pelvic_series_uid,
                "spine_bone_pct":      c.spine_bone_pct,
                "pelvic_bone_pct":     c.pelvic_bone_pct,
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
        print(f"  annotations: {c.annotations}")
