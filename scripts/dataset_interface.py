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
    manifest.json                 per-record metadata
    splits/
      test.json                   fixed test holdout (flat token list)
      cv_5fold.json               5-fold CV on trainval pool (schema v3)

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

    def exists(self) -> bool:
        return self.ct_path.exists() and self.label_path.exists()

    def aligned(self) -> bool:
        """CT and label share the same affine by construction in export_hf.py."""
        return True

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


# ── Main dataset class ───────────────────────────────────────────────────────

class CTSpinoPelvic1K:
    """Directory-backed dataset with rich per-case metadata."""

    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._load()

    def _load(self) -> None:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json missing under {self.root}. "
                "Re-run scripts/export_hf.py.")
        manifest = json.loads(manifest_path.read_text())
        raw_records = manifest.get("records", [])

        # Per-record split assignment: "test" vs "trainval"
        # Source: splits/test.json (flat token list).  Anyone not in test is
        # treated as trainval (the CV folds further subdivide).
        token_to_split: Dict[str, str] = {}
        test_path = self.root / "splits" / "test.json"
        if test_path.exists():
            for tok in json.loads(test_path.read_text()):
                token_to_split[str(tok)] = "test"

        # Full 5-fold CV structure (trainval pool).  Optional — not every
        # downstream consumer needs folds.
        self.cv: Optional[Dict] = None
        cv_path = self.root / "splits" / "cv_5fold.json"
        if cv_path.exists():
            self.cv = json.loads(cv_path.read_text())

        self.cases: List[Case] = []
        for r in raw_records:
            token = str(r.get("token", ""))
            cfg   = str(r.get("config", ""))
            split = r.get("split") or token_to_split.get(token) or \
                    ("test" if token in token_to_split else "trainval")
            self.cases.append(Case(
                token               = token,
                config              = cfg,
                match_type          = r.get("match_type", ""),
                ct_path             = self.root / r.get("ct_file", ""),
                label_path          = self.root / r.get("label_file", ""),
                split               = split,
                lstv_label          = r.get("lstv_label", ""),
                lstv_pelvic         = r.get("lstv_pelvic", ""),
                lstv_vertebral      = r.get("lstv_vertebral", ""),
                lstv_agreement      = r.get("lstv_agreement"),
                lstv_confusion_zone = bool(r.get("lstv_confusion_zone", False)),
                lstv_class          = int(r.get("lstv_class", 0)),
                has_l6              = bool(r.get("has_l6", False)),
                position            = r.get("position", "unknown"),
                spine_series_uid    = r.get("spine_series_uid"),
                pelvic_series_uid   = r.get("pelvic_series_uid"),
                spine_bone_pct      = r.get("spine_bone_pct"),
                pelvic_bone_pct     = r.get("pelvic_bone_pct"),
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
            allow_patterns = ["manifest.json", "splits/**", "README.md",
                              "splits_summary.json"],
        )
        return cls(local_dir)

    # ── Filtering ─────────────────────────────────────────────────────────
    def filter(self,
               config:        Optional[str]  = None,
               match_type:    Optional[str]  = None,
               lstv_label:    Optional[str]  = None,
               split:         Optional[str]  = None,
               aligned_only:  bool = False,
               present_only:  bool = False) -> List[Case]:
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

        Raises RuntimeError if cv_5fold.json wasn't shipped with the dataset.
        Each case appears via its patient token across ALL config records —
        a patient with both spine_only + pelvic_native records will get both
        records yielded together.
        """
        if self.cv is None:
            raise RuntimeError(
                "splits/cv_5fold.json not found. The dataset shipped without "
                "5-fold CV; either re-export with scripts/export_hf.py or run "
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
        `fold(i)` instead; this method exists only so benchmarks that do
        `_, _, test = ds.splits()` keep working.
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
        lines = [
            "CTSpinoPelvic1K",
            f"  root:            {self.root}",
            f"  records:         {len(self.cases)}  (present on disk: {n_present})",
            f"  configs:         {dict(cfg)}",
            f"  splits:          {dict(sp)}  (patient-level)",
            f"  match_types:     {dict(mt)}",
            f"  lstv_class dist: {dict(lstv)}",
            f"  cv folds:        {self.n_folds}",
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

        # Resolve the case list
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

        # Apply config + existence filter
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
