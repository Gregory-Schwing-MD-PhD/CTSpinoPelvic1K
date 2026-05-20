"""
scripts/review/diff.py — label diff + inter-rater agreement (IRR).

Pure numpy (arrays in, dicts out) so it is unit-testable without any
NIfTI/ITK-SNAP/cloud dependency. Two jobs:

  * label_diff(pseudo, corrected)  — what a reviewer changed: per-class
    added/removed voxels, per-class Dice, changed bbox, which regions were
    touched. Goes into the review record.
  * irr(label_a, label_b)          — agreement between the two reviewers of
    a case: per-class Dice with a configurable threshold τ. Drives the
    auto-finalize vs needs-adjudication decision and is the paper's
    inter-rater statistic.

Dice convention: 2|A∩B| / (|A|+|B|); both-empty -> 1.0 (agree on absence);
one-empty -> 0.0. Classes 0 (background) and IGNORE_LABEL are excluded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402  (sibling module; see __init__ docstring)

DEFAULT_TAU = 0.9
DEFAULT_MODE = "per_class_min"   # per_class_min | overall


def _dice(bool_a, bool_b) -> float:
    import numpy as np
    a = np.asarray(bool_a, dtype=bool)
    b = np.asarray(bool_b, dtype=bool)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return 1.0
    inter = int((a & b).sum())
    return (2.0 * inter) / (na + nb)


def _present_classes(*arrays) -> List[int]:
    import numpy as np
    seen: set = set()
    for arr in arrays:
        seen |= {int(v) for v in np.unique(np.asarray(arr))}
    return sorted(seen - {0, schema.IGNORE_LABEL})


def _bbox_pir(changed_mask) -> Optional[List[List[int]]]:
    import numpy as np
    m = np.asarray(changed_mask, dtype=bool)
    if not m.any():
        return None
    out = []
    for ax in range(m.ndim):
        axes = tuple(i for i in range(m.ndim) if i != ax)
        idx = np.where(m.any(axis=axes))[0]
        out.append([int(idx[0]), int(idx[-1])])
    return out


def per_class_dice(a, b, classes: Optional[List[int]] = None
                   ) -> Dict[str, float]:
    """Per-class Dice over the classes present in either label (bg/IGNORE
    excluded). A class present in only one label scores 0.0; both-absent
    classes are simply not in the union and are omitted."""
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    if classes is None:
        classes = _present_classes(a, b)
    return {str(c): _dice(a == c, b == c) for c in classes}


def label_diff(pseudo, corrected,
               classes: Optional[List[int]] = None) -> dict:
    """What changed from `pseudo` to `corrected`."""
    import numpy as np
    pseudo = np.asarray(pseudo)
    corrected = np.asarray(corrected)
    if pseudo.shape != corrected.shape:
        raise ValueError(f"shape mismatch: pseudo {pseudo.shape} vs "
                         f"corrected {corrected.shape}")
    if classes is None:
        classes = _present_classes(pseudo, corrected)

    per_class: Dict[str, dict] = {}
    for c in classes:
        b = pseudo == c
        a = corrected == c
        per_class[str(c)] = {
            "name": schema.CLASS_NAMES.get(c, str(c)),
            "added": int((a & ~b).sum()),
            "removed": int((b & ~a).sum()),
            "dice": _dice(b, a),
        }

    changed = pseudo != corrected
    pcd = [v["dice"] for v in per_class.values()]
    regions = sorted({
        reg for reg, cls in schema.REGION_CLASSES.items()
        for c in classes
        if c in cls and (per_class[str(c)]["added"]
                         + per_class[str(c)]["removed"]) > 0
    })
    return {
        "n_voxels_changed": int(changed.sum()),
        "macro_dice": (sum(pcd) / len(pcd)) if pcd else 1.0,
        "foreground_dice": _dice(
            (pseudo > 0) & (pseudo != schema.IGNORE_LABEL),
            (corrected > 0) & (corrected != schema.IGNORE_LABEL),
        ),
        "per_class": per_class,
        "changed_bbox_pir": _bbox_pir(changed),
        "regions_touched": regions,
    }


def irr(label_a, label_b, tau: float = DEFAULT_TAU,
        mode: str = DEFAULT_MODE,
        classes: Optional[List[int]] = None) -> dict:
    """Inter-rater agreement between two reviewers' corrected labels.

    mode="per_class_min" (default): agree iff the WORST per-class Dice ≥ τ
      — i.e. every annotated class must agree, so a disagreement on any
      single structure (e.g. L5 vs sacrum boundary) blocks auto-finalize.
    mode="overall": agree iff binary-foreground Dice ≥ τ (coarser).
    """
    pcd = per_class_dice(label_a, label_b, classes)
    if not pcd:                       # neither reviewer has foreground
        metric = 1.0
    elif mode == "overall":
        import numpy as np
        a = np.asarray(label_a)
        b = np.asarray(label_b)
        metric = _dice((a > 0) & (a != schema.IGNORE_LABEL),
                       (b > 0) & (b != schema.IGNORE_LABEL))
    else:
        metric = min(pcd.values())
    return {
        "mode": mode,
        "tau": tau,
        "per_class_dice": pcd,
        "metric": metric,
        "min_class_dice": (min(pcd.values()) if pcd else 1.0),
        "agree": bool(metric >= tau),
    }


def agree(label_a, label_b, tau: float = DEFAULT_TAU,
          mode: str = DEFAULT_MODE) -> bool:
    return irr(label_a, label_b, tau=tau, mode=mode)["agree"]
