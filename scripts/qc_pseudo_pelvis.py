"""
qc_pseudo_pelvis.py — triage the MODEL-pseudolabelled pelves for human review.

The spine_only cases in v2 carry a pelvis the model guessed (no radiologist traced
it), so there is no ground truth to score against. Instead we score each pseudo
pelvis on ANATOMICAL PLAUSIBILITY — cheap, GT-free checks tuned to the failure modes
the model actually produces — and emit a triage-sorted CSV so review targets the
worst first instead of eyeballing all ~440.

Design notes (v2 of this script):
  * Fragmentation is scored by LARGEST-COMPONENT FRACTION (LCF = biggest-blob voxels
    / total class voxels), a CONTINUOUS 0..1 severity, NOT a binary "components>1"
    flag (which fired on a single stray voxel and saturated the score).
  * Volume outliers are DATA-DRIVEN: a class volume is flagged only if it falls outside
    the cohort's own Tukey fences (Q1-1.5*IQR, Q3+1.5*IQR), not fixed clinical ranges.
  * An explicit L/R check: left_hip must sit on the patient-left of the sacrum and
    right_hip on the patient-right (world RAS X); a wrong-sided or near-absent hip is
    flagged as a merge/swap (the model's most common pelvis error).
  * triage_score is a weighted sum of continuous severities, so it SPREADS and ranks.

Two passes: per-case raw metrics are computed in parallel (cases are independent),
then the cohort IQR fences are derived and the flags + score assigned.

Usage:
  python scripts/qc_pseudo_pelvis.py \
      --v2_dir  data/hf_export_v2 \
      --out_csv data/hf_export_v2/qc/pseudo_pelvis_triage.csv
  # optional: --bone_hu 200 --tokens 0267 --workers 24
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("qc_pseudo_pelvis")

SACRUM, LHIP, RHIP = 7, 8, 9
LUMBAR = (1, 2, 3, 4, 5, 6)
PELVIS = (SACRUM, LHIP, RHIP)
PELVIS_NAME = {SACRUM: "sacrum", LHIP: "left_hip", RHIP: "right_hip"}


def _voxvol_cm3(affine: np.ndarray) -> float:
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(np.prod(spacing)) / 1000.0


def _largest_component_fraction(mask: np.ndarray) -> float:
    """Fraction of the class in its single biggest 3D blob. 1.0 = one clean component;
    lower = more fragmented / scattered false-positive specks."""
    if not mask.any():
        return 0.0
    struct = ndimage.generate_binary_structure(3, 3)        # 26-connectivity
    lab, n = ndimage.label(mask, structure=struct)
    if n <= 1:
        return 1.0
    sizes = np.bincount(lab.ravel())[1:]                     # drop background bin
    return float(sizes.max() / sizes.sum())


def _world_centroid(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    if not mask.any():
        return None
    com = ndimage.center_of_mass(mask)
    return apply_affine(affine, np.asarray(com))             # (x, y, z) RAS


# ---------------------------------------------------------------------------
# Per-case RAW metrics (parallel worker). No flags / no score yet — those need
# the cohort distribution and are assigned in the second pass.
# ---------------------------------------------------------------------------
def raw_metrics(ct: np.ndarray, lbl: np.ndarray, affine: np.ndarray,
                *, bone_hu: int = 200) -> Dict[str, object]:
    vv = _voxvol_cm3(affine)
    out: Dict[str, object] = {}
    cents: Dict[int, Optional[np.ndarray]] = {}
    for c in PELVIS:
        m = lbl == c
        name = PELVIS_NAME[c]
        out[f"{name}_cm3"] = round(float(m.sum()) * vv, 1)
        out[f"{name}_lcf"] = round(_largest_component_fraction(m), 3)
        out[f"{name}_present"] = int(bool(m.any()))
        cents[c] = _world_centroid(m, affine)

    # L/R hip symmetry.
    lv, rv = out["left_hip_cm3"], out["right_hip_cm3"]
    out["lr_asym"] = round(abs(lv - rv) / (0.5 * (lv + rv)), 3) if (lv and rv) else None

    # Lateralization: left hip should be patient-LEFT of sacrum (RAS -X), right hip
    # patient-RIGHT (+X). Record world-X of each so the 2nd pass can flag wrong sides.
    out["sacrum_cx"] = None if cents[SACRUM] is None else round(float(cents[SACRUM][0]), 1)
    out["left_hip_cx"] = None if cents[LHIP] is None else round(float(cents[LHIP][0]), 1)
    out["right_hip_cx"] = None if cents[RHIP] is None else round(float(cents[RHIP][0]), 1)

    # Soft bone-fit (HU is unreliable on contrast/soft-tissue scans — one input only).
    pelvis_mask = np.isin(lbl, PELVIS)
    out["bone_fit"] = round(float((ct[pelvis_mask] > bone_hu).mean()), 3) \
        if pelvis_mask.any() else None

    # Sacrum should be inferior to the lumbar column (world Z; +Z superior).
    sz = _world_centroid(lbl == SACRUM, affine)
    lz = _world_centroid(np.isin(lbl, LUMBAR), affine)
    out["sacrum_z"] = None if sz is None else round(float(sz[2]), 1)
    out["lumbar_z"] = None if lz is None else round(float(lz[2]), 1)
    return out


# ---------------------------------------------------------------------------
# Second pass: cohort-relative flags + weighted, continuous triage score.
# ---------------------------------------------------------------------------
def _tukey_fences(values: List[float]) -> Tuple[float, float]:
    if len(values) < 4:
        return (-np.inf, np.inf)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def assign_flags(row: Dict[str, object],
                 fences: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
    flags: List[str] = []
    score = 0.0
    for c in PELVIS:
        name = PELVIS_NAME[c]
        if not row[f"{name}_present"]:
            flags.append(f"missing_{name}"); score += 3.0
            continue
        lcf = row[f"{name}_lcf"]
        if lcf < 0.90:                              # continuous fragmentation severity
            flags.append(f"frag_{name}"); score += (1.0 - lcf) * 4.0
        lo, hi = fences[name]
        v = row[f"{name}_cm3"]
        if v < lo or v > hi:                        # data-driven volume outlier
            flags.append(f"vol_{name}"); score += 1.0

    # L/R symmetry.
    asym = row.get("lr_asym")
    if asym is not None and asym > 0.30:
        flags.append("lr_asym"); score += min(asym, 1.0) * 1.5

    # Lateralization: flag a hip on the wrong side of the sacrum (merge/swap).
    sx, lx, rx = row.get("sacrum_cx"), row.get("left_hip_cx"), row.get("right_hip_cx")
    if sx is not None and lx is not None and rx is not None:
        # expect lx < sx < rx (left hip patient-left = smaller world-X)
        if not (lx < sx < rx):
            flags.append("lr_wrong_side"); score += 3.0

    # Sacrum must be inferior to the lumbar column.
    sz, lz = row.get("sacrum_z"), row.get("lumbar_z")
    if sz is not None and lz is not None and sz > lz:
        flags.append("sacrum_above_lumbar"); score += 2.5

    # Bone-fit (soft).
    bf = row.get("bone_fit")
    if bf is not None and bf < 0.35:
        flags.append("low_bone_fit"); score += (0.35 - bf) * 4.0

    row["flags"] = ";".join(flags)
    row["n_flags"] = len(flags)
    row["triage_score"] = round(score, 2)
    return row


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine


def _score_record(payload):
    """Worker: load one case + compute raw metrics. Returns ("ok", row) or
    ("skip", token, reason). Top-level / picklable for ProcessPoolExecutor."""
    v2_dir_str, bone_hu, r = payload
    v2_dir = Path(v2_dir_str)
    tok = str(r.get("token", "?"))
    ct_p, lbl_p = v2_dir / r["ct_file"], v2_dir / r["label_file"]
    if not ct_p.exists() or not lbl_p.exists():
        return ("skip", tok, "missing CT/label")
    ct, aff = _load(ct_p)
    lbl, _ = _load(lbl_p)
    if ct.shape[:3] != lbl.shape[:3]:
        return ("skip", tok, "shape mismatch")
    m = raw_metrics(ct, np.rint(lbl).astype(np.int16), aff, bone_hu=bone_hu)
    return ("ok", {"token": tok, "config": r.get("config", ""),
                   "review_priority": r.get("review_priority", ""),
                   "label_file": r["label_file"], **m})


def run(v2_dir: Path, out_csv: Path, *, bone_hu: int = 200,
        tokens: Optional[set] = None, workers: int = 8) -> List[dict]:
    """Score every model-pseudolabelled-pelvis case in the v2 tree (parallel)."""
    from concurrent.futures import ProcessPoolExecutor

    manifest = json.loads((v2_dir / "manifest.json").read_text())
    records = manifest["records"] if isinstance(manifest, dict) else manifest
    scoped = [r for r in records
              if (r.get("prov_pelvis") == "pseudo" or r.get("config") == "spine_only")
              and (tokens is None or str(r.get("token")) in tokens)]
    workers = max(1, min(workers, len(scoped) or 1))
    log.info("triaging %d model-pseudolabelled-pelvis case(s) with %d worker(s)",
             len(scoped), workers)

    # Pass 1 (parallel): raw metrics.
    rows: List[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_score_record, [(str(v2_dir), bone_hu, r) for r in scoped]):
            done += 1
            if res[0] == "skip":
                log.warning("token=%s skipped: %s", res[1], res[2])
            else:
                rows.append(res[1])
            if done % 50 == 0 or done == len(scoped):
                log.info("  [%d/%d] metrics", done, len(scoped))

    # Pass 2: cohort Tukey fences per class, then flags + score.
    fences = {PELVIS_NAME[c]: _tukey_fences(
                  [r[f"{PELVIS_NAME[c]}_cm3"] for r in rows if r[f"{PELVIS_NAME[c]}_present"]])
              for c in PELVIS}
    log.info("volume fences (cm3): %s",
             {k: (round(lo, 1), round(hi, 1)) for k, (lo, hi) in fences.items()})
    for r in rows:
        assign_flags(r, fences)

    rows.sort(key=lambda d: d["triage_score"], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        cols = list(rows[0].keys())
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    n_flag = sum(1 for r in rows if r["n_flags"])
    log.info("wrote %s — %d/%d flagged; score range %.2f..%.2f", out_csv, n_flag,
             len(rows), rows[-1]["triage_score"] if rows else 0.0,
             rows[0]["triage_score"] if rows else 0.0)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2_dir", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--bone_hu", type=int, default=200)
    ap.add_argument("--tokens", default="", help="comma-separated subset to score")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel workers (default 0 = os.cpu_count())")
    args = ap.parse_args()
    import os
    import re
    want = {t for t in re.split(r"[,:;\s]+", args.tokens.strip()) if t} or None
    workers = args.workers or (os.cpu_count() or 8)
    run(args.v2_dir, args.out_csv, bone_hu=args.bone_hu, tokens=want, workers=workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
