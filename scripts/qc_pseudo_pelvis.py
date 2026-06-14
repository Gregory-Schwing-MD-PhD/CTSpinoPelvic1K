"""
qc_pseudo_pelvis.py — triage the MODEL-pseudolabelled pelves for human review.

The spine_only cases in v2 carry a pelvis that the model guessed (no radiologist
ever traced it), so there is no ground truth to score against. Instead we score
each pseudo pelvis on ANATOMICAL PLAUSIBILITY — cheap, GT-free sanity checks that
catch the failure modes a model actually produces — and emit a triage-sorted CSV
so you review the worst first instead of eyeballing all ~440.

Checks per case (sacrum=7, left_hip=8, right_hip=9; lumbar spine=1..6 is manual GT):
  - present            each of sacrum / L-hip / R-hip exists at all
  - vol_cm3            per-class volume (implausibly small/large -> suspect)
  - n_components       each bone should be ONE blob; >1 == fragmentation
  - lr_asym            |L-hip - R-hip| / mean — the two hips should be ~symmetric
  - bone_fit           fraction of the pseudo pelvis on bone-HU (CT>200). A SOFT
                       signal only (contrast/soft-tissue scans shift HU), but a very
                       low value still flags a mask floating off bone.
  - sacrum_below_lumbar  the sacrum centroid must sit INFERIOR to the lumbar column
                       (world Z); a sacrum that lands up among the vertebrae is wrong
  - triage_score       weighted sum of the flags; higher = review sooner

Usage
-----
  python scripts/qc_pseudo_pelvis.py \
      --v2_dir  data/hf_export_v2 \
      --out_csv data/hf_export_v2/qc/pseudo_pelvis_triage.csv
  # optional: --bone_hu 200  --tokens 12,34  --top_overlays 25 --nifti_dir ...
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

# Plausible per-class volume window (cm^3), generous — only the gross outliers
# should trip it. Adult sacrum ~40-130, hemipelvis (ilium portion segmented) ~40-160.
VOL_RANGE_CM3 = {SACRUM: (20.0, 180.0), LHIP: (20.0, 220.0), RHIP: (20.0, 220.0)}


def _voxvol_cm3(affine: np.ndarray) -> float:
    """Physical volume of one voxel in cm^3 from the affine's column norms."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(np.prod(spacing)) / 1000.0


def _largest_n_components(mask: np.ndarray) -> int:
    """Number of 3D connected components (26-conn) in a binary mask."""
    if not mask.any():
        return 0
    struct = ndimage.generate_binary_structure(3, 3)
    _, n = ndimage.label(mask, structure=struct)
    return int(n)


def _world_z(mask: np.ndarray, affine: np.ndarray) -> Optional[float]:
    """World (RAS) Z of a mask centroid; None if empty. +Z = superior."""
    if not mask.any():
        return None
    com = ndimage.center_of_mass(mask)
    return float(apply_affine(affine, np.asarray(com))[2])


def qc_one(ct: np.ndarray, lbl: np.ndarray, affine: np.ndarray,
           *, bone_hu: int = 200) -> Dict[str, object]:
    """Compute the plausibility metrics + triage score for one pseudo pelvis."""
    vv = _voxvol_cm3(affine)
    out: Dict[str, object] = {}
    flags: List[str] = []
    score = 0.0

    # Per-class presence, volume, fragmentation.
    for c in PELVIS:
        m = lbl == c
        vol = float(m.sum()) * vv
        ncomp = _largest_n_components(m)
        name = PELVIS_NAME[c]
        out[f"{name}_cm3"] = round(vol, 1)
        out[f"{name}_ncomp"] = ncomp
        if not m.any():
            flags.append(f"missing_{name}"); score += 3.0
            continue
        lo, hi = VOL_RANGE_CM3[c]
        if vol < lo or vol > hi:
            flags.append(f"vol_{name}"); score += 1.0
        if ncomp > 1:
            flags.append(f"frag_{name}"); score += min(ncomp - 1, 3) * 0.7

    # Left/right hip symmetry.
    lv, rv = out["left_hip_cm3"], out["right_hip_cm3"]
    if lv and rv:
        asym = abs(lv - rv) / (0.5 * (lv + rv))
        out["lr_asym"] = round(asym, 3)
        if asym > 0.30:
            flags.append("lr_asym"); score += min(asym, 1.0) * 1.5
    else:
        out["lr_asym"] = None

    # Bone fit — SOFT signal (HU is unreliable on contrast/soft-tissue scans).
    pelvis_mask = np.isin(lbl, PELVIS)
    if pelvis_mask.any():
        bf = float((ct[pelvis_mask] > bone_hu).mean())
        out["bone_fit"] = round(bf, 3)
        if bf < 0.35:
            flags.append("low_bone_fit"); score += (0.35 - bf) * 4.0
    else:
        out["bone_fit"] = None

    # Sacrum must be inferior to the lumbar column (world Z).
    sac_z = _world_z(lbl == SACRUM, affine)
    lum_z = _world_z(np.isin(lbl, LUMBAR), affine)
    out["sacrum_z"] = None if sac_z is None else round(sac_z, 1)
    out["lumbar_z"] = None if lum_z is None else round(lum_z, 1)
    if sac_z is not None and lum_z is not None and sac_z > lum_z:
        flags.append("sacrum_above_lumbar"); score += 2.5

    out["flags"] = ";".join(flags) if flags else ""
    out["n_flags"] = len(flags)
    out["triage_score"] = round(score, 2)
    return out


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine


def run(v2_dir: Path, out_csv: Path, *, bone_hu: int = 200,
        tokens: Optional[set] = None) -> List[dict]:
    """Score every model-pseudolabelled-pelvis case in the v2 tree."""
    manifest = json.loads((v2_dir / "manifest.json").read_text())
    records = manifest["records"] if isinstance(manifest, dict) else manifest

    # Pseudo PELVIS = prov_pelvis == "pseudo" (model filled the pelvis). Falls back
    # to config==spine_only for manifests written before the prov flip.
    scoped = [r for r in records
              if (r.get("prov_pelvis") == "pseudo"
                  or r.get("config") == "spine_only")
              and (tokens is None or str(r.get("token")) in tokens)]
    log.info("triaging %d model-pseudolabelled-pelvis case(s)", len(scoped))

    rows: List[dict] = []
    for i, r in enumerate(scoped, 1):
        tok = str(r.get("token", "?"))
        ct_p, lbl_p = v2_dir / r["ct_file"], v2_dir / r["label_file"]
        if not ct_p.exists() or not lbl_p.exists():
            log.warning("[%d/%d] token=%s missing CT/label — skip", i, len(scoped), tok)
            continue
        ct, aff = _load(ct_p)
        lbl, _ = _load(lbl_p)
        if ct.shape[:3] != lbl.shape[:3]:
            log.warning("[%d/%d] token=%s shape mismatch — skip", i, len(scoped), tok)
            continue
        m = qc_one(ct, np.rint(lbl).astype(np.int16), aff, bone_hu=bone_hu)
        m = {"token": tok, "config": r.get("config", ""),
             "review_priority": r.get("review_priority", ""),
             "label_file": r["label_file"], **m}
        rows.append(m)
        if i % 25 == 0 or i == len(scoped):
            log.info("  [%d/%d] scored", i, len(scoped))

    # Worst first.
    rows.sort(key=lambda d: d["triage_score"], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    n_flag = sum(1 for r in rows if r["n_flags"])
    log.info("wrote %s — %d/%d cases flagged (triage_score>0); top score %.2f",
             out_csv, n_flag, len(rows), rows[0]["triage_score"] if rows else 0.0)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2_dir", required=True, type=Path,
                    help="the v2 export tree (ct/, labels/, manifest.json)")
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--bone_hu", type=int, default=200,
                    help="bone-HU threshold for the SOFT bone_fit signal (default 200)")
    ap.add_argument("--tokens", default="", help="comma-separated subset to score")
    args = ap.parse_args()
    import re
    want = {t for t in re.split(r"[,:;\s]+", args.tokens.strip()) if t} or None
    run(args.v2_dir, args.out_csv, bone_hu=args.bone_hu, tokens=want)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
