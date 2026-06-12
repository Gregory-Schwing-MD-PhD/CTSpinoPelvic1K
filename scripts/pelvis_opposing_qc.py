"""
pelvis_opposing_qc.py — validate a PSEUDOLABELLED pelvis against the SAME
patient's REAL pelvis ground truth from the OPPOSING acquisition.

Why this works without registration
------------------------------------
The bony pelvis is a RIGID body. Flipping a patient prone<->supine changes the
pelvis's POSE in the scanner (roughly a flip about the L-R axis) but NOT its
shape: the sacrum and both hemipelves are bone and do not deform, and the SI
joint / pubic symphysis give only a degree or two. So we do NOT compare voxels
(different poses) — we compare POSE-INVARIANT shape descriptors:

  * each structure's 3 PCA principal-axis lengths (extents_mm) — invariant to
    rotation and translation,
  * each structure's volume (mm^3) — invariant to pose,
  * the inter-hip mediolateral width.

A CORRECT pseudo pelvis must match the patient's own GT pelvis on all of these
to within a few percent (rigid bone, same patient). A large mismatch is the
failure we care about:
  * extent / volume off  -> irregular borders, over- / under-segmentation, leak
  * left_hip<->right_hip laterality disagreeing with GT -> an L/R SWAP

Which cases this applies to
---------------------------
The "separate" patients: a token that has a `spine_only` record (spine GT, the
pelvis PSEUDOLABELLED, on one scan) AND a `pelvic_native` record (real pelvis GT,
on the OTHER scan). Fused patients have a single shared scan and a real-GT pelvis,
so there is nothing to validate. The pseudo pelvis is read from the dense tree
(v2), the GT pelvis from the base tree (v1); pass the same dir for both if one
tree already holds both configs.

Usage
-----
  python scripts/pelvis_opposing_qc.py \
      --pseudo_tree data/hf_export_v2 \
      --gt_tree     data/hf_export \
      --out_csv     data/pelvis_opposing_qc.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.pelvis_opposing_qc")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from viz_pelvic_dimensions import (                              # noqa: E402
    _load_pir_int, compute_structure_pca, compute_inter_hip_extent)

SACRUM, LEFT_HIP, RIGHT_HIP = 7, 8, 9
PELVIS = {"sacrum": SACRUM, "left_hip": LEFT_HIP, "right_hip": RIGHT_HIP}


def _load_manifest(p: Path) -> List[dict]:
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _index_by_token(recs: List[dict], configs: set) -> Dict[str, dict]:
    """token -> the first record whose config is in `configs` (and has a label)."""
    out: Dict[str, dict] = {}
    for r in recs:
        if r.get("config") in configs and r.get("label_file"):
            out.setdefault(str(r.get("token", "?")), r)
    return out


def _pct(a: Optional[float], b: Optional[float]) -> float:
    """Signed % difference of a (pseudo) vs b (gt), relative to gt. NaN if n/a."""
    if a is None or b is None or b == 0:
        return float("nan")
    return 100.0 * (a - b) / b


def _descriptors(label_path: Path) -> Optional[dict]:
    """Pose-invariant pelvic descriptors from a label volume."""
    lbl, aff = _load_pir_int(label_path)
    d: dict = {"inter_hip_mm": compute_inter_hip_extent(lbl, aff)}
    for name, lid in PELVIS.items():
        pca = compute_structure_pca(lbl, aff, lid, name)
        if pca is None:
            d[name] = None
            continue
        ext = sorted(pca.extents_mm.values(), reverse=True)  # 3 principal lengths
        d[name] = {"ext": ext, "vol": pca.volume_mm3(),
                   "cx": float(pca.centroid_mm[0])}          # world-X for laterality
    return d


def _lr_sign(desc: dict) -> Optional[int]:
    """sign(left_hip.cx - right_hip.cx). RAS+: +X = patient RIGHT, so a correct
    labelling has left_hip on -X => sign = -1. Returns None if a hip is absent."""
    lh, rh = desc.get("left_hip"), desc.get("right_hip")
    if not lh or not rh:
        return None
    return -1 if lh["cx"] < rh["cx"] else 1


def _worst_ext_pct(a: dict, b: dict) -> float:
    """Largest |%| difference across the 3 principal axes."""
    worst = float("nan")
    for ea, eb in zip(a["ext"], b["ext"]):
        p = abs(_pct(ea, eb))
        if p == p and (worst != worst or p > worst):
            worst = p
    return worst


def _eval_pair(token: str, pseudo_lbl: Path, gt_lbl: Path,
               tol_pct: float) -> dict:
    ps = _descriptors(pseudo_lbl)
    gt = _descriptors(gt_lbl)
    row: dict = {"token": token, "status": "ok"}
    flags: List[str] = []

    for name in PELVIS:
        a, b = ps.get(name), gt.get(name)
        if a is None or b is None:
            row[f"{name}_vol_pct"] = ""
            row[f"{name}_ext_pct"] = ""
            if (a is None) != (b is None):
                flags.append(f"{name}_presence")   # in one pelvis but not the other
            continue
        vol_p = _pct(a["vol"], b["vol"])
        ext_p = _worst_ext_pct(a, b)
        row[f"{name}_vol_pct"] = round(vol_p, 1) if vol_p == vol_p else ""
        row[f"{name}_ext_pct"] = round(ext_p, 1) if ext_p == ext_p else ""
        if (vol_p == vol_p and abs(vol_p) > tol_pct) or \
           (ext_p == ext_p and ext_p > tol_pct):
            flags.append(f"{name}_shape")

    ih = _pct(ps.get("inter_hip_mm"), gt.get("inter_hip_mm"))
    row["inter_hip_pct"] = round(ih, 1) if ih == ih else ""
    if ih == ih and abs(ih) > tol_pct:
        flags.append("inter_hip")

    sp, sg = _lr_sign(ps), _lr_sign(gt)
    row["lr_sign_pseudo"] = sp if sp is not None else ""
    row["lr_sign_gt"] = sg if sg is not None else ""
    row["lr_swap_vs_gt"] = int(sp is not None and sg is not None and sp != sg)
    if row["lr_swap_vs_gt"]:
        flags.append("LR_SWAP")

    row["flags"] = ";".join(flags)
    row["flag"] = int(bool(flags))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo_tree", required=True, type=Path,
                    help="Dense tree (v2) holding spine_only pseudo pelves.")
    ap.add_argument("--gt_tree", required=True, type=Path,
                    help="Base tree (v1) holding pelvic_native real-GT pelves. "
                         "May be the SAME dir as --pseudo_tree.")
    ap.add_argument("--out_csv", type=Path,
                    default=Path("pelvis_opposing_qc.csv"))
    ap.add_argument("--tol_pct", type=float, default=15.0,
                    help="Flag a structure if any pose-invariant descriptor "
                         "differs from GT by more than this %% (default 15).")
    ap.add_argument("--manifest_name", default="manifest.json")
    args = ap.parse_args()

    pseudo_recs = _load_manifest(args.pseudo_tree / args.manifest_name)
    gt_recs = _load_manifest(args.gt_tree / args.manifest_name)
    if not pseudo_recs:
        log.error("no records in %s", args.pseudo_tree / args.manifest_name)
        return 2
    if not gt_recs:
        log.error("no records in %s", args.gt_tree / args.manifest_name)
        return 2

    pseudo = _index_by_token(pseudo_recs, {"spine_only"})
    gt = _index_by_token(gt_recs, {"pelvic_native"})
    paired = sorted(set(pseudo) & set(gt))
    log.info("pseudo pelves (spine_only): %d  |  GT pelves (pelvic_native): %d  "
             "|  paired patients: %d", len(pseudo), len(gt), len(paired))
    if not paired:
        log.warning("no patient has BOTH a spine_only (pseudo pelvis) and a "
                    "pelvic_native (GT pelvis) record — nothing to cross-check. "
                    "This cohort only exists for 'separate' patients.")
        return 0

    rows: List[dict] = []
    for tok in paired:
        ps_lbl = args.pseudo_tree / pseudo[tok]["label_file"]
        gt_lbl = args.gt_tree / gt[tok]["label_file"]
        if not ps_lbl.exists() or not gt_lbl.exists():
            rows.append({"token": tok, "status": "missing_label", "flag": "",
                         "flags": ""})
            continue
        try:
            rows.append(_eval_pair(tok, ps_lbl, gt_lbl, args.tol_pct))
        except Exception as exc:                                # noqa: BLE001
            log.warning("token=%s failed: %s", tok, exc)
            rows.append({"token": tok, "status": "fail", "flag": "",
                         "flags": str(exc)})

    cols = ["token", "status",
            "sacrum_vol_pct", "sacrum_ext_pct",
            "left_hip_vol_pct", "left_hip_ext_pct",
            "right_hip_vol_pct", "right_hip_ext_pct",
            "inter_hip_pct",
            "lr_sign_pseudo", "lr_sign_gt", "lr_swap_vs_gt",
            "flags", "flag"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows -> %s", len(rows), args.out_csv)

    ok = [r for r in rows if r.get("status") == "ok"]
    flagged = [r for r in ok if r.get("flag") == 1]
    swaps = [r for r in ok if r.get("lr_swap_vs_gt") == 1]
    log.info("=" * 72)
    log.info("OPPOSING-POSITION pelvis cross-check (pseudo vs same-patient GT, "
             "pose-invariant descriptors)")
    log.info("  evaluated pairs : %d", len(ok))
    log.info("  flagged (>%.0f%% on any descriptor, or L/R swap) : %d",
             args.tol_pct, len(flagged))
    log.info("  L/R SWAPS vs GT : %d", len(swaps))
    if swaps:
        log.info("    swap tokens   : %s",
                 ", ".join(r["token"] for r in swaps[:20]))

    def _med(key: str) -> str:
        vals = sorted(abs(float(r[key])) for r in ok
                      if isinstance(r.get(key), (int, float)) or
                      (isinstance(r.get(key), str) and r[key] not in ("", None)))
        if not vals:
            return "—"
        return f"{vals[len(vals) // 2]:.1f}"

    for key in ("sacrum_vol_pct", "sacrum_ext_pct", "left_hip_vol_pct",
                "left_hip_ext_pct", "right_hip_vol_pct", "right_hip_ext_pct",
                "inter_hip_pct"):
        log.info("  median |%-18s| = %s %%", key, _med(key))
    log.info("=" * 72)
    log.info("Bone is rigid and barely moves prone<->supine, so a CORRECT pseudo "
             "pelvis matches the patient's own GT pelvis within a few %% on every "
             "descriptor. Large vol/ext %% = irregular borders / leak / miss; "
             "lr_swap_vs_gt=1 = left_hip<->right_hip swapped relative to GT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
