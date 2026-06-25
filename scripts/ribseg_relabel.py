"""
ribseg_relabel.py — relabel the rib cage with RibSeg v2 (PointNet++), REPLACING the
TotalSegmentator ribs (which mis-number the lower ribs) in the v3 tree -> v4.

Per case:
  CT -> bone point cloud (>=200 HU) -> RibSeg PointNet++ -> per-point rib label
  (0 = non-rib, 1..24 = ribs) -> voxelize back to the CT grid -> map RibSeg's rib
  number to the canonical id (rib_left/right_N -> 34..57, scripts/label_scheme.py)
  -> merge onto the v3 label (clearing the old TS ribs, never touching vertebrae/
  pelvis GT) -> v4 label.

Sharded like build_v3_totalseg (--shard_id/--n_shards), resumable via per-case markers.

RibSeg is the yanx27 PointNet++ lineage (pure-Python ops) — it runs in the existing
TotalSegmentator container (CUDA torch + nibabel). Clone the repo + drop the weights,
and bind it in (see slurm/ribseg.sh): --ribseg_dir points at it.

============================  OPEN ITEMS (VERIFY on the grid)  ==================
[A] WEIGHTS: log/<log_dir>/checkpoints/best_model.pth must exist (README has no link).
[B] MODEL: --ribseg_model + get_model() signature + forward call -> see run_ribseg().
[C] LABEL CONVENTION: which of 1..24 are left vs right -> RIBSEG_SIDE_FIRST below;
    the script geometrically AUDITS it per case and warns on mismatch.
================================================================================
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.ribseg")

HU_THRESH = 200          # RibSeg's binarization (data_prepare.py: source[source>=200]=1)
NPOINT = 30000           # RibSeg's per-forward point count
N_RIBS = 12              # per side
RIB_IDS = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + N_RIBS + 1))  # 34..57

# [C] RibSeg labels ribs 1..24. ASSUMED convention (VERIFY): 1..12 = right_1..12
# (cranial->caudal), 13..24 = left_1..12. The per-case audit re-checks side from the
# x-centroid and warns if a "right" rib sits on the left, etc.
RIBSEG_RIGHT = range(1, 13)        # 1..12 -> right
RIBSEG_LEFT = range(13, 25)        # 13..24 -> left


def ribseg_num_to_canonical(k: int) -> Optional[int]:
    """RibSeg rib label k in 1..24 -> canonical id (34..57), or None for non-rib."""
    if k in RIBSEG_RIGHT:
        return LS.rib_id("right", k)            # right_k
    if k in RIBSEG_LEFT:
        return LS.rib_id("left", k - 12)        # left_(k-12)
    return None


# ----------------------------------------------------------------------------
# CT -> bone point cloud
# ----------------------------------------------------------------------------
def ct_to_bone_points(ct_path: Path) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Return (coords[N,3] int voxel indices of bone>=HU_THRESH, ct shape)."""
    img = nib.load(str(ct_path))
    vol = np.asanyarray(img.dataobj)
    coords = np.argwhere(vol >= HU_THRESH).astype(np.int64)
    return coords, vol.shape[:3]


# ----------------------------------------------------------------------------
# RibSeg PointNet++ inference  ([B] VERIFY the model call against the repo)
# ----------------------------------------------------------------------------
def load_ribseg_model(ribseg_dir: Path, model_name: str, log_dir: str, device: str):
    import importlib
    import torch
    if str(ribseg_dir) not in sys.path:
        sys.path.insert(0, str(ribseg_dir))
    mod = importlib.import_module(f"models.{model_name}")
    # num classes = 25 (background + 24 ribs). VERIFY get_model signature in the repo.
    net = mod.get_model(25, normal_channel=False)
    ckpt = torch.load(str(ribseg_dir / "log" / log_dir / "checkpoints" / "best_model.pth"),
                      map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    net.load_state_dict(state)
    return net.to(device).eval()


def run_ribseg(coords: np.ndarray, net, device: str) -> np.ndarray:
    """coords[N,3] -> per-point labels[N] in 0..24. Replicates inference.py's prep:
    centroid-normalize + scale by max distance, run in NPOINT-sized chunks."""
    import torch
    pts = coords.astype(np.float32)
    pts = pts - pts.mean(0)
    scale = np.max(np.linalg.norm(pts, axis=1)) or 1.0
    pts = pts / scale

    labels = np.zeros(len(pts), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(pts), NPOINT):
            chunk = pts[s:s + NPOINT]
            n = len(chunk)
            if n < NPOINT:                       # pad the tail to NPOINT, drop padding after
                chunk = np.vstack([chunk, np.zeros((NPOINT - n, 3), np.float32)])
            x = torch.from_numpy(chunk).to(device).float().unsqueeze(0).transpose(2, 1)  # (1,3,NPOINT)
            # [B] VERIFY: pointnet2_part_seg_msg.forward(xyz, cls_label) returns (logits, ...).
            # Single category -> a zero one-hot of width 1. CLNet may differ.
            cls = torch.zeros(1, 1, device=device)
            out = net(x, cls)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (1,NPOINT,25)
            pred = logits.argmax(-1).squeeze(0).cpu().numpy()[:n]
            labels[s:s + n] = pred
    return labels


# ----------------------------------------------------------------------------
# voxelize + audit + merge
# ----------------------------------------------------------------------------
def points_to_canonical_volume(coords: np.ndarray, labels: np.ndarray,
                               shape: Tuple[int, int, int]) -> np.ndarray:
    """Scatter RibSeg per-point labels (1..24) into a canonical-id volume (34..57)."""
    out = np.zeros(shape, dtype=np.int16)
    for k in range(1, 25):
        cid = ribseg_num_to_canonical(k)
        if cid is None:
            continue
        m = labels == k
        if m.any():
            c = coords[m]
            out[c[:, 0], c[:, 1], c[:, 2]] = cid
    return out


def audit_sides(rib_vol: np.ndarray, affine) -> None:
    """[C] Sanity-check: every 'right' rib's centroid should be on the opposite world-x
    side from every 'left' rib. Warn (don't fail) so a wrong RIBSEG_* map is caught."""
    def world_x(cid):
        ijk = np.argwhere(rib_vol == cid)
        if not len(ijk):
            return None
        return float(nib.affines.apply_affine(affine, ijk.mean(0))[0])
    rx = [world_x(LS.rib_id("right", n)) for n in range(1, N_RIBS + 1)]
    lx = [world_x(LS.rib_id("left", n)) for n in range(1, N_RIBS + 1)]
    rx = [v for v in rx if v is not None]
    lx = [v for v in lx if v is not None]
    if rx and lx and not (max(rx) < min(lx) or min(rx) > max(lx)):
        log.warning("SIDE AUDIT: right/left rib centroids overlap in world-x — the "
                    "RIBSEG_RIGHT/LEFT convention is likely wrong; check on a full-ribcage case.")


def merge_into_v3(v3_path: Path, rib_vol: np.ndarray, out_path: Path) -> int:
    """Clear the old TS ribs (34..57) in the v3 label, then write RibSeg ribs only on
    background (never overwriting vertebrae/pelvis GT). Returns rib voxel count."""
    img = nib.load(str(v3_path))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    lab[np.isin(lab, list(RIB_IDS))] = 0                      # drop TS ribs
    place = (rib_vol > 0) & (lab == 0)                        # ribs only on bg
    lab[place] = rib_vol[place]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lab, img.affine, img.header), str(out_path))
    return int(place.sum())


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def process_case(ct_path: Path, v3_label_path: Path, out_label_path: Path,
                 net, device: str) -> Dict[str, object]:
    coords, shape = ct_to_bone_points(ct_path)
    if not len(coords):
        log.warning("%s: no bone>=%dHU — skipping", ct_path.name, HU_THRESH)
        return {"ct": ct_path.name, "rib_vox": 0, "status": "no_bone"}
    labels = run_ribseg(coords, net, device)
    rib_vol = points_to_canonical_volume(coords, labels, shape)
    img = nib.load(str(v3_label_path))
    audit_sides(rib_vol, img.affine)
    n = merge_into_v3(v3_label_path, rib_vol, out_label_path)
    log.info("%s: %d rib voxels (%d distinct ribs)", ct_path.name, n,
             len(set(int(v) for v in np.unique(rib_vol)) - {0}))
    return {"ct": ct_path.name, "rib_vox": n, "status": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3_dir", type=Path, required=True, help="v3 tree (ct/ + labels/)")
    ap.add_argument("--out_dir", type=Path, required=True, help="v4 tree to write")
    ap.add_argument("--ribseg_dir", type=Path, required=True, help="cloned RibSeg repo (+ log/<log_dir> weights)")
    ap.add_argument("--ribseg_model", default="pointnet2_part_seg_msg", help="[B] model module under models/")
    ap.add_argument("--log_dir", default="c2_a", help="RibSeg checkpoint dir under log/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true")
    a = ap.parse_args()

    cts = sorted((a.v3_dir / "ct").glob("*.nii.gz"))
    if a.n_shards > 1:
        cts = [c for i, c in enumerate(cts) if i % a.n_shards == a.shard_id]
    if a.limit:
        cts = cts[:a.limit]
    log.info("shard %d/%d: %d cases", a.shard_id, a.n_shards, len(cts))
    (a.out_dir / "labels").mkdir(parents=True, exist_ok=True)
    done_dir = a.out_dir / "_ribseg_done"; done_dir.mkdir(parents=True, exist_ok=True)

    net = load_ribseg_model(a.ribseg_dir, a.ribseg_model, a.log_dir, a.device)

    n_ok = n_skip = 0
    for ct in cts:
        cid = ct.name[:-len(".nii.gz")]
        lbl = a.v3_dir / "labels" / f"{cid}_label.nii.gz"
        out = a.out_dir / "labels" / f"{cid}_label.nii.gz"
        marker = done_dir / f"{cid}.json"
        if not a.no_resume and marker.exists() and out.exists():
            n_skip += 1; continue
        if not lbl.exists():
            log.warning("%s: no v3 label — skipping", cid); continue
        qc = process_case(ct, lbl, out, net, a.device)
        marker.write_text(str(qc))
        n_ok += int(qc["status"] == "ok")
    log.info("done: %d processed, %d resumed/skipped", n_ok, n_skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
