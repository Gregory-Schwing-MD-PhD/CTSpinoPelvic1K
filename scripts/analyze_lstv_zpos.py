#!/usr/bin/env python3
"""
analyze_lstv_zpos.py — Test the "consistent Z-position" assumption that a
CoordConv-style positional prior relies on, and quantify how much LSTV
(lumbarization / sacralization) violates it.

Why this exists
===============
A CoordConv Z-channel encodes each voxel's box-normalized superior-inferior
position in [-1, +1]. It can only help vertebra *numbering* if position
predicts label. The decisive case is the vertebra immediately superior to
the sacrum: it is L5 in a normal spine but L6 in a lumbarized spine, yet it
sits at ~the same box position. If the box-normalized Z of that bottom
vertebra is the same whether it's L5 or L6, a coordinate channel CANNOT tell
them apart — which is exactly the LSTV distinction we care about.

What it computes
================
For each record's label NIfTI, for each lumbar vertebra (L1..L6) and the
sacrum, the superior-inferior (SI) centroid in two coordinate systems:

  1. box_norm_si in [-1,+1]  — what a CoordConv Z-channel encodes
     (centroid index / full image SI extent; +1 = inferior). Uses the FULL
     image grid (= the CT FOV, since labels are voxel-aligned), NOT the
     label bounding box, so it matches what the network's coordinate channel
     would see.
  2. sup_to_sacrum_mm        — anatomy-anchored: world-space mm the vertebra
     sits SUPERIOR to the sacrum centroid (the robust signal your
     post-processing uses). Positive = above the sacrum.

Headline test
=============
The "lowest lumbar" per record (most-inferior lumbar centroid) is L5 in
normals and L6 in lumbarization. We compare box_norm_si of that bottom
vertebra grouped by its true value (5 vs 6). Heavy overlap => CoordConv
cannot disambiguate L5/L6 by position => keep the merge + sacrum-anchored
counting. The sup_to_sacrum_mm view should, by contrast, stay tight.

Source label scheme (HF export, pre-merge):
  0 bg, 1-4 L1-L4, 5 L5, 6 L6, 7 sacrum, 8 left_hip, 9 right_hip, 10 ignore

Usage
=====
  python scripts/analyze_lstv_zpos.py \
      --hf_dir ~/CTSpinoPelvic1K/data/hf_export \
      --splits ~/CTSpinoPelvic1K/data/hf_export/splits_5fold.json \
      --out_dir ~/CTSpinoPelvic1K/data/hf_export/qc_zpos

Writes <out_dir>/zpos_centroids.csv (+ per-class stats to stdout) and, if
matplotlib is available, violin PNGs. No scipy required (pure numpy).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("analyze_lstv_zpos")

# Source label values -> readable names (pre-merge HF export scheme).
VERT_LABELS = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6", 7: "sacrum"}
LUMBAR_VALUES = (1, 2, 3, 4, 5, 6)
SACRUM_VALUE = 7


def _load_manifest_records(hf_dir: Path) -> List[Dict]:
    records: List[Dict] = []
    for fn in ("manifest_train.json", "manifest_validation.json", "manifest_test.json"):
        p = hf_dir / fn
        if not p.exists():
            log.warning("manifest not found: %s", p)
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("records", data.get("cases", list(data.values())))
        if isinstance(data, list):
            split = fn.replace("manifest_", "").replace(".json", "")
            for r in data:
                if isinstance(r, dict):
                    r = dict(r)
                    r.setdefault("_split", split)
                    records.append(r)
    return records


def _si_axis_and_sign(affine: np.ndarray) -> Tuple[int, float]:
    """Return (voxel_axis_that_maps_to_world_Z, sign).

    World Z is superior (+) in nibabel RAS+. sign>0 means increasing that
    voxel index moves SUPERIOR; sign<0 means it moves INFERIOR.
    """
    z_row = affine[2, :3]
    si_axis = int(np.argmax(np.abs(z_row)))
    sign = float(np.sign(z_row[si_axis]) or 1.0)
    return si_axis, sign


def _centroid_vox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Mean voxel index per axis for a boolean mask, or None if empty."""
    nz = np.nonzero(mask)
    if nz[0].size == 0:
        return None
    return np.array([c.mean() for c in nz], dtype=np.float64)


def _box_norm_si(centroid_vox: np.ndarray, shape, si_axis: int, sign: float) -> float:
    """Box-normalized SI position in [-1,+1], +1 = inferior, over the full grid."""
    n = shape[si_axis]
    frac = centroid_vox[si_axis] / max(n - 1, 1)          # 0..1 along voxel axis
    infer_frac = (1.0 - frac) if sign > 0 else frac        # 0=superior .. 1=inferior
    return 2.0 * infer_frac - 1.0


def _world_z(centroid_vox: np.ndarray, affine: np.ndarray) -> float:
    """World-space superior coordinate (mm) of a voxel centroid (RAS+ Z)."""
    return float((affine[:3, :3] @ centroid_vox + affine[:3, 3])[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_dir", required=True, type=Path)
    ap.add_argument("--splits", type=Path, default=None,
                    help="splits_5fold.json for patient_subtypes (normal vs LSTV). "
                         "Optional: falls back to L6-presence heuristic.")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--max_cases", type=int, default=0,
                    help="0 = all; otherwise cap (for a quick smoke run).")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.hf_dir / "qc_zpos")
    out_dir.mkdir(parents=True, exist_ok=True)

    subtypes: Dict[str, str] = {}
    if args.splits and args.splits.exists():
        subtypes = json.loads(args.splits.read_text()).get("patient_subtypes", {})
        log.info("loaded %d patient subtypes from %s", len(subtypes), args.splits)

    records = _load_manifest_records(args.hf_dir)
    log.info("loaded %d manifest records", len(records))

    rows: List[Dict] = []
    n_done = 0
    for rec in records:
        tok = str(rec.get("token") or rec.get("patient_token") or "")
        label_rel = rec.get("label_file") or rec.get("label") or ""
        if not tok or not label_rel:
            continue
        label_path = args.hf_dir / label_rel
        if not label_path.exists():
            continue

        img = nib.load(str(label_path))
        arr = np.asarray(img.dataobj)
        si_axis, sign = _si_axis_and_sign(img.affine)

        # sacrum anchor (world Z), if present in this record
        sac_c = _centroid_vox(arr == SACRUM_VALUE)
        sac_z = _world_z(sac_c, img.affine) if sac_c is not None else None

        present_lumbar: List[Tuple[int, float, float]] = []  # (value, box_norm, mm)
        for v in LUMBAR_VALUES + (SACRUM_VALUE,):
            c = _centroid_vox(arr == v)
            if c is None:
                continue
            bn = _box_norm_si(c, arr.shape, si_axis, sign)
            mm = (_world_z(c, img.affine) - sac_z) if sac_z is not None else None
            subtype = subtypes.get(tok, "")
            if not subtype:
                # fallback: lumbarization iff an L6 voxel exists anywhere
                subtype = "lumb" if (arr == 6).any() else "normal"
            is_lstv = subtype not in ("", "normal")
            rows.append({
                "token": tok, "config": rec.get("config", ""),
                "split": rec.get("_split", ""), "subtype": subtype,
                "is_lstv": int(is_lstv), "class": VERT_LABELS[v], "value": v,
                "box_norm_si": round(bn, 4),
                "sup_to_sacrum_mm": (round(mm, 2) if mm is not None else ""),
            })
            if v in LUMBAR_VALUES:
                present_lumbar.append((v, bn, mm if mm is not None else float("nan")))

        # tag the most-inferior lumbar (largest box_norm) as the "lowest lumbar"
        if present_lumbar:
            low_v, low_bn, low_mm = max(present_lumbar, key=lambda t: t[1])
            rows.append({
                "token": tok, "config": rec.get("config", ""),
                "split": rec.get("_split", ""),
                "subtype": subtypes.get(tok, "") or ("lumb" if (arr == 6).any() else "normal"),
                "is_lstv": "", "class": "LOWEST_LUMBAR", "value": low_v,
                "box_norm_si": round(low_bn, 4),
                "sup_to_sacrum_mm": (round(low_mm, 2) if not np.isnan(low_mm) else ""),
            })

        n_done += 1
        if args.max_cases and n_done >= args.max_cases:
            break
        if n_done % 100 == 0:
            log.info("processed %d records ...", n_done)

    log.info("processed %d records with labels", n_done)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = out_dir / "zpos_centroids.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    # ── Stats ──────────────────────────────────────────────────────────────
    def _stats(vals):
        a = np.array([v for v in vals if v is not None and v == v], dtype=float)
        return (len(a), a.mean(), a.std()) if a.size else (0, float("nan"), float("nan"))

    print("\n=== box_norm_si per class  (normal vs LSTV)  +1=inferior ===")
    print(f"{'class':<14}{'n_norm':>7}{'mean':>8}{'std':>7}   |{'n_lstv':>7}{'mean':>8}{'std':>7}")
    by = defaultdict(lambda: {"normal": [], "lstv": []})
    for r in rows:
        if r["class"] == "LOWEST_LUMBAR":
            continue
        key = "lstv" if r["is_lstv"] == 1 else "normal"
        by[r["class"]][key].append(r["box_norm_si"])
    for cls in ["L1", "L2", "L3", "L4", "L5", "L6", "sacrum"]:
        if cls not in by:
            continue
        nn, mn, sn = _stats(by[cls]["normal"]); nl, ml, sl = _stats(by[cls]["lstv"])
        print(f"{cls:<14}{nn:>7}{mn:>8.3f}{sn:>7.3f}   |{nl:>7}{ml:>8.3f}{sl:>7.3f}")

    # ── Headline: is the LOWEST lumbar's position separable into L5 vs L6? ──
    low_by_val = defaultdict(lambda: {"box": [], "mm": []})
    for r in rows:
        if r["class"] != "LOWEST_LUMBAR":
            continue
        low_by_val[int(r["value"])]["box"].append(r["box_norm_si"])
        if r["sup_to_sacrum_mm"] != "":
            low_by_val[int(r["value"])]["mm"].append(float(r["sup_to_sacrum_mm"]))

    print("\n=== HEADLINE: bottom vertebra (just above sacrum) — L5 vs L6 ===")
    for v in (5, 6):
        nb, mb, sb = _stats(low_by_val[v]["box"])
        nm, mm_, sm = _stats(low_by_val[v]["mm"])
        nm_lbl = VERT_LABELS.get(v, str(v))
        print(f"  bottom=={nm_lbl}: n={nb:<4} box_norm {mb:.3f}+/-{sb:.3f}   "
              f"sup_to_sacrum_mm {mm_:.1f}+/-{sm:.1f}")
    b5, b6 = low_by_val[5]["box"], low_by_val[6]["box"]
    if b5 and b6:
        m5, s5 = np.mean(b5), np.std(b5); m6, s6 = np.mean(b6), np.std(b6)
        sep = abs(m5 - m6) / (s5 + s6 + 1e-9)   # >1.5 ~ separable, <1 ~ overlapping
        verdict = ("SEPARABLE -> a Z-channel could help" if sep > 1.5
                   else "OVERLAPPING -> CoordConv CANNOT tell L5 from L6 by position")
        print(f"  separation (|dmean|/(std5+std6)) = {sep:.2f}  => {verdict}")

    # ── Violin plots (optional) ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def violin(metric: str, ylabel: str, fname: str):
            order = ["L1", "L2", "L3", "L4", "L5", "L6", "sacrum"]
            fig, ax = plt.subplots(figsize=(11, 6))
            pos = 0; ticks = []; labels = []
            for cls in order:
                for grp, color, off in (("normal", "#4C78A8", -0.18), ("lstv", "#E45756", 0.18)):
                    vals = [r[metric] for r in rows
                            if r["class"] == cls and r[metric] != ""
                            and ((r["is_lstv"] == 1) == (grp == "lstv"))]
                    vals = [float(v) for v in vals]
                    if len(vals) >= 2:
                        vp = ax.violinplot(vals, positions=[pos + off], widths=0.3,
                                           showmeans=True)
                        for b in vp["bodies"]:
                            b.set_facecolor(color); b.set_alpha(0.6)
                ticks.append(pos); labels.append(cls); pos += 1
            ax.set_xticks(ticks); ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per vertebra - blue=normal, red=LSTV")
            ax.grid(axis="y", alpha=0.3)
            out = out_dir / fname; fig.tight_layout(); fig.savefig(out, dpi=130)
            plt.close(fig); log.info("wrote %s", out)

        violin("box_norm_si", "box-normalized SI (+1=inferior) [CoordConv view]",
               "violin_box_norm_si.png")
        violin("sup_to_sacrum_mm", "mm superior to sacrum [anatomy-anchored]",
               "violin_sup_to_sacrum_mm.png")
    except Exception as e:
        log.warning("skipping plots (matplotlib unavailable or failed): %s", e)
        log.warning("CSV is written; plot %s elsewhere.", csv_path)


if __name__ == "__main__":
    main()
