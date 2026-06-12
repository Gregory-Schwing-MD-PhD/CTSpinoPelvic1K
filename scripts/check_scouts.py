"""
check_scouts.py — are the COLONOG scout/topogram series on disk, and do they
reach high enough to count ribs?

Two questions, one tool:

  (a) RECOVERABILITY — the downloader pulls every CT-modality series (scouts
      included; see download_tcia_colonog.py), and tcia_index flags them
      `is_scout` but never deletes them. So they should be sitting in the raw
      TCIA download. This confirms it: load the patient index, filter to scout
      records, and stat each `series_dir` (exists + has DICOMs). If a manifest
      is given, it also reports coverage among the patients that actually made
      it into the dataset (join on patient_token).

  (b) CRANIOCAUDAL EXTENT — a scout only resolves "T13 vs L1-with-a-lumbar-rib"
      if it shows enough of the rib cage. For a sample it reads the single
      projection header and computes the scout's superior-inferior extent (mm)
      and z-range, classifies AP (coronal) vs lateral (sagittal), and compares
      the scout's z-range to the matched axial CT's — so we can see whether the
      scout reaches MORE CRANIAL than the abdominal scan (i.e. up toward the
      ribs). `--dump_png` renders the sampled scouts so you can eyeball them.

Run inside the project container (needs pydicom + numpy):

  singularity exec --bind "$(pwd):/workspace,$DATA_DIR:/data" "$SIF_PATH" \
    python3 /workspace/scripts/check_scouts.py \
      --tcia_dir /data/tcia \
      --manifest /data/hf_export_v2/manifest.json \
      --sample 12 --dump_png /data/scout_samples
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ on path

import numpy as np  # noqa: E402
import pydicom      # noqa: E402

from tcia_index import build_tcia_patient_index  # noqa: E402
from patient_db import TciaSeriesRecord           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("check_scouts")


# ── disk check ───────────────────────────────────────────────────────────────

def _dcm_files(series_dir: Path) -> List[Path]:
    files = sorted(series_dir.glob("*.dcm"))
    return files or sorted(series_dir.rglob("*.dcm"))


def _on_disk(rec: TciaSeriesRecord) -> Tuple[bool, int]:
    d = Path(rec.series_dir)
    if not d.is_dir():
        return False, 0
    return True, len(_dcm_files(d))


# ── geometry ─────────────────────────────────────────────────────────────────

def _scout_geometry(series_dir: Path) -> Optional[dict]:
    """Superior-inferior extent + orientation of a single projection scout,
    from one DICOM header. z here is LPS-z == RAS-S (LPS->RAS only flips x,y),
    so it is directly comparable to the axial CT's RAS-z."""
    files = _dcm_files(series_dir)
    if not files:
        return None
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        ps = [float(x) for x in ds.PixelSpacing]          # [row_sp, col_sp]
        rows, cols = int(ds.Rows), int(ds.Columns)
    except Exception as exc:                              # noqa: BLE001
        log.debug("geometry read failed for %s: %s", series_dir.name, exc)
        return None

    row_cos = np.array(iop[:3])      # increasing column index (image →)
    col_cos = np.array(iop[3:])      # increasing row index (image ↓)
    normal = np.cross(row_cos, col_cos)

    # z (S) of the four image corners → robust SI extent regardless of rotation
    c0 = ipp
    c_r = ipp + (rows - 1) * ps[0] * col_cos
    c_c = ipp + (cols - 1) * ps[1] * row_cos
    c_rc = c_r + (cols - 1) * ps[1] * row_cos
    zs = [c0[2], c_r[2], c_c[2], c_rc[2]]
    z_lo, z_hi = float(min(zs)), float(max(zs))

    # orientation from the dominant normal axis (LPS): Y=A/P → coronal (AP),
    # X=L/R → sagittal (lateral).
    ax = int(np.argmax(np.abs(normal)))
    view = {0: "lateral", 1: "AP", 2: "axial"}.get(ax, "?")
    return {"view": view, "rows": rows, "cols": cols,
            "si_extent_mm": z_hi - z_lo, "z_lo": z_lo, "z_hi": z_hi}


def _ct_zrange(rec: TciaSeriesRecord) -> Optional[Tuple[float, float]]:
    sp = rec.spatial
    if sp is None:
        return None
    try:
        oz = float(sp.origin[2])
        dz = float(sp.spacing[2])
        dir_zz = float(sp.direction[8])               # dir[2,2] of flattened 3x3
        end = oz + (sp.nz - 1) * dz * (1.0 if dir_zz >= 0 else -1.0)
        return (min(oz, end), max(oz, end))
    except Exception:                                  # noqa: BLE001
        return None


def _best_ct(recs: List[TciaSeriesRecord]) -> Optional[TciaSeriesRecord]:
    cts = [r for r in recs if r.is_ct_quality and r.spatial is not None]
    return max(cts, key=lambda r: r.n_dcm) if cts else None


# ── png dump (best effort) ───────────────────────────────────────────────────

def _dump_png(series_dir: Path, out_png: Path) -> bool:
    files = _dcm_files(series_dir)
    if not files:
        return False
    try:
        ds = pydicom.dcmread(str(files[0]), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + inter
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (arr * 255).astype(np.uint8)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(img).save(str(out_png))
        except Exception:                              # noqa: BLE001
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(str(out_png), img, cmap="gray")
        return True
    except Exception as exc:                            # noqa: BLE001
        log.debug("png dump failed for %s: %s", series_dir.name, exc)
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def _manifest_tokens(path: Path) -> Optional[set]:
    try:
        data = json.loads(path.read_text())
        recs = data if isinstance(data, list) else data.get("records", [])
        toks = {str(r.get("token")) for r in recs if r.get("token") is not None}
        return toks or None
    except Exception as exc:                            # noqa: BLE001
        log.warning("could not read manifest %s: %s", path, exc)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcia_dir", required=True, type=Path,
                    help="raw COLONOG download root (one folder per series).")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional manifest.json — restrict coverage report to "
                         "the patients (tokens) actually in the dataset.")
    ap.add_argument("--sample", type=int, default=10,
                    help="how many on-disk scouts to inspect for geometry.")
    ap.add_argument("--dump_png", type=Path, default=None,
                    help="render the sampled scouts to this dir for eyeballing.")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force_rebuild", action="store_true")
    a = ap.parse_args()

    grouped, _ = build_tcia_patient_index(
        a.tcia_dir, workers=a.workers, force_rebuild=a.force_rebuild)
    if not grouped:
        log.error("empty index — check --tcia_dir.")
        return 1

    ds_tokens = _manifest_tokens(a.manifest) if a.manifest else None

    # ── (a) on-disk scout coverage ───────────────────────────────────────────
    n_pat = n_pat_in_ds = 0
    n_pat_with_scout = n_pat_in_ds_with_scout = 0
    scout_recs: List[Tuple[TciaSeriesRecord, int]] = []
    for uid, recs in grouped.items():
        tok = recs[0].patient_token
        in_ds = ds_tokens is None or str(tok) in ds_tokens
        n_pat += 1
        n_pat_in_ds += int(in_ds)
        present = []
        for r in recs:
            if not r.is_scout:
                continue
            ok, n = _on_disk(r)
            if ok and n >= 1:
                present.append((r, n))
        if present:
            n_pat_with_scout += 1
            n_pat_in_ds_with_scout += int(in_ds)
            if in_ds:
                scout_recs.extend(present)

    log.info("=" * 64)
    log.info("SCOUT ON-DISK CHECK   tcia_dir=%s", a.tcia_dir)
    log.info("-" * 64)
    log.info("patients indexed                    : %d", n_pat)
    log.info("patients with >=1 on-disk scout     : %d  (%.1f%%)",
             n_pat_with_scout, 100 * n_pat_with_scout / max(n_pat, 1))
    if ds_tokens is not None:
        log.info("patients in dataset (manifest)      : %d", n_pat_in_ds)
        log.info("  └─ of those, with on-disk scout   : %d  (%.1f%%)  <-- coverage",
                 n_pat_in_ds_with_scout,
                 100 * n_pat_in_ds_with_scout / max(n_pat_in_ds, 1))
    log.info("on-disk scout series (dataset pts)  : %d", len(scout_recs))

    # ── (b) craniocaudal extent on a sample ──────────────────────────────────
    if not scout_recs:
        log.info("no on-disk scouts to sample.")
        return 0

    sample = scout_recs[: a.sample]
    log.info("-" * 64)
    log.info("CRANIOCAUDAL EXTENT  (sample of %d)", len(sample))
    log.info("-" * 64)
    log.info("%-6s %-8s %-9s %-10s %-10s %s",
             "token", "view", "n_dcm", "SI_mm", "CT_SI_mm", "scout taller by (extra cranial cover)")
    from collections import Counter
    view_counts: Counter = Counter()
    taller = 0
    for r, n in sample:
        g = _scout_geometry(Path(r.series_dir))
        if g is None:
            log.info("%-6s  (geometry unreadable)", r.patient_token)
            continue
        view_counts[g["view"]] += 1
        ct = _best_ct(grouped[r.patient_uid])
        ctz = _ct_zrange(ct) if ct else None
        # FRAME-INDEPENDENT comparison: a COLONOG localizer and its axial series
        # usually DON'T share a z-origin (separate prone/supine acquisitions reset
        # table position), so comparing absolute z is meaningless. Compare the SI
        # EXTENTS instead: the axial already covers lumbar+sacrum, so any extra
        # scout height is added cranial coverage — i.e. toward the lower thorax /
        # ribs. (Definitive check is still the PNG.)
        if ctz:
            ct_si = ctz[1] - ctz[0]
            delta = g["si_extent_mm"] - ct_si
            verdict = f"{ct_si:7.0f}    {'+' if delta >= 0 else ''}{delta:.0f}mm"
            taller += int(delta > 50)                  # >5cm taller than the CT
        else:
            ct_si = float("nan")
            verdict = "   no CT spatial"
        log.info("%-6s %-8s %-9d %-10.0f %-10.0f %s",
                 r.patient_token, g["view"], n, g["si_extent_mm"], ct_si, verdict)
        if a.dump_png:
            png = a.dump_png / f"{r.patient_token}_{g['view']}_{r.series_uid[-8:]}.png"
            if _dump_png(Path(r.series_dir), png):
                log.info("        png -> %s", png)

    log.info("-" * 64)
    log.info("sample views: %s", dict(view_counts))
    log.info("scouts >5cm taller than their axial CT (extra cranial cover -> ribs): %d / %d",
             taller, len(sample))
    log.info("=" * 64)
    log.info("READ: SI_mm ~500+ AND taller than the abdominal CT => whole-torso "
             "scout, ribs almost certainly in frame (countable). Confirm on the "
             "PNGs. SI_mm ~300 and no extra height => abdomen/pelvis only "
             "(relative count). Absolute z is NOT compared (localizer/axial "
             "frames-of-reference differ).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
