"""
sweep_stray_vertebrae.py — find v4 labels where a VERTEBRA class has voxels far from its body.

WHY. Case 0730 carried ~2.8k voxels of L4 and ~2.1k of L5 floating 230-260 mm ABOVE their real
bodies, at thoracic level: ribs the pipeline mislabelled as lumbar. That defect is invisible to the
rib annotator and UNFIXABLE by them -- service._normalize_spine force-restores the spine on every
rib submission and never writes a rib over a vertebra, so their (correct) relabelling is discarded,
and rib_spine_gap then measures their ribs against the phantom vertebra and BLOCKS the submit. The
case simply cannot be submitted until the source label is fixed. One of these cost a full session.

THE SCREEN. A real vertebra spans ~25-55 mm cranio-caudally. A class whose voxels span far more
than that has something detached and distant. Reported per class:

    span_mm      full cranio-caudal extent of the class
    outlier_mm   distance from the class's main slab to its farthest voxel

Cheap by construction: one pass of per-plane np.unique along the superior-inferior axis (no
connected components, no full-volume masks), so it survives the 145M-voxel labels.

    python scripts/sweep_stray_vertebrae.py                      # all of v4 -> csv
    python scripts/sweep_stray_vertebrae.py --limit 25           # smoke test
    python scripts/sweep_stray_vertebrae.py --span_mm 80         # tighter/looser screen
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sweep")

# vertebrae only: sacrum (26) / coccyx (27) / S1 (29) are legitimately tall or irregular, and the
# pelvis (30-33) is not a vertebra. T13 (28) is included -- it is a vertebra when present.
VERT_IDS = [*range(1, 26), 28]
NAMES = {v: k for k, v in LS.label_dict().items()}


def class_z_profile(lab: np.ndarray, affine):
    """{class: (z_min_mm, z_max_mm, [z of each occupied plane])} via one pass of per-plane unique."""
    si = int(np.argmax(np.abs(affine[:3, :3][2, :])))
    n = lab.shape[si]
    pts = np.zeros((n, 3), dtype=float)          # world z of every plane along the SI axis
    pts[:, si] = np.arange(n)
    zs = nib.affines.apply_affine(affine, pts)[:, 2]
    planes: dict[int, list[int]] = {}
    for i in range(n):
        sl = [slice(None)] * 3
        sl[si] = i
        ids = np.unique(lab[tuple(sl)])
        for c in ids.tolist():
            if c in VERT_IDS:
                planes.setdefault(int(c), []).append(i)
    return {c: (zs[idx].min(), zs[idx].max(), zs[idx]) for c, idx in planes.items()}


def scan_label(path, span_mm: float):
    """Scan one label with a near-zero memory footprint.

    A .nii.gz cannot be mmap'd, so nibabel materialises the whole ~290 MB array. On a box with no
    page file (commit limit == physical RAM) that fails outright once other processes have reserved
    the commit -- the first full sweep died of MemoryError at ~case 600. So: gunzip to a temp .nii,
    let nibabel mmap it, and touch only one 2D plane at a time.
    """
    import gc
    import gzip
    import shutil
    import tempfile

    p = str(path)
    tmp = None
    try:
        if p.endswith(".gz"):
            fd, tmp = tempfile.mkstemp(suffix=".nii")
            os.close(fd)
            with gzip.open(p, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)
            img = nib.load(tmp, mmap=True)
        else:
            img = nib.load(p, mmap=True)
        return _scan(img.dataobj, img.affine, span_mm, [])
    finally:
        try:
            img.uncache(); del img
        except Exception:
            pass
        gc.collect()
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _scan(lab, affine, span_mm, out):
    for c, (zlo, zhi, zarr) in class_z_profile(lab, affine).items():
        span = float(zhi - zlo)
        if span <= span_mm:
            continue
        # distance from the densest slab to the farthest occupied plane: a stray blob shows up as a
        # big gap, a merely tall vertebra does not.
        med = float(np.median(zarr))
        outlier = float(max(abs(zhi - med), abs(zlo - med)))
        out.append({"class_id": c, "name": NAMES.get(c, str(c)),
                    "span_mm": round(span, 1), "outlier_mm": round(outlier, 1),
                    "z_min": round(float(zlo), 1), "z_max": round(float(zhi), 1)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K"))
    ap.add_argument("--revision", default="v4")
    ap.add_argument("--labels_dir", default=None,
                    help="scan a LOCAL dir of *_label.nii.gz instead of the hub")
    ap.add_argument("--span_mm", type=float, default=90.0,
                    help="flag a vertebra class spanning more than this (default 90)")
    ap.add_argument("--out", default="stray_vertebrae.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    if a.labels_dir:
        files = sorted(Path(a.labels_dir).glob("*_label.nii.gz"))
        fetch = lambda p: p                                        # noqa: E731
    else:
        from huggingface_hub import HfApi, hf_hub_download
        tok = os.environ.get("HF_TOKEN")
        api = HfApi(token=tok)
        files = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset", revision=a.revision)
                       if f.startswith("labels/") and f.endswith(".nii.gz"))
        fetch = lambda f: hf_hub_download(a.repo, f, repo_type="dataset",  # noqa: E731
                                          revision=a.revision, token=tok)
    if a.limit:
        files = files[:a.limit]
    log.info("scanning %d labels (span > %.0f mm)", len(files), a.span_mm)

    # Resume + incremental write: the first full run died of MemoryError at ~case 600 and lost
    # everything, because results were only written at the end. Every scanned case is recorded
    # immediately, so a crash costs one case, not the run.
    cols = ["case", "class_id", "name", "span_mm", "outlier_mm", "z_min", "z_max"]
    done_path = Path(a.out).with_suffix(".done.txt")
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    if done:
        log.info("resuming: %d cases already scanned", len(done))
    fresh = not Path(a.out).exists()
    out_fh = open(a.out, "a", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=cols, extrasaction="ignore")
    if fresh:
        writer.writeheader(); out_fh.flush()
    done_fh = open(done_path, "a")

    n_flagged = n_hits = n_err = 0
    for i, f in enumerate(files, 1):
        case = Path(str(f)).name.replace("_label.nii.gz", "")
        if case in done:
            continue
        try:
            hits = scan_label(fetch(f), a.span_mm)
        except Exception as exc:
            n_err += 1
            log.warning("%s: %s", case, (str(exc) or type(exc).__name__)[:120])
            continue
        for h in hits:
            h["case"] = case
            writer.writerow(h); n_hits += 1
        out_fh.flush()
        done_fh.write(case + "\n"); done_fh.flush()
        if hits:
            n_flagged += 1
            log.info("FLAG %s: %s", case,
                     ", ".join(f"{h['name']} span {h['span_mm']}mm out {h['outlier_mm']}mm"
                               for h in hits))
        if i % 25 == 0:
            log.info("  %d/%d seen, %d flagged, %d errors", i, len(files), n_flagged, n_err)

    out_fh.close(); done_fh.close()
    log.info("done: %d cases flagged, %d class hits, %d errors -> %s",
             n_flagged, n_hits, n_err, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
