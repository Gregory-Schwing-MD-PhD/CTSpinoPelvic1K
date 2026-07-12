"""
scripts/postprocess_halo.py — deterministic HALO cleanup for corrected rib labels.

THE DEFECT. When an annotator renumbers a rib (e.g. paints rib_left_6 as rib_left_7), a thin residue
of the OLD label survives on the surface of the NEW one. Measured across every occurrence in the
corpus, that residue is unmistakable: a fragment of rib N, under ~2000 voxels, FUSED TO rib N+-1,
sitting >100 mm from the spine. It is not a rib; it is halo.

THE FIX. The correct (neighbouring) class engulfs it: relabel the fragment to the adjacent rib it is
fused to. Two independent signals must BOTH hold -- size AND fused-to-an-adjacent-number -- so a
small-but-real rib is never silently absorbed.

WHY IT LIVES HERE, NOT IN THE LEDGER. The annotator's submitted label is the PRIMARY RECORD: it is
what the human actually drew, and it must stay immutable for inter-rater agreement, the data
descriptor, and auditability. So this is a *build step*, not an edit: raw annotation -> deterministic,
versioned post-processing -> release label. Same input + same commit = same output, always.

REPRODUCIBILITY. Every run writes a MANIFEST (CSV) with one row per engulfed fragment -- case, slot,
rib, voxels, the neighbour that absorbed it -- plus the git commit. That manifest is the auditable
answer to "how many examples were cleaned, and which?"; no number needs to be remembered.

  python scripts/postprocess_halo.py IN.nii.gz --out OUT.nii.gz [--manifest halo.csv] [--dry-run]
  python scripts/postprocess_halo.py --dir LABELS/ --out-dir CLEAN/ --manifest halo.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS          # noqa: E402
import review_anatomy_qc as RA     # noqa: E402   (single source of truth for the halo rule)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_HERE.parent), text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def engulf_halo(lab: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """Relabel every halo speck into the adjacent rib it is fused to. Pure: returns (cleaned, log).
    A fragment qualifies only if it is BOTH small (< RA.HALO_MAX_VOX) AND fused to rib N+-1 -- the
    same rule the QC uses to stop blocking on it, so the gate and the cleanup can never disagree."""
    out = lab.copy()
    log: List[dict] = []
    objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))
    lo_id, hi_id = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
    for rid in range(lo_id, hi_id + 1):
        o = objs[rid - 1] if rid - 1 < len(objs) else None
        if o is None:
            continue
        if not RA.is_halo_speck(lab, rid, o):
            continue
        # which adjacent rib absorbs it: the neighbour it touches most
        pad = tuple(slice(max(0, o[i].start - 3), min(lab.shape[i], o[i].stop + 3)) for i in range(3))
        sub = lab[pad]
        m = (sub == rid)
        dil = ndimage.binary_dilation(m, iterations=2)
        adj = RA._adjacent_rib_ids(rid)
        touched = sub[dil & (sub > 0) & (sub != rid)]
        # Only a REAL rib may absorb the halo (never a fellow fragment -- otherwise two small pieces
        # would engulf each other and the result would depend on iteration order).
        cand = [int(v) for v in touched
                if int(v) in adj and int((lab == int(v)).sum()) >= RA.HALO_MAX_VOX]
        if not cand:
            continue
        winner = int(np.bincount(np.asarray(cand)).argmax())     # most-contacted real neighbour
        n = int(m.sum())
        out[pad][m] = winner                                     # the correct class engulfs the halo
        names = RA._id2name()
        log.append({"rib": names.get(rid, rid), "rib_id": rid, "voxels": n,
                    "engulfed_by": names.get(winner, winner), "engulfed_by_id": winner})
    return out, log


def _process(path: Path, out_path: Path, dry: bool) -> List[dict]:
    import nibabel as nib
    img = nib.load(str(path))
    lab = np.asanyarray(img.dataobj)
    cleaned, log = engulf_halo(lab)
    if not dry and log:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(cleaned.astype(lab.dtype), img.affine, img.header), str(out_path))
    elif not dry:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(cleaned.astype(lab.dtype), img.affine, img.header), str(out_path))
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?", help="a single label .nii.gz")
    ap.add_argument("--out", default=None, help="output label (single-file mode)")
    ap.add_argument("--dir", default=None, help="directory of labels to clean")
    ap.add_argument("--out-dir", default=None, help="output directory (--dir mode)")
    ap.add_argument("--manifest", default="halo_manifest.csv",
                    help="CSV audit trail: one row per engulfed fragment")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args(argv)

    commit = _git_commit()
    rows: List[dict] = []
    files: List[Tuple[Path, Path]] = []
    if a.dir:
        src = Path(a.dir)
        dst = Path(a.out_dir or (str(src) + "_clean"))
        files = [(p, dst / p.name) for p in sorted(src.glob("*.nii.gz"))]
    elif a.label:
        files = [(Path(a.label), Path(a.out or "clean.nii.gz"))]
    else:
        ap.error("give a label file or --dir")

    n_files = n_engulfed = n_vox = 0
    for src_p, dst_p in files:
        log = _process(src_p, dst_p, a.dry_run)
        if log:
            n_files += 1
        for e in log:
            e.update(case=src_p.stem.replace(".nii", ""), commit=commit)
            rows.append(e)
            n_engulfed += 1
            n_vox += e["voxels"]
        print(f"{src_p.name}: {len(log)} halo fragment(s) engulfed"
              f"{' [dry-run]' if a.dry_run else ''}")

    if rows and not a.dry_run:
        mf = Path(a.manifest)
        with mf.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "rib", "rib_id", "voxels",
                                              "engulfed_by", "engulfed_by_id", "commit"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nmanifest -> {mf}")
    print(f"\nSUMMARY (commit {commit}): {n_engulfed} halo fragment(s), {n_vox:,} voxels, "
          f"across {n_files}/{len(files)} label(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
