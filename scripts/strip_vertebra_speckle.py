"""scripts/strip_vertebra_speckle.py — remove stray label fragments without touching bone
that the SCAN cut rather than the annotator.

WHY THIS EXISTS. Pseudolabelled spines carry speckle: 0344's L5 is 112 connected
components with the main body holding 99.3% of the voxels. Fragments of L3, L4 and L5 sit
up at z~660 among the ribs, and that is what made a 6th rib appear to articulate with L3 --
the nearest-vertebra search does not care that a candidate is four voxels.

THE RULE, AND THE TRAP IT AVOIDS. The obvious cleanup is "keep the largest connected
component per vertebra". That is wrong, and a hand-annotated case proved it: a vertebra
clipped by the top of the field of view legitimately splits into two pieces, and on 0344's
T6 the larger held only 60%. Largest-component-wins would have deleted 40% of a vertebra a
human had just drawn.

So the discriminator is not size, it is WHY the piece is disconnected:

    touches a face of the volume  ->  the scan cut it. KEEP, at any size.
    small and interior            ->  nobody scanned a four-voxel bone. DROP.

_touches_face already exists in qc_rib_vertebra_incidence and is imported rather than
reimplemented, so the two places that ask this question cannot drift apart.

Dry run by default; --apply writes. Backups go to <out>/pre_speckle/.

    python scripts/strip_vertebra_speckle.py --labels data/v5_final
    python scripts/strip_vertebra_speckle.py --labels data/v5_final --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))
from qc_rib_vertebra_incidence import _touches_face                # noqa: E402

THORACIC_BASE = 7
NAMES = {**{THORACIC_BASE + n: f"T{n}" for n in range(1, 13)},
         **{19 + n: f"L{n}" for n in range(1, 7)},
         26: "sacrum", 29: "S1"}
MIN_KEEP_VOX = 200          # below this AND interior -> speckle


def clean(lab, min_keep):
    """-> (new label array, [dropped records]). Never touches face-touching pieces."""
    out = lab.copy()
    dropped = []
    for vid, nm in NAMES.items():
        m = lab == vid
        if not m.any():
            continue
        cc, ncc = ndimage.label(m)
        if ncc <= 1:
            continue
        sizes = ndimage.sum(m, cc, range(1, ncc + 1))
        keep_big = int(np.argmax(sizes)) + 1
        for i in range(1, ncc + 1):
            if i == keep_big:
                continue
            piece = cc == i
            vox = int(sizes[i - 1])
            if vox >= min_keep or _touches_face(piece):
                continue                      # substantial, or cut by the scan
            out[piece] = 0
            dropped.append({"label": nm, "id": int(vid), "voxels": vox,
                            "z": float(np.argwhere(piece)[:, 2].mean())})
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", default="")
    ap.add_argument("--min-keep", type=int, default=MIN_KEEP_VOX)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="qc_speckle")
    a = ap.parse_args()

    labdir = Path(a.labels)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    backup = out / "pre_speckle"

    report, n_changed, n_vox = [], 0, 0
    for stem in stems:
        fp = labdir / f"{stem}_label.nii.gz"
        if not fp.exists():
            continue
        img = nib.load(str(fp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        new, dropped = clean(lab, a.min_keep)
        if not dropped:
            continue
        n_changed += 1
        v = sum(d["voxels"] for d in dropped)
        n_vox += v
        by = {}
        for d in dropped:
            by.setdefault(d["label"], []).append(d["voxels"])
        pretty = "  ".join(f"{k}x{len(vv)}({sum(vv)}vox)" for k, vv in sorted(by.items()))
        print(f"  {stem}: {len(dropped)} fragment(s), {v} vox   {pretty[:90]}")
        report.append({"case": stem, "n_fragments": len(dropped), "voxels": v,
                       "dropped": dropped})
        if a.apply:
            # only the vertebra ids may change, and only downward (fragments -> background)
            diff = {int(x) for x in np.unique(lab[lab != new])}
            assert diff <= set(NAMES), f"{stem}: non-vertebra ids changed: {sorted(diff)}"
            assert (new > 0).sum() < (lab > 0).sum(), f"{stem}: nothing removed"
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / fp.name).exists():
                shutil.copy2(fp, backup / fp.name)
            nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), img.affine,
                                     img.header), str(fp))

    p = out / "speckle_report.json"
    p.write_text(json.dumps({"labels": str(labdir), "applied": a.apply,
                             "min_keep_vox": a.min_keep,
                             "cases_changed": n_changed, "voxels_removed": n_vox,
                             "cases": report}, indent=1))
    print(f"\n  {n_changed} case(s) with speckle, {n_vox} voxel(s) total")
    print("  " + ("APPLIED" if a.apply else "DRY RUN -- pass --apply to write"))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
