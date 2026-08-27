"""apply_hardware_class.py — write the reviewed class into the label, for the cases that are real.

seed_hardware.py proposed a class from geometry, and geometry could not name most of what is
here: a femoral stem came back as generic `hardware` because the subtype block has no
arthroplasty class, and would have come back as `screw_rod` if it had been allowed to guess,
since a stem is long and thin and that is all "linear" tests.

The manifest carries the reviewed answer. This applies it: every metal voxel in a confirmed
case takes the reviewed class, and nothing is written for a case read as artefact.

METAL OUTRANKS BONE, as everywhere else in this pipeline. A bone segmenter absorbs an
implant it can see, so on these cases the metal is already inside a vertebra, hip or femur
label and naming it is a subtraction rather than an addition. The vertebra or the femur loses
those voxels; nothing is deleted, and the count is reported per structure so the loss is
visible rather than silent.

Writes a label for ITK-SNAP review and nothing else. v5 is opened read-only.

    python scripts/apply_hardware_class.py --manifest hardware_review/hardware_manifest.csv \\
        --proposals data/hardware_fix --labels data/v5_final --out data/hardware_final
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
        **{v: f"L{v - 19}" for v in range(20, 26)},
        26: "sacrum", 29: "S1", 30: "left_hip", 31: "right_hip",
        32: "femur_left", 33: "femur_right"}
HW_CLASS = {76: "hardware", 77: "hardware_cage", 78: "hardware_screw_rod",
            79: "hardware_plate", 80: "hardware_arthroplasty", 81: "hardware_si_screw",
            82: "hardware_osteosynthesis"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", default="",
                    help="path to a label to use verbatim for one case, as CASE=path")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    over = {}
    for spec in filter(None, a.override.split(",")):
        cid, _, path = spec.partition("=")
        over[cid.strip()] = Path(path.strip())

    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["verdict"] == "instrumentation"]
    print(f"  {len(rows)} confirmed case(s)\n")

    report = []
    for r in rows:
        cid = r["case"]
        cls = int(r["class_id"])
        lp = Path(a.labels) / f"{cid}_label.nii.gz"
        mp = Path(a.proposals) / f"{cid}_hardware_only.nii.gz"

        if cid in over:
            src = over[cid]
            if not src.exists():
                print(f"  ! {cid}: override {src} missing")
                continue
            img = nib.load(str(src))
            new = np.asanyarray(img.dataobj).astype(np.int16)
            ids = sorted(int(v) for v in np.unique(new) if int(v) in HW_CLASS)
            nib.save(nib.Nifti1Image(new, img.affine, img.header),
                     str(out / f"{cid}_label_hw.nii.gz"))
            print(f"  {cid}  used the reviewed label verbatim "
                  f"({', '.join(HW_CLASS[i] for i in ids) or 'no hardware ids'})")
            report.append({"case": cid, "class": HW_CLASS.get(cls, cls),
                           "source": "reviewed label", "taken_from": {}})
            continue

        if not lp.exists() or not mp.exists():
            print(f"  ! {cid}: label or mask missing")
            continue
        img = nib.load(str(lp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        hw = np.asanyarray(nib.load(str(mp)).dataobj).astype(np.int16) > 0
        if hw.shape != lab.shape:
            print(f"  ! {cid}: shape mismatch")
            continue

        taken = {}
        for v in np.unique(lab[hw]):
            if int(v) == 0:
                continue
            taken[NAME.get(int(v), str(int(v)))] = int((lab[hw] == v).sum())
        new = lab.copy()
        new[hw] = cls
        # the label gains exactly the mask voxels that were background, and loses nothing:
        # a metal voxel already inside a structure changes id, it does not disappear
        added = int((hw & (lab == 0)).sum())
        assert (new > 0).sum() == (lab > 0).sum() + added, "voxels lost"

        nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), img.affine, img.header),
                 str(out / f"{cid}_label_hw.nii.gz"))
        print(f"  {cid}  {HW_CLASS[cls]:<24} {int(hw.sum()):>8,} voxels; "
              f"taken from {taken or 'background only'}")
        report.append({"case": cid, "class": HW_CLASS[cls], "voxels": int(hw.sum()),
                       "source": "mask + reviewed class", "taken_from": taken})

    (out / "applied.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {len(report)} label(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
