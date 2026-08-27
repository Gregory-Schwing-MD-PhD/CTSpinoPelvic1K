"""qc_hardware_consolidation.py — what the hardware subtraction left behind.

Metal outranks bone in v6, so naming an implant TAKES voxels from whatever structure held
them. On 1003 that is 271,006 voxels out of `femur_right`. The label that remains is
whatever the segmenter had called femur and the metal did not claim, and there is no reason
for that remainder to be a sensible object.

FOUR WAYS IT CAN BE WRONG, and this looks for all four:

  EMPTIED      a structure lost every voxel it had. The label is gone from the file while
               the manifest still says the case has a right femur.
  GUTTED       a structure kept only a few per cent of itself. Present, and no longer the
               thing it is named after.
  SHATTERED    what remains is many disconnected pieces rather than one. A prosthesis
               replacing the middle of a bone leaves a proximal and a distal island, which
               is anatomically honest; twenty islands is not.
  DUST         small isolated remnants -- a few voxels of "femur" stranded inside the
               implant, or clinging to its surface. These are the dangling labels: too
               small to be anatomy, and they will be counted as anatomy by anything that
               measures volumes or fits shapes.

Reports before it changes anything. --apply removes only DUST, and only below a size floor,
because that is the one category where the right answer is not a judgement. Emptied, gutted
and shattered structures are reported for a person to decide about: a femur that is
genuinely half implant SHOULD be a small label, and deleting it would hide the finding.

    python scripts/qc_hardware_consolidation.py --labels data/hardware_final \\
        --original data/v5_final --out qc_hardware/consolidation.csv
    python scripts/qc_hardware_consolidation.py ... --apply --dust-mm3 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

HW_IDS = (76, 77, 78, 79, 80, 81, 82)
NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
        **{v: f"L{v - 19}" for v in range(20, 26)},
        26: "sacrum", 29: "S1", 30: "left_hip", 31: "right_hip",
        32: "femur_left", 33: "femur_right",
        76: "hardware", 77: "cage", 78: "screw_rod", 79: "plate",
        80: "arthroplasty", 81: "si_screw", 82: "osteosynthesis"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="the hardware-coded labels")
    ap.add_argument("--original", required=True, help="the labels before the subtraction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dust-mm3", type=float, default=30.0)
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label_hw.nii.gz"))
    print(f"  {len(files)} hardware-coded label(s); dust floor {a.dust_mm3:.0f} mm3\n")

    rows = []
    for f in files:
        cid = f.name[:4]
        img = nib.load(str(f))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        op = Path(a.original) / f"{cid}_label.nii.gz"
        orig = (np.asanyarray(nib.load(str(op)).dataobj).astype(np.int16)
                if op.exists() else None)

        hw = np.isin(lab, HW_IDS)
        touched = set()
        if orig is not None:
            touched = {int(v) for v in np.unique(orig[hw]) if int(v) and int(v) not in HW_IDS}

        cleaned = lab.copy()
        removed = 0
        for v in sorted(touched):
            now = int((lab == v).sum())
            was = int((orig == v).sum()) if orig is not None else 0
            kept = 100.0 * now / max(was, 1)

            state, detail = "ok", ""
            if now == 0:
                state = "EMPTIED"
                detail = f"lost all {was:,} voxels"
            else:
                cc, n = ndimage.label(lab == v)
                sizes = np.sort(ndimage.sum(lab == v, cc, range(1, n + 1)))[::-1]
                dust = [s for s in sizes if s * vox < a.dust_mm3]
                # MASS BEFORE COUNT. Judging on piece count first called 405 specks
                # around one intact femur "shattered", while reporting in the same
                # line that the largest piece still held 100% of it. A whole object
                # plus a cloud of debris is DUST; a structure genuinely broken into
                # parts is one where no piece holds most of the mass.
                main = 100.0 * sizes[0] / max(now, 1)
                dust_mm3 = float(sum(dust)) * vox
                if kept < 5:
                    state = "GUTTED"
                    detail = f"{kept:.1f}% of it remains ({now:,} of {was:,} voxels)"
                elif main < 90:
                    state = "SHATTERED"
                    detail = (f"largest piece holds only {main:.0f}% across "
                              f"{n} pieces")
                elif dust:
                    state = "DUST"
                    detail = (f"{len(dust)} speck(s), {dust_mm3:.1f} mm3 total, "
                              f"{100.0 * float(sum(dust)) / max(now, 1):.2f}% of the "
                              f"structure; main piece holds {main:.0f}%")
                if dust and a.apply:
                    for i in range(1, n + 1):
                        if ndimage.sum(lab == v, cc, i) * vox < a.dust_mm3:
                            cleaned[cc == i] = 0
                            removed += int((cc == i).sum())

            rows.append({"case": cid, "structure": NAME.get(v, str(v)),
                         "was": was, "now": now, "kept_pct": round(kept, 1),
                         "pieces": (0 if now == 0 else n),
                         "dust_pieces": (0 if now == 0 else len(dust)),
                         "dust_mm3": (0.0 if now == 0 else
                                      round(float(sum(dust)) * vox, 1)),
                         "main_piece_pct": (0.0 if now == 0 else round(main, 1)),
                         "state": state, "detail": detail})
            if state != "ok":
                print(f"  {cid}  {NAME.get(v, v):<14} {state:<10} {detail}")

        if a.apply and removed:
            nib.save(nib.Nifti1Image(cleaned.astype(img.get_data_dtype()), img.affine,
                                     img.header), str(f))
            print(f"  {cid}  removed {removed:,} dust voxels")

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "structure", "was", "now",
                                           "kept_pct", "pieces", "dust_pieces",
                                           "dust_mm3", "main_piece_pct",
                                           "state", "detail"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    print("\n  " + "\n  ".join(f"{n:>4}  {s}" for s, n in
                               Counter(r["state"] for r in rows).most_common()))
    print(f"\n  wrote {dst}")
    if not a.apply:
        print("  (report only -- pass --apply to remove DUST, nothing else)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
