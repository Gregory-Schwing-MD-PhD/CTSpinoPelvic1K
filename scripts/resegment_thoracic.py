"""resegment_thoracic.py — segment the thoracic levels a case images but does not label.

WHY THIS EXISTS. 0068 carries L1-L5, the sacrum, S1, both hips, both femurs and twelve
labelled ribs, and not one thoracic vertebra -- while 48 mm of vertebral column sits imaged
above L1's superior endplate. The thoracic was deferred at annotation time because the case
carries a lumbosacral construct; the metal turns out to span -322 to -23 mm relative to L1's
top and NOT ONE VOXEL of it is above L1, so the segment that needs drawing is clean.

TOTALSEGMENTATOR SEPARATES THE VERTEBRAE. IT DOES NOT NAME THEM.
This is the whole design. TS numbers vertebrae by counting down from the top of what it can
see, and on an FOV-limited abdominal scan the top is not the top of the spine -- it is
wherever the technologist started. Its labels here are an inference from a count that cannot
be made, which is the failure this dataset exists to document; adopting them would write
that failure into the release.

What TS is good at, and is used for, is telling one vertebra from the next: where a body
ends, the disc, where the next begins. That is a local judgement and it does not need a
count. The IDENTITY then comes from below, where it is already known: L1 is labelled, so
the vertebra immediately above L1 is T12, the one above that T11, and so on. Nothing is
counted from a vertebra nobody can see.

TWO RULES THAT ARE NOT NEGOTIABLE.
  * Additions land on BACKGROUND ONLY. Existing v5 voxels are never overwritten, moved or
    reclassified. This script cannot damage a label it did not create; the earlier automated
    passes that gave one vertebra's spinous process to its neighbour were exactly what this
    forbids.
  * NEVER REORIENT WHEN WRITING. The superior direction is read off the affine and the work
    is done in the label's own array order. Canonicalising to RAS and saving silently
    transposes a label away from the CT it belongs to.

Output goes to a PROPOSAL path, never over v5. It is a starting point for correction in
ITK-SNAP, not a release artefact.

    python scripts/resegment_thoracic.py --case 0068 \
        --ct data/hf_export_v5/ct/0068_ct.nii.gz \
        --label data/v5_final/0068_label.nii.gz \
        --out data/thoracic_fix
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

log = logging.getLogger("resegment_thoracic")

# v5 is VerSe-native: C1-C7 = 1-7, T1-T12 = 8-19, L1-L6 = 20-25.
V5_T1, V5_T12 = 8, 19
V5_L1 = 20
V5_NAME = {v: f"T{v - V5_T1 + 1}" for v in range(V5_T1, V5_T12 + 1)}

# every vertebra class TS knows, so the run does not depend on guessing which levels are in
# view -- the ones that are not simply come back empty
TS_VERTEBRAE = ([f"vertebrae_C{i}" for i in range(1, 8)]
                + [f"vertebrae_T{i}" for i in range(1, 13)]
                + [f"vertebrae_L{i}" for i in range(1, 6)]
                + ["vertebrae_S1"])


def superior_axis(affine: np.ndarray) -> tuple[int, int]:
    """(array axis that runs most superiorly, +1 if increasing index goes superior).

    Read from the affine rather than assumed. These CTs sit on disk as ('P','I','R'), so
    axis 2 is not superior and axis 1 runs the wrong way; a hard-coded axis would quietly
    graft the new levels onto the bottom of the scan.
    """
    col = affine[2, :3]                       # world +z (superior) per array axis
    ax = int(np.argmax(np.abs(col)))
    return ax, int(np.sign(col[ax]) or 1)


def sup_index(mask: np.ndarray, ax: int, sgn: int) -> int:
    """Index, along `ax`, of the most SUPERIOR slice the mask occupies."""
    idx = np.where(mask.any(axis=tuple(i for i in range(3) if i != ax)))[0]
    return int(idx.max() if sgn > 0 else idx.min())


def is_superior(i: np.ndarray | int, ref: int, sgn: int) -> np.ndarray | bool:
    return (i > ref) if sgn > 0 else (i < ref)


def run_ts(ct_path: Path, device: str):
    """TS 'total', every vertebra class, on the CT's own grid."""
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map

    name_to_ts = {n: i for i, n in class_map["total"].items()}
    valid = [n for n in TS_VERTEBRAE if n in name_to_ts]
    pred = totalsegmentator(input=nib.load(str(ct_path)), output=None, task="total",
                            ml=True, device=device, roi_subset=valid, verbose=False)
    arr = np.asarray(pred.dataobj).astype(np.int32)
    name_val = {n: name_to_ts[n] for n in valid}
    present = set(int(v) for v in np.unique(arr)) - {0}
    if present and not (present & set(name_val.values())):
        # TS compacts the ids when roi_subset is used on some versions
        name_val = {n: k for k, n in enumerate(valid, start=1)}
    return arr, name_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--min-voxels", type=int, default=400,
                    help="ignore a TS vertebra smaller than this above L1 (FOV specks)")
    ap.add_argument("--highest", default="T8",
                    help="do not name anything above this level; a run that wants to go "
                         "higher than the FOV-limited GT convention has to say so")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lab_img = nib.load(a.label)
    lab = np.asanyarray(lab_img.dataobj).astype(np.int32)     # NATIVE order, not canonical
    ax, sgn = superior_axis(lab_img.affine)
    zmm = float(np.linalg.norm(lab_img.affine[:3, ax]))
    log.info(f"{a.case}: {lab.shape}, axcodes {nib.aff2axcodes(lab_img.affine)}, "
             f"superior = axis {ax} {'+' if sgn > 0 else '-'}, {zmm:.2f} mm/slice")

    if not (lab == V5_L1).any():
        log.error("no L1 in this label -- identity cannot be anchored from below, refusing")
        return 2
    already = sorted(int(v) for v in np.unique(lab) if V5_T1 <= v <= V5_T12)
    if already:
        log.warning(f"thoracic labels already present: {[V5_NAME[v] for v in already]}")

    z_l1 = sup_index(lab == V5_L1, ax, sgn)
    n = lab.shape[ax]
    z_end = (n - 1) if sgn > 0 else 0
    log.info(f"L1's superior slice is index {z_l1}; the scan ends at {z_end} "
             f"({abs(z_end - z_l1) * zmm:.0f} mm above it)")

    ts, name_val = run_ts(Path(a.ct), a.device)
    if ts.shape != lab.shape:
        log.error(f"TS returned {ts.shape} against a label of {lab.shape}; refusing to graft")
        return 3

    # index of every voxel along the superior axis, broadcast to the volume
    shape_i = [1, 1, 1]
    shape_i[ax] = n
    zi = np.arange(n).reshape(shape_i)
    above = np.broadcast_to(is_superior(zi, z_l1, sgn), lab.shape)
    free = (lab == 0) & above                       # background, above L1: the only place
                                                    # this script is ever allowed to write

    # --- order TS's vertebrae by how far above L1 they sit -----------------------------
    found = []
    for name, val in name_val.items():
        m = (ts == val) & free
        cnt = int(m.sum())
        if cnt < a.min_voxels:
            continue
        idx = np.where(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        found.append({"ts_name": name, "voxels": cnt,
                      "lo": int(idx.min()), "hi": int(idx.max()),
                      "centre": float(idx.mean())})
    if not found:
        log.error("TS found no vertebra above L1 -- nothing to graft")
        return 4
    # nearest to L1 first
    found.sort(key=lambda d: d["centre"], reverse=(sgn < 0))

    highest = a.highest.upper()
    floor_id = V5_T1 + int(highest[1:]) - 1
    add = np.zeros_like(lab)
    rows = []
    for k, d in enumerate(found):
        v5 = V5_T12 - k
        if v5 < floor_id:
            log.warning(f"  stopping: the next level up would be {V5_NAME.get(v5, v5)}, "
                        f"above the --highest {highest} convention")
            break
        m = (ts == name_val[d["ts_name"]]) & free
        add[m] = v5
        touches = (d["hi"] == z_end) if sgn > 0 else (d["lo"] == z_end)
        rows.append({**d, "assigned": V5_NAME[v5], "v5_id": v5,
                     "mm_above_L1": round(abs(d["centre"] - z_l1) * zmm, 1),
                     "extent_mm": round((d["hi"] - d["lo"] + 1) * zmm, 1),
                     "truncated_by_fov": bool(touches)})
        flag = "  TRUNCATED at the edge of the scan" if touches else ""
        log.info(f"  {V5_NAME[v5]:<4} <- TS {d['ts_name']:<14} {d['voxels']:>8,} vox  "
                 f"centre {rows[-1]['mm_above_L1']:>5.1f} mm above L1  "
                 f"height {rows[-1]['extent_mm']:>5.1f} mm{flag}")
        if d["ts_name"] != f"vertebrae_{V5_NAME[v5]}":
            log.info(f"       (TS called it {d['ts_name']}; the name is discarded, the "
                     f"separation is kept)")

    merged = lab.copy()
    merged[add > 0] = add[add > 0]                  # `add` only ever lives on background
    assert (merged[lab > 0] == lab[lab > 0]).all(), "an existing v5 voxel was modified"

    # SAME affine, SAME header, SAME array order as the label that came in
    dst = out / f"{a.case}_label_proposed.nii.gz"
    nib.save(nib.Nifti1Image(merged.astype(lab_img.get_data_dtype()),
                             lab_img.affine, lab_img.header), str(dst))
    meta = {"case": a.case, "superior_axis": ax, "superior_sign": sgn,
            "mm_per_slice": zmm, "l1_superior_index": z_l1, "scan_end_index": z_end,
            "identity": "anchored on the labelled L1 below; TS names discarded",
            "levels": rows, "added_voxels": int((add > 0).sum())}
    (out / f"{a.case}_thoracic.json").write_text(json.dumps(meta, indent=1) + "\n")
    log.info(f"wrote {dst}  (+{meta['added_voxels']:,} voxels, {len(rows)} level(s))")
    log.info("this is a PROPOSAL -- correct it in ITK-SNAP before it goes anywhere near v5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
