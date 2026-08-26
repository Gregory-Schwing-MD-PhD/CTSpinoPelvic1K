"""scripts/seed_hardware.py — turn detected instrumentation into a labelled seed to correct.

scan_hardware.py finds the metal and REPORTS it; its docstring says the subtype call is
"left to a reader, because a threshold cannot tell a cage from a plate lying against a
body". That is true of a threshold. It is not true of the component's SHAPE, which that
script already computes and then throws away, so a reader is being asked to do by hand a
call that the geometry mostly makes on its own.

WHAT SEPARATES THEM IS WHERE THEY SIT, then how long and thin they are:

    NOT AN IMPLANT AT ALL   more than 15 mm from any labelled bone. These are colonography
                            series and the stool is tagged with oral contrast that saturates
                            the scanner exactly as titanium does; on 0068, seventeen of the
                            nineteen dense components in the volume are 24-75 mm out in the
                            colon. This test comes first because the search shell does not
                            exclude them.
    a screw or a rod        LINEAR and POSTERIOR: a screw enters through the pedicle, a rod
                            lies behind the lamina. Long against its own width.
    an interbody cage       in the BODY COLUMN, between the endplates and anterior to the
                            canal, and big enough to be an implant rather than a fleck.
    a plate                 PLANAR -- two broad axes and a thin one -- against a body.

Bridging an interspace is recorded but is no longer what makes a cage a cage: a screw whose
tip reaches the endplate touches the vertebra above it too.

MEASURED IN MILLIMETRES, not in singular values. A singular value scales with the number of
voxels in the component and cannot be compared with anything; the extent along each
principal direction is a length, and a length can be checked against the sizes implants
actually come in.

THIS IS A SEED, NOT AN ANSWER. Every component is written with the class the geometry
proposes and a reason recorded beside it, so a reader is confirming or overruling a specific
claim rather than starting from an empty mask. Threshold blooming makes the seed run a
little wide of the true implant surface, which is the right direction to be wrong in when
the next step is a human eraser.

TWO THINGS IT WILL NOT DO.
  * It writes on BACKGROUND ONLY. A pedicle screw lies inside a vertebra that already has a
    label, and whether metal should take precedence over bone is a release convention, not
    a decision for a script -- so the overlap is measured and reported, never applied.
  * It never reorients. Work is done in the label's own array order with the superior axis
    read off the affine.

    python scripts/seed_hardware.py --case 0068 \
        --ct data/hf_export_v5/ct/0068_ct.nii.gz \
        --label data/v5_final/0068_label.nii.gz --out data/hardware_fix
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

log = logging.getLogger("seed_hardware")

# kept identical to scan_hardware.py: the detection is the same detection
METAL_HU = 1800.0
DILATE_MM = 12.0
MIN_VOX = 40

THORACIC = list(range(8, 20))
LUMBAR = list(range(20, 26))
SACRAL = [26, 29]
SPINE_IDS = THORACIC + LUMBAR + SACRAL

HW_GENERIC, HW_CAGE, HW_SCREW_ROD, HW_PLATE = 76, 77, 78, 79
HW_NAME = {76: "hardware", 77: "hardware_cage", 78: "hardware_screw_rod", 79: "hardware_plate"}

VERT_NAME = {**{v: f"T{v - 7}" for v in THORACIC},
             **{v: f"L{v - 19}" for v in LUMBAR}, 26: "sacrum", 29: "S1"}


# distances and sizes the calls turn on, all in mm
OFF_BONE_MM = 15.0      # further than this from any labelled bone and it is bowel, not metal
LINEAR = 3.0            # length against width, above which a thing is a screw or a rod
PLANAR = 3.0            # width against thickness, above which it is a plate
CAGE_MIN_MM = 15.0      # an implant that spans a disc space is at least this long


def classify(ext, dist_mm, in_body_column):
    """(v5 id, reason) from where the component sits and how big it is.

    `ext` is the physical extent in mm along each of the component's own principal
    directions, longest first. `dist_mm` is the shortest distance from the component to any
    labelled bone. `in_body_column` says whether its centre lies anterior of the back of the
    vertebral bodies, which is where a disc space is and where a cage must therefore be.
    """
    length, width, thick = [float(x) for x in ext]
    if dist_mm > OFF_BONE_MM:
        return None, (f"{dist_mm:.0f} mm from the nearest labelled bone -- tagged stool "
                      f"or contrast, not instrumentation")
    if length / max(width, 1e-6) >= LINEAR:
        return HW_SCREW_ROD, (f"linear on bone, {length:.0f} mm long against {width:.0f} mm "
                              f"wide" + ("" if in_body_column else ", and posterior to the "
                                         "vertebral bodies"))
    if width / max(thick, 1e-6) >= PLANAR:
        return HW_PLATE, (f"planar: {length:.0f} x {width:.0f} mm broad and only "
                          f"{thick:.0f} mm thick")
    if in_body_column and length >= CAGE_MIN_MM:
        return HW_CAGE, (f"a {length:.0f} x {width:.0f} x {thick:.0f} mm block in the body "
                         f"column, between the endplates and anterior to the canal")
    if not in_body_column:
        return HW_SCREW_ROD, (f"posterior to the vertebral bodies, where screws and rods "
                              f"are and cages are not")
    return HW_GENERIC, (f"on bone, {length:.0f} x {width:.0f} x {thick:.0f} mm, no shape "
                        f"or place that names it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hu", type=float, default=METAL_HU)
    ap.add_argument("--min-voxels", type=int, default=MIN_VOX)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lab_img = nib.load(a.label)
    lab = np.asanyarray(lab_img.dataobj).astype(np.int16)     # NATIVE order
    ct = np.asanyarray(nib.load(a.ct).dataobj)
    if ct.shape != lab.shape:
        log.error(f"CT {ct.shape} against label {lab.shape}; refusing")
        return 2
    sp = np.array(lab_img.header.get_zooms()[:3], float)
    log.info(f"{a.case}: {lab.shape}, {sp.round(3)} mm, "
             f"axcodes {nib.aff2axcodes(lab_img.affine)}")

    spine = np.isin(lab, SPINE_IDS)
    if not spine.any():
        log.error("no spine labels; the search region cannot be built")
        return 3

    # CROP BEFORE DILATING. The shell is the spine grown by 12 mm; growing it across the
    # whole 512-cubed volume allocates and scans a hundred and fifty million voxels fifteen
    # times over for a region that occupies a fraction of the frame.
    pad = np.ceil(DILATE_MM / sp).astype(int) + 2
    idx = np.argwhere(spine)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.array(lab.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    log.info(f"spine bounding box {tuple(int(h - l) for l, h in zip(lo, hi))} "
             f"of {lab.shape}")

    spine_c, lab_c, ct_c = spine[sl], lab[sl], ct[sl]
    it = max(1, int(round(DILATE_MM / max(sp.min(), 1e-6))))
    region = ndimage.binary_dilation(spine_c, iterations=it)
    bright = (ct_c > a.hu) & region
    log.info(f"metal threshold {a.hu:.0f} HU inside a {DILATE_MM:.0f} mm shell: "
             f"{int(bright.sum()):,} voxels")
    # WHY THE SHELL MATTERS HERE. These are CT colonography series: the stool is tagged with
    # oral contrast that saturates just like metal does. Thresholding the whole volume finds
    # it in quantity and calls it instrumentation.
    whole = int((ct > a.hu).sum())
    log.info(f"  (the same threshold over the WHOLE volume hits {whole:,} voxels -- "
             f"{whole - int(bright.sum()):,} of them are not near the spine)")
    if not bright.any():
        log.info("no instrumentation found")
        return 0

    # distance to the nearest labelled BONE, in mm. This is what drops the tagged stool:
    # the search shell reaches 12 mm out from the spine and a loop of tagged colon lying
    # against the psoas gets inside it, saturating the scanner just as titanium does.
    bone_c = lab_c > 0
    dist_c = ndimage.distance_transform_edt(~bone_c, sampling=sp)

    cc, ncc = ndimage.label(bright)
    seed_c = np.zeros_like(lab_c)
    rows = []
    for i in range(1, ncc + 1):
        m = cc == i
        vox = int(m.sum())
        if vox < a.min_voxels:
            continue
        near = ndimage.binary_dilation(m, iterations=max(1, it // 4))
        touched = sorted({int(v) for v in np.unique(lab_c[near]) if v in SPINE_IDS})

        # world mm, so the extents are lengths and "anterior" is anterior
        idx = np.argwhere(m) + np.array([sl[k].start for k in range(3)])
        w = (lab_img.affine @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
        q = w - w.mean(0)
        vt = np.linalg.svd(q, full_matrices=False)[2]
        ext = [float((q @ vt[k]).max() - (q @ vt[k]).min()) for k in range(3)]

        # the body column, from the vertebrae THIS component touches. Over the whole spine
        # the anterior-posterior range is a lordosis, not a vertebra, and the back of the
        # body column computed from it means nothing.
        in_body = False
        if touched:
            vm = np.isin(lab, touched)
            vw = (lab_img.affine @ np.c_[np.argwhere(vm),
                                         np.ones(int(vm.sum()))].T).T[:, 1]
            back = vw.max() - 0.60 * (vw.max() - vw.min())
            in_body = bool(w[:, 1].mean() > back)

        hw, why = classify(ext, float(dist_c[m].min()), in_body)
        rows.append({"voxels": vox, "mm3": round(vox * float(np.prod(sp)), 1),
                     "touches": [VERT_NAME.get(t, str(t)) for t in touched],
                     "extent_mm": [round(x, 1) for x in ext],
                     "dist_to_bone_mm": round(float(dist_c[m].min()), 1),
                     "in_body_column": in_body,
                     "proposed": HW_NAME.get(hw, "not instrumentation"),
                     "v5_id": hw, "why": why})
        if hw is None:                       # rejected: never written, but kept in the report
            log.info(f"  {vox:>6,} vox  -- skipped: {why}")
            continue
        seed_c[m] = hw
        log.info(f"  {vox:>6,} vox  -> {HW_NAME[hw]:<19} "
                 f"touches {','.join(rows[-1]['touches']) or 'nothing'}  ({why})")
    rows = [r for r in rows if r["v5_id"] is not None] or []
    if not rows:
        log.info(f"nothing above {a.min_voxels} voxels -- beam-hardening speckle only")
        return 0

    seed = np.zeros_like(lab)
    seed[sl] = seed_c

    # what would be lost by writing on background only
    clash = (seed > 0) & (lab > 0)
    n_clash = int(clash.sum())
    if n_clash:
        who = {VERT_NAME.get(int(v), str(int(v))): int((lab[clash] == v).sum())
               for v in np.unique(lab[clash])}
        log.warning(f"{n_clash:,} of {int((seed > 0).sum()):,} hardware voxels "
                    f"({100.0 * n_clash / int((seed > 0).sum()):.1f}%) fall inside an "
                    f"existing label: {who}")
        log.warning("  those are NOT written. Whether metal outranks bone is a release "
                    "convention, so this script reports the overlap and leaves it.")

    hdr = lab_img.header
    merged = lab.copy()
    free = (seed > 0) & (lab == 0)
    merged[free] = seed[free]
    assert (merged[lab > 0] == lab[lab > 0]).all(), "an existing v5 voxel was modified"
    nib.save(nib.Nifti1Image(merged.astype(lab_img.get_data_dtype()), lab_img.affine, hdr),
             str(out / f"{a.case}_label_hardware.nii.gz"))

    # THE FILE THAT IS ACTUALLY USEFUL WHEN THE METAL IS ALREADY LABELLED AS BONE.
    # A segmenter handed a bright implant against an endplate absorbs it into the vertebra,
    # so on an instrumented case the background-only file above can be an exact copy of its
    # input. Naming the hardware then means taking voxels back from the vertebrae that hold
    # them, which is a different kind of edit and gets a different file: reviewable on its
    # own, and undone by deleting it.
    taken = {}
    if n_clash:
        re_lab = lab.copy()
        hit = seed > 0
        for v in np.unique(lab[clash]):
            taken[VERT_NAME.get(int(v), str(int(v)))] = int((lab[clash] == v).sum())
        re_lab[hit] = seed[hit]
        nib.save(nib.Nifti1Image(re_lab.astype(lab_img.get_data_dtype()), lab_img.affine,
                                 hdr), str(out / f"{a.case}_label_hardware_reassigned.nii.gz"))
        log.info(f"also wrote {a.case}_label_hardware_reassigned.nii.gz -- metal outranks "
                 f"bone, taking {taken} back from the vertebrae that hold them")

    nib.save(nib.Nifti1Image(seed.astype(lab_img.get_data_dtype()), lab_img.affine, hdr),
             str(out / f"{a.case}_hardware_only.nii.gz"))
    meta = {"case": a.case, "hu": a.hu, "components": rows,
            "written_on_background": int(free.sum()),
            "overlapping_existing_labels": n_clash,
            "taken_from": taken,
            "note": "proposed classes from component shape; confirm or overrule in ITK-SNAP"}
    (out / f"{a.case}_hardware.json").write_text(json.dumps(meta, indent=1) + "\n")
    log.info(f"wrote {a.case}_label_hardware.nii.gz (+{int(free.sum()):,} voxels on "
             f"background) and the hardware-only mask beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
