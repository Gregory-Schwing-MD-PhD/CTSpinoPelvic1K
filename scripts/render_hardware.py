"""render_hardware.py — look at the instrumentation before naming it.

The shape test in seed_hardware.py called both components of 0068 compact and therefore
cages. Compactness was measured as the ratio of the first two singular values of the voxel
cloud, and that number is honest but the wrong question: a screw-rod-screw construct is a Z,
not a line, so PCA on the whole assembly reads as compact even though every piece of it is
long and thin. A ratio near 2 does not distinguish a cage from a bent construct.

WHAT ACTUALLY SEPARATES THEM IS WHERE THEY SIT, not how round they are:

    an interbody cage    lies in the DISC SPACE, between the two endplates, anterior to
                         the canal, inside the footprint of the vertebral body
    a pedicle screw      enters POSTERIORLY through the pedicle and runs forward into the
                         body; it is lateral to the midline and it crosses the plane of
                         the canal
    a rod                lies posterior to everything, lateral, and spans levels

So this measures the anterior-posterior position of each component against the vertebral
body and the canal, its true physical length along its own principal axis, and its
thickness -- and renders it, because a picture settles what a table of ratios argues about.

READ ONLY. It writes images and a table; it does not touch a label.

    python scripts/render_hardware.py --case 0068 \
        --ct thoracic_fix/0068/0068_ct.nii.gz \
        --label thoracic_fix/0068/0068_label_proposed.nii.gz \
        --hardware thoracic_fix/0068/hu2500/0068_hardware_only.nii.gz \
        --out qc_hardware_render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LUMBAR = list(range(20, 26))
VERT_NAME = {**{v: f"T{v - 7}" for v in range(8, 20)},
             **{v: f"L{v - 19}" for v in LUMBAR}, 26: "sacrum", 29: "S1"}


def axes_of(affine):
    """(array axis, sign) for world +x(R), +y(A), +z(S) — read, never assumed."""
    out = []
    for w in range(3):
        col = affine[w, :3]
        ax = int(np.argmax(np.abs(col)))
        out.append((ax, int(np.sign(col[ax]) or 1)))
    return out                                   # [(R...), (A...), (S...)]


def world_of(idx, affine):
    """voxel indices (N,3) -> world mm (N,3)."""
    h = np.concatenate([idx, np.ones((len(idx), 1))], 1)
    return (affine @ h.T).T[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lab_img = nib.load(a.label)
    lab = np.asanyarray(lab_img.dataobj).astype(np.int16)
    hw = np.asanyarray(nib.load(a.hardware).dataobj).astype(np.int16)
    ct = np.asanyarray(nib.load(a.ct).dataobj)
    aff = lab_img.affine
    sp = np.array(lab_img.header.get_zooms()[:3], float)
    print(f"{a.case}: {lab.shape}, axcodes {nib.aff2axcodes(aff)}")

    # ---- reference frame from the anatomy, in WORLD mm ------------------------------
    # the canal is not labelled, so the posterior edge of the vertebral BODY is the
    # landmark: anything anterior to it is in the disc/body column, anything posterior to
    # it is in the pedicle/lamina/rod territory
    ref = {}
    for v in (23, 24):                                   # L4, L5
        m = lab == v
        if not m.any():
            continue
        w = world_of(np.argwhere(m), aff)
        ref[v] = {"centre": w.mean(0), "min": w.min(0), "max": w.max(0)}
        print(f"  {VERT_NAME[v]}: centre {w.mean(0).round(1)}, "
              f"A-P span {w[:, 1].min():.0f}..{w[:, 1].max():.0f} mm")
    if 23 not in ref or 24 not in ref:
        print("  ! need both L4 and L5 to build the frame")
        return 2

    # the vertebral BODY is the anterior part; take the anterior 60% of the A-P span of the
    # pair as "body column", which is where a cage must live
    a_lo = min(ref[23]["min"][1], ref[24]["min"][1])
    a_hi = max(ref[23]["max"][1], ref[24]["max"][1])
    body_front = a_hi - 0.60 * (a_hi - a_lo)             # anterior of this = body column
    mid_x = (ref[23]["centre"][0] + ref[24]["centre"][0]) / 2
    print(f"  A-P range of L4+L5: {a_lo:.0f}..{a_hi:.0f} mm; "
          f"body column is anterior of {body_front:.0f} mm; midline x={mid_x:.0f}")

    cc, ncc = ndimage.label(hw > 0)
    rows = []
    for i in range(1, ncc + 1):
        m = cc == i
        if m.sum() < 40:
            continue
        idx = np.argwhere(m)
        w = world_of(idx, aff)
        c = w.mean(0)
        q = w - c
        u, s, vt = np.linalg.svd(q, full_matrices=False)
        # TRUE PHYSICAL EXTENT along each principal direction, not a singular value:
        # a singular value scales with the number of voxels and says nothing in mm
        ext = [float((q @ vt[k]).max() - (q @ vt[k]).min()) for k in range(3)]
        rows.append({
            "component": i, "voxels": int(m.sum()),
            "length_mm": round(ext[0], 1), "width_mm": round(ext[1], 1),
            "thick_mm": round(ext[2], 1),
            "aspect": round(ext[0] / max(ext[1], 1e-6), 2),
            "centre_world": [round(float(x), 1) for x in c],
            "off_midline_mm": round(float(c[0] - mid_x), 1),
            "anterior_mm": round(float(c[1]), 1),
            "in_body_column": bool(c[1] > body_front),
            "principal_axis": [round(float(x), 2) for x in vt[0]],
            "touches": sorted({VERT_NAME.get(int(v), str(int(v)))
                               for v in np.unique(lab[ndimage.binary_dilation(m, iterations=4)])
                               if int(v) in VERT_NAME}),
        })
    rows.sort(key=lambda r: -r["voxels"])

    print(f"\n{len(rows)} component(s):")
    for r in rows:
        where = "in the body column (disc/body)" if r["in_body_column"] else \
                "POSTERIOR to the body column (pedicle/lamina territory)"
        print(f"  #{r['component']}  {r['voxels']:>6,} vox   "
              f"{r['length_mm']:>5.1f} x {r['width_mm']:>4.1f} x {r['thick_mm']:>4.1f} mm "
              f"(aspect {r['aspect']:.1f})")
        print(f"        {r['off_midline_mm']:+.0f} mm off midline, {where}, "
              f"touches {','.join(r['touches'])}")

    (out / f"{a.case}_hardware_geometry.json").write_text(
        json.dumps({"case": a.case, "body_column_anterior_of": round(body_front, 1),
                    "midline_x": round(mid_x, 1), "components": rows}, indent=1) + "\n")

    # ---- pictures --------------------------------------------------------------------
    # crop to the hardware plus a generous margin of context
    idx = np.argwhere(hw > 0)
    pad = (np.array([45.0, 45.0, 45.0]) / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.array(lab.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    ct_c, hw_c, lab_c = ct[sl], hw[sl], lab[sl]

    (rax, rsg), (aax, asg), (sax, ssg) = axes_of(aff)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    views = [("sagittal", rax), ("coronal", aax), ("axial", sax)]
    for col, (name, axis) in enumerate(views):
        ct_p = ct_c.max(axis=axis)
        hw_p = hw_c.max(axis=axis)
        # put superior (or anterior for the axial) at the top
        if axis != sax:
            k = sax if sax < axis else sax - 1
            ct_p, hw_p = np.moveaxis(ct_p, k, 0), np.moveaxis(hw_p, k, 0)
            if ssg > 0:
                ct_p, hw_p = ct_p[::-1], hw_p[::-1]
        for row, over in enumerate((False, True)):
            A = ax[row, col]
            A.imshow(ct_p, cmap="gray", vmin=200, vmax=2200, aspect="equal")
            if over:
                A.imshow(np.ma.masked_where(hw_p == 0, hw_p), cmap="autumn_r",
                         vmin=70, vmax=80, alpha=0.85, aspect="equal")
            A.set_title(f"{name}{' — hardware' if over else ' — bone MIP'}", fontsize=9)
            A.axis("off")
    fig.suptitle(f"{a.case}: instrumentation, maximum-intensity projections", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{a.case}_hardware_mip.png", bbox_inches="tight")
    print(f"\nwrote {out / f'{a.case}_hardware_mip.png'}")

    # ---- the hardware ALONE, in 3-D --------------------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render3d import surface_mesh, fit_camera, render

    verts, normals = surface_mesh(hw_c > 0, sp, step=1, smooth=1.0)
    if len(verts):
        cams = [("from the left", (1, 0, 0)), ("from the front", (0, -1, 0)),
                ("from above", (0, 0, -1))]
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), dpi=140)
        for k, (name, d) in enumerate(cams):
            up = (0, 0, 1) if k < 2 else (0, 1, 0)
            cam = fit_camera(verts, d, up, 620, 620)
            img, _ = render([(verts, normals, (232, 106, 92))], cam, bg=(250, 249, 246))
            ax[k].imshow(img)
            ax[k].set_title(name, fontsize=10)
            ax[k].axis("off")
        fig.suptitle(f"{a.case}: the instrumentation alone", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"{a.case}_hardware_3d.png", bbox_inches="tight")
        print(f"wrote {out / f'{a.case}_hardware_3d.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
