"""scripts/render_hardware.py — the instrumentation the dataset holds, one panel per subtype.

WHAT THIS FIGURE IS FOR. The hardware section says the cohort forced three identifiers the
subtype block did not have, and that the implants had been sitting inside bone labels. This
shows the four objects that exist in the release, rendered from the released labels the
same way the anchor figure is (render_turntable.render, first-hit surface): a bilateral hip
arthroplasty, femoral-neck osteosynthesis, sacroiliac screws, and threaded interbody cages.

METAL IS DRAWN THROUGH BONE. A first-hit render shows whatever surface is nearest the eye,
and most of an implant is inside a bone -- a femoral stem inside the shaft, cages inside an
interspace, SI screws inside the ilium and sacrum. Rendered honestly, the cages would not be
visible at all. So the bone is rendered once, muted, and the hardware is rendered again on
its own and composited on top, the way metal reads on a radiograph: the bone is context, the
metal is always visible. Nothing about the labels is changed by this; it is only a drawing
convention, and the caption says so.

EACH PANEL IS FRAMED ON ITS IMPLANT, not on the skeleton: a bounding box around the hardware
voxels padded by a fixed margin in millimetres, so a 135,000 mm3 arthroplasty and a 2,600 mm3
cage pair each fill their panel. The panels therefore have DIFFERENT scales, and a 5 cm bar
is drawn in each so the reader can see that.

    python scripts/render_hardware.py --labels data/zenodo_deposit/labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_turntable import render, BG                              # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})

HW = {76: "hardware", 77: "cage", 78: "screw / rod", 79: "plate",
      80: "arthroplasty", 81: "sacroiliac screw", 82: "osteosynthesis"}
C_HW = {80: np.array([214, 69, 65], np.float32),      # arthroplasty
        82: np.array([232, 140, 40], np.float32),     # osteosynthesis
        81: np.array([48, 110, 190], np.float32),     # SI screw
        77: np.array([60, 160, 90], np.float32),      # cage
        76: np.array([160, 60, 180], np.float32),
        78: np.array([160, 60, 180], np.float32),
        79: np.array([160, 60, 180], np.float32)}
C_BONE = np.array([200, 200, 196], np.float32)
C_SACRUM = np.array([160, 160, 158], np.float32)

# The four subtypes present in the release, one exemplar each, with the view that shows
# the object: hips and the femoral neck from the front; cages from the side, because from
# the front an interbody device is edge-on behind the vertebral body.
DEFAULT_CASES = ("0974:(a) bilateral hip arthroplasty:0;"
                 "0247:(b) femoral-neck screws:0;"
                 "1035:(c) SI screws:0;"
                 "0068:(d) interbody cages, oblique:40")


def frame(lab, margin_mm, zooms):
    """Zero everything outside a box around the hardware voxels, padded by margin_mm."""
    hw = np.isin(lab, list(HW))
    if not hw.any():
        return lab
    # SLICE, do not zero: the render is the size of the array it is given, and zeroing
    # outside the box left every implant in the corner of a panel of empty volume
    sl = []
    for ax in range(3):
        idx = np.nonzero(hw.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        pad = int(round(margin_mm / max(zooms[ax], 1e-6)))
        sl.append(slice(max(0, idx.min() - pad), min(lab.shape[ax], idx.max() + pad + 1)))
    return lab[tuple(sl)]


def composite(lab, angle, cut=False):
    """Bone first-hit in muted grey, hardware first-hit on top: metal through bone.

    cut=True is the honest alternative for an implant that sits INSIDE the column: the
    half of the volume between the eye and the implant's mid-plane is removed and one
    first-hit render is taken, so the cut faces of the bodies show and the cage sits in
    the interspace where it is. Drawn through bone, a cage looks stuck to the surface.
    """
    # A lateral view is an axis swap, not a rotation: ndimage.rotate with reshape=False
    # clips the corners of a non-square box, and 90 degrees is exact as a transpose.
    if int(angle) == 90:
        lab, angle = np.ascontiguousarray(np.transpose(lab, (1, 0, 2))[::-1]), 0
    elif angle:
        # an oblique view rotates in-plane with reshape=False, which clips whatever the
        # crop box's corners held; pad the box to a square wide enough for its diagonal
        side = int(np.ceil(np.hypot(*lab.shape[:2])))
        px, py = (side - lab.shape[0]) // 2, (side - lab.shape[1]) // 2
        lab = np.pad(lab, ((px, side - lab.shape[0] - px), (py, side - lab.shape[1] - py), (0, 0)))
    if cut:
        # the eye looks along +axis1 and the first occupied voxel wins, so the near half
        # is the low index side; remove it up to the implant's mid-plane
        # cut THROUGH the implant, at the plane holding the most metal: paired cages sit
        # either side of the midline, and a cut between them shows disc and no cage
        per_plane = np.isin(lab, list(HW)).sum(axis=(0, 2))
        mid = int(np.argmax(per_plane)) if per_plane.any() else lab.shape[1] // 2
        lab = lab.copy(); lab[:, :mid, :] = 0
        cols = {int(v): C_BONE for v in np.unique(lab) if v}
        cols[26] = C_SACRUM; cols.update(C_HW)
        rgb = render(lab, cols, angle, 255)
        return rgb.astype(np.float32)
    bone = lab.copy(); bone[np.isin(bone, list(HW))] = 0
    metal = lab.copy(); metal[~np.isin(metal, list(HW))] = 0
    cols = {int(v): C_BONE for v in np.unique(bone) if v}
    cols[26] = C_SACRUM
    rgb = render(bone, cols, angle, 255)
    if rgb is None:
        rgb = np.full((lab.shape[2], lab.shape[0], 3), BG, np.uint8)
    rgb = rgb.astype(np.float32)
    hw = render(metal, {k: v for k, v in C_HW.items()}, angle, 255)
    if hw is not None:
        hw = hw.astype(np.float32)
        on = np.any(np.abs(hw - BG[None, None, :]) > 2, axis=2)
        rgb[on] = hw[on]
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--pattern", default="{case}_label.nii.gz",
                    help="filename under --labels; hardware_review/verify uses {case}_label_hw.nii.gz")
    ap.add_argument("--cases", default=DEFAULT_CASES,
                    help="semicolon list of case:caption:angle_deg")
    ap.add_argument("--margin_mm", type=float, default=35.0)
    ap.add_argument("--mm_per_px", type=float, default=0.6)
    ap.add_argument("--panel_in", type=float, default=1.75)
    ap.add_argument("--height_in", type=float, default=2.6)
    ap.add_argument("--out", default="paper/mpda/figures")
    ap.add_argument("--name", default="fig_hardware")
    a = ap.parse_args()

    items = [c.split(":") for c in a.cases.split(";") if c.strip()]
    panels, present = [], set()
    for spec in items:
        case, caption, ang = spec[:3]
        cut = len(spec) > 3 and spec[3] == "cut"
        f = Path(a.labels) / a.pattern.format(case=case)
        img = nib.as_closest_canonical(nib.load(str(f)))
        lab = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()[:3]
        ids = [int(v) for v in np.unique(lab) if int(v) in HW]
        present.update(ids)
        lab = frame(lab, a.margin_mm, zooms)
        rgb = composite(lab, float(ang), cut=cut)
        # rows are craniocaudal (zooms[2]); columns are x for a frontal view and y for a
        # lateral one, and the two in-plane zooms are equal on every CT here
        fy, fx = zooms[2] / a.mm_per_px, zooms[0] / a.mm_per_px
        rgb = ndimage.zoom(rgb, (fy, fx, 1.0), order=1, mode="nearest")
        panels.append((case, caption, rgb))
        print(f"  {case}: hardware ids {ids}  panel {rgb.shape[1]}x{rgb.shape[0]} px")

    # ONE HEIGHT FOR EVERY PANEL, widths from each image's aspect: a fixed grid left the
    # wide arthroplasty strip floating mid-cell and the square cage panel towering over it.
    aspects = [p_.shape[1] / p_.shape[0] for _, _, p_ in panels]
    fig_w = a.panel_in * len(items)
    fig = plt.figure(figsize=(fig_w, a.height_in))
    gs = fig.add_gridspec(1, len(items), width_ratios=aspects, wspace=0.06,
                          left=0.01, right=0.99, top=0.88, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(items))]

    px_per_5cm = 50.0 / a.mm_per_px
    for ax, (case, caption, rgb) in zip(axes, panels):
        ax.imshow(np.clip(rgb, 0, 255).astype(np.uint8), interpolation="bilinear")
        h, w = rgb.shape[:2]
        # 5 cm bar, bottom left of each panel: the panels are at different scales
        ax.plot([w * 0.06, w * 0.06 + px_per_5cm], [h * 0.95, h * 0.95], color="k", lw=1.4)
        ax.text(w * 0.06, h * 0.93, "5 cm", fontsize=6.5, va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(caption.replace("--", "–"), fontsize=7.5)

    handles = [mp.Patch(color=C_HW[i] / 255, label=f"{i} {HW[i]}")
               for i in (80, 82, 81, 77) if i in present]
    handles.append(mp.Patch(color=C_BONE / 255, label="bone (metal drawn through it)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    out = Path(a.out) / f"{a.name}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
