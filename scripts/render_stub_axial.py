"""scripts/render_stub_axial.py — is that stub a lumbar rib, or a mislabelled transverse process?

A coronal MIP cannot answer this. Projected, a hypoplastic rib and a transverse process
occupy the same place and the same silhouette. What separates them is only visible in
cross-section:

    transverse process   CONTINUOUS with the vertebral arch -- one piece of bone
    lumbar rib           a SEPARATE ossicle across a costovertebral joint space

So this draws axial slices through the structure, plus the contralateral side at the same
level as the built-in control: whatever the other side has there is, by definition, what a
transverse process looks like in this patient at this level.

It also prints the numbers that back the picture up:

    voxels, and the gap to the vertebra              a rib has a joint, a TP has none
    the contralateral vertebra's own TP voxel count  is the vertebra MISSING a process on
                                                     the stub side? if so the stub is that
                                                     process, wearing a rib label

    python scripts/render_stub_axial.py --labels data/v5_final --case 0315 --struct 75
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy import ndimage                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

INK, SURFACE = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "font.size": 8})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--case", required=True)
    ap.add_argument("--struct", type=int, required=True, help="label id of the stub")
    ap.add_argument("--vert", type=int, default=20, help="vertebra it sits on (L1=20)")
    ap.add_argument("--out", default="stub_axial")
    ap.add_argument("--n", type=int, default=6, help="slices through the stub")
    a = ap.parse_args()

    fp = Path(a.labels) / f"{a.case}_label.nii.gz"
    img = nib.as_closest_canonical(nib.load(str(fp)))       # RAS+: x=R, y=A, z=S
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zoom = np.array(img.header.get_zooms()[:3], float)

    m = lab == a.struct
    if not m.any():
        print(f"  label {a.struct} absent from {fp.name}")
        return 1
    vm = lab == a.vert

    # numbers first -- the picture is corroboration, not the argument
    zs = np.nonzero(m.any(axis=(0, 1)))[0]
    xs = np.nonzero(m.any(axis=(1, 2)))[0]
    # RAS+ after as_closest_canonical: x INCREASES toward the patient's RIGHT
    vmid = float(np.average(np.arange(lab.shape[0]), weights=vm.sum(axis=(1, 2))))
    side = "right" if xs.mean() > vmid else "left"
    d = ndimage.distance_transform_edt(~vm, sampling=zoom)
    gap = float(d[m].min())
    print(f"  {a.case}  label {a.struct} on the patient's {side}")
    print(f"    spacing         {zoom[0]:.2f} x {zoom[1]:.2f} x {zoom[2]:.2f} mm")
    print(f"    voxels          {int(m.sum())}")
    print(f"    z-extent        {len(zs)} slices ({len(zs)*zoom[2]:.0f} mm)")
    # in VOXELS, because "1.4 mm" means adjacent at 1.4mm spacing and a real gap at 0.5mm
    print(f"    gap to vertebra {gap:.2f} mm = {gap/zoom.min():.1f} voxels   "
          f"({'ADJACENT -> could be a carved-off process' if gap <= 1.5 * zoom.min() else 'separated -> joint space, rib'})")

    # THE DISCRIMINATOR: at the stub's own levels, how far laterally does the vertebra
    # reach on each side? A vertebra missing a transverse process is SHORT on that side,
    # and adding the stub back would restore the symmetry. Comparing whole-vertebra
    # voxel halves does not work -- the body dominates and swamps a missing process.
    zsl = slice(zs.min(), zs.max() + 1)
    vsub, msub = vm[:, :, zsl], m[:, :, zsl]
    vxs = np.nonzero(vsub.any(axis=(1, 2)))[0]
    reach_r = (vxs.max() - vmid) * zoom[0]
    reach_l = (vmid - vxs.min()) * zoom[0]
    both = vsub | msub
    bxs = np.nonzero(both.any(axis=(1, 2)))[0]
    with_r = (bxs.max() - vmid) * zoom[0]
    with_l = (vmid - bxs.min()) * zoom[0]
    print(f"    lateral reach of {a.vert} at those levels:  left {reach_l:5.1f} mm   "
          f"right {reach_r:5.1f} mm   (asymmetry {abs(reach_l-reach_r):.1f} mm)")
    print(f"    same, counting the stub as vertebra:   left {with_l:5.1f} mm   "
          f"right {with_r:5.1f} mm   (asymmetry {abs(with_l-with_r):.1f} mm)")
    print(f"    -> if folding the stub in makes the vertebra SYMMETRIC, it is that "
          f"vertebra's process.\n       If it makes it LOPSIDED, the process is already "
          f"there and the stub is a rib.")

    pick = zs[np.linspace(0, len(zs) - 1, min(a.n, len(zs))).astype(int)]
    # crop around the VERTEBRA plus the stub, so the contralateral side is in frame as
    # the control -- cropping around the stub alone leaves out the thing to compare with
    axs = np.nonzero((m | vm).any(axis=(1, 2)))[0]
    x0, x1 = max(0, axs.min() - 25), min(lab.shape[0], axs.max() + 25)
    ys = np.nonzero((m | vm).any(axis=(0, 2)))[0]
    y0, y1 = max(0, ys.min() - 25), min(lab.shape[1], ys.max() + 25)

    fig, axes = plt.subplots(1, len(pick), figsize=(2.5 * len(pick), 3.0), dpi=180)
    for ax, z in zip(np.atleast_1d(axes), pick):
        sl = lab[x0:x1, y0:y1, z]
        ax.imshow((sl > 0).T, origin="lower", cmap="Greys", vmin=0, vmax=2.2,
                  aspect=zoom[1] / zoom[0], interpolation="nearest")
        for ids, col, lw in (([a.vert], "#2a78d6", .8), ([a.struct], "#d81f26", 1.4)):
            mm = np.isin(sl, ids)
            if mm.any():
                ax.contour(mm.T, levels=[.5], colors=[col], linewidths=lw)
        ax.set_title(f"z={z}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{a.case}  label {a.struct} (red) vs vertebra {a.vert} (blue)  —  patient's "
                 f"{side}; gap {gap:.2f} mm ({gap/zoom.min():.1f} vox), {int(m.sum())} vox; "
                 f"lateral reach L {reach_l:.0f} / R {reach_r:.0f} mm   [image right = patient right]",
                 fontsize=8)
    fig.tight_layout()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.case}_{a.struct}_axial.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
