"""Draw the pelvic-incidence construction so a failing case can be looked at.

For each case: a mid-sagittal projection of the sacrum, S1 and femur labels, with the
fitted S1 plate, its normal, the femoral-head axis and the radius line drawn on top, and
the resulting PI/SS/PT in the title. A number that is wrong for a visible reason stops
being a mystery.

    python render_pi.py <labels_dir> <out_dir> <case,case,...> [tag]
"""
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/wsu/home/go/go24/go2432/CTSpinoPelvic1K/scripts")
import extract_surgical_morphometrics as E

S1, SACRUM, HIP_L, HIP_R, FEM_L, FEM_R = 29, 26, 30, 31, 32, 33
L5 = 24


def main():
    lab_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cases = sys.argv[3].split(",")
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    for cs in cases:
        f = lab_dir / f"{cs}_label.nii.gz"
        if not f.exists():
            continue
        img = nib.as_closest_canonical(nib.load(str(f)))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.asarray(img.header.get_zooms()[:3], float)
        have = {v: (lab == v) for v in (S1, SACRUM, HIP_L, HIP_R, FEM_L, FEM_R, L5)}

        # femoral heads and the plate, exactly as the extractor computes them
        fem = None
        if have[FEM_L].any() and have[FEM_R].any():
            cl = E._femoral_head(have[FEM_L], have[HIP_L], sp)
            cr = E._femoral_head(have[FEM_R], have[HIP_R], sp)
            if cl is not None and cr is not None:
                fem = (cl + cr) / 2
        s1c, s1n = (E._endplate(have[S1], sp, True) if have[S1].any() else (None, None))
        if s1c is None or fem is None:
            continue

        v = fem - s1c
        vs = np.array([0.0, v[1], v[2]])
        ns = np.array([0.0, s1n[1], s1n[2]])
        PI = 180.0 - E._angle(ns, vs)
        SS = E._angle(ns, np.array([0.0, 0.0, 1.0]))
        PT = PI - SS

        # MID-SAGITTAL PROJECTION, taken about the femoral-head midline rather than the
        # volume's centre: on a field-limited scan the volume centre is not the patient's.
        xmid = int(round(fem[0] / sp[0]))
        half = max(3, int(round(12.0 / sp[0])))
        lo, hi = max(xmid - half, 0), min(xmid + half + 1, lab.shape[0])

        fig, ax = plt.subplots(figsize=(6.2, 6.6))
        colours = {SACRUM: "#c8ccd2", S1: "#e08a3c", L5: "#6fa8b8",
                   FEM_L: "#9bb08a", FEM_R: "#9bb08a", HIP_L: "#d9d2c0", HIP_R: "#d9d2c0"}
        for vid, col in colours.items():
            m = have.get(vid)
            if m is None or not m.any():
                continue
            proj = m[lo:hi].any(axis=0)          # (y, z)
            if not proj.any():
                continue
            ys, zs = np.nonzero(proj)
            ax.scatter(ys * sp[1], zs * sp[2], s=0.6, c=col, marker="s",
                       linewidths=0, alpha=0.75, zorder=1)

        # the plate: a segment through its centroid, perpendicular to the fitted normal
        pdir = np.array([-ns[2], ns[1]])
        pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
        L = 22.0
        ax.plot([s1c[1] - L*pdir[0], s1c[1] + L*pdir[0]],
                [s1c[2] - L*pdir[1], s1c[2] + L*pdir[1]],
                color="#b3202c", lw=2.4, zorder=4, label="fitted S1 plate")
        nn = ns[1:] / (np.linalg.norm(ns[1:]) + 1e-9)
        ax.arrow(s1c[1], s1c[2], 26*nn[0], 26*nn[1], width=0.7, color="#b3202c",
                 length_includes_head=True, zorder=4, alpha=0.9)
        ax.plot([s1c[1], fem[1]], [s1c[2], fem[2]], color="#1c3f8f", lw=1.8,
                ls="--", zorder=4, label="plate midpoint to hip axis")
        ax.scatter([fem[1]], [fem[2]], s=90, c="#1c3f8f", marker="o",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.scatter([s1c[1]], [s1c[2]], s=70, c="#b3202c", marker="o",
                   edgecolors="white", linewidths=1.2, zorder=5)

        ax.set_aspect("equal")
        ax.set_xlabel("anterior (mm)  →")
        ax.set_ylabel("superior (mm)  →")
        ax.set_title(f"{cs}{tag}\nPI {PI:.1f}°   SS {SS:.1f}°   PT {PT:.1f}°"
                     f"   plate tilt {E._angle(s1n, np.array([0.,0.,1.])):.1f}°",
                     fontsize=10)
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cs}_pi.png", dpi=125)
        plt.close(fig)
        print(f"  {cs}: PI {PI:6.1f}  SS {SS:5.1f}  PT {PT:6.1f}", flush=True)


if __name__ == "__main__":
    main()
