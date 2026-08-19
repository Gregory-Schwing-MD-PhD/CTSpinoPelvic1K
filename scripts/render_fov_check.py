"""scripts/render_fov_check.py — how much spine is in the field, and how much of it is labelled?

THE QUESTION THIS ANSWERS. Sixteen cases have no thoracic vertebra labelled at all. That
is either work to do or nothing to do, and the label alone cannot tell you which: a case
with no thoracic labels looks identical to a case whose field of view stops below the
thorax. Only the CT knows.

So the background is a coronal MIP of the CT thresholded at bone, and the labelled
vertebrae are drawn over it. Unlabelled vertebral bodies then appear as bone with no
outline -- visible, countable, and obviously annotatable. A case whose bone simply stops
appears as bone that stops.

The number in the title is the one that decides it: millimetres of BONE above the highest
labelled vertebra. Near zero means the field ends there and there is nothing to add; a
large number means that much spine is sitting in the scan unlabelled.

    python scripts/render_fov_check.py --labels data/v5_final --ct data/hf_export_v4/ct \\
        --cases 0033,0068,... --out fov_check
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
BONE_HU = 250.0
INK, SURFACE = "#0b0b0b", "#fcfcfb"
LBL, RIBL, RIBR = "#ff2fd0", "#2a78d6", "#eb6834"

plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "font.size": 8})


def one(args):
    stem, lab_path, ct_path, out = args
    try:
        limg = nib.as_closest_canonical(nib.load(lab_path))
        lab = np.asanyarray(limg.dataobj).astype(np.int16)
        zoom = np.array(limg.header.get_zooms()[:3], float)
        cimg = nib.as_closest_canonical(nib.load(ct_path))
        ct = np.asanyarray(cimg.dataobj).astype(np.float32)
    except Exception as exc:                                       # noqa: BLE001
        return f"  ! {stem}: {type(exc).__name__}"
    if ct.shape != lab.shape:
        return f"  ! {stem}: shape mismatch"

    # the spine sits near the midline; a full-width MIP buries it under ribs, arms and
    # table, so the projection is limited to a central AP slab
    ys = np.nonzero((lab > 0).any(axis=(0, 2)))[0]
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    else:
        y0, y1 = 0, ct.shape[1]
    bone = (ct[:, y0:y1] > BONE_HU).max(axis=1)

    fig, ax = plt.subplots(figsize=(4.6, 7.4), dpi=165)
    ax.imshow(bone.T, origin="lower", cmap="bone_r", vmin=0, vmax=1.55,
              aspect=zoom[2] / zoom[0], interpolation="nearest")

    named, top_z = [], None
    for vid, nm in ([(THORACIC_BASE + n, f"T{n}") for n in range(1, 13)]
                    + list(LUMBAR.items()) + [(26, "sac"), (29, "S1")]):
        m = lab == vid
        if m.sum() < 500:
            continue
        mip = m.max(axis=1)
        ax.contour(mip.T, levels=[.5], colors=[LBL], linewidths=1.1)
        yy, xx = np.nonzero(mip.T)
        ax.text(xx.mean(), yy.mean(), nm, color=LBL, fontsize=6.5,
                fontweight="bold", ha="center", va="center", zorder=6)
        if nm not in ("sac", "S1"):
            named.append(nm)
            z = np.nonzero(m.any(axis=(0, 1)))[0].max()
            top_z = z if top_z is None else max(top_z, z)

    for base, colour in ((LS.RIB_LEFT_OFFSET, RIBL), (LS.RIB_RIGHT_OFFSET, RIBR)):
        m = np.isin(lab, [base + n for n in range(1, 13)])
        if m.any():
            ax.contour(m.max(axis=1).T, levels=[.5], colors=[colour], linewidths=.6)

    # THE NUMBER: bone above the highest labelled vertebra
    bz = np.nonzero(bone.any(axis=0))[0]
    hdr = f"{stem}   labelled: {', '.join(named) if named else 'NONE'}"
    if len(bz) and top_z is not None:
        gap_mm = float(bz.max() - top_z) * zoom[2]
        hdr += f"\n{gap_mm:.0f} mm of bone above the highest labelled vertebra"
        ax.axhline(top_z, color="#12a15a", lw=1.1, ls=(0, (5, 3)))
    elif len(bz):
        hdr += "\nno vertebra labelled at all"
    ax.set_title(hdr, loc="left", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = Path(out) / f"{stem}_fov.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return f"  {stem}: labelled {len(named)} vertebrae -> {p.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--ct", default="data/hf_export_v4/ct")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="fov_check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in [c.strip() for c in a.cases.split(",") if c.strip()]:
        lp = Path(a.labels) / f"{s}_label.nii.gz"
        cp = Path(a.ct) / f"{s}_ct.nii.gz"
        if lp.exists() and cp.exists():
            jobs.append((s, str(lp), str(cp), str(out)))
        else:
            print(f"  ! {s}: missing {'label' if not lp.exists() else 'ct'}")
    print(f"{len(jobs)} case(s)\n", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
