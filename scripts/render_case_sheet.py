"""scripts/render_case_sheet.py — draw a case exactly as it is now, no QC filter.

render_rib_review only draws cases the QC put on a dispute list, and colours by whether a
rib is disputed. This draws whatever you name, and its job is to show WHAT IS LABELLED --
which vertebrae exist, which ribs exist, and where the lumbar-rib classes (74/75) ended
up. That is the picture you need when the claim under test is "there is no labelled
thoracic vertebra near these ribs", because the way to check that claim is to look at
which thoracic bodies are drawn at all.

Vertebrae absent from the label are absent from the picture, which is the point.

    python scripts/render_case_sheet.py --labels data/v5_final --cases 0231,0389,0720
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

BLUE, ORANGE, GREEN, INK, SURFACE = "#2a78d6", "#eb6834", "#12a15a", "#0b0b0b", "#fcfcfb"
THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": "#101014",
                     "savefig.facecolor": SURFACE, "text.color": INK, "font.size": 8})


def panel(ax, path: Path, stem: str):
    img = nib.as_closest_canonical(nib.load(str(path)))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zoom = np.array(img.header.get_zooms()[:3], float)

    spine = list(range(THORACIC_BASE + 1, THORACIC_BASE + 13)) + list(LUMBAR) + [26, 29]
    ax.imshow(np.isin(lab, spine).max(axis=1).T, origin="lower", cmap="Greys",
              vmin=0, vmax=1.6, aspect=zoom[2] / zoom[0], interpolation="nearest")

    have_t = []
    for vid, name in ([(THORACIC_BASE + n, f"T{n}") for n in range(1, 13)]
                      + list(LUMBAR.items())):
        m = lab == vid
        if not m.any():
            continue
        if name.startswith("T"):
            have_t.append(name)
        ys, xs = np.nonzero(m.max(axis=1).T)
        ax.text(xs.mean(), ys.mean(), name, color="#e8e8e6", fontsize=5.5,
                ha="center", va="center", fontweight="bold", zorder=5)

    for side, base, colour in (("left", LS.RIB_LEFT_OFFSET, BLUE),
                               ("right", LS.RIB_RIGHT_OFFSET, ORANGE)):
        for n in range(1, 13):
            m = lab == base + n
            if not m.any():
                continue
            mip = m.max(axis=1)
            ax.contour(mip.T, levels=[.5], colors=[colour], linewidths=.8)
            ys, xs = np.nonzero(mip.T)
            k = np.argmin(xs) if side == "left" else np.argmax(xs)
            ax.text(xs[k], ys[k], str(n), color=colour, fontsize=6,
                    fontweight="bold", ha="center", va="center", zorder=6)

    # the lumbar-rib classes, drawn loud -- they are the thing that just changed
    for cid, tag in ((LS.LUMBAR_RIB_LEFT, "L-lum"), (LS.LUMBAR_RIB_RIGHT, "R-lum")):
        m = lab == cid
        if not m.any():
            continue
        mip = m.max(axis=1)
        ax.contour(mip.T, levels=[.5], colors=[GREEN], linewidths=2.0)
        ys, xs = np.nonzero(mip.T)
        k = np.argmin(xs) if "L-" in tag else np.argmax(xs)
        ax.text(xs[k], ys[k], tag, color=GREEN, fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=7)

    ax.set_title(f"{stem}\nthoracic bodies labelled: "
                 f"{', '.join(have_t) if have_t else 'NONE'}",
                 loc="left", color=INK, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="case_sheets")
    a = ap.parse_args()
    stems = [c.strip() for c in a.cases.split(",") if c.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(stems), figsize=(4.4 * len(stems), 6.6), dpi=150)
    for ax, stem in zip(np.atleast_1d(axes), stems):
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}"); continue
        panel(ax, fp, stem)
        print(f"  drew {stem}")
    fig.tight_layout()
    p = out / ("_".join(stems) + ".png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
