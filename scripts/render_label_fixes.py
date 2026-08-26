"""scripts/render_label_fixes.py — before and after, on the labels that were changed.

Numbers in a report are not evidence that a fix was right. This renders the same anatomy
twice from the same camera -- once from the backup taken before the change, once from what
is on disk now -- and colours each vertebra separately, so a piece that swapped from one
bone's colour to its neighbour's is visible as a colour change rather than as a claim.

The camera is fitted on the BEFORE volume and reused for the AFTER, so the two panels are
registered: anything that appears to move between them moved in the data.

    python scripts/render_label_fixes.py --case 0196 --before data/v5_pre_break_backup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render3d import surface_mesh, fit_camera, render              # noqa: E402

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1"}
for v in range(34, 58):
    NAME[v] = f"rib {'L' if v < 46 else 'R'}{(v - 34) % 12 + 1}"

# one colour per structure, so a renumbered piece changes colour between the panels
PALETTE = [(214, 108, 92), (86, 132, 184), (206, 168, 84), (120, 166, 122),
           (168, 122, 178), (196, 140, 108), (110, 168, 176), (188, 156, 176)]
PAPER = (244, 242, 236)


def load(p):
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asanyarray(img.dataobj), np.array(img.header.get_zooms()[:3], float)


def scene(lab, sp, ids, cam=None, view=None, size=640):
    groups, allv = [], []
    for i, vid in enumerate(ids):
        m = lab == vid
        if m.sum() < 300:
            continue
        idx = np.argwhere(m)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = np.minimum(idx.max(0) + 3, np.array(m.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        v, n = surface_mesh(m[sl], sp, step=1)
        if v is None:
            continue
        v = v + lo * sp
        groups.append((v, n, PALETTE[i % len(PALETTE)]))
        allv.append(v)
    if not groups:
        return None, None
    if cam is None:
        cam = fit_camera(np.concatenate(allv), width=size, height=size, margin=1.12, **view)
    img, _ = render(groups, cam, bg=PAPER, supersample=2)
    return img, cam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--before", default="data/v5_pre_break_backup")
    ap.add_argument("--after", default="data/hf_export_v5/labels")
    ap.add_argument("--out", default=None)
    ap.add_argument("--view", default="lateral", choices=("lateral", "anterior"))
    a = ap.parse_args()

    bpath = Path(a.before) / f"{a.case}_label.nii.gz"
    apath = Path(a.after) / f"{a.case}_label.nii.gz"
    if not bpath.exists():
        print(f"  ! no backup for {a.case} in {a.before}")
        return 1
    before, sp = load(bpath)
    after, _ = load(apath)

    changed = before != after
    if not changed.any():
        print(f"  ! nothing changed in {a.case}")
        return 1
    # the structures involved: whatever wore a changed voxel, either side of the change
    ids = sorted({int(v) for v in np.unique(before[changed]) if v} |
                 {int(v) for v in np.unique(after[changed]) if v})
    print(f"  {a.case}: {int(changed.sum())} voxel(s) changed across "
          f"{', '.join(NAME.get(i, str(i)) for i in ids)}")

    view = (dict(direction=(1, 0, 0), up=(0, 0, 1)) if a.view == "lateral"
            else dict(direction=(0, -1, 0), up=(0, 0, 1)))
    img_b, cam = scene(before, sp, ids, view=view)
    if img_b is None:
        print("  ! nothing to render")
        return 1
    img_a, _ = scene(after, sp, ids, cam=cam)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0))
    for ax, im, ttl in ((axes[0], img_b, "before"), (axes[1], img_a, "after")):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=13, weight="600", color="#1A1C18")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor=np.array(PALETTE[i % len(PALETTE)]) / 255,
                          markeredgecolor="none", markersize=11,
                          label=NAME.get(v, str(v))) for i, v in enumerate(ids)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(ids), 6),
               frameon=False, fontsize=10)
    fig.suptitle(f"case {a.case} — {int(changed.sum())} voxels changed",
                 fontsize=12, color="#1A1C18")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = a.out or f"scratchpad/fix_{a.case}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, facecolor="#f4f2ec")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
