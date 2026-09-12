"""Show the S1 carve on a midsagittal slice, where the cut plane is actually visible.

A 3D render shows that something is wrong with 0428 but not WHAT: the blue surface hides
the plane that produced it. The carve is a plane through the sacrum, so the view that shows
it is a midsagittal slice, where S1 and the sacrum below it are two regions meeting along a
line and the line is the estimate under suspicion.

Drawn beside 0704, which passes, so the comparison is like for like rather than against a
description of what a correct carve would look like.

Volumes are canonicalised for ANALYSIS before slicing -- stored orientation is ('P','I','R'),
so without it the "sagittal" slice is not sagittal and the "height" is a width.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch

S1, SACRUM = 29, 26
LABELS = "data/zenodo_deposit/labels"
CASES = [("0704", "0704 — carve accepted"), ("0428", "0428 — carve failed")]
OUT = sys.argv[1] if len(sys.argv) > 1 else "s1_failure.png"

fig, axes = plt.subplots(1, len(CASES), figsize=(9.2, 5.4))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

for ax, (case, title) in zip(np.atleast_1d(axes), CASES):
    img = nib.as_closest_canonical(nib.load(f"{LABELS}/{case}_label.nii.gz"))
    a = np.asanyarray(img.dataobj)
    zx, zy, zz = (abs(v) for v in img.header.get_zooms()[:3])

    sac_any = (a == S1) | (a == SACRUM)
    if not sac_any.any():
        ax.set_title(f"{title}: no sacrum"); ax.axis("off"); continue

    # midsagittal through the sacrum's own centre of mass, not the volume's
    xs = np.nonzero(sac_any.any(axis=(1, 2)))[0]
    xmid = int(round(xs.mean()))
    sl_s1 = (a[xmid] == S1)
    sl_sac = (a[xmid] == SACRUM)

    ys = np.nonzero((sl_s1 | sl_sac).any(axis=1))[0]
    zs = np.nonzero((sl_s1 | sl_sac).any(axis=0))[0]
    pad = 12
    y0, y1 = max(0, ys[0] - pad), min(sl_s1.shape[0], ys[-1] + pad)
    z0, z1 = max(0, zs[0] - pad), min(sl_s1.shape[1], zs[-1] + pad)

    rgb = np.ones((y1 - y0, z1 - z0, 3), float)
    rgb[sl_sac[y0:y1, z0:z1]] = (0.62, 0.62, 0.64)          # sacrum below the cut
    rgb[sl_s1[y0:y1, z0:z1]] = (0.11, 0.42, 0.72)           # what the carve called S1

    # display superior-up, anterior-left
    ax.imshow(np.transpose(rgb, (1, 0, 2))[::-1], aspect=zz / zy, interpolation="nearest")

    s1z = np.nonzero(sl_s1.any(axis=0))[0]
    allz = np.nonzero((sl_s1 | sl_sac).any(axis=0))[0]
    frac = (s1z[-1] - s1z[0] + 1) / (allz[-1] - allz[0] + 1) if s1z.size else 0.0
    mm = (s1z[-1] - s1z[0] + 1) * zz if s1z.size else 0.0
    whole = (allz[-1] - allz[0] + 1) * zz

    ax.set_title(f"{title}\nS1 = {mm:.0f} mm of {whole:.0f} mm sacrum  ({frac:.0%})",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.legend(handles=[Patch(color=(0.11, 0.42, 0.72), label="labelled S1 (the carve)"),
                    Patch(color=(0.62, 0.62, 0.64), label="sacrum below it")],
           loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.suptitle("The S1 carve on a midsagittal slice: a plane estimate, and where it lands",
             fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
