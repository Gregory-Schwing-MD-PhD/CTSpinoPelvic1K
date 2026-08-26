"""split_and_renumber_0068.py — six lumbar bodies under five labels, and the fix.

WHAT IS WRONG. 0068's spine labels are pseudolabels that no human checked
(`prov_spine: pseudo`). The label called L1 covers TWO vertebral bodies, so five lumbar
labels span six free bodies and every level below the merge is named one too high. The
consequence is not cosmetic: the interbody cage that the label puts at L4-L5 is really at
L5-L6, and the case is a six-lumbar (L6) record that the manifest currently calls
`has_l6: false` with `n_lumbar_labels: 5`.

WHERE THE NAMES COME FROM. Not from counting down from the top -- this scan starts
mid-thoracic and there is no top to count from, which is the failure this dataset exists to
document. They come from the LAST RIB: rib_left_12 and rib_right_12 both articulate with one
vertebra at about 5 mm, and the vertebra bearing the twelfth rib is T12 by definition.
Everything below is then counted off a landmark that is actually in the image.

HOW THE MERGED LABEL IS SPLIT. Not with a flat plane at the disc: that cuts the posterior
elements at an arbitrary height and leaves half a lamina on the wrong vertebra. The two
BODIES are found first, in the anterior column where the disc is a real gap, and they become
seeds; every voxel of the merged label then goes to whichever seed is nearer. The posterior
elements follow their own body, which is what an annotator would do by hand.

A PROPOSAL, NOT A RELEASE. Writes to its own file. The fused L5-L6 interspace is a genuine
iatrogenic fusion -- there is no image boundary in the disc space because the cage fills it
-- so the split there is inferred from the endplates that are visible above and below, and
that is exactly the judgement a reader should check.

    python scripts/split_and_renumber_0068.py --label thoracic_fix/0068/0068_label_FULL.nii.gz \\
        --ct thoracic_fix/0068/0068_ct.nii.gz --out thoracic_fix/0068
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

LUMBAR = list(range(20, 26))          # L1..L6
THOR = list(range(8, 20))
NAME = {**{v: f"T{v - 7}" for v in THOR}, **{v: f"L{v - 19}" for v in LUMBAR},
        26: "sacrum", 29: "S1", 77: "hardware_cage"}
RIB12 = (45, 57)                      # rib_left_12, rib_right_12
T12 = 19


def bodies_in(mask, aff, zmm, ax, others, frac=0.45, min_frac=0.18):
    """How many distinct vertebral BODIES a label covers, and their z centres.

    The body is the anterior part; the posterior elements run continuously past a disc and
    would join two bodies into one blob. Restricting to the anterior `frac` of the label's
    own anterior-posterior extent leaves the disc as a real gap.
    """
    idx = np.argwhere(mask)
    w = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
    a_hi, a_lo = w[:, 1].max(), w[:, 1].min()
    keep = w[:, 1] > a_hi - frac * (a_hi - a_lo)
    sub = np.zeros_like(mask)
    sub[tuple(idx[keep].T)] = True
    sub = ndimage.binary_erosion(sub, iterations=2)
    cc, n = ndimage.label(sub)
    if n == 0:
        return 0, []
    sizes = ndimage.sum(sub, cc, range(1, n + 1))
    big = [i + 1 for i, s in enumerate(sizes) if s >= min_frac * sizes.max()]
    centres = []
    for i in big:
        z = np.where((cc == i).any(axis=others))[0]
        centres.append(float(z.mean()))
    order = np.argsort(centres)
    return len(big), [(big[k], centres[k]) for k in order], cc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="0068")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = nib.load(a.label)
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    aff = img.affine
    col = aff[2, :3]
    ax = int(np.argmax(np.abs(col)))
    sgn = int(np.sign(col[ax]) or 1)
    zmm = float(np.linalg.norm(aff[:3, ax]))
    others = tuple(i for i in range(3) if i != ax)

    # ---- the anchor: the vertebra bearing the twelfth rib -------------------------
    if not any((lab == r).any() for r in RIB12):
        print("  ! no twelfth rib in this label; the anchor does not exist, refusing")
        return 2
    print(f"  anchor: rib 12 present, articulating with the label called {NAME[T12]}")

    # ---- how many bodies does each lumbar label cover? ----------------------------
    print(f"\n  {'label':<7} {'bodies':>7}   body centres (mm above the sacrum)")
    print("  " + "-" * 58)
    sac = lab == 26
    z_sac = float(np.where(sac.any(axis=others))[0].mean())
    counts = {}
    for v in [x for x in LUMBAR if (lab == x).any()]:
        n, blobs, _ = bodies_in(lab == v, aff, zmm, ax, others)
        counts[v] = n
        cs = ", ".join(f"{abs(c - z_sac) * zmm:.0f}" for _, c in blobs)
        flag = "   <- TWO BODIES UNDER ONE LABEL" if n > 1 else ""
        print(f"  {NAME[v]:<7} {n:>7}   {cs}{flag}")
    total = sum(counts.values())
    print(f"\n  {len(counts)} lumbar labels covering {total} free lumbar bodies")
    if total == len(counts):
        print("  nothing to renumber; every label holds exactly one body")
        return 0

    merged = [v for v, n in counts.items() if n > 1]
    if len(merged) != 1 or counts[merged[0]] != 2:
        print(f"  ! expected exactly one label holding exactly two bodies, got {counts}")
        return 3
    m = merged[0]
    print(f"  the merge is at {NAME[m]}; the cage sits between the two lowest bodies, "
          f"which after renumbering are L{total - 1} and L{total}")

    # ---- split the merged label on nearest-body ----------------------------------
    _, blobs, cc = bodies_in(lab == m, aff, zmm, ax, others)
    seeds = np.zeros_like(lab)
    for k, (comp, _) in enumerate(blobs, start=1):        # upper body first
        seeds[cc == comp] = k
    # every voxel of the merged label joins the nearer body. A flat plane at the disc
    # would cut the laminae at an arbitrary height; this lets the posterior elements
    # follow their own body, which is what a hand annotator does.
    _, ind = ndimage.distance_transform_edt(seeds == 0, return_indices=True,
                                            sampling=[zmm if i == ax else
                                                      float(np.linalg.norm(aff[:3, i]))
                                                      for i in range(3)])
    nearest = seeds[tuple(ind)]
    upper = (lab == m) & (nearest == 1)
    lower = (lab == m) & (nearest == 2)
    print(f"  split {NAME[m]}: {int(upper.sum()):,} voxels to the upper body, "
          f"{int(lower.sum()):,} to the lower")

    # ---- renumber ----------------------------------------------------------------
    # counting DOWN from the anchor: the merged label becomes two levels and every
    # lumbar label below it shifts one place caudally
    new = lab.copy()
    present = sorted([x for x in LUMBAR if (lab == x).any()])
    shift = [v for v in present if v > m]
    for v in sorted(shift, reverse=True):                  # highest id first, no clobber
        new[lab == v] = v + 1
    new[upper] = m
    new[lower] = m + 1
    moved = {NAME[v]: NAME[v + 1] for v in shift}
    print(f"\n  renumbered: {NAME[m]} -> {NAME[m]} + {NAME[m + 1]}; "
          + ", ".join(f"{k} -> {v}" for k, v in moved.items()))

    final = [x for x in LUMBAR if (new == x).any()]
    print(f"  lumbar labels now: {', '.join(NAME[v] for v in final)}")

    dst = out / f"{a.case}_label_RENUMBERED.nii.gz"
    nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), aff, img.header), str(dst))
    meta = {"case": a.case, "anchor": "twelfth rib -> T12",
            "lumbar_bodies": total, "merged_label": NAME[m],
            "split": {"upper_voxels": int(upper.sum()), "lower_voxels": int(lower.sum())},
            "renumbering": {NAME[m]: [NAME[m], NAME[m + 1]], **moved},
            "cage_level": f"L{total - 1}-L{total}",
            "has_l6": total >= 6, "n_lumbar_labels": total}
    (out / f"{a.case}_renumber.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
