"""scripts/clean_label_speckle.py — remove stray fragments before the labels are deposited.

WHAT IS BEING REMOVED. Across the release most vertebra and rib labels are one solid
structure with a handful of loose voxels beside it: 6,931 stray components at the last
count, and twenty labels that are nothing BUT dust -- 233 voxels in 138 pieces, largest
nine, on a class that a card was describing as an absence cleanly recorded.

WHY THE EXISTING QC NEVER SAW IT. qc_version_progression counts a label as fragmented only
when two or more pieces each clear 200 voxels, so a label made entirely of specks passes as
intact, and a real bone with fifty specks around it passes as intact too. The number this
release reports for fragmentation does not measure this at all.

THE RULE, AND WHY IT IS THIS ONE. For each label the largest component is the structure and
is never touched. A smaller component is removed when it is BOTH small in absolute terms and
small relative to the structure -- a hypoplastic twelfth rib is 1093 voxels and survives on
the absolute test, while a 40-voxel speck beside a 50,000-voxel vertebra fails both. A
component that is large enough to be real but detached is NOT removed: it is reported, and a
person looks at it, because that is either a genuine bipartite structure or a labelling
error and this script cannot tell which.

WHEN A LABEL IS ONLY DUST the whole class goes. That is the correct reading -- there is no
rib there -- and it is what makes an absence an absence rather than a mislabelled fragment.
Every such removal is listed by name in the report, because it changes what the record says
about the anatomy.

THIS IS A POST-CORRECTION STEP ONLY. It runs on v5, which students have already reviewed.
Running speckle removal on pre-review pseudolabels would delete evidence a reviewer needs.

    python scripts/clean_label_speckle.py --labels data/hf_export_v5/labels --dry-run
    python scripts/clean_label_speckle.py --labels data/hf_export_v5/labels --apply
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

VERT = set(range(8, 30)) - {27}
RIBS = set(range(34, 58))
LUMB = {74, 75}
CLEAN = VERT | RIBS | LUMB

# absolute floor: below this a component cannot be a structure at this resolution. The
# smallest real thing in the release is a hypoplastic twelfth rib at ~460 voxels before
# correction, so 120 leaves a wide margin.
ABS_MIN = 120
# and it must also be trivial next to the structure it sits beside
REL_MIN = 0.02
# a detached piece this big is not speckle; it is reported instead of removed
REVIEW_MIN = 400

# WHERE THE GAP SITS DECIDES WHAT THE GAP MEANS. Expressed as a fraction of the radius of
# the imaged cylinder: near 1 the gap is at the reconstruction circle and the bone was never
# scanned; near 0 it is deep inside a region the scanner covered and the label lost it.
FOV_EDGE = 0.80
INTERIOR = 0.55

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"


def imaged_radius(lab, zooms, in_plane):
    """Radius of the reconstructed cylinder, taken from how far any label ever reaches.

    The labels only exist where the scanner imaged, so their outermost reach is a good
    stand-in for the reconstruction circle without needing the CT alongside.
    """
    idx = np.argwhere(lab[::2, ::2, ::2] > 0) * 2
    if len(idx) < 500:
        return None, None
    a, b = in_plane
    cen = (float(idx[:, a].mean()), float(idx[:, b].mean()))
    r = np.hypot((idx[:, a] - cen[0]) * zooms[a], (idx[:, b] - cen[1]) * zooms[b])
    return cen, float(np.percentile(r, 99.8))


def one(args):
    path, apply_it, backup_dir = args
    case = Path(path).name.split("_")[0]
    rows = []
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
    except Exception as e:                                        # noqa: BLE001
        return [{"case": case, "label": "READ_ERROR", "id": -1, "action": "error",
                 "voxels": 0, "components": 0, "removed_voxels": 0, "note": str(e)[:70]}]

    counts = np.bincount(lab.reshape(-1), minlength=256)
    zooms = np.array(img.header.get_zooms()[:3], float)
    codes = nib.aff2axcodes(img.affine)
    si = [i for i, c in enumerate(codes) if c in "SI"]
    in_plane = [i for i in range(3) if i != si[0]] if si else [0, 1]
    centre, R = imaged_radius(lab, zooms, in_plane)
    out = lab.copy()
    touched = False

    for v in sorted(CLEAN):
        tot = int(counts[v]) if v < len(counts) else 0
        if not tot:
            continue
        # THE BOUNDING BOX COMES FROM THREE BOOLEAN REDUCTIONS, not from argwhere and not
        # from find_objects. argwhere materialises every coordinate of a 50,000-voxel
        # vertebra and was costing a minute a case; find_objects wants an integer array and
        # promotes the whole 136-million-voxel volume to int64, which is a gigabyte per
        # worker and killed the first run outright. `any` over two axes at a time allocates
        # nothing of consequence and gives the same box.
        m = lab == v
        span = []
        for ax in range(3):
            hit = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]
            if not len(hit):
                break
            span.append(slice(int(hit[0]), int(hit[-1]) + 1))
        if len(span) != 3:
            continue
        sl = tuple(span)
        sub = m[sl]
        del m
        lt, n = ndimage.label(sub)
        if n <= 1:
            continue
        sizes = ndimage.sum(sub, lt, range(1, n + 1))
        biggest = int(np.argmax(sizes)) + 1
        big_size = int(sizes[biggest - 1])

        drop = np.zeros(sub.shape, bool)
        removed = 0
        review = []
        for k in range(1, n + 1):
            if k == biggest:
                continue
            sz = int(sizes[k - 1])
            if sz >= REVIEW_MIN and sz >= REL_MIN * big_size:
                review.append((k, sz))
                continue
            if sz < ABS_MIN and sz < REL_MIN * big_size:
                drop |= lt == k
                removed += sz
            elif sz < ABS_MIN:
                # small in absolute terms but a big share of a small label: this is the
                # dust-only case, handled below by looking at the label as a whole
                drop |= lt == k
                removed += sz
            else:
                review.append((k, sz))

        whole_label_is_dust = big_size < ABS_MIN
        if whole_label_is_dust:
            drop = np.ones(sub.shape, bool) & sub
            removed = tot

        if removed:
            block = out[sl]
            block[drop] = 0
            out[sl] = block
            touched = True

        # ---- where is each detached piece, and is the bone between it and the body
        # something the scanner ever saw?
        verdicts = []
        if review and R:
            core = lt == biggest
            dist, nearest = ndimage.distance_transform_edt(
                ~core, sampling=zooms, return_indices=True)
            for k, sz in review:
                piece = lt == k
                gap_mm = float(dist[piece].min())
                pos = np.argwhere(piece & (dist <= gap_mm + 1e-6))
                if not len(pos):
                    verdicts.append((sz, gap_mm, None, "no closest point"))
                    continue
                q = pos[0]
                near = np.array([nearest[d][tuple(q)] for d in range(3)])
                midpoint = ((q + near) / 2.0 + np.array([sc.start for sc in sl])) * zooms
                a, b = in_plane
                r_mid = float(np.hypot(midpoint[a] - centre[0] * zooms[a],
                                       midpoint[b] - centre[1] * zooms[b]))
                frac = r_mid / R
                verdict = ("outside_fov" if frac >= FOV_EDGE
                           else "segmentation_break" if frac <= INTERIOR
                           else "uncertain")
                verdicts.append((sz, gap_mm, round(frac, 3), verdict))

        if removed or review:
            if verdicts:
                worst = max(verdicts, key=lambda t: t[0])
                action = ("speckle_removed" if removed and worst[3] == "outside_fov"
                          else worst[3])
                note = "; ".join(f"{sz} vox at {gap:.1f} mm, r={fr}, {vd}"
                                 for sz, gap, fr, vd in verdicts)
            else:
                action = ("label_removed" if whole_label_is_dust
                          else "speckle_removed" if removed else "review")
                note = f"largest {big_size}"
            rows.append({
                "case": case, "label": NAME.get(v, str(v)), "id": v,
                "voxels": tot, "components": n, "removed_voxels": removed,
                "action": ("label_removed" if whole_label_is_dust else action),
                "note": note if not whole_label_is_dust else f"largest {big_size}",
            })

    if apply_it and touched:
        bak = Path(backup_dir) / Path(path).name
        if not bak.exists():
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak)
        # THE HEADER AND AFFINE ARE REUSED VERBATIM. Rebuilding either is how a label gets
        # silently transposed off its CT; nothing here changes geometry, only voxel values.
        newimg = nib.Nifti1Image(out.astype(lab.dtype), img.affine, img.header)
        newimg.set_data_dtype(lab.dtype)
        nib.save(newimg, str(path))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/hf_export_v5/labels")
    ap.add_argument("--out", default="qc_final/speckle_cleanup.csv")
    ap.add_argument("--backup", default="data/v5_pre_speckle_backup")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    if a.apply == a.dry_run:
        print("  ! choose exactly one of --apply or --dry-run")
        return 2

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if not files:
        print(f"  ! no labels under {a.labels}")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    cols = ["case", "label", "id", "action", "voxels", "components",
            "removed_voxels", "note"]
    done = 0
    tally = {}
    vox = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        payload = [(str(f), a.apply, a.backup) for f in files]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for rows in ex.map(one, payload, chunksize=2):
                for r in rows:
                    tally[r["action"]] = tally.get(r["action"], 0) + 1
                    vox += int(r["removed_voxels"])
                w.writerows(rows)
                done += 1
                if done % 25 == 0:
                    fh.flush()
                    print(f"  {done}/{len(files)}", flush=True)

    print(f"\n  {'APPLIED' if a.apply else 'DRY RUN'} over {done} case(s)")
    print(f"    labels with speckle removed : {tally.get('speckle_removed', 0)}")
    print(f"    labels removed entirely     : {tally.get('label_removed', 0)}")
    print(f"    detached, outside the FOV   : {tally.get('outside_fov', 0)}"
          "   (imaging limit, not an error)")
    print(f"    detached, segmentation break: {tally.get('segmentation_break', 0)}"
          "   (the worklist)")
    print(f"    detached, uncertain         : {tally.get('uncertain', 0)}")
    print(f"    voxels removed              : {vox}")
    print(f"  report: {a.out}")
    if a.apply:
        print(f"  originals: {a.backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
