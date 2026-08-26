"""scripts/audit_deposit_labels.py — measure the deposited labels. Read only. No exceptions.

WHY THIS EXISTS SEPARATELY. The earlier fragment figures came from a script that measured and
modified in the same pass, so its numbers described a state that no longer exists, and its
printed summary did not match its own report file. Documentation built on that is worthless
however carefully it is worded.

This opens files for reading and never writes to them. There is no --apply, no backup
directory, no code path that constructs a Nifti1Image. It runs on a directory the caller
names, and the intended target is the assembled deposit itself, so that every number in the
known-issues file was measured on the exact bytes a downloader will receive.

WHAT IT MEASURES, per label, per case:

    voxels, connected components, and the size of the largest
    for each detached component: its size, its distance to the largest, and where that gap
    sits as a fraction of the radius of the imaged cylinder

The last of those separates a rib that leaves the reconstruction circle and returns -- where
the missing bone was never scanned -- from a structure that lost continuity inside a region
the scanner covered.

    python scripts/audit_deposit_labels.py --labels data/zenodo_deposit/labels \
        --out qc_final/deposit_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

VERT = set(range(8, 30)) - {27}
RIBS = set(range(34, 58))
LUMB = {74, 75}
CHECK = VERT | RIBS | LUMB

PIECE_MIN = 100          # report any detached piece at least this big
FOV_EDGE = 0.80          # at or beyond this fraction of the imaged radius: the circle
INTERIOR = 0.55          # at or within this: comfortably inside the imaged region

NAME = {**{i: "T%d" % (i - 7) for i in range(8, 20)},
        **{i: "L%d" % (i - 19) for i in range(20, 26)},
        26: "sacrum", 28: "T13", 29: "S1", 74: "rib_lumbar_left", 75: "rib_lumbar_right"}
for v in range(34, 58):
    NAME[v] = f"rib_{'left' if v < 46 else 'right'}_{(v - 34) % 12 + 1}"


def audit(path: str) -> list[dict]:
    case = Path(path).name.split("_")[0]
    try:
        img = nib.load(str(path))
        lab = np.asanyarray(img.dataobj)
    except Exception as e:                                       # noqa: BLE001
        return [{"case": case, "label": "READ_ERROR", "id": -1, "voxels": 0,
                 "components": 0, "largest": 0, "piece_voxels": 0, "gap_mm": "",
                 "radius_fraction": "", "verdict": "read_error", "note": str(e)[:60]}]

    zooms = np.array(img.header.get_zooms()[:3], float)
    codes = nib.aff2axcodes(img.affine)
    si = [i for i, c in enumerate(codes) if c in "SI"]
    in_plane = [i for i in range(3) if i != si[0]] if si else [0, 1]

    # the imaged cylinder, inferred from how far any label ever reaches; the labels only
    # exist where the scanner imaged, so their outermost reach stands in for the circle
    fg = np.argwhere(lab[::2, ::2, ::2] > 0) * 2
    centre = R = None
    if len(fg) >= 500:
        a, b = in_plane
        centre = (float(fg[:, a].mean()), float(fg[:, b].mean()))
        rr = np.hypot((fg[:, a] - centre[0]) * zooms[a], (fg[:, b] - centre[1]) * zooms[b])
        R = float(np.percentile(rr, 99.8))

    counts = np.bincount(lab.reshape(-1), minlength=256)
    rows = []
    for v in sorted(CHECK):
        tot = int(counts[v]) if v < len(counts) else 0
        if not tot:
            continue
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
        lt, n = ndimage.label(sub)
        sizes = ndimage.sum(sub, lt, range(1, n + 1)) if n else np.array([])
        big_k = int(np.argmax(sizes)) + 1 if n else 0
        big = int(sizes[big_k - 1]) if n else 0

        base = {"case": case, "label": NAME.get(v, str(v)), "id": v, "voxels": tot,
                "components": n, "largest": big}
        if n <= 1:
            rows.append({**base, "piece_voxels": 0, "gap_mm": "", "radius_fraction": "",
                         "verdict": "single_component", "note": ""})
            continue

        core = lt == big_k
        dist, nearest = ndimage.distance_transform_edt(~core, sampling=zooms,
                                                       return_indices=True)
        off = np.array([q.start for q in sl])
        loose = 0
        for k in range(1, n + 1):
            if k == big_k:
                continue
            sz = int(sizes[k - 1])
            if sz < PIECE_MIN:
                loose += sz
                continue
            piece = lt == k
            gap = float(dist[piece].min())
            frac, verdict = "", "detached"
            if R:
                pos = np.argwhere(piece & (dist <= gap + 1e-6))
                if len(pos):
                    q = pos[0]
                    near = np.array([nearest[d][tuple(q)] for d in range(3)])
                    mid = ((q + near) / 2.0 + off) * zooms
                    a, b = in_plane
                    r_mid = float(np.hypot(mid[a] - centre[0] * zooms[a],
                                           mid[b] - centre[1] * zooms[b]))
                    frac = round(r_mid / R, 3)
                    verdict = ("at_reconstruction_circle" if frac >= FOV_EDGE
                               else "inside_imaged_volume" if frac <= INTERIOR
                               else "near_edge_uncertain")
            rows.append({**base, "piece_voxels": sz, "gap_mm": round(gap, 1),
                         "radius_fraction": frac, "verdict": verdict, "note": ""})
        if loose:
            rows.append({**base, "piece_voxels": loose, "gap_mm": "", "radius_fraction": "",
                         "verdict": "loose_voxels",
                         "note": f"voxels in pieces smaller than {PIECE_MIN}"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--out", default="qc_final/deposit_audit.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if not files:
        print(f"  ! no labels under {a.labels}")
        return 1
    print(f"  auditing {len(files)} volume(s) in {a.labels} -- read only")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    cols = ["case", "label", "id", "voxels", "components", "largest", "piece_voxels",
            "gap_mm", "radius_fraction", "verdict", "note"]
    done = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for rows in ex.map(audit, [str(f) for f in files], chunksize=1):
                w.writerows(rows)
                done += 1
                if done % 25 == 0:
                    fh.flush()
                    print(f"  {done}/{len(files)}", flush=True)
    print(f"  wrote {a.out} ({done} volumes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
