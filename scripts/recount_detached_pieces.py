"""Recount KNOWN_ISSUES section 4 against the labels that are actually shipping.

Those figures -- 523 detached pieces across 408 records, 21,474 components in total -- were
measured on v5. v6 changed thirteen records: eleven gained hardware that took voxels away
from the bone labels around it, 0068 was renumbered and gained three thoracic levels, and
1035's hips were lateralised, which merged what used to read as two large pieces into one.

Every one of those edits moves a component count. The numbers are probably close, and
"probably close" is not what a caveat section is for: a reader checks these against their own
run and a mismatch makes them doubt the rest of the file.

An edge-touching piece is reported separately from an interior one, because they are
different facts. A structure leaving the field of view is correctly labelled and merely cut;
a break inside the imaged volume is a defect.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/zenodo_deposit")
    ap.add_argument("--out", default="detached_pieces.json")
    ap.add_argument("--names", default=None, help="dataset_labels.json, for readable names")
    a = ap.parse_args()
    src = Path(a.src)

    names = {}
    nf = Path(a.names) if a.names else (src / "dataset_labels.json")
    if nf.exists():
        j = json.loads(nf.read_text(encoding="utf-8"))
        names = j.get("id_to_name", j) if isinstance(j, dict) else {}

    files = sorted((src / "labels").glob("*_label.nii.gz"))
    print(f"  {len(files)} label volume(s)")

    total_components = 0
    clean = 0
    detached = []          # (case, label id, piece voxels, touches an edge)
    per_record = Counter()

    for i, p in enumerate(files, 1):
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        shape = np.array(arr.shape)
        objs = ndi.find_objects(arr.astype(np.int32))
        for v in (int(x) for x in np.unique(arr) if x):
            sl = objs[v - 1] if v - 1 < len(objs) else None
            if sl is None:
                continue
            # components cannot leave the mask's own extent, so cropping to it is exact
            m = arr[sl] == v
            lab, n = ndi.label(m)
            total_components += n
            if n == 1:
                clean += 1
                continue
            sizes = np.bincount(lab.ravel())[1:]
            main = int(np.argmax(sizes)) + 1
            off = np.array([x.start for x in sl])
            for k in range(1, n + 1):
                if k == main:
                    continue
                # the edge test belongs in the ORIGINAL frame: a piece at the edge of its
                # own crop says nothing, a piece at the edge of the scan is the distinction
                idx = np.argwhere(lab == k) + off
                edge = bool((idx == 0).any() or (idx == shape - 1).any())
                detached.append((p.name[:4], v, int(sizes[k - 1]), edge))
                per_record[p.name[:4]] += 1
        if i % 100 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    at_edge = sum(1 for *_, e in detached if e)
    inside = len(detached) - at_edge

    # a speck along a label boundary and a vertebra split at the pedicle are both "a
    # detached piece"; only the size tells them apart, so bucket before reporting
    BUCKETS = ((0, 10, "under 10 voxels -- a speck"),
               (10, 100, "10-99 -- boundary roughness"),
               (100, 1000, "100-999 -- a fragment"),
               (1000, 10 ** 12, "1000+ -- a real piece of bone"))
    sizes = np.array([sz for _, _, sz, _ in detached]) if detached else np.array([])
    print("")
    print("  detached pieces by size:")
    for lo, hi, name in BUCKETS:
        k = int(((sizes >= lo) & (sizes < hi)).sum()) if sizes.size else 0
        print(f"    {name:<34} {k:>7,}  ({k / max(len(detached), 1):5.1%})")
    big = [(c, v, sz) for c, v, sz, _ in detached if sz >= 1000]
    print(f"  records carrying a piece of 1000+ voxels: "
          f"{len({c for c, _, _ in big})}")
    print(f"\n  detached pieces        : {len(detached)} across {len(per_record)} record(s)")
    print(f"    touching a scan edge : {at_edge} ({at_edge / max(len(detached), 1):.0%})"
          "  -- the structure leaves the volume; the label is right and the anatomy is cut")
    print(f"    inside the volume    : {inside} ({inside / max(len(detached), 1):.0%})"
          "  -- a genuine break")
    print(f"  single-component labels: {clean}")
    print(f"  label components in all: {total_components}")

    top = Counter(names.get(str(v), str(v)) for _, v, _, _ in detached)
    print("\n  commonest labels among the detached pieces:")
    for k, c in top.most_common(6):
        print(f"    {k:<24} {c}")

    Path(a.out).write_text(json.dumps(
        {"detached": len(detached), "records": len(per_record), "at_edge": at_edge,
         "inside": inside, "single_component_labels": clean,
         "total_components": total_components,
         "by_label": dict(top.most_common())}, indent=1), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
