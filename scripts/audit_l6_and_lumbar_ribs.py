"""How many records actually carry L6, T13 and lumbar ribs, counted in the label volumes?

Three sources disagree and at most one can be right. The deposit description claims 17 records
with L6 and 15 with a lumbar rib. The shipped manifest says has_l6 is true in 1 record and
has_lumbar_rib in none, with n_lumbar_labels zero in 799 of 802 -- which cannot be true of a
lumbar spine dataset and marks those fields as broken rather than surprising.

Users filter on exactly these fields. A field that reads false everywhere silently removes
the entire population someone came here to study, and they would have no way to notice.

So count in the volumes, which cannot be stale: id 25 is L6 and 28 is T13 in the VerSe
numbering this release inherits, 74 and 75 are the left and right lumbar rib classes, and
20-25 are the lumbar column. Reports what the manifest claims beside what the voxels show.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

L6, T13 = 25, 28
LUMBAR = list(range(20, 26))
RIB_LUMBAR = (74, 75)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v6")
    ap.add_argument("--out", default="data/l6_audit.json")
    a = ap.parse_args()
    src = Path(a.src)

    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    claim = {str(r["volume_id"]): r for r in recs}

    files = sorted((src / "labels").glob("*_label.nii.gz"))
    print(f"  scanning {len(files)} volume(s)")

    has_l6, has_t13, has_rib = [], [], []
    n_lumbar = Counter()
    truth = {}
    for i, p in enumerate(files, 1):
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        present = {int(v) for v in np.unique(arr) if v}
        cid = p.name[:4]
        nl = sum(1 for v in LUMBAR if v in present)
        n_lumbar[nl] += 1
        rec = {"l6": L6 in present, "t13": T13 in present,
               "lumbar_rib": any(v in present for v in RIB_LUMBAR),
               "n_lumbar": nl}
        truth[cid] = rec
        if rec["l6"]:
            has_l6.append(cid)
        if rec["t13"]:
            has_t13.append(cid)
        if rec["lumbar_rib"]:
            has_rib.append(cid)
        if i % 200 == 0:
            print(f"    {i}/{len(files)}", flush=True)

    print(f"\n  L6 (id 25) present        : {len(has_l6)} record(s)")
    print(f"    {', '.join(has_l6[:24])}{' ...' if len(has_l6) > 24 else ''}")
    print(f"  T13 (id 28) present       : {len(has_t13)} record(s)")
    if has_t13:
        print(f"    {', '.join(has_t13[:24])}")
    print(f"  lumbar rib (74/75) present: {len(has_rib)} record(s)")
    if has_rib:
        print(f"    {', '.join(has_rib[:24])}{' ...' if len(has_rib) > 24 else ''}")
    print(f"  lumbar labels per record  : "
          f"{dict(sorted(n_lumbar.items()))}")

    # what the manifest asserts about the same records
    m_l6 = {c for c, r in claim.items() if r.get("has_l6")}
    m_rib = {c for c, r in claim.items() if r.get("has_lumbar_rib")}
    print(f"\n  manifest has_l6 true         : {len(m_l6)}  {sorted(m_l6)}")
    print(f"  manifest has_lumbar_rib true : {len(m_rib)}  {sorted(m_rib)}")
    print(f"  volumes with L6 the manifest misses    : "
          f"{sorted(set(has_l6) - m_l6)[:20]}")
    print(f"  manifest claims L6 the volume lacks    : "
          f"{sorted(m_l6 - set(has_l6))}")
    print(f"  volumes with a lumbar rib the manifest misses: "
          f"{sorted(set(has_rib) - m_rib)[:20]}")

    Path(a.out).write_text(json.dumps(truth, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
