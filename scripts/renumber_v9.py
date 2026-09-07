"""scripts/renumber_v9.py -- close the gap in the label space (v9).

Through v8 the lumbar ribs sat at 74/75 and the hardware block at 76..82, above a retired
and never-populated block (58..73). v9 removes the gap: lumbar ribs become 58/59 and the
hardware block 60..66, so the identifier space is contiguous from 0 to 66 (27 coccyx and
28 T13 are VerSe identifiers and stay). The partial-annotation sentinel (255), declared
and never used, is dropped from the scheme.

    python scripts/renumber_v9.py --labels data/zenodo_deposit/labels --out data/zenodo_deposit/labels_v9
    python scripts/renumber_v9.py --manifest data/zenodo_deposit/manifest.json

The volume pass writes to --out and never touches the input; swap the directories by hand
once the census on --out matches. Arrays are read and written on the grid they are stored
on (nothing is reoriented: see never-reorient-when-writing-labels). Before remapping, every
volume is checked to contain no identifier in 58..73 and no 255, so the remap cannot
collide with anything already present. The identifiers present in each volume (after the
remap) are written to <out>_presence.json, so the census questions -- which record lacks
T12, which lacks S1 -- are answered by the same pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

OLD_TO_NEW = {74: 58, 75: 59, 76: 60, 77: 61, 78: 62, 79: 63, 80: 64, 81: 65, 82: 66}
FORBIDDEN_BEFORE = set(range(58, 74)) | {255}

nib.openers.Opener.default_compresslevel = 3     # 6 by default; the archive is re-zipped anyway


def lut() -> np.ndarray:
    t = np.arange(256, dtype=np.int64)
    for a, b in OLD_TO_NEW.items():
        t[a] = b
    return t


def one(args):
    f, dst = args
    f, dst = Path(f), Path(dst)
    img = nib.load(str(f))
    arr = np.asanyarray(img.dataobj)
    present = set(np.unique(arr).tolist())
    bad = present & FORBIDDEN_BEFORE
    if bad:
        return f.name, None, f"carries {sorted(bad)} before the remap"
    if arr.max() > 255:
        return f.name, None, "identifier above 255"
    out = lut()[arr.astype(np.int64)].astype(arr.dtype)
    nib.Nifti1Image(out, img.affine, img.header).to_filename(str(dst / f.name))
    after = sorted(OLD_TO_NEW.get(int(i), int(i)) for i in present)
    return f.name, after, None


def remap_volumes(src: Path, dst: Path, workers: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*_label.nii.gz"))
    presence = {}
    errors = []
    with Pool(workers) as pool:
        for k, (name, after, err) in enumerate(pool.imap_unordered(one, [(str(f), str(dst)) for f in files]), 1):
            if err:
                errors.append(f"{name}: {err}")
            else:
                presence[name.split("_")[0]] = after
            if k % 50 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    if errors:
        sys.exit("refused:\n  " + "\n  ".join(errors))
    Path(str(dst) + "_presence.json").write_text(json.dumps(presence, indent=0), encoding="utf-8")
    changed = sum(1 for v in presence.values() if set(v) & set(OLD_TO_NEW.values()))
    print(f"wrote {len(files)} volumes to {dst}; {changed} carry a remapped identifier")
    for name, i in [("T12", 19), ("L5", 24), ("S1", 29), ("sacrum", 26)]:
        missing = sorted(t for t, v in presence.items() if i not in v)
        print(f"  records lacking {name} ({i}): {len(missing)} {missing[:12]}")


def remap_manifest(path: Path) -> None:
    m = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for r in m:
        ids = r.get("hardware_label_ids")
        if ids:
            r["hardware_label_ids"] = [OLD_TO_NEW.get(i, i) for i in ids]
            n += 1
    path.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"manifest: hardware_label_ids remapped in {n} records")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.labels:
        remap_volumes(Path(a.labels), Path(a.out or (a.labels + "_v9")), a.workers)
    if a.manifest:
        remap_manifest(Path(a.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
