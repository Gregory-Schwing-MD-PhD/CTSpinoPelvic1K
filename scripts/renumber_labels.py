"""scripts/renumber_labels.py -- apply a published-version remap to every label volume.

    python scripts/renumber_labels.py --map v10 --labels data/zenodo_deposit/labels --out data/zenodo_deposit/labels_v10
    python scripts/renumber_labels.py --map v10 --manifest data/zenodo_deposit/manifest.json

The remaps live in label_scheme (OLD_TO_NEW_V9, OLD_TO_NEW_V10) so that the code and the
release notes cannot disagree about what moved where. The volume pass writes to --out and
never touches the input; swap the directories once the census on --out matches (see
zenodo/finalize_v9.py, which also builds labels.zip and SHA256SUMS.txt). Arrays are read
and written on the grid they are stored on (nothing is reoriented). Before remapping,
every volume is checked to contain no identifier outside the OLD scheme (above --old-max,
or 255), so nothing can be remapped by accident. The identifiers present in each volume
after the remap are written to <out>_presence.json.

Supersedes scripts/renumber_v9.py, which carried the v9 table inline.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402

MAPS = {"v9": (LS.OLD_TO_NEW_V9, 82), "v10": (LS.OLD_TO_NEW_V10, 66)}   # (table, old MAX_ID)

nib.openers.Opener.default_compresslevel = 3


def lut(table):
    t = np.arange(256, dtype=np.int64)
    for a, b in table.items():
        t[a] = b
    return t


def one(args):
    f, dst, name = args
    table, old_max = MAPS[name]
    f, dst = Path(f), Path(dst)
    img = nib.load(str(f))
    arr = np.asanyarray(img.dataobj)
    present = set(int(i) for i in np.unique(arr))
    bad = {i for i in present if i > old_max}
    if bad:
        return f.name, None, f"carries {sorted(bad)}, outside the old scheme"
    out = lut(table)[arr.astype(np.int64)].astype(arr.dtype)
    nib.Nifti1Image(out, img.affine, img.header).to_filename(str(dst / f.name))
    return f.name, sorted(table.get(i, i) for i in present), None


def remap_volumes(src, dst, name, workers):
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*_label.nii.gz"))
    presence, errors = {}, []
    with Pool(workers) as pool:
        for k, (fn, after, err) in enumerate(pool.imap_unordered(one, [(str(f), str(dst), name) for f in files]), 1):
            if err:
                errors.append(f"{fn}: {err}")
            else:
                presence[fn.split("_")[0]] = after
            if k % 100 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    if errors:
        sys.exit("refused:\n  " + "\n  ".join(errors))
    Path(str(dst) + "_presence.json").write_text(json.dumps(presence, indent=0), encoding="utf-8")
    print(f"wrote {len(files)} volumes to {dst}; max identifier "
          f"{max(max(v) for v in presence.values())} (scheme max {LS.MAX_ID})")


def remap_manifest(path, name):
    table = MAPS[name][0]
    m = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for r in m:
        ids = r.get("hardware_label_ids")
        if ids:
            r["hardware_label_ids"] = [table.get(i, i) for i in ids]
            n += 1
    path.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"manifest: hardware_label_ids remapped in {n} records")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, choices=sorted(MAPS))
    ap.add_argument("--labels")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.labels:
        remap_volumes(Path(a.labels), Path(a.out or (a.labels + "_" + a.map)), a.map, a.workers)
    if a.manifest:
        remap_manifest(Path(a.manifest), a.map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
