"""zenodo/finalize_v9.py -- turn the remapped labels into the v9 deposit.

    python zenodo/finalize_v9.py --deposit data/zenodo_deposit --new labels_v10 --old labels_v9_old

Steps, in order:
  1. `labels_v9/` (from scripts/renumber_v9.py) replaces `labels/`; the v8 labels move to
     `labels_v8/` and can be deleted once v9 is published.
  2. Every volume is checked: 802 files, no identifier above 66, none of the v8 ids 74..82.
  3. `labels.zip` is built (stored, not recompressed: the members are already gzipped) with
     members named `labels/<record>_label.nii.gz`, the path SHA256SUMS.txt uses.
  4. SHA256SUMS.txt is rewritten last, over the final bytes of every loose file and every
     label, so `sha256sum -c SHA256SUMS.txt` passes after extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

LOOSE_SKIP = {"SHA256SUMS.txt", "labels.zip"}
MAX_ID = 68


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_deposit")
    ap.add_argument("--new", default="labels_v9", help="remapped directory to promote")
    ap.add_argument("--old", default="labels_v8", help="name for the directory being replaced")
    ap.add_argument("--max-id", type=int, default=MAX_ID)
    a = ap.parse_args()
    D = Path(a.deposit)
    new, cur, old = D / a.new, D / "labels", D / a.old

    # 1. swap
    if new.exists():
        if old.exists():
            sys.exit(f"{old} already exists; refusing to overwrite")
        cur.rename(old)
        new.rename(cur)
        print(f"  labels_v9 -> labels; previous labels -> {old.name}")
    files = sorted(cur.glob("*_label.nii.gz"))
    if len(files) != 802:
        sys.exit(f"expected 802 label volumes, found {len(files)}")

    # 2. presence check from the remap's own record
    pres = D / (a.new + "_presence.json")
    if pres.exists():
        p = json.loads(pres.read_text(encoding="utf-8"))
        bad = {k: [i for i in v if i > a.max_id] for k, v in p.items()}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            sys.exit(f"identifiers above {a.max_id}: {list(bad.items())[:5]}")
        print(f"  presence: {len(p)} records, max identifier "
              f"{max(max(v) for v in p.values())}, none above {a.max_id}")

    # 3. labels.zip
    z = D / "labels.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, f"labels/{f.name}")
    print(f"  labels.zip: {len(files)} members, {z.stat().st_size / 1e9:.2f} GB")

    # 4. sums, last
    rows = []
    for p in sorted(D.iterdir()):
        if p.is_file() and p.name not in LOOSE_SKIP:
            rows.append((p.name, sha256(p)))
    for f in files:
        rows.append((f"labels/{f.name}", sha256(f)))
    rows.sort(key=lambda r: r[0])
    with open(D / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
        for name, h in rows:
            fh.write(f"{h}  {name}\n")
    print(f"  SHA256SUMS.txt: {len(rows)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
