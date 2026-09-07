"""zenodo_newversion.py -- publish a NEW VERSION of the existing CTSpinoPelvic1K record.

NOT a new deposit. CTSpinoPelvic1K is already on Zenodo as record 22139643 under concept DOI
10.5281/zenodo.22139642, and the concept DOI is what the paper cites. Creating a fresh
deposit would mint an unrelated DOI and orphan that citation; a new version keeps the concept
DOI stable and mints a version DOI beneath it.

A new version inherits the previous version's files, so they are deleted from the draft
before the new ones go up -- otherwise the record ends up holding both releases at once.

    python zenodo_newversion.py --dir data/zenodo_v7 --record 22139643 --version v7
    python zenodo_newversion.py ... --publish        # irreversible; do it deliberately
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--record", default="22139643")
    ap.add_argument("--version", required=True)
    ap.add_argument("--metadata", default="zenodo/zenodo.json")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.zenodo_token"))
    ap.add_argument("--publish", action="store_true",
                    help="publish the draft. IRREVERSIBLE: a published record cannot be "
                         "deleted or edited, only superseded by another version.")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    tok = Path(a.token_file).read_text().strip()
    s = requests.Session()
    s.params = {"access_token": tok}

    src = Path(a.dir)
    lab_dir = src / "labels"
    zip_path = src / "labels.zip"
    if lab_dir.is_dir() and not zip_path.exists():
        import zipfile
        n = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(lab_dir.iterdir()):
                if f.is_file():
                    zf.write(f, f"labels/{f.name}")
                    n += 1
        print(f"  packed {n} labels -> labels.zip "
              f"({zip_path.stat().st_size/1e9:.2f} GB)")
    files = sorted(p for p in src.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"  deposit  : {src}  ({len(files)} files, {total/1e9:.2f} GB)")

    # ---- new version of the existing record ---------------------------------
    r = s.post(f"{API}/deposit/depositions/{a.record}/actions/newversion")
    if r.status_code not in (201, 200):
        print(f"  newversion failed: {r.status_code} {r.text[:400]}")
        return 1
    draft_url = r.json()["links"]["latest_draft"]
    draft = s.get(draft_url).json()
    dep_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"  draft    : {dep_id}   (concept {draft.get('conceptdoi')})")

    # ---- clear inherited files ----------------------------------------------
    existing = s.get(f"{API}/deposit/depositions/{dep_id}/files").json()
    if isinstance(existing, list) and existing:
        print(f"  clearing {len(existing)} inherited file(s) from the draft")
        for f in existing:
            s.delete(f"{API}/deposit/depositions/{dep_id}/files/{f['id']}")

    # ---- upload -------------------------------------------------------------
    for i, p in enumerate(files, 1):
        rel = p.relative_to(src).as_posix()
        with open(p, "rb") as fh:
            up = s.put(f"{bucket}/{rel}", data=fh)
        if up.status_code not in (200, 201):
            print(f"  upload FAILED {rel}: {up.status_code} {up.text[:200]}")
            return 1
        got = up.json().get("checksum", "").replace("md5:", "")
        if got and got != md5(p):
            print(f"  CHECKSUM MISMATCH on {rel}")
            return 1
        if i % 50 == 0 or i == len(files):
            print(f"    {i}/{len(files)}  {rel}", flush=True)

    # ---- metadata ------------------------------------------------------------
    meta = json.loads(Path(a.metadata).read_text())
    meta["version"] = a.version
    meta["publication_date"] = time.strftime("%Y-%m-%d")
    if a.note:
        meta["description"] = meta.get("description", "") + a.note
    r = s.put(f"{API}/deposit/depositions/{dep_id}",
              data=json.dumps({"metadata": meta}),
              headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        print(f"  metadata failed: {r.status_code} {r.text[:400]}")
        return 1
    print(f"  metadata : version={a.version}")

    d = s.get(f"{API}/deposit/depositions/{dep_id}").json()
    print(f"  draft DOI: {d.get('doi') or d.get('metadata', {}).get('prereserve_doi', {}).get('doi')}")
    print(f"  review at: https://zenodo.org/uploads/{dep_id}")

    if not a.publish:
        print("\n  NOT published (no --publish). The draft is complete and reviewable.")
        return 0

    r = s.post(f"{API}/deposit/depositions/{dep_id}/actions/publish")
    if r.status_code not in (202, 200, 201):
        print(f"  publish failed: {r.status_code} {r.text[:400]}")
        return 1
    pub = r.json()
    print("\n  PUBLISHED")
    print(f"    version DOI : {pub.get('doi')}")
    print(f"    concept DOI : {pub.get('conceptdoi')}")
    print(f"    record      : https://zenodo.org/records/{pub.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
