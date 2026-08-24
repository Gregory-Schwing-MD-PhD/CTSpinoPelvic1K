"""zenodo/upload.py — create the Zenodo deposition, upload the files, stop before publishing.

    export ZENODO_TOKEN=...            # zenodo.org/account/settings/applications/tokens
    python zenodo/upload.py --dir data/zenodo_deposit --reserve-doi
    # inspect the draft in the browser, then publish there, or:
    python zenodo/upload.py --publish 1234567

IT DOES NOT PUBLISH. Reserving a DOI and uploading files are reversible; publishing is not.
A published Zenodo record cannot be deleted or edited, only superseded by a new version, so
the last step is left to a human who has looked at the draft. `--publish` exists for when
that has happened and is deliberately a separate invocation with the record id typed out.

RESERVE THE DOI BEFORE THE PAPER IS FINAL. `--reserve-doi` returns a DOI that is quotable
immediately and resolves once the record is published, which is what lets \\datasetdoi in the
manuscript stop being a placeholder. Deleting the draft destroys that DOI and its files, so
do not reserve twice.

SCOPE. Zenodo's default limit is 50 GB per record; this deposit is 1.8 GB and needs no quota
request. That is only true because the CT images are not in it -- see README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

BASE = "https://zenodo.org/api"


def _tok():
    t = os.environ.get("ZENODO_TOKEN")
    if not t:
        print("  ! ZENODO_TOKEN is not set.")
        print("  Create one at zenodo.org/account/settings/applications/tokens with the")
        print("  deposit:write and deposit:actions scopes, then export it.")
        sys.exit(2)
    return t


def _check(r, what):
    if r.status_code >= 300:
        print(f"  ! {what} failed: HTTP {r.status_code}")
        try:
            print(f"    {json.dumps(r.json(), indent=2)[:900]}")
        except Exception:                                             # noqa: BLE001
            print(f"    {r.text[:500]}")
        sys.exit(1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/zenodo_deposit")
    ap.add_argument("--metadata", default="zenodo/zenodo.json")
    ap.add_argument("--reserve-doi", action="store_true")
    ap.add_argument("--publish", default=None, metavar="RECORD_ID",
                    help="publish an existing draft. Irreversible.")
    a = ap.parse_args()
    tok = _tok()
    params = {"access_token": tok}

    if a.publish:
        print(f"  publishing record {a.publish} -- this cannot be undone")
        r = _check(requests.post(f"{BASE}/deposit/depositions/{a.publish}/actions/publish",
                                 params=params, timeout=120), "publish")
        d = r.json()
        print(f"  published: {d.get('doi_url') or d.get('doi')}")
        return 0

    src = Path(a.dir)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"  {len(files)} file(s), {total / 1e9:.2f} GB from {src}")
    if total > 50e9:
        print("  ! over Zenodo's 50 GB default; request a quota increase first")
        return 1

    meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))

    r = _check(requests.post(f"{BASE}/deposit/depositions", params=params,
                             json={}, timeout=60), "create deposition")
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"  deposition {dep_id}")

    if a.reserve_doi:
        meta = dict(meta)
        meta["prereserve_doi"] = True

    r = _check(requests.put(f"{BASE}/deposit/depositions/{dep_id}",
                            params=params, json={"metadata": meta}, timeout=60),
               "set metadata")
    got = r.json().get("metadata", {})
    doi = (got.get("prereserve_doi") or {}).get("doi")
    if doi:
        print(f"  reserved DOI: {doi}")
        print(f"  put this in the manuscript: {B_DOI_HINT}")

    # bucket upload: streams, so a 1.8 GB deposit does not need to fit in memory
    for i, p in enumerate(files, 1):
        rel = p.relative_to(src).as_posix()
        with p.open("rb") as fh:
            _check(requests.put(f"{bucket}/{rel}", data=fh, params=params, timeout=None),
                   f"upload {rel}")
        if i % 50 == 0 or i == len(files):
            print(f"    {i}/{len(files)}", flush=True)

    print(f"\n  draft ready: https://zenodo.org/deposit/{dep_id}")
    print("  NOT PUBLISHED. Look at the draft, then publish in the browser or run")
    print(f"    python zenodo/upload.py --publish {dep_id}")
    return 0


B_DOI_HINT = r"\newcommand{\datasetdoi}{<the reserved DOI>}"

if __name__ == "__main__":
    sys.exit(main())
