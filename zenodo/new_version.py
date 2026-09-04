"""zenodo/new_version.py -- a new version of the published record, changing only named files.

    export ZENODO_TOKEN=...
    python zenodo/new_version.py --record 22242745 --version v8 \
        --dir data/zenodo_deposit \
        --replace manifest.json dataset_labels.json README.md KNOWN_ISSUES.md SHA256SUMS.txt \
        --changelog "Castellvi grades are the consensus of two readers; ..."
    # inspect https://zenodo.org/deposit/<draft id>, then:
    python zenodo/upload.py --publish <draft id>

WHAT A NEW VERSION IS ON ZENODO. POST .../actions/newversion on the latest published record
returns a DRAFT that already carries every file of the old version. Files that did not
change (labels.zip, 1.2 GB) stay as they are; the files named in --replace are deleted from
the draft and uploaded fresh from --dir. The concept DOI is unchanged; the draft gets its
own version DOI on publish.

IT DOES NOT PUBLISH, for the same reason upload.py does not: publishing cannot be undone.
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
    t = os.environ.get("ZENODO_TOKEN") or (Path.home() / ".zenodo_token").read_text().strip()
    if not t:
        sys.exit("ZENODO_TOKEN not set and ~/.zenodo_token missing")
    return t


def _check(r, what):
    if r.status_code >= 300:
        print(f"  ! {what}: HTTP {r.status_code}\n    {r.text[:800]}")
        sys.exit(1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="id of the latest PUBLISHED version")
    ap.add_argument("--version", required=True)
    ap.add_argument("--dir", default="data/zenodo_deposit")
    ap.add_argument("--replace", nargs="+", required=True, help="filenames to replace")
    ap.add_argument("--changelog", default="", help="prepended to the description")
    ap.add_argument("--draft", default=None, help="resume into an existing draft id")
    a = ap.parse_args()
    params = {"access_token": _tok()}

    if a.draft:
        dep = _check(requests.get(f"{BASE}/deposit/depositions/{a.draft}", params=params,
                                  timeout=60), "open draft").json()
    else:
        r = _check(requests.post(f"{BASE}/deposit/depositions/{a.record}/actions/newversion",
                                 params=params, timeout=120), "newversion")
        latest_draft = r.json()["links"]["latest_draft"]
        dep = _check(requests.get(latest_draft, params=params, timeout=60), "open draft").json()
    dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    print(f"  draft {dep_id}  (concept {dep.get('conceptrecid')})")

    # metadata: keep everything, bump the version, prepend the changelog
    meta = dict(dep["metadata"])
    meta["version"] = a.version
    meta.pop("doi", None); meta.pop("prereserve_doi", None)
    if a.changelog:
        meta["description"] = f"<p><b>{a.version}:</b> {a.changelog}</p>" + meta.get("description", "")
    _check(requests.put(f"{BASE}/deposit/depositions/{dep_id}", params=params,
                        json={"metadata": meta}, timeout=60), "set metadata")

    # files: delete the named ones from the draft, upload the local ones
    files = _check(requests.get(f"{BASE}/deposit/depositions/{dep_id}/files", params=params,
                                timeout=60), "list files").json()
    byname = {f["filename"]: f for f in files}
    for name in a.replace:
        if name in byname:
            _check(requests.delete(byname[name]["links"]["self"], params=params, timeout=60),
                   f"delete {name}")
        src = Path(a.dir) / name
        if not src.is_file():
            sys.exit(f"  ! {src} missing")
        with open(src, "rb") as fh:
            _check(requests.put(f"{bucket}/{name}", data=fh, params=params, timeout=600),
                   f"upload {name}")
        print(f"    replaced {name} ({src.stat().st_size:,} bytes)")
    kept = sorted(set(byname) - set(a.replace))
    print(f"  kept from {a.record}: {kept}")
    print(f"\n  draft ready: https://zenodo.org/deposit/{dep_id}")
    print(f"  NOT PUBLISHED. Inspect it, then: python zenodo/upload.py --publish {dep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
