"""zenodo_fix_title.py -- bring the deposit title in line with the manuscript.

The v7 record says "ribs and femurs" where the manuscript, the title page and the label
documentation all say "femora". Zenodo metadata stays editable after publication, so this is
one field rather than a new version -- and it must NOT become a new version, because the
manuscript cites 10.5281/zenodo.22242745 as the exact release it describes. A new version
would mint a new DOI and leave that citation pointing at a superseded record.

The edit/publish cycle on an already-published record keeps the same DOI: the edit action
reopens the metadata, the publish action closes it again. Nothing about the files changes.

    python scripts/zenodo_fix_title.py            # show what would change
    python scripts/zenodo_fix_title.py --apply    # do it
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

RECORD = "22242745"
API = "https://zenodo.org/api/deposit/depositions"


def token() -> str:
    for src in (os.environ.get("ZENODO_TOKEN"),):
        if src:
            return src.strip()
    for p in (Path.home() / ".zenodo_token",
              Path(__file__).resolve().parents[1] / ".env"):
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if p.name == ".env":
            for line in txt.splitlines():
                if line.startswith("ZENODO_TOKEN"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        else:
            return txt.strip()
    sys.exit("no Zenodo token found (env ZENODO_TOKEN, ~/.zenodo_token, or repo .env)")


def call(method: str, url: str, tok: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:500]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    tok = token()

    dep = call("GET", f"{API}/{RECORD}", tok)
    old = dep["metadata"]["title"]
    new = old.replace("femurs", "femora")

    print(f"record {RECORD}   doi {dep.get('doi')}   version "
          f"{dep['metadata'].get('version')}   state {dep.get('state')}")
    print(f"  before: {old}")
    print(f"  after : {new}")
    if old == new:
        sys.exit("\nnothing to change -- the title already says femora")
    if not a.apply:
        print("\ndry run; pass --apply to write it")
        return

    call("POST", f"{API}/{RECORD}/actions/edit", tok)      # reopen, same DOI
    md = dict(dep["metadata"], title=new)
    call("PUT", f"{API}/{RECORD}", tok, {"metadata": md})
    out = call("POST", f"{API}/{RECORD}/actions/publish", tok)

    check = call("GET", f"{API}/{RECORD}", tok)
    print(f"\npublished. doi {out.get('doi')} (unchanged: {out.get('doi') == dep.get('doi')})")
    print(f"title now: {check['metadata']['title']}")


if __name__ == "__main__":
    main()
