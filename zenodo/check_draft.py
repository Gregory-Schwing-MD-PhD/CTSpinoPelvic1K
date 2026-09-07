"""zenodo/check_draft.py -- compare a draft deposition's files with a local SHA256SUMS + sizes list.

    ZENODO_TOKEN=... python zenodo/check_draft.py --deposition 22647933 --sizes sizes.txt

sizes.txt lines: "<bytes> <md5> <filename>" (made on the machine that holds the files:
    for f in *; do printf '%s %s %s\n' "$(stat -c %s "$f")" "$(md5sum "$f" | cut -d' ' -f1)" "$f"; done).
Exit 0 only when every local file is present with the same size and md5.
"""
import argparse, json, os, sys
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--deposition", required=True)
ap.add_argument("--sizes", required=True)
a = ap.parse_args()
tok = os.environ.get("ZENODO_TOKEN") or sys.exit("ZENODO_TOKEN not set")
r = requests.get(f"https://zenodo.org/api/deposit/depositions/{a.deposition}/files",
                 params={"access_token": tok}, timeout=60)
r.raise_for_status()
remote = {f["filename"]: (int(f["filesize"]), f.get("checksum", "")) for f in r.json()}
bad = 0
for line in open(a.sizes):
    parts = line.split()
    if len(parts) < 3:
        continue
    size, md5, name = int(parts[0]), parts[1], parts[2]
    rs = remote.get(name)
    ok = rs is not None and rs[0] == size and rs[1].replace("md5:", "") == md5
    print(("ok  " if ok else "BAD ") + f"{name:28s} local {size:>12d} {md5[:8]}  remote {rs}")
    bad += 0 if ok else 1
print(f"{len(remote)} remote files; {bad} mismatches")
sys.exit(1 if bad else 0)
