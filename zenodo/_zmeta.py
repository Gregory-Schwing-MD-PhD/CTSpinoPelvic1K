"""Push the corrected byline to the draft, and see what it says about the README.

Two things at once, because they touch the same record:

  THE BYLINE. zenodo.json now carries all fourteen manuscript authors instead of one. This
  PUTs the metadata; it does not touch files, so an upload in progress in the browser is
  unaffected.

  THE README. It is reported as not rendering in the preview. Zenodo's previewer handles
  Markdown, so the likely causes are that the file is not on the draft yet, that it is not
  the file the previewer defaults to (the preview slot shows one chosen file, usually the
  first or largest), or that its type was not detected. Listing what the draft holds
  distinguishes those without guessing.

Prints the metadata back as the server stored it, since what was sent and what was accepted
are not always the same thing.
"""
import json
import os

import requests

t = os.environ["ZENODO_TOKEN"]
P = {"access_token": t}
DEP = 22139643
BASE = f"https://zenodo.org/api/deposit/depositions/{DEP}"

meta = json.loads(open("zenodo/zenodo.json", encoding="utf-8").read())
r = requests.put(BASE, params=P, json={"metadata": meta}, timeout=90)
print(f"  PUT metadata: HTTP {r.status_code}")
if r.status_code >= 300:
    print("  " + r.text[:400])
    raise SystemExit(1)

got = r.json().get("metadata", {})
cr = got.get("creators", [])
print(f"  creators now on the record: {len(cr)}")
for c in cr:
    print(f"    {c.get('name')}")
print(f"  licence: {got.get('license')}")
print(f"  title  : {str(got.get('title'))[:70]}")

print("\n  files on the draft:")
rf = requests.get(f"{BASE}/files", params=P, timeout=60)
files = rf.json() if rf.status_code < 300 else []
for f in sorted(files, key=lambda x: str(x.get("filename"))):
    print(f"    {str(f.get('filename')):22} {(f.get('filesize') or 0) / 1e6:9.2f} MB")
print(f"  {len(files)} file(s)")
names = {str(f.get("filename")) for f in files}
if "README.md" not in names:
    print("\n  README.md is NOT on the draft yet -- nothing to render.")
else:
    print("\n  README.md is present, so the previewer has a file to work with.")
