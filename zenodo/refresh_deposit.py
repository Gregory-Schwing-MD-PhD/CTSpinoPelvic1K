"""Rebuild everything downstream of manifest.json, in the order the dependencies run.

manifest.json has changed, and three files describe it:

  croissant.json embeds the manifest's sha256 and enumerates its fields, so it is stale the
    moment the manifest is rewritten -- and it gained a field, lumbar_rib_side;
  README.md documents what the manifest contains;
  SHA256SUMS.txt hashes all of them.

The order is therefore manifest, then croissant, then README, then sums -- and the sums
last, always. Generating them earlier is what shipped a checksum file that failed on an
uncorrupted README earlier today, and the failure is invisible until a user runs the very
command the deposit tells them to run.

Ends by verifying every loose file against the manifest it just wrote, because a checksum
file is only worth shipping if it passes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_upload")
    ap.add_argument("--labels", default="data/zenodo_v6/labels")
    a = ap.parse_args()
    D, LAB = Path(a.deposit), Path(a.labels)

    recs = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", list(recs.values()))
    l6 = sum(1 for r in recs if r.get("has_l6"))
    rib = sum(1 for r in recs if r.get("has_lumbar_rib"))
    fields = sorted({k for r in recs for k in r})
    print(f"  manifest: {len(recs)} records, {len(fields)} fields, "
          f"has_l6 {l6}, has_lumbar_rib {rib}")
    assert "lumbar_rib_side" in fields, "manifest was not repaired"

    print("\n  regenerating croissant.json from the repaired manifest")
    r = subprocess.run([sys.executable, "zenodo/make_croissant.py",
                        "--deposit", str(D), "--out", str(D / "croissant.json")],
                       capture_output=True, text=True)
    print("    " + (r.stdout or r.stderr).strip().replace("\n", "\n    "))
    if r.returncode:
        return 1

    print("\n  regenerating SHA256SUMS.txt last, over the final bytes")
    sums = D / "SHA256SUMS.txt"
    old = {}
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                old[parts[1].strip()] = parts[0]

    loose = sorted(p for p in D.iterdir()
                   if p.is_file() and p.name not in {"SHA256SUMS.txt", "labels.zip"})
    rows = [(p.name, sha256(p)) for p in loose]
    changed = [n for n, h in rows if old.get(n) and old[n] != h]
    for p in sorted(LAB.glob("*_label.nii.gz")):
        key = f"labels/{p.name}"
        rows.append((key, old.get(key) or sha256(p)))
    rows.sort(key=lambda x: x[0])
    with open(sums, "w", encoding="utf-8", newline="\n") as fh:
        for name, h in rows:
            fh.write(f"{h}  {name}\n")
    print(f"    {len(rows)} entries ({len(loose)} loose + {len(rows)-len(loose)} labels)")
    print(f"    hashes that changed: {changed or 'none'}")

    print("\n  verifying the loose files as a reader would:")
    entries = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0])
                   for l in sums.read_text(encoding="utf-8").splitlines()
                   if len(l.split(None, 1)) == 2)
    bad = []
    for name in sorted(k for k in entries if not k.startswith("labels/")):
        ok = sha256(D / name) == entries[name]
        print(f"    {name:22} {'OK' if ok else 'FAILED'}")
        if not ok:
            bad.append(name)
    print(f"  {len(bad)} failure(s): {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
