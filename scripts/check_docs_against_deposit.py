"""Does the documentation name anything the deposit does not contain?

Two bugs of this shape have already shipped and been caught by hand: the README described
`fetch_from_tcia.py`, which the assembler was not copying, and KNOWN_ISSUES.md told the
reader to filter on `hardware_labelled`, a manifest field that does not exist. Both read
perfectly well. Both are dead ends for whoever follows them.

So check it mechanically instead of by eye. Every backticked token in the prose is either a
file in the deposit, a field in the manifest, or a word -- and the first two can be verified.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Backticked tokens that are deliberately not files or fields: filename PATTERNS, package
# and tool names, and literal values. Everything outside this set has to resolve, because
# the whole point is that a reader who follows a backticked name gets somewhere.
PROSE = {"null", "true", "false", "None", "NaN", "int16", "uint8", "float32",
         "nii.gz", "PIR", "RAS", "LPS", "dcm2niix", "pip", "python", "bash",
         "sha256sum", "v5", "v6", "main", "ct", "labels", "qc",
         # tools and packages the reader installs, not files we ship
         "tcia_utils", "nibabel", "scipy", "numpy", "huggingface_hub", "sha256sum",
         # scripts that live in the source repository and are named as such
         "mirror_ct_to_hf.py", "assemble_deposit.py", "upload.py",
         # filename patterns, not filenames
         "NNNN_label.nii.gz", "<id>_ct.nii.gz", "<id>_spine_ct.nii.gz",
         "<id>_pelvic_ct.nii.gz"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_deposit")
    ap.add_argument("--docs", nargs="*", default=["README.md", "KNOWN_ISSUES.md"])
    a = ap.parse_args()

    dep = Path(a.deposit)
    files = {str(p.relative_to(dep)).replace("\\", "/") for p in dep.rglob("*") if p.is_file()}
    files |= {p.name for p in dep.iterdir() if p.is_file()}

    man = json.loads((dep / "manifest.json").read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    fields = {k for r in recs for k in r}

    # STRUCTURE NAMES COUNT TOO. `hardware_arthroplasty` is snake_case and is not a manifest
    # field, but it is a real name in dataset_labels.json and a reader who looks it up finds
    # it. Only a name that resolves NOWHERE is a broken reference.
    dl = dep / "dataset_labels.json"
    names = set()
    if dl.exists():
        j = json.loads(dl.read_text(encoding="utf-8"))
        for v in j.values() if isinstance(j, dict) else []:
            if isinstance(v, dict):
                names |= {str(x) for x in v.values()} | {str(x) for x in v}
    print(f"  deposit: {len(files)} path(s), manifest has {len(fields)} field(s), "
          f"{len(names)} structure name(s)")

    bad = 0
    for name in a.docs:
        p = dep / name
        if not p.exists():
            print(f"\n  {name}: NOT IN THE DEPOSIT")
            bad += 1
            continue
        txt = p.read_text(encoding="utf-8")
        toks = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+)`", txt)))
        unknown = []
        for t in toks:
            if t in PROSE or t in files or t in fields or t in names:
                continue
            # a filename that is not in the deposit is the dangerous case
            if re.search(r"\.(py|json|md|txt|sh|nii\.gz|csv)$", t):
                unknown.append((t, "file named in the docs is not in the deposit"))
            # a snake_case token that is not a field is probably a field that moved
            elif re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)+", t):
                unknown.append((t, "looks like a manifest field and is not one"))
        print(f"\n  {name}: {len(toks)} backticked token(s)")
        for t, why in unknown:
            print(f"    MISSING  `{t}`  -- {why}")
            bad += 1
        if not unknown:
            print("    every file and field it names exists")

    # --- the licence, which the deposit states in several places ------------------------
    print()
    print("  licence, as stated in each place it appears:")
    stated = {}
    lic = dep / "LICENSE"
    if lic.exists():
        t = lic.read_text(encoding="utf-8")
        m = re.search(r"CC BY[A-Z-]*[ -]?4\.0", t)
        stated["LICENSE"] = m.group(0) if m else "not found"
    zj = Path("zenodo/zenodo.json")
    if zj.exists():
        j = json.loads(zj.read_text(encoding="utf-8"))
        code = (j.get("license") or j.get("metadata", {}).get("license") or "")
        stated["zenodo.json"] = str(code).upper().replace("CC-", "CC ").replace("-4.0", " 4.0")
    rd = dep / "README.md"
    if rd.exists():
        for m in set(re.findall(r"CC BY[A-Z-]*[ -]?4\.0", rd.read_text(encoding="utf-8"))):
            stated.setdefault("README.md", m)
            if m != stated["README.md"]:
                stated["README.md (also)"] = m
    norm = lambda v: v.upper().replace("-", " ").replace("  ", " ").strip()
    for k, v in stated.items():
        print(f"    {k:<22} {v}")
    if len({norm(v) for v in stated.values()}) > 1:
        print("    THE DEPOSIT DISAGREES WITH ITSELF ABOUT ITS OWN LICENCE")
        bad += 1
    elif stated:
        print("    all agree")

    print()
    print("  the docs and the deposit agree" if not bad
          else f"  {bad} problem(s) found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
