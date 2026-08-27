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
    print(f"  deposit: {len(files)} path(s), manifest has {len(fields)} field(s)")

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
            if t in PROSE or t in files or t in fields:
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

    print()
    print("  the docs and the deposit agree" if not bad
          else f"  {bad} reference(s) point at nothing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
