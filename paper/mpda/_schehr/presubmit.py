"""Pre-submission check: read the typeset reference list back out of the PDF and look for
the things that actually stop a submission."""
import re
import sys
import io
from pathlib import Path

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MPDA = Path(__file__).resolve().parent.parent
tex = (MPDA / "main.tex").read_text(encoding="utf-8")

print("=" * 90)
print("PLACEHOLDERS AND UNFINISHED MARKERS IN THE SOURCE")
print("=" * 90)
pats = {
    "PENDING": r"PENDING", "TODO": r"TODO", "XXX": r"XXXX?", "FIXME": r"FIXME",
    "[CO-AUTHOR]": r"\[CO-AUTHOR\]", "TBD": r"\bTBD\b",
    "placeholder DOI": r"10\.5281/zenodo\.XXXX|zenodo\.0+\b",
}
hit = False
for name, pat in pats.items():
    for m in re.finditer(pat, tex):
        ln = tex[:m.start()].count("\n") + 1
        ctx = re.sub(r"\s+", " ", tex[max(0, m.start() - 90):m.end() + 90])
        print(f"  line {ln:5d}  [{name}] ...{ctx}...")
        hit = True
if not hit:
    print("  none")

print()
print("=" * 90)
print("KEY SUBMISSION FIELDS")
print("=" * 90)
for label, pat in [
    ("dataset DOI", r"\\newcommand\{\\datasetdoi\}\{([^}]*)\}"),
    ("dataset version", r"\\newcommand\{\\datasetversion\}\{([^}]*)\}"),
    ("review/blinding flag", r"\\newif\\ifreview\s*\\review(true|false)"),
    ("caption-list flag", r"\\newif\\ifcaptionlist\s*\\captionlist(true|false)"),
    ("document class", r"\\documentclass\[([^\]]*)\]"),
]:
    m = re.search(pat, tex)
    print(f"  {label:22s} {m.group(1) if m else 'NOT FOUND'}")

eth = re.search(r"\\section\*?\{Ethics\}(.{0,400})", tex, re.S)
if eth:
    print(f"  ethics statement       {re.sub(r'[[:space:]]+', ' ', re.sub(chr(92)+'s+', ' ', eth.group(1)))[:220]}")

print()
print("=" * 90)
print("TYPESET REFERENCE LIST (from the two-column PDF)")
print("=" * 90)
pdf = MPDA / "CTSpinoPelvic1K_reprint_check.pdf"
d = pymupdf.open(pdf)
txt = "".join(p.get_text() for p in d)
i = txt.find("1J. Lian")
if i < 0:
    i = txt.find("Lian, N. Levine")
body = txt[i:]
body = re.sub(r"-\n", "", body)
body = re.sub(r"\n", " ", body)
refs = re.split(r"(?=(?<![\d.])\b(?:[1-9]|[1-3]\d|40)(?=[A-Z]))", body)
n = 0
for r in refs:
    r = r.strip()
    if len(r) < 30:
        continue
    n += 1
    print(f"  {r[:185]}")
    if n >= 40:
        break
print(f"\n  entries printed: {n}")
