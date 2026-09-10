"""Set Ashley's updated reference list beside the manuscript's, matched on DOI."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPDA = HERE.parent

ash = (HERE / "ashley_refs.txt").read_text(encoding="utf-8")
ash_dois = {}
for para in ash.split("\n\n"):
    m = re.search(r"doi\.org/([^\s]+)", para)
    if m:
        ash_dois[m.group(1).lower().rstrip(".")] = re.sub(r"\s+", " ", para)[:90]

bib = (MPDA / "ctspinopelvic1k.bib").read_text(encoding="utf-8")
mine = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
    d = re.search(r"\bdoi\s*=\s*\{([^}]*)\}", m.group(2))
    t = re.search(r"\btitle\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(2))
    mine[m.group(1)] = ((d.group(1).lower() if d else ""), re.sub(r"\s+", " ", re.sub(r"[{}]", "", t.group(1))) if t else "")

mine_dois = {v[0]: k for k, v in mine.items() if v[0]}

print(f"Ashley: {len(ash_dois)} refs with a DOI   Manuscript: {len(mine)} refs\n")

print("=== in the manuscript but NOT in Ashley's list ===")
for k, (d, t) in mine.items():
    if not d or d not in ash_dois:
        print(f"  {k:18s} {t[:72]}")

print("\n=== in Ashley's list but NOT in the manuscript ===")
for d, txt in ash_dois.items():
    if d not in mine_dois:
        print(f"  {d:42s} {txt[:80]}")
