import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
s = (HERE / "main.tex").read_text(encoding="utf-8")
live = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))
cited = []
for g in re.findall(r"\\cite\{([^}]*)\}", live):
    for k in g.split(","):
        k = k.strip()
        if k and k not in cited:
            cited.append(k)
bib = set(re.findall(r"^@\w+\{([^,]+),", (HERE / "ctspinopelvic1k.bib").read_text(encoding="utf-8"), re.M))
items = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
print("cited:", len(cited), "| bib entries:", len(bib), "| printed bibitems:", len(items))
print("cited but missing from .bib :", [k for k in cited if k not in bib])
print("cited but not printed       :", [k for k in cited if k not in items])
print("printed but never cited     :", sorted(items - set(cited)))
