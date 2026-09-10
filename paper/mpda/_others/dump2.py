import re
from pathlib import Path

s = (Path(__file__).resolve().parent.parent / "main.tex").read_text(encoding="utf-8")
body = s[s.index(r"\section{Introduction}"):s.index(r"\begin{thebibliography}")]
body = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", "[FLOAT]", body, flags=re.S)
paras = [p.strip() for p in body.split("\n\n") if p.strip() and not p.lstrip().startswith("%")]
for p in sorted(paras, key=len, reverse=True)[:6]:
    print("=" * 100)
    print(f"[{len(p)} chars]")
    print(p)
