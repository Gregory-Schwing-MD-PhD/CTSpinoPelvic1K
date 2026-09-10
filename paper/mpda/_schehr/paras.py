import re
from pathlib import Path

s = (Path(__file__).resolve().parent.parent / "main.tex").read_text(encoding="utf-8")
body = s[s.index(r"\section{Introduction}"):s.index(r"\begin{thebibliography}")]
body = re.sub(r"\\begin\{(figure\*?|table\*?|tikzpicture)\}.*?\\end\{\1\}", "", body, flags=re.S)
paras = [p.strip() for p in body.split("\n\n") if p.strip() and not p.lstrip().startswith("%")]
for p in sorted(paras, key=len, reverse=True)[:14]:
    print(f"{len(p):5d}  {re.sub(chr(92)+'s+', ' ', p)[:120]}")
print("\ntotal body chars:", sum(len(p) for p in paras), "in", len(paras), "paragraphs")
