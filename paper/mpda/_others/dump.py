import re
from pathlib import Path

s = (Path(__file__).resolve().parent.parent / "main.tex").read_text(encoding="utf-8")
body = s[s.index(r"\section{Introduction}"):s.index(r"\begin{thebibliography}")]
body = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", "[FLOAT]", body, flags=re.S)
paras = [p.strip() for p in body.split("\n\n") if p.strip()]
want = ["Existing public collections cannot address this", "The gap was in plain sight",
        "A proxy for the preoperative view", "Comparison with related material",
        "The result is a pseudolabel whose review was triaged", "Future directions"]
for p in paras:
    flat = re.sub(r"\s+", " ", p)
    if any(w in flat for w in want):
        print("=" * 100)
        print(f"[{len(p)} chars]")
        print(p)
