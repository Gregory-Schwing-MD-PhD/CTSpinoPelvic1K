"""Print, with character counts, every paragraph that states a limitation, so the
consolidation can be measured rather than guessed."""
import re
from pathlib import Path

s = (Path(__file__).resolve().parent.parent / "main.tex").read_text(encoding="utf-8")
start = s.index(r"\section{Potential Applications and Limitations}")
end = s.index(r"\section*{Data Availability}")
block = s[start:end]
block = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", "[FLOAT]", block, flags=re.S)
paras = [p.strip() for p in block.split("\n\n") if p.strip()]
total = 0
for i, p in enumerate(paras):
    flat = re.sub(r"\s+", " ", p)
    total += len(p)
    print(f"[{i:2d}] {len(p):5d}  {flat[:150]}")
print("\nSection V + Discussion + Conclusion:", total, "chars")
