r"""fix_table_rows.py -- restore the row terminators in the completeness table.

The table was inserted through a shell heredoc and its LaTeX row separators arrived as a
single backslash instead of a double. TeX then read every `\hline` as sitting inside an
unterminated row and raised "Misplaced \noalign" forty-one times, while nonstopmode carried
on and produced a PDF with the table silently broken.

Only lines inside the tabular block are touched, and only where the line ends in exactly one
backslash.
"""
from pathlib import Path

p = Path(__file__).parent / "main.tex"
lines = p.read_text(encoding="utf-8").split("\n")

start = next(i for i, l in enumerate(lines) if l.startswith(r"\begin{tabular}{llrr}"))
end = next(i for i, l in enumerate(lines) if i > start and l.startswith(r"\end{tabular}"))

fixed = 0
for i in range(start + 1, end):
    l = lines[i].rstrip()
    if l.endswith("\\") and not l.endswith("\\\\"):
        lines[i] = l + "\\"
        fixed += 1

p.write_text("\n".join(lines), encoding="utf-8")
print(f"repaired {fixed} row terminators (lines {start+1}-{end})")
