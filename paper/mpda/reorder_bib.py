r"""reorder_bib.py -- sort the inline thebibliography into order of first citation.
REVTeX numbers \bibitem entries in the order they appear, so the list must follow the text."""
import re
from pathlib import Path

p = Path(__file__).parent / "main.tex"
s = p.read_text(encoding="utf-8")
head, rest = s.split("\\begin{thebibliography}", 1)
open_line, rest = rest.split("\n", 1)
bib_body, tail = rest.split("\\end{thebibliography}", 1)

body_nc = "\n".join(l for l in head.split("\n") if not l.lstrip().startswith("%"))
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body_nc):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in order:
            order.append(k)

items = re.split(r"(?=\\bibitem\{)", bib_body)
pre = items[0]
entries = {re.match(r"\\bibitem\{([a-z0-9]+)\}", it).group(1): it.rstrip() + "\n\n" for it in items[1:]}
missing = [k for k in order if k not in entries]; unused = [k for k in entries if k not in order]
assert not missing and not unused, (missing, unused)
new_body = pre + "".join(entries[k] for k in order)
p.write_text(head + "\\begin{thebibliography}" + open_line + "\n" + new_body + "\\end{thebibliography}" + tail, encoding="utf-8")
print("bibliography reordered:", len(order), "entries")
