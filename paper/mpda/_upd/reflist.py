"""Reconstruct Ashley's final reference list (insertions kept, deletions dropped) and set it
beside the manuscript's current list."""
import re
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HERE = Path(__file__).resolve().parent


def final_text(p):
    """Text of a paragraph with tracked changes accepted."""
    out = []
    for e in p.iter():
        if e.tag == W + "t":
            parent_is_del = False
            out.append(e.text or "")
        elif e.tag == W + "delText":
            pass  # deletion: drop it
    return "".join(out)


doc = ET.parse(HERE / "word/document.xml").getroot()
paras = [re.sub(r"\s+", " ", final_text(p)).strip() for p in doc.iter(W + "p")]

start = next((i for i, t in enumerate(paras) if t.strip().upper() == "REFERENCES"), None)
print("REFERENCES at paragraph", start)
refs = []
if start is not None:
    for t in paras[start + 1:]:
        if not t:
            continue
        if t.lower().startswith("from mendeley"):
            break
        refs.append(t)

out = []
for r in refs:
    out.append(r)
(HERE / "ashley_refs.txt").write_text("\n\n".join(out), encoding="utf-8")
print(f"{len(refs)} reference paragraphs captured")
for r in refs[:6]:
    print("  ", r[:150])
